//! Reverse-mode autodiff with a flat (struct-of-arrays) tape.
//!
//! The tape is owned by a `Tape` struct rather than living in module-global
//! state. Callers (stanwasm-runtime, stanwasm-codegen) instantiate one tape per
//! model compilation. The tape is used at compile time to build the
//! computation graph; it is NOT used in the per-draw sampling hot path —
//! that is replaced by AOT-compiled model wasm.

#![forbid(unsafe_code)]

use std::f64::consts::PI;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Leaf = 0,
    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
    Neg = 5,
    Exp = 6,
    Log = 7,
    Sin = 8,
    Cos = 9,
    Sqrt = 10,
    Pow = 11,
    Abs = 12,
    Lgamma = 13,
    AddC = 14,
    SubC = 15,
    RsubC = 16,
    MulC = 17,
    DivC = 18,
    RdivC = 19,
    Phi = 20,
    Erf = 21,
    Erfc = 22,
    Tan = 23,
    Asin = 24,
    Acos = 25,
    Atan = 26,
    Digamma = 27,
}


/// Value numbering key: opcode plus both arguments. Two nodes with the same
/// key compute the same number on a tape that never mutates a value, so the
/// second can reuse the first.
type Vn = (u8, u32, u32, u64);

/// A cheap, deterministic hasher. The default one pulls SipHash into the wasm
/// bundle for no benefit here, and the table must be exact rather than
/// direct-mapped: a loop-invariant subexpression shared by only some elements
/// of a vectorised statement would leave the recorded blocks unequal, and the
/// AOT emitter could no longer re-roll them.
#[derive(Default)]
pub struct VnHasher(u64);

impl std::hash::Hasher for VnHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 = (self.0 ^ *b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
    fn write_u32(&mut self, v: u32) {
        self.0 = (self.0 ^ v as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0 ^ v).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    fn write_u8(&mut self, v: u8) {
        self.write_u32(v as u32);
    }
}

type VnMap = std::collections::HashMap<Vn, u32, std::hash::BuildHasherDefault<VnHasher>>;
const TAPE_DEFAULT_CAP: usize = 65536;

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
const TWO_OVER_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

pub struct Tape {
    val: Vec<f64>,
    grad: Vec<f64>,
    op: Vec<Op>,
    arg1: Vec<u32>,
    arg2i: Vec<u32>,
    arg2f: Vec<f64>,

    /// Common-subexpression table. A vectorised statement re-records every
    /// loop-invariant term once per element without it — `student_t` over
    /// `vector[N]` recorded `lgamma(nu/2)` N times.
    vn: VnMap,
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

impl Tape {
    pub fn new() -> Self {
        Self {
            val: Vec::with_capacity(TAPE_DEFAULT_CAP),
            grad: Vec::with_capacity(TAPE_DEFAULT_CAP),
            op: Vec::with_capacity(TAPE_DEFAULT_CAP),
            arg1: Vec::with_capacity(TAPE_DEFAULT_CAP),
            arg2i: Vec::with_capacity(TAPE_DEFAULT_CAP),
            arg2f: Vec::with_capacity(TAPE_DEFAULT_CAP),
            vn: VnMap::default(),
        }
    }

    pub fn reset(&mut self) {
        self.val.clear();
        self.grad.clear();
        self.op.clear();
        self.arg1.clear();
        self.arg2i.clear();
        self.arg2f.clear();
        self.vn.clear();
    }

    /// Reset every gradient to zero. Used between consecutive `backward()`
    /// calls on the same recorded tape (the replay path used at sampling time).
    pub fn reset_grads(&mut self) {
        for g in self.grad.iter_mut() {
            *g = 0.0;
        }
    }

    /// Replay the forward pass with new leaf values: the first `params.len()` leaves
    /// take them, later ones stay. Valid only if the graph is parameter-independent.
    pub fn forward_replay(&mut self, params: &[f64]) {
        for (i, p) in params.iter().enumerate() {
            self.val[i] = *p;
        }
        for k in params.len()..self.val.len() {
            let op = self.op[k];
            let a1 = self.arg1[k] as usize;
            let a2i = self.arg2i[k] as usize;
            let a2f = self.arg2f[k];
            self.val[k] = match op {
                Op::Leaf => self.val[k],
                Op::Add => self.val[a1] + self.val[a2i],
                Op::Sub => self.val[a1] - self.val[a2i],
                Op::Mul => self.val[a1] * self.val[a2i],
                Op::Div => self.val[a1] / self.val[a2i],
                Op::Neg => -self.val[a1],
                Op::Exp => self.val[a1].exp(),
                Op::Log => self.val[a1].ln(),
                Op::Sin => self.val[a1].sin(),
                Op::Cos => self.val[a1].cos(),
                Op::Sqrt => self.val[a1].sqrt(),
                Op::Pow => self.val[a1].powf(a2f),
                Op::Abs => self.val[a1].abs(),
                Op::Lgamma => lgamma(self.val[a1]),
                Op::AddC => self.val[a1] + a2f,
                Op::SubC => self.val[a1] - a2f,
                Op::RsubC => a2f - self.val[a1],
                Op::MulC => self.val[a1] * a2f,
                Op::DivC => self.val[a1] / a2f,
                Op::RdivC => a2f / self.val[a1],
                Op::Phi => phi_cdf(self.val[a1]),
                Op::Erf => erf(self.val[a1]),
                Op::Erfc => 1.0 - erf(self.val[a1]),
                Op::Tan => self.val[a1].tan(),
                Op::Asin => self.val[a1].asin(),
                Op::Acos => self.val[a1].acos(),
                Op::Atan => self.val[a1].atan(),
                Op::Digamma => digamma(self.val[a1]),
            };
        }
    }

    pub fn len(&self) -> usize {
        self.val.len()
    }

    pub fn is_empty(&self) -> bool {
        self.val.is_empty()
    }

    fn push(&mut self, v: f64, op: Op, a1: u32, a2i: u32, a2f: f64) -> u32 {
        // A leaf is a fresh input even when it repeats a value, so it is the
        // one thing never shared.
        if op != Op::Leaf {
            let key: Vn = (op as u8, a1, a2i, a2f.to_bits());
            if let Some(&i) = self.vn.get(&key) {
                return i;
            }
            let idx = self.push_raw(v, op, a1, a2i, a2f);
            self.vn.insert(key, idx);
            return idx;
        }
        self.push_raw(v, op, a1, a2i, a2f)
    }

    fn push_raw(&mut self, v: f64, op: Op, a1: u32, a2i: u32, a2f: f64) -> u32 {
        let idx = self.val.len() as u32;
        self.val.push(v);
        self.grad.push(0.0);
        self.op.push(op);
        self.arg1.push(a1);
        self.arg2i.push(a2i);
        self.arg2f.push(a2f);
        idx
    }

    // ---- node creation ----

    pub fn new_var(&mut self, v: f64) -> u32 {
        self.push(v, Op::Leaf, 0, 0, 0.0)
    }

    pub fn value(&self, i: u32) -> f64 {
        self.val[i as usize]
    }

    pub fn grad_at(&self, i: u32) -> f64 {
        self.grad[i as usize]
    }

    pub fn add(&mut self, a: u32, b: u32) -> u32 {
        let v = self.val[a as usize] + self.val[b as usize];
        self.push(v, Op::Add, a, b, 0.0)
    }

    pub fn sub(&mut self, a: u32, b: u32) -> u32 {
        let v = self.val[a as usize] - self.val[b as usize];
        self.push(v, Op::Sub, a, b, 0.0)
    }

    pub fn mul(&mut self, a: u32, b: u32) -> u32 {
        let v = self.val[a as usize] * self.val[b as usize];
        self.push(v, Op::Mul, a, b, 0.0)
    }

    pub fn div(&mut self, a: u32, b: u32) -> u32 {
        let v = self.val[a as usize] / self.val[b as usize];
        self.push(v, Op::Div, a, b, 0.0)
    }

    pub fn neg(&mut self, a: u32) -> u32 {
        let v = -self.val[a as usize];
        self.push(v, Op::Neg, a, 0, 0.0)
    }

    pub fn exp(&mut self, a: u32) -> u32 {
        let v = self.val[a as usize].exp();
        self.push(v, Op::Exp, a, 0, 0.0)
    }

    pub fn log(&mut self, a: u32) -> u32 {
        let v = self.val[a as usize].ln();
        self.push(v, Op::Log, a, 0, 0.0)
    }

    pub fn sin(&mut self, a: u32) -> u32 {
        let v = self.val[a as usize].sin();
        self.push(v, Op::Sin, a, 0, 0.0)
    }

    pub fn cos(&mut self, a: u32) -> u32 {
        let v = self.val[a as usize].cos();
        self.push(v, Op::Cos, a, 0, 0.0)
    }

    pub fn sqrt(&mut self, a: u32) -> u32 {
        let v = self.val[a as usize].sqrt();
        self.push(v, Op::Sqrt, a, 0, 0.0)
    }

    pub fn pow(&mut self, a: u32, n: f64) -> u32 {
        let v = self.val[a as usize].powf(n);
        self.push(v, Op::Pow, a, 0, n)
    }

    pub fn abs(&mut self, a: u32) -> u32 {
        let v = self.val[a as usize].abs();
        self.push(v, Op::Abs, a, 0, 0.0)
    }

    pub fn lgamma(&mut self, a: u32) -> u32 {
        let v = lgamma(self.val[a as usize]);
        self.push(v, Op::Lgamma, a, 0, 0.0)
    }

    pub fn phi(&mut self, a: u32) -> u32 {
        let v = phi_cdf(self.val[a as usize]);
        self.push(v, Op::Phi, a, 0, 0.0)
    }

    pub fn erf(&mut self, a: u32) -> u32 {
        let v = erf(self.val[a as usize]);
        self.push(v, Op::Erf, a, 0, 0.0)
    }

    pub fn erfc(&mut self, a: u32) -> u32 {
        let v = 1.0 - erf(self.val[a as usize]);
        self.push(v, Op::Erfc, a, 0, 0.0)
    }

    pub fn tan(&mut self, a: u32) -> u32 {
        let v = self.val[a as usize].tan();
        self.push(v, Op::Tan, a, 0, 0.0)
    }

    pub fn asin(&mut self, a: u32) -> u32 {
        let v = self.val[a as usize].asin();
        self.push(v, Op::Asin, a, 0, 0.0)
    }

    pub fn acos(&mut self, a: u32) -> u32 {
        let v = self.val[a as usize].acos();
        self.push(v, Op::Acos, a, 0, 0.0)
    }

    pub fn atan(&mut self, a: u32) -> u32 {
        let v = self.val[a as usize].atan();
        self.push(v, Op::Atan, a, 0, 0.0)
    }

    pub fn digamma(&mut self, a: u32) -> u32 {
        let v = digamma(self.val[a as usize]);
        self.push(v, Op::Digamma, a, 0, 0.0)
    }

    // ---- scalar-constant variants (avoid creating leaf nodes for constants) ----

    pub fn add_c(&mut self, a: u32, c: f64) -> u32 {
        let v = self.val[a as usize] + c;
        self.push(v, Op::AddC, a, 0, c)
    }

    pub fn sub_c(&mut self, a: u32, c: f64) -> u32 {
        let v = self.val[a as usize] - c;
        self.push(v, Op::SubC, a, 0, c)
    }

    pub fn rsub_c(&mut self, a: u32, c: f64) -> u32 {
        let v = c - self.val[a as usize];
        self.push(v, Op::RsubC, a, 0, c)
    }

    pub fn mul_c(&mut self, a: u32, c: f64) -> u32 {
        let v = self.val[a as usize] * c;
        self.push(v, Op::MulC, a, 0, c)
    }

    pub fn div_c(&mut self, a: u32, c: f64) -> u32 {
        let v = self.val[a as usize] / c;
        self.push(v, Op::DivC, a, 0, c)
    }

    pub fn rdiv_c(&mut self, a: u32, c: f64) -> u32 {
        let v = c / self.val[a as usize];
        self.push(v, Op::RdivC, a, 0, c)
    }

    // ---- backward pass ----

    pub fn backward(&mut self, root: u32) {
        self.grad[root as usize] = 1.0;
        let mut i = self.val.len();
        while i > 0 {
            i -= 1;
            let g = self.grad[i];
            let op = self.op[i];
            let a1 = self.arg1[i] as usize;
            let a2i = self.arg2i[i] as usize;
            let a2f = self.arg2f[i];

            match op {
                Op::Leaf => {}
                Op::Add => {
                    self.grad[a1] += g;
                    self.grad[a2i] += g;
                }
                Op::Sub => {
                    self.grad[a1] += g;
                    self.grad[a2i] -= g;
                }
                Op::Mul => {
                    let va = self.val[a1];
                    let vb = self.val[a2i];
                    self.grad[a1] += g * vb;
                    self.grad[a2i] += g * va;
                }
                Op::Div => {
                    let va = self.val[a1];
                    let vb = self.val[a2i];
                    self.grad[a1] += g / vb;
                    self.grad[a2i] -= g * va / (vb * vb);
                }
                Op::Neg => {
                    self.grad[a1] -= g;
                }
                Op::Exp => {
                    self.grad[a1] += g * self.val[i];
                }
                Op::Log => {
                    self.grad[a1] += g / self.val[a1];
                }
                Op::Sin => {
                    self.grad[a1] += g * self.val[a1].cos();
                }
                Op::Cos => {
                    self.grad[a1] -= g * self.val[a1].sin();
                }
                Op::Sqrt => {
                    self.grad[a1] += g / (2.0 * self.val[i]);
                }
                Op::Pow => {
                    let va = self.val[a1];
                    self.grad[a1] += g * a2f * va.powf(a2f - 1.0);
                }
                Op::Abs => {
                    let sign = if self.val[a1] >= 0.0 { 1.0 } else { -1.0 };
                    self.grad[a1] += g * sign;
                }
                Op::Lgamma => {
                    self.grad[a1] += g * digamma(self.val[a1]);
                }
                Op::AddC | Op::SubC => {
                    self.grad[a1] += g;
                }
                Op::RsubC => {
                    self.grad[a1] -= g;
                }
                Op::MulC => {
                    self.grad[a1] += g * a2f;
                }
                Op::DivC => {
                    self.grad[a1] += g / a2f;
                }
                Op::RdivC => {
                    let va = self.val[a1];
                    self.grad[a1] -= g * a2f / (va * va);
                }
                Op::Phi => {
                    let x = self.val[a1];
                    self.grad[a1] += g * INV_SQRT_2PI * (-0.5 * x * x).exp();
                }
                Op::Erf => {
                    let x = self.val[a1];
                    self.grad[a1] += g * TWO_OVER_SQRT_PI * (-(x * x)).exp();
                }
                Op::Erfc => {
                    let x = self.val[a1];
                    self.grad[a1] -= g * TWO_OVER_SQRT_PI * (-(x * x)).exp();
                }
                Op::Tan => {
                    // d/dx tan(x) = sec²(x) = 1 + tan²(x)
                    let t = self.val[i];
                    self.grad[a1] += g * (1.0 + t * t);
                }
                Op::Asin => {
                    let x = self.val[a1];
                    self.grad[a1] += g / (1.0 - x * x).sqrt();
                }
                Op::Acos => {
                    let x = self.val[a1];
                    self.grad[a1] -= g / (1.0 - x * x).sqrt();
                }
                Op::Atan => {
                    let x = self.val[a1];
                    self.grad[a1] += g / (1.0 + x * x);
                }
                Op::Digamma => {
                    self.grad[a1] += g * trigamma(self.val[a1]);
                }
            }
        }
    }

    // ---- introspection (used by codegen to walk the recorded tape) ----

    pub fn op_at(&self, i: u32) -> Op {
        self.op[i as usize]
    }
    pub fn arg1_at(&self, i: u32) -> u32 {
        self.arg1[i as usize]
    }
    pub fn arg2i_at(&self, i: u32) -> u32 {
        self.arg2i[i as usize]
    }
    pub fn arg2f_at(&self, i: u32) -> f64 {
        self.arg2f[i as usize]
    }
}

// ---- special functions (free standing — also used as primal helpers) ----

/// Stirling-series log-Gamma.
pub fn lgamma(x: f64) -> f64 {
    let mut z = x;
    let mut r = 0.0;
    while z < 10.0 {
        r -= z.ln();
        z += 1.0;
    }
    let zinv = 1.0 / z;
    let zinv2 = zinv * zinv;
    r + (z - 0.5) * z.ln() - z
        + 0.5 * (2.0 * PI).ln()
        + zinv * (1.0 / 12.0 + zinv2 * (-1.0 / 360.0 + zinv2 / 1260.0))
}

/// Asymptotic-series digamma (Ψ).
pub fn digamma(x: f64) -> f64 {
    let mut xx = x;
    let mut result = 0.0;
    while xx < 6.0 {
        result -= 1.0 / xx;
        xx += 1.0;
    }
    let r = 1.0 / xx;
    let r2 = r * r;
    result + xx.ln() - 0.5 * r - r2 * (1.0 / 12.0 - r2 * (1.0 / 120.0 - r2 / 252.0))
}

/// Asymptotic-series trigamma (Ψ').
pub fn trigamma(x: f64) -> f64 {
    let mut xx = x;
    let mut result = 0.0;
    while xx < 6.0 {
        result += 1.0 / (xx * xx);
        xx += 1.0;
    }
    let r = 1.0 / xx;
    let r2 = r * r;
    result + r + 0.5 * r2 + r2 * r * (1.0 / 6.0 - r2 * (1.0 / 30.0 - r2 / 42.0))
}

/// Standard normal CDF (Abramowitz & Stegun 26.2.17).
pub fn phi_cdf(x: f64) -> f64 {
    if x >= 0.0 {
        let t = 1.0 / (1.0 + 0.231_641_9 * x);
        let poly = t
            * (0.319_381_530
                + t * (-0.356_563_782
                    + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
        1.0 - INV_SQRT_2PI * (-0.5 * x * x).exp() * poly
    } else {
        1.0 - phi_cdf(-x)
    }
}

/// Error function (Abramowitz & Stegun 7.1.26, max error ~1.5e-7).
pub fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-(ax * ax)).exp())
}

/// Convenience: trace `f` on a fresh tape, return (value, gradient).
pub fn log_prob_grad<F>(params: &[f64], f: F) -> (f64, Vec<f64>)
where
    F: FnOnce(&mut Tape, &[u32]) -> u32,
{
    let mut tape = Tape::new();
    let xs: Vec<u32> = params.iter().map(|v| tape.new_var(*v)).collect();
    let root = f(&mut tape, &xs);
    tape.backward(root);
    let lp = tape.value(root);
    let grads: Vec<f64> = xs.iter().map(|i| tape.grad_at(*i)).collect();
    (lp, grads)
}
