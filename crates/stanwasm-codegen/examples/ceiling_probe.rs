//! What the AOT emitter could reach on a matrix product, if a tape node could
//! hold a vector. Emits, by hand, the wasm the emitter *would* produce for
//! `y ~ normal(X * beta, sigma)` under six designs, so each one's payoff can be
//! measured before any of it is built.
//!
//!   cargo run -p stanwasm-codegen --example ceiling_probe -- 5000 4 target/ceiling
//!   cd ts && node --experimental-strip-types tests/ceiling_probe.ts ../target/ceiling
//!
//! Not a test: it answers one design question and is kept so the numbers in
//! that document can be reproduced.
//!
//!   split_strided  today: `X*beta` and the density are separate re-rolled
//!                  blocks, so four passes over N, and `mu` is written at the
//!                  producing block's stride (2K) and read back at that stride
//!   split_packed   ... with block-owned slots in position-major order, so
//!                  every inter-block read is stride 1
//!   unfused        ... with the two blocks merged into one pass
//!   fused          ... with the mean as one contraction node
//!   fused_loc      ... with the log-density accumulator chain in wasm locals
//!   simd           ... widened to f64x2 over two rows, which needs
//!                  column-major X, privatised beta adjoints and a
//!                  reassociated accumulator
//!
//! Conventions copied from the real emitter so the comparison is fair:
//! loop-invariants (sigma, prefix) are re-read from scratch every iteration,
//! iteration-local values are recomputed in the backward pass and their
//! adjoint locals cleared, anything a later block reads keeps an address, and
//! the adjoint half of the scratch buffer is zeroed per call, sized by the
//! variant's node count.

use wasm_encoder::*;

const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

const P_PARAMS: u32 = 0;
const P_GRADS: u32 = 1;
const P_SCRATCH: u32 = 3;

const F64_BASE: u32 = 4;
const F64_COUNT: u32 = 60;
const I32_BASE: u32 = F64_BASE + F64_COUNT;
const I32_COUNT: u32 = 8;
const V128_BASE: u32 = I32_BASE + I32_COUNT;
const V128_COUNT: u32 = 24;

const L_SIGMA: u32 = F64_BASE;
const L_ACC: u32 = F64_BASE + 2;
const L_MU: u32 = F64_BASE + 3;
const L_T0: u32 = F64_BASE + 6;
const L_T1: u32 = F64_BASE + 7;
const L_DMU: u32 = F64_BASE + 8;
const L_DACC: u32 = F64_BASE + 9;
const L_DSIG: u32 = F64_BASE + 10;
/// The density block's five iteration-local primals, then their adjoints.
const L_R: u32 = F64_BASE + 40;
const L_Z: u32 = F64_BASE + 41;
const L_Z2: u32 = F64_BASE + 42;
const L_H: u32 = F64_BASE + 43;
const L_T: u32 = F64_BASE + 44;
const L_DR: u32 = F64_BASE + 45;
const L_DZ: u32 = F64_BASE + 46;
const L_DZ2: u32 = F64_BASE + 47;
const L_DH: u32 = F64_BASE + 48;
const L_DT: u32 = F64_BASE + 49;
/// 2K primal then 2K adjoint locals for the unfused mean chain.
const L_MEAN: u32 = F64_BASE + 12;

const I_IV: u32 = I32_BASE;
const I_ROW: u32 = I32_BASE + 1;
const I_MU: u32 = I32_BASE + 2;
const I_XC: u32 = I32_BASE + 3;
const I_XR: u32 = I32_BASE + 4;
const I_Y: u32 = I32_BASE + 5;

const W_ACC: u32 = V128_BASE;
const W_SIG: u32 = V128_BASE + 1;
const W_PREFIX: u32 = V128_BASE + 2;
const W_MU: u32 = V128_BASE + 3;
const W_T0: u32 = V128_BASE + 4;
const W_DSIG: u32 = V128_BASE + 6;
const W_DBETA: u32 = V128_BASE + 7;
const W_R: u32 = V128_BASE + 12;
const W_Z: u32 = V128_BASE + 13;
const W_Z2: u32 = V128_BASE + 14;
const W_H: u32 = V128_BASE + 15;
const W_T: u32 = V128_BASE + 16;
const W_DR: u32 = V128_BASE + 17;
const W_DZ: u32 = V128_BASE + 18;
const W_DZ2: u32 = V128_BASE + 19;
const W_DH: u32 = V128_BASE + 20;
const W_DT: u32 = V128_BASE + 21;

#[derive(Clone, Copy)]
struct V {
    name: &'static str,
    /// The mean and the density are separate loops, as separate blocks are today.
    split: bool,
    /// Slot distance between consecutive rows' `mu`: the producing block's
    /// length today, 1 under a position-major layout.
    packed: bool,
    fused: bool,
    acc_local: bool,
    simd: bool,
}

const SCRATCH_ABS: u32 = 65536;

struct Lay {
    mu_stride: u64,
    acc: u64,
    sigma: u64,
    prefix: u64,
    dmu: u64,
    dacc: u64,
    dbeta: u64,
    dsigma: u64,
    dprefix: u64,
    adj: u64,
    adj_len: u64,
    total: u64,
    x_col: u32,
    x_row: u32,
    y: u32,
    end: u32,
}

fn layout(v: V, n: u32, k: u32) -> Lay {
    let (nn, kk) = (n as u64, k as u64);
    let nodes_per_row = if v.fused { 7 } else { 2 * kk + 6 };
    let mu_stride = if v.split && !v.packed { 2 * kk } else { 1 };
    let acc = nn * mu_stride;
    let adj = acc + nn + 4;
    let need = nn * mu_stride + nn + kk + 8;
    let adj_len = (nodes_per_row * nn).max(need);
    let total = adj + adj_len;
    // Data sits past the largest variant's scratch so every variant agrees on
    // where X and y are.
    let data_base = 2 * kk * nn + nn + 4 + (2 * kk + 6) * nn;
    let x_col = SCRATCH_ABS + (data_base * 8) as u32;
    let x_row = x_col + n * k * 8;
    let y = x_row + n * k * 8;
    Lay {
        mu_stride,
        acc,
        sigma: acc + nn + 1,
        prefix: acc + nn + 2,
        dmu: adj,
        dacc: adj + nn * mu_stride,
        dbeta: adj + nn * mu_stride + nn + 1,
        dsigma: adj + nn * mu_stride + nn + 1 + kk,
        dprefix: adj + nn * mu_stride + nn + 2 + kk,
        adj,
        adj_len,
        total,
        x_col,
        x_row,
        y,
        end: y + n * 8,
    }
}

fn ma(slot: u64) -> MemArg {
    MemArg {
        offset: slot * 8,
        align: 3,
        memory_index: 0,
    }
}
fn ma16(slot: u64) -> MemArg {
    MemArg {
        offset: slot * 8,
        align: 4,
        memory_index: 0,
    }
}
fn v2(a: f64, b: f64) -> i128 {
    (((b.to_bits() as u128) << 64) | a.to_bits() as u128) as i128
}

struct E(Function);

#[allow(dead_code)]
impl E {
    fn i(&mut self, x: Instruction) -> &mut Self {
        self.0.instruction(&x);
        self
    }
    fn get(&mut self, l: u32) -> &mut Self {
        self.i(Instruction::LocalGet(l))
    }
    fn set(&mut self, l: u32) -> &mut Self {
        self.i(Instruction::LocalSet(l))
    }
    fn c(&mut self, v: f64) -> &mut Self {
        self.i(Instruction::F64Const(v.into()))
    }
    fn ld(&mut self, p: u32, slot: u64) -> &mut Self {
        self.get(p).i(Instruction::F64Load(ma(slot)))
    }
    fn vld(&mut self, p: u32, slot: u64) -> &mut Self {
        self.get(p).i(Instruction::V128Load(ma16(slot)))
    }
    fn add(&mut self) -> &mut Self {
        self.i(Instruction::F64Add)
    }
    fn sub(&mut self) -> &mut Self {
        self.i(Instruction::F64Sub)
    }
    fn mul(&mut self) -> &mut Self {
        self.i(Instruction::F64Mul)
    }
    fn div(&mut self) -> &mut Self {
        self.i(Instruction::F64Div)
    }
    fn neg(&mut self) -> &mut Self {
        self.i(Instruction::F64Neg)
    }
    fn vadd(&mut self) -> &mut Self {
        self.i(Instruction::F64x2Add)
    }
    fn vsub(&mut self) -> &mut Self {
        self.i(Instruction::F64x2Sub)
    }
    fn vmul(&mut self) -> &mut Self {
        self.i(Instruction::F64x2Mul)
    }
    fn vdiv(&mut self) -> &mut Self {
        self.i(Instruction::F64x2Div)
    }
    fn splat(&mut self) -> &mut Self {
        self.i(Instruction::F64x2Splat)
    }
    fn vconst(&mut self, a: f64, b: f64) -> &mut Self {
        self.i(Instruction::V128Const(v2(a, b)))
    }
    fn accum(&mut self, p: u32, slot: u64, what: impl FnOnce(&mut E)) -> &mut Self {
        self.get(p);
        self.ld(p, slot);
        what(self);
        self.add();
        self.i(Instruction::F64Store(ma(slot)))
    }
    fn store(&mut self, p: u32, slot: u64, what: impl FnOnce(&mut E)) -> &mut Self {
        self.get(p);
        what(self);
        self.i(Instruction::F64Store(ma(slot)))
    }
    fn vstore(&mut self, p: u32, slot: u64, what: impl FnOnce(&mut E)) -> &mut Self {
        self.get(p);
        what(self);
        self.i(Instruction::V128Store(ma16(slot)))
    }
    /// `dst = <absolute base> + iv * scale`
    fn ptr_abs(&mut self, dst: u32, base: u32, scale: u32) -> &mut Self {
        self.i(Instruction::I32Const(base as i32))
            .get(I_IV)
            .i(Instruction::I32Const(scale as i32))
            .i(Instruction::I32Mul)
            .i(Instruction::I32Add)
            .set(dst)
    }
    fn ptr_rel(&mut self, dst: u32, base: u32, scale: u32) -> &mut Self {
        self.get(base)
            .get(I_IV)
            .i(Instruction::I32Const(scale as i32))
            .i(Instruction::I32Mul)
            .i(Instruction::I32Add)
            .set(dst)
    }
    fn loop_over(&mut self, n: u32, step: i32, forward: bool, body: impl FnOnce(&mut E)) {
        self.i(Instruction::I32Const(if forward {
            0
        } else {
            n as i32 - step
        }))
        .set(I_IV)
        .i(Instruction::Loop(BlockType::Empty));
        body(self);
        self.get(I_IV).i(Instruction::I32Const(step));
        if forward {
            self.i(Instruction::I32Add)
                .i(Instruction::LocalTee(I_IV))
                .i(Instruction::I32Const(n as i32))
                .i(Instruction::I32LtU);
        } else {
            self.i(Instruction::I32Sub)
                .i(Instruction::LocalTee(I_IV))
                .i(Instruction::I32Const(0))
                .i(Instruction::I32GeS);
        }
        self.i(Instruction::BrIf(0)).i(Instruction::End);
    }
}

fn build(v: V, n: u32, k: u32) -> (Vec<u8>, Lay) {
    let l = layout(v, n, k);
    let (kk, nn) = (k as u64, n as u64);
    let step = if v.simd { 2 } else { 1 };
    let dadj = L_MEAN + 2 * k;

    let mut types = TypeSection::new();
    types.ty().function([ValType::F64], [ValType::F64]);
    types.ty().function(
        [ValType::I32, ValType::I32, ValType::I32, ValType::I32],
        [ValType::F64],
    );
    let mut imports = ImportSection::new();
    imports.import(
        "stan",
        "memory",
        EntityType::Memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        }),
    );
    imports.import("Math", "exp", EntityType::Function(0));
    imports.import("Math", "log", EntityType::Function(0));
    let mut funcs = FunctionSection::new();
    funcs.function(1);
    let mut exports = ExportSection::new();
    exports.export("log_prob_grad", ExportKind::Func, 2);

    let mut decl = vec![(F64_COUNT, ValType::F64), (I32_COUNT, ValType::I32)];
    if v.simd {
        decl.push((V128_COUNT, ValType::V128));
    }
    let mut e = E(Function::new(decl));

    // ---- prologue ---------------------------------------------------------
    e.ld(P_PARAMS, kk).i(Instruction::Call(0)).set(L_SIGMA);
    e.store(P_SCRATCH, l.sigma, |e| {
        e.get(L_SIGMA);
    });
    e.c(LOG_SQRT_2PI);
    e.get(L_SIGMA).i(Instruction::Call(1));
    e.add().neg().set(L_T0);
    e.store(P_SCRATCH, l.prefix, |e| {
        e.get(L_T0);
    });
    e.ld(P_PARAMS, kk).set(L_ACC);
    for j in 0..kk {
        e.get(L_ACC).c(-LOG_SQRT_2PI);
        e.ld(P_PARAMS, j).ld(P_PARAMS, j).mul().c(-0.5).mul();
        e.add().add().set(L_ACC);
    }
    e.get(L_ACC).get(L_SIGMA).sub().set(L_ACC);
    e.get(P_SCRATCH)
        .i(Instruction::I32Const((l.adj * 8) as i32))
        .i(Instruction::I32Add)
        .i(Instruction::I32Const(0))
        .i(Instruction::I32Const((l.adj_len * 8) as i32))
        .i(Instruction::MemoryFill(0));
    if v.simd {
        e.vconst(0.0, 0.0).set(W_ACC);
        e.ld(P_SCRATCH, l.sigma).splat().set(W_SIG);
        e.ld(P_SCRATCH, l.prefix).splat().set(W_PREFIX);
        e.vconst(0.0, 0.0).set(W_DSIG);
        for j in 0..kk {
            e.vconst(0.0, 0.0).set(W_DBETA + j as u32);
        }
    } else if !v.acc_local {
        e.store(P_SCRATCH, l.acc, |e| {
            e.get(L_ACC);
        });
    }

    // ---- loop bodies ------------------------------------------------------
    let ptrs = |e: &mut E| {
        e.ptr_rel(I_ROW, P_SCRATCH, 8);
        e.ptr_rel(I_MU, P_SCRATCH, 8 * l.mu_stride as u32);
        e.ptr_abs(I_Y, l.y, 8);
        if v.fused && !v.simd {
            e.ptr_abs(I_XR, l.x_row, 8 * k);
        } else {
            e.ptr_abs(I_XC, l.x_col, 8);
        }
    };

    // mean: 2K scalar nodes, or one contraction, into L_MU / W_MU
    let mean = |e: &mut E| {
        if v.simd {
            for j in 0..kk {
                if j > 0 {
                    e.get(W_MU);
                }
                e.vld(I_XC, j * nn);
                e.ld(P_PARAMS, j).splat();
                e.vmul();
                if j > 0 {
                    e.vadd();
                }
                e.set(W_MU);
            }
        } else if v.fused {
            for j in 0..kk {
                e.get(I_XR).i(Instruction::F64Load(ma(j)));
                e.ld(P_PARAMS, j).mul();
                if j > 0 {
                    e.add();
                }
            }
            e.set(L_MU);
        } else {
            for j in 0..kk {
                e.get(I_XC).i(Instruction::F64Load(ma(j * nn)));
                e.ld(P_PARAMS, j).mul().set(L_MEAN + j as u32);
            }
            e.get(L_MEAN).c(0.0).add().set(L_MEAN + k);
            for j in 1..k {
                e.get(L_MEAN + k + j - 1)
                    .get(L_MEAN + j)
                    .add()
                    .set(L_MEAN + k + j);
            }
            e.get(L_MEAN + 2 * k - 1).set(L_MU);
        }
    };

    let mean_store = |e: &mut E| {
        if v.simd {
            e.vstore(I_MU, 0, |e| {
                e.get(W_MU);
            });
        } else {
            e.store(I_MU, 0, |e| {
                e.get(L_MU);
            });
        }
    };

    // One local.set per tape node, as the emitter emits it.
    let density = |e: &mut E| {
        if v.simd {
            e.vld(I_Y, 0).get(W_MU).vsub().set(W_R);
            e.get(W_R).get(W_SIG).vdiv().set(W_Z);
            e.get(W_Z).get(W_Z).vmul().set(W_Z2);
            e.get(W_Z2).vconst(-0.5, -0.5).vmul().set(W_H);
            e.get(W_PREFIX).get(W_H).vadd().set(W_T);
            e.get(W_ACC).get(W_T).vadd().set(W_ACC);
        } else {
            e.get(I_Y)
                .i(Instruction::F64Load(ma(0)))
                .get(L_MU)
                .sub()
                .set(L_R);
            e.get(L_R).ld(P_SCRATCH, l.sigma).div().set(L_Z);
            e.get(L_Z).get(L_Z).mul().set(L_Z2);
            e.get(L_Z2).c(-0.5).mul().set(L_H);
            e.ld(P_SCRATCH, l.prefix).get(L_H).add().set(L_T);
            if v.acc_local {
                e.get(L_ACC).get(L_T).add().set(L_ACC);
            } else {
                e.store(I_ROW, l.acc + 1, |e| {
                    e.get(I_ROW).i(Instruction::F64Load(ma(l.acc)));
                    e.get(L_T);
                    e.add();
                });
            }
        }
    };

    // Density backward, node by node: recompute the block's iteration-local
    // primals, clear their adjoint locals, then one step per node in reverse.
    // Leaves d_mu in L_DMU / W_T0.
    let density_back = |e: &mut E| {
        if v.simd {
            e.vld(I_MU, 0).set(W_MU);
            e.vld(I_Y, 0).get(W_MU).vsub().set(W_R);
            e.get(W_R).get(W_SIG).vdiv().set(W_Z);
            e.get(W_Z).get(W_Z).vmul().set(W_Z2);
            e.get(W_Z2).vconst(-0.5, -0.5).vmul().set(W_H);
            e.get(W_PREFIX).get(W_H).vadd().set(W_T);
            for w in [W_DR, W_DZ, W_DZ2, W_DH, W_DT] {
                e.vconst(0.0, 0.0).set(w);
            }
            e.get(W_DT).vconst(1.0, 1.0).vadd().set(W_DT);
            e.get(W_DH).get(W_DT).vadd().set(W_DH);
            e.get(W_DZ2)
                .get(W_DH)
                .vconst(-0.5, -0.5)
                .vmul()
                .vadd()
                .set(W_DZ2);
            e.get(W_DZ).get(W_DZ2).get(W_Z).vmul().vadd().set(W_DZ);
            e.get(W_DZ).get(W_DZ2).get(W_Z).vmul().vadd().set(W_DZ);
            e.get(W_DR).get(W_DZ).get(W_SIG).vdiv().vadd().set(W_DR);
            e.get(W_DSIG);
            e.get(W_DZ)
                .get(W_R)
                .vmul()
                .get(W_SIG)
                .get(W_SIG)
                .vmul()
                .vdiv();
            e.vsub().set(W_DSIG);
            e.get(W_DR).vconst(-1.0, -1.0).vmul().set(W_T0);
        } else {
            e.get(I_MU).i(Instruction::F64Load(ma(0))).set(L_MU);
            e.get(I_Y)
                .i(Instruction::F64Load(ma(0)))
                .get(L_MU)
                .sub()
                .set(L_R);
            e.get(L_R).ld(P_SCRATCH, l.sigma).div().set(L_Z);
            e.get(L_Z).get(L_Z).mul().set(L_Z2);
            e.get(L_Z2).c(-0.5).mul().set(L_H);
            e.ld(P_SCRATCH, l.prefix).get(L_H).add().set(L_T);
            for w in [L_DR, L_DZ, L_DZ2, L_DH, L_DT] {
                e.c(0.0).set(w);
            }
            if v.acc_local {
                e.c(1.0).set(L_DACC);
            } else {
                e.get(I_ROW)
                    .i(Instruction::F64Load(ma(l.dacc + 1)))
                    .set(L_DACC);
                e.accum(I_ROW, l.dacc, |e| {
                    e.get(L_DACC);
                });
            }
            e.get(L_DT).get(L_DACC).add().set(L_DT);
            e.accum(P_SCRATCH, l.dprefix, |e| {
                e.get(L_DT);
            });
            e.get(L_DH).get(L_DT).add().set(L_DH);
            e.get(L_DZ2).get(L_DH).c(-0.5).mul().add().set(L_DZ2);
            e.get(L_DZ).get(L_DZ2).get(L_Z).mul().add().set(L_DZ);
            e.get(L_DZ).get(L_DZ2).get(L_Z).mul().add().set(L_DZ);
            e.get(L_DR)
                .get(L_DZ)
                .ld(P_SCRATCH, l.sigma)
                .div()
                .add()
                .set(L_DR);
            e.get(P_SCRATCH);
            e.ld(P_SCRATCH, l.dsigma);
            e.get(L_DZ).get(L_R).mul();
            e.ld(P_SCRATCH, l.sigma).ld(P_SCRATCH, l.sigma).mul();
            e.div().sub();
            e.i(Instruction::F64Store(ma(l.dsigma)));
            e.get(L_DR).neg().set(L_DMU);
        }
    };

    // mean backward: consumes d_mu from L_DMU / W_T0
    let mean_back = |e: &mut E| {
        if v.simd {
            for j in 0..kk {
                e.get(W_DBETA + j as u32);
                e.get(W_T0).vld(I_XC, j * nn).vmul();
                e.vadd().set(W_DBETA + j as u32);
            }
        } else if v.fused {
            for j in 0..kk {
                e.accum(P_SCRATCH, l.dbeta + j, |e| {
                    e.get(L_DMU).get(I_XR).i(Instruction::F64Load(ma(j))).mul();
                });
            }
        } else {
            for j in 0..kk {
                e.get(I_XC).i(Instruction::F64Load(ma(j * nn)));
                e.ld(P_PARAMS, j).mul().set(L_MEAN + j as u32);
            }
            e.get(L_MEAN).c(0.0).add().set(L_MEAN + k);
            for j in 1..k {
                e.get(L_MEAN + k + j - 1)
                    .get(L_MEAN + j)
                    .add()
                    .set(L_MEAN + k + j);
            }
            for j in 0..2 * k {
                e.c(0.0).set(dadj + j);
            }
            e.get(dadj + 2 * k - 1)
                .get(L_DMU)
                .add()
                .set(dadj + 2 * k - 1);
            for j in (1..k).rev() {
                e.get(dadj + k + j - 1)
                    .get(dadj + k + j)
                    .add()
                    .set(dadj + k + j - 1);
                e.get(dadj + j).get(dadj + k + j).add().set(dadj + j);
            }
            e.get(dadj).get(dadj + k).add().set(dadj);
            for j in 0..kk {
                e.accum(P_SCRATCH, l.dbeta + j, |e| {
                    e.get(dadj + j as u32)
                        .get(I_XC)
                        .i(Instruction::F64Load(ma(j * nn)))
                        .mul();
                });
            }
        }
    };

    // ---- forward ----------------------------------------------------------
    if v.split {
        e.loop_over(n, step, true, |e| {
            ptrs(e);
            mean(e);
            mean_store(e);
        });
        e.loop_over(n, step, true, |e| {
            ptrs(e);
            if v.simd {
                e.vld(I_MU, 0).set(W_MU);
            } else {
                e.get(I_MU).i(Instruction::F64Load(ma(0))).set(L_MU);
            }
            density(e);
        });
    } else {
        e.loop_over(n, step, true, |e| {
            ptrs(e);
            mean(e);
            mean_store(e);
            density(e);
        });
    }

    if !(v.acc_local || v.simd) {
        e.store(P_SCRATCH, l.dacc + nn, |e| {
            e.c(1.0);
        });
    }

    // ---- backward ---------------------------------------------------------
    if v.split {
        e.loop_over(n, step, false, |e| {
            ptrs(e);
            density_back(e);
            e.accum(I_MU, l.dmu, |e| {
                e.get(L_DMU);
            });
        });
        e.loop_over(n, step, false, |e| {
            ptrs(e);
            e.get(I_MU).i(Instruction::F64Load(ma(l.dmu))).set(L_DMU);
            mean_back(e);
        });
    } else {
        e.loop_over(n, step, false, |e| {
            ptrs(e);
            density_back(e);
            mean_back(e);
        });
    }

    // ---- epilogue ---------------------------------------------------------
    if v.simd {
        for j in 0..kk {
            e.store(P_SCRATCH, l.dbeta + j, |e| {
                e.get(W_DBETA + j as u32)
                    .i(Instruction::F64x2ExtractLane(0));
                e.get(W_DBETA + j as u32)
                    .i(Instruction::F64x2ExtractLane(1));
                e.add();
            });
        }
        e.store(P_SCRATCH, l.dsigma, |e| {
            e.get(W_DSIG).i(Instruction::F64x2ExtractLane(0));
            e.get(W_DSIG).i(Instruction::F64x2ExtractLane(1));
            e.add();
        });
        e.store(P_SCRATCH, l.dprefix, |e| {
            e.c(n as f64);
        });
        e.get(L_ACC);
        e.get(W_ACC).i(Instruction::F64x2ExtractLane(0));
        e.get(W_ACC).i(Instruction::F64x2ExtractLane(1));
        e.add().add().set(L_ACC);
    }
    if v.acc_local || v.simd {
        e.c(1.0).set(L_T0);
    } else {
        e.ld(P_SCRATCH, l.dacc).set(L_T0);
    }
    e.ld(P_SCRATCH, l.dsigma).set(L_DSIG);
    e.get(L_DSIG).get(L_T0).sub().set(L_DSIG);
    e.get(L_DSIG);
    e.ld(P_SCRATCH, l.dprefix).neg().get(L_SIGMA).div();
    e.add().set(L_DSIG);
    e.get(L_T0).get(L_DSIG).get(L_SIGMA).mul().add().set(L_T1);
    for j in 0..kk {
        e.get(P_GRADS);
        e.ld(P_SCRATCH, l.dbeta + j);
        e.ld(P_PARAMS, j).neg().get(L_T0).mul();
        e.add();
        e.i(Instruction::F64Store(ma(j)));
    }
    e.get(P_GRADS).get(L_T1).i(Instruction::F64Store(ma(kk)));
    if v.acc_local || v.simd {
        e.get(L_ACC);
    } else {
        e.ld(P_SCRATCH, l.acc + nn);
    }
    e.i(Instruction::End);

    let mut code = CodeSection::new();
    code.function(&e.0);
    let mut m = Module::new();
    m.section(&types);
    m.section(&imports);
    m.section(&funcs);
    m.section(&exports);
    m.section(&code);
    (m.finish(), l)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: u32 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(5000);
    let k: u32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(4);
    let out = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "target/ceiling".into());
    std::fs::create_dir_all(&out).unwrap();
    assert!(n.is_multiple_of(2), "the widened loop has no scalar tail");

    let base = V {
        name: "",
        split: false,
        packed: false,
        fused: false,
        acc_local: false,
        simd: false,
    };
    let variants = [
        V {
            name: "split_strided",
            split: true,
            ..base
        },
        V {
            name: "split_packed",
            split: true,
            packed: true,
            ..base
        },
        V {
            name: "unfused",
            ..base
        },
        V {
            name: "fused",
            fused: true,
            ..base
        },
        V {
            name: "fused_loc",
            fused: true,
            acc_local: true,
            ..base
        },
        V {
            name: "simd",
            fused: true,
            acc_local: true,
            simd: true,
            ..base
        },
    ];
    let mut meta = String::from("{\n  \"variants\": {\n");
    let mut end = 0;
    for v in variants {
        let (bytes, l) = build(v, n, k);
        std::fs::write(format!("{out}/{}.wasm", v.name), &bytes).unwrap();
        meta += &format!(
            "    \"{}\": {{ \"bytes\": {}, \"slots\": {} }},\n",
            v.name,
            bytes.len(),
            l.total
        );
        end = end.max(l.end);
        assert_eq!(
            l.x_col,
            layout(base, n, k).x_col,
            "variants disagree on the data base"
        );
    }
    meta.truncate(meta.trim_end().len() - 1);
    let l = layout(base, n, k);
    meta += &format!(
        "\n  }},\n  \"scratch\": {SCRATCH_ABS}, \"x_col\": {}, \"x_row\": {}, \"y\": {}, \"end\": {end}, \"n\": {n}, \"k\": {k}\n}}\n",
        l.x_col, l.x_row, l.y
    );
    std::fs::write(format!("{out}/meta.json"), &meta).unwrap();
    print!("{meta}");
}
