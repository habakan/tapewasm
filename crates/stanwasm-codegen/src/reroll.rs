//! Finding the loops a vectorised Stan statement leaves on the tape.
//!
//! Tracing `y ~ normal(alpha + beta * x, sigma)` over `vector[N]` records the
//! same handful of ops N times over, so the straight-line emitter produces a
//! function that grows with the data. Past a few thousand nodes V8 stops
//! optimising it and the AOT path loses to plain tape replay.
//!
//! The recorded shape is very regular: every integer argument of the k-th copy
//! is `arg0 + k * stride` for a stride fixed per argument. Intra-block
//! references have stride `len`, a strided read into a vector produced earlier
//! has stride 1, and anything loop-invariant (a shared `sigma`, a CSE'd
//! `log(sigma)`) has stride 0. Detection is therefore just: same opcodes, and
//! every integer argument affine in the repeat index.

use stanwasm_autodiff::{Op, Tape};

/// Longest block considered. A vectorised density is well under this; the cap
/// keeps detection from going quadratic on a tape with no structure.
const MAX_BLOCK: u32 = 96;

/// Fewer repeats than this is not worth a loop: the prologue and induction
/// variable cost more than the unrolled copies.
const MIN_REPS: u32 = 4;

/// A run of `reps` identical blocks of `len` nodes starting at `start`.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub start: u32,
    pub len: u32,
    pub reps: u32,
    /// Per-node argument strides, `len` entries in block order.
    pub strides: Vec<ArgStrides>,
    /// Per-node f64 argument: `None` if constant across repeats, else the
    /// `reps` values in order.
    pub consts: Vec<Option<Vec<f64>>>,
}

impl Block {
    /// One past the last node covered.
    pub fn end(&self) -> u32 {
        self.start + self.len * self.reps
    }
}

/// How a node's integer arguments move from one repeat to the next.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArgStrides {
    pub arg1: u32,
    pub arg2i: u32,
}

/// Which arguments an opcode actually reads. Unused slots hold stale values on
/// the tape, so comparing them would reject blocks that are in fact identical.
pub(crate) fn uses(op: Op) -> (bool, bool, bool) {
    match op {
        Op::Leaf => (false, false, true),
        Op::Add | Op::Sub | Op::Mul | Op::Div => (true, true, false),
        Op::AddC
        | Op::SubC
        | Op::RsubC
        | Op::MulC
        | Op::DivC
        | Op::RdivC
        | Op::Pow => (true, false, true),
        _ => (true, false, false),
    }
}

/// The stride between `a` and `b` if one exists. Arguments only ever move
/// forward, and a repeat cannot reference a node it has not reached yet.
fn stride_of(a: u32, b: u32) -> Option<u32> {
    b.checked_sub(a)
}

/// Repeat count, per-node strides, and per-node moving constants.
type Probe = (u32, Vec<ArgStrides>, Vec<Option<Vec<f64>>>);

/// Check that `len`-node blocks at `start` repeat, and collect the strides.
/// Returns the largest repeat count of at least two, or `None`.
fn probe(tape: &Tape, start: u32, len: u32) -> Option<Probe> {
    let n = tape.len() as u32;
    if start + 2 * len > n {
        return None;
    }
    // Strides come from the first two copies, then every later copy is checked
    // against them: two copies can agree by accident, a hundred cannot.
    let mut strides = Vec::with_capacity(len as usize);
    for j in 0..len {
        let (k0, k1) = (start + j, start + len + j);
        if tape.op_at(k0) != tape.op_at(k1) {
            return None;
        }
        let (u1, u2, _) = uses(tape.op_at(k0));
        let arg1 = if u1 {
            stride_of(tape.arg1_at(k0), tape.arg1_at(k1))?
        } else {
            0
        };
        let arg2i = if u2 {
            stride_of(tape.arg2i_at(k0), tape.arg2i_at(k1))?
        } else {
            0
        };
        strides.push(ArgStrides { arg1, arg2i });
    }

    let mut reps = 2;
    'outer: while start + (reps + 1) * len <= n {
        for j in 0..len {
            let k0 = start + j;
            let k = start + reps * len + j;
            if tape.op_at(k) != tape.op_at(k0) {
                break 'outer;
            }
            let (u1, u2, _) = uses(tape.op_at(k0));
            let s = strides[j as usize];
            if u1 && tape.arg1_at(k) != tape.arg1_at(k0) + reps * s.arg1 {
                break 'outer;
            }
            if u2 && tape.arg2i_at(k) != tape.arg2i_at(k0) + reps * s.arg2i {
                break 'outer;
            }
        }
        reps += 1;
    }

    // A constant that moves needs a table; one that does not folds into the
    // instruction immediate.
    let mut consts = Vec::with_capacity(len as usize);
    for j in 0..len {
        let k0 = start + j;
        let (_, _, uf) = uses(tape.op_at(k0));
        if !uf {
            consts.push(None);
            continue;
        }
        let read = |k: u32| {
            if tape.op_at(k) == Op::Leaf {
                tape.value(k)
            } else {
                tape.arg2f_at(k)
            }
        };
        let first = read(k0);
        let varies = (1..reps).any(|i| read(start + i * len + j).to_bits() != first.to_bits());
        consts.push(if varies {
            Some((0..reps).map(|i| read(start + i * len + j)).collect())
        } else {
            None
        });
    }

    // Two shapes the emitter has no form for: a leaf reads the parameter
    // buffer by absolute index, and `Pow` folds `exponent - 1` into an
    // immediate for its backward step.
    for j in 0..len {
        let op = tape.op_at(start + j);
        if op == Op::Leaf {
            return None;
        }
        if op == Op::Pow && consts[j as usize].is_some() {
            return None;
        }
    }

    Some((reps, strides, consts))
}

/// Partition the tape into re-rollable runs, in tape order and non-overlapping.
/// Nodes not covered by any block stay straight-line.
pub fn detect(tape: &Tape) -> Vec<Block> {
    let n = tape.len() as u32;
    let mut out: Vec<Block> = Vec::new();
    let mut k = 0u32;
    while k < n {
        let mut best: Option<Block> = None;
        for len in 1..=MAX_BLOCK.min(n - k) {
            let Some((reps, strides, consts)) = probe(tape, k, len) else {
                continue;
            };
            if reps < MIN_REPS {
                continue;
            }
            let cand = Block {
                start: k,
                len,
                reps,
                strides,
                consts,
            };
            // Cover as much tape as possible; on a tie the shorter block wins,
            // since its loop body is smaller.
            let better = match &best {
                None => true,
                Some(b) => cand.len * cand.reps > b.len * b.reps,
            };
            if better {
                best = Some(cand);
            }
        }
        match best {
            Some(b) => {
                k = b.end();
                out.push(b);
            }
            None => k += 1,
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod tests_support {
    use stanwasm_autodiff::Tape;

    /// `y ~ normal(alpha + beta * x, sigma)` over N points: the mean is two
    /// element-wise runs and the density is one block per point.
    pub fn linreg_tape(n: usize) -> (Tape, u32) {
        use stanwasm_runtime::{data_from_json, Model};
        let src = r#"data { int<lower=0> N; vector[N] x; vector[N] y; }
parameters { real alpha; real beta; real<lower=0> sigma; }
model {
  alpha ~ normal(0, 10); beta ~ normal(0, 10); sigma ~ exponential(1);
  y ~ normal(alpha + beta * x, sigma);
}"#;
        let xs: Vec<String> = (0..n).map(|i| format!("{}", i as f64 * 0.1)).collect();
        let ys: Vec<String> = (0..n).map(|i| format!("{}", 1.0 + i as f64 * 0.2)).collect();
        let data = format!(
            "{{\"N\": {n}, \"x\": [{}], \"y\": [{}]}}",
            xs.join(","),
            ys.join(",")
        );
        let model = Model::parse_and_load(src, data_from_json(&data).unwrap()).unwrap();
        let mut tape = Tape::new();
        let leaves: Vec<u32> = [0.1, 0.9, 0.2].iter().map(|p| tape.new_var(*p)).collect();
        let root = model.trace_forward(&mut tape, &leaves, true).unwrap();
        (tape, root)
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::linreg_tape;
    use super::*;

    #[test]
    fn finds_the_vectorised_runs() {
        let (tape, _) = linreg_tape(64);
        let blocks = detect(&tape);
        assert!(!blocks.is_empty(), "no blocks found");
        let covered: u32 = blocks.iter().map(|b| b.len * b.reps).sum();
        assert!(
            covered * 100 / tape.len() as u32 > 80,
            "only {covered}/{} nodes covered by {} blocks",
            tape.len(),
            blocks.len()
        );
    }

    #[test]
    fn blocks_do_not_overlap_and_stay_in_range() {
        let (tape, _) = linreg_tape(64);
        let mut prev_end = 0;
        for b in detect(&tape) {
            assert!(b.start >= prev_end, "overlapping blocks");
            assert!(b.end() <= tape.len() as u32, "block runs past the tape");
            assert_eq!(b.strides.len(), b.len as usize);
            assert_eq!(b.consts.len(), b.len as usize);
            prev_end = b.end();
        }
    }

    #[test]
    fn strides_predict_every_repeat() {
        let (tape, _) = linreg_tape(64);
        for b in detect(&tape) {
            for i in 0..b.reps {
                for j in 0..b.len {
                    let k0 = b.start + j;
                    let k = b.start + i * b.len + j;
                    assert_eq!(tape.op_at(k), tape.op_at(k0));
                    let (u1, u2, _) = uses(tape.op_at(k0));
                    let s = b.strides[j as usize];
                    if u1 {
                        assert_eq!(tape.arg1_at(k), tape.arg1_at(k0) + i * s.arg1);
                    }
                    if u2 {
                        assert_eq!(tape.arg2i_at(k), tape.arg2i_at(k0) + i * s.arg2i);
                    }
                }
            }
        }
    }

    #[test]
    fn scales_with_n() {
        // The block structure should not change with N; only the repeat count.
        let (t1, _) = linreg_tape(50);
        let (t2, _) = linreg_tape(100);
        let (b1, b2) = (detect(&t1), detect(&t2));
        assert_eq!(
            b1.iter().map(|b| b.len).collect::<Vec<_>>(),
            b2.iter().map(|b| b.len).collect::<Vec<_>>(),
            "block shapes differ between N=50 and N=100"
        );
        assert!(
            b2.iter().map(|b| b.reps).sum::<u32>() > b1.iter().map(|b| b.reps).sum::<u32>(),
            "repeat counts did not grow with N"
        );
    }
}

