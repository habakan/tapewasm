//! Finding the loops a vectorised Stan statement leaves on the tape.
//!
//! Tracing `y ~ normal(alpha + beta * x, sigma)` over `vector[N]` records the
//! same handful of ops N times over, so the straight-line emitter produces a
//! function that grows with the data. Past a few thousand nodes V8 stops
//! optimising it and the AOT path loses to plain tape replay.
//!
//! The recorded shape is mostly regular: an integer argument of the k-th copy
//! is usually `arg0 + k * stride` for a stride fixed per argument. Intra-block
//! references have stride `len`, a strided read into a vector produced earlier
//! has stride 1, and anything loop-invariant (a shared `sigma`, a CSE'd
//! `log(sigma)`) has stride 0.
//!
//! One argument commonly breaks that: a gather like `mu[g[i]]` in a
//! hierarchical model reads wherever the data says, and no stride describes it.
//! Rejecting those blocks would leave exactly the models people write most
//! unrolled, so a bounded number of arguments may instead be *tabled*: the
//! emitter stages their slot indices and reads one per iteration.

use stanwasm_autodiff::{Op, Tape};

/// Longest block considered. A vectorised density is well under this; the cap
/// keeps detection from going quadratic on a tape with no structure.
const MAX_BLOCK: u32 = 96;

/// Fewer repeats than this is not worth a loop: the prologue and induction
/// variable cost more than the unrolled copies.
const MIN_REPS: u32 = 4;

/// How far a *candidate* is followed. Detection asks about `MAX_BLOCK` lengths
/// at every node it does not cover, and a periodic region answers "still
/// repeating" to most of them, so following each to the end walked the tape
/// `MAX_BLOCK` times over. A candidate that gets this far is periodic; the
/// shortest period describes it, and only that one is then measured exactly.
const PROBE_REPS: u32 = 8;

/// A run of `reps` identical blocks of `len` nodes starting at `start`.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub start: u32,
    pub len: u32,
    pub reps: u32,
    /// Per-node argument relations, `len` entries in block order.
    pub args: Vec<ArgRels>,
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

/// At most this many arguments per block may need an index table. A gather
/// costs one, and a statement can hold several: an LDA term reads two gathers
/// per topic, so its block wants ten. Refusing them left that model re-rolled
/// into fragments — 2.0 MB of wasm against 0.6 at this bound, and 576 µs per
/// gradient against 372, since the smaller module is also the one V8 keeps
/// optimising. Nothing in posteriordb asks for more than twelve.
pub(crate) const MAX_TABLED: usize = 12;

/// How one integer argument moves from one repeat to the next.
#[derive(Debug, Clone, PartialEq)]
pub enum ArgRel {
    /// `tape.argN_at(start + j) + i * stride`.
    Affine(u32),
    /// One absolute slot index per repeat.
    Tabled(Vec<u32>),
}

impl ArgRel {
    pub fn is_tabled(&self) -> bool {
        matches!(self, ArgRel::Tabled(_))
    }
}

/// How a node's integer arguments move from one repeat to the next.
#[derive(Debug, Clone, PartialEq)]
pub struct ArgRels {
    pub arg1: ArgRel,
    pub arg2i: ArgRel,
}

/// Which arguments an opcode actually reads. Unused slots hold stale values on
/// the tape, so comparing them would reject blocks that are in fact identical.
pub(crate) fn uses(op: Op) -> (bool, bool, bool) {
    match op {
        Op::Leaf => (false, false, true),
        // `arg2i` is a handle into the tape's extent table, not a slot, and a
        // run of coefficients is not one immediate.
        Op::DotC | Op::Sum => (true, false, false),
        Op::Add | Op::Sub | Op::Mul | Op::Div => (true, true, false),
        Op::AddC | Op::SubC | Op::RsubC | Op::MulC | Op::DivC | Op::RdivC | Op::Pow => {
            (true, false, true)
        }
        _ => (true, false, false),
    }
}

/// Repeat count, per-node argument relations, and per-node moving constants.
type Probe = (u32, Vec<ArgRels>, Vec<Option<Vec<f64>>>);

/// Check that `len`-node blocks at `start` repeat, and describe how their
/// arguments move. Returns `None` if the opcodes do not repeat, or if too many
/// arguments would need an index table.
fn probe(tape: &Tape, start: u32, len: u32, max_reps: u32) -> Option<Probe> {
    let n = tape.len() as u32;
    if start + 2 * len > n {
        return None;
    }

    // How far the opcode pattern repeats. Arguments cannot bound this any more:
    // a tabled one is allowed to be arbitrary.
    let ops = tape.ops();
    let (head, width) = (start as usize, len as usize);
    let mut reps = 1;
    while reps < max_reps && start + (reps + 1) * len <= n {
        let next = (start + reps * len) as usize;
        if ops[next..next + width] != ops[head..head + width] {
            break;
        }
        reps += 1;
    }
    if reps < 2 {
        return None;
    }

    // Three shapes the emitter has no form for. Rejected before the argument
    // walk below, which costs `len * reps` reads: most of the candidates this
    // is asked about die here, and detection probes `MAX_BLOCK` of them at
    // every node it does not cover.
    for j in 0..len {
        // A leaf reads the parameter buffer by absolute index, and a reduction
        // walks a run the loop emitter has no form for — it is one node for a
        // whole vectorised statement, so it never wants re-rolling anyway.
        match tape.op_at(start + j) {
            Op::Leaf | Op::Sum => return None,
            // A contraction is emitted unrolled, so every repeat has to
            // contract the same number of elements the same distance apart.
            Op::DotC => {
                let e0 = tape.extent_at(start + j);
                if (1..reps).any(|i| {
                    let e = tape.extent_at(start + i * len + j);
                    (e.len, e.stride) != (e0.len, e0.stride)
                }) {
                    return None;
                }
            }
            _ => {}
        }
    }

    // An argument is affine when one stride explains every repeat, and tabled
    // otherwise. The write target is structural and always affine, so a block
    // is describable as long as few enough reads need a table.
    let classify = |read: &dyn Fn(u32) -> u32| -> ArgRel {
        let v0 = read(start);
        let v1 = read(start + len);
        if let Some(stride) = v1.checked_sub(v0) {
            if (2..reps).all(|i| read(start + i * len) == v0 + i * stride) {
                return ArgRel::Affine(stride);
            }
        }
        ArgRel::Tabled((0..reps).map(|i| read(start + i * len)).collect())
    };

    let mut args = Vec::with_capacity(len as usize);
    let mut tabled = 0usize;
    for j in 0..len {
        let k0 = start + j;
        let (u1, u2, _) = uses(tape.op_at(k0));
        let arg1 = if u1 {
            classify(&|base| tape.arg1_at(base + j))
        } else {
            ArgRel::Affine(0)
        };
        let arg2i = if u2 {
            classify(&|base| tape.arg2i_at(base + j))
        } else {
            ArgRel::Affine(0)
        };
        tabled += arg1.is_tabled() as usize + arg2i.is_tabled() as usize;
        if tabled > MAX_TABLED {
            return None;
        }
        args.push(ArgRels { arg1, arg2i });
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
        // `Pow` folds `exponent - 1` into an immediate for its backward step,
        // so a moving exponent has no form.
        if varies && tape.op_at(k0) == Op::Pow {
            return None;
        }
        consts.push(if varies {
            Some((0..reps).map(|i| read(start + i * len + j)).collect())
        } else {
            None
        });
    }

    Some((reps, args, consts))
}

/// How many of a block's positions read a value the previous repeat wrote.
/// Each one needs an address rather than a wasm local, so a block that starts
/// mid-statement pays for every value straddling the boundary it chose.
fn carried(tape: &Tape, start: u32, len: u32) -> u32 {
    let mut n = 0;
    for j in 0..len {
        let k = start + j;
        let (u1, u2, _) = uses(tape.op_at(k));
        for (used, a) in [(u1, tape.arg1_at(k)), (u2, tape.arg2i_at(k))] {
            if used && a < start && a + len >= start {
                n += 1;
            }
        }
    }
    n
}

/// Slide a block forward to the boundary that leaves the fewest values
/// straddling it. Greedy detection takes the first offset where the opcodes
/// repeat, which can be mid-statement when a preceding node happens to match.
fn rotate(tape: &Tape, b: &Block) -> Option<Block> {
    let base = carried(tape, b.start, b.len);
    if base == 0 {
        return None;
    }
    let r = (1..b.len).min_by_key(|&r| carried(tape, b.start + r, b.len))?;
    if carried(tape, b.start + r, b.len) >= base {
        return None;
    }
    let (reps, args, consts) = probe(tape, b.start + r, b.len, u32::MAX)?;
    // The nodes left in front stay straight-line; giving up more than one
    // repeat's worth of coverage is not a trade worth making.
    if reps < MIN_REPS || (reps + 1) * b.len < b.reps * b.len {
        return None;
    }
    Some(Block {
        start: b.start + r,
        len: b.len,
        reps,
        args,
        consts,
    })
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
            let Some((reps, args, consts)) = probe(tape, k, len, PROBE_REPS) else {
                continue;
            };
            if reps < MIN_REPS {
                continue;
            }
            let cand = Block {
                start: k,
                len,
                reps,
                args,
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
            // Lengths ascend, so the first one still repeating at the cap is the
            // shortest period here; a longer one can only describe the same
            // region with a bigger body.
            if best.as_ref().is_some_and(|b| b.reps == PROBE_REPS) {
                break;
            }
        }
        match best {
            Some(b) => {
                let b = extend(tape, b);
                let b = rotate(tape, &b).unwrap_or(b);
                k = b.end();
                out.push(b);
            }
            None => k += 1,
        }
    }
    out
}

/// Follow the chosen block to its real end. Only the winner pays for this, so
/// the walk costs the tape once rather than `MAX_BLOCK` times.
fn extend(tape: &Tape, b: Block) -> Block {
    if b.reps < PROBE_REPS {
        return b;
    }
    match probe(tape, b.start, b.len, u32::MAX) {
        Some((reps, args, consts)) => Block {
            reps,
            args,
            consts,
            ..b
        },
        None => b,
    }
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
        let ys: Vec<String> = (0..n)
            .map(|i| format!("{}", 1.0 + i as f64 * 0.2))
            .collect();
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

    /// A hierarchical model whose group index is irregular, the way real data
    /// is: `mu[g[i]]` cannot be described by a stride.
    pub fn gather_tape(n: usize) -> (Tape, u32) {
        use stanwasm_runtime::{data_from_json, Model};
        let src = r#"data { int<lower=0> N; int<lower=1> G; array[N] int<lower=1> g; vector[N] y; }
parameters { vector[G] mu; real<lower=0> sigma; }
model {
  mu ~ normal(0, 5); sigma ~ exponential(1);
  for (i in 1:N) y[i] ~ normal(mu[g[i]], sigma);
}"#;
        let mut seed: u64 = 12345;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223) & 0xffff_ffff;
            seed as f64 / 4294967296.0
        };
        let gs: Vec<String> = (0..n)
            .map(|_| format!("{}", 1 + (rnd() * 8.0) as u32))
            .collect();
        let ys: Vec<String> = (0..n)
            .map(|i| format!("{}", (i as f64).sin() * 2.0))
            .collect();
        let data = format!(
            "{{\"N\": {n}, \"G\": 8, \"g\": [{}], \"y\": [{}]}}",
            gs.join(","),
            ys.join(",")
        );
        let model = Model::parse_and_load(src, data_from_json(&data).unwrap()).unwrap();
        let mut tape = Tape::new();
        let init = [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, 0.05, 0.5];
        let leaves: Vec<u32> = init.iter().map(|p| tape.new_var(*p)).collect();
        let root = model.trace_forward(&mut tape, &leaves, true).unwrap();
        (tape, root)
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::gather_tape;
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
            assert_eq!(b.args.len(), b.len as usize);
            assert_eq!(b.consts.len(), b.len as usize);
            prev_end = b.end();
        }
    }

    /// Whatever `detect` claims about an argument has to reproduce the tape:
    /// an affine one from its stride, a tabled one from its recorded indices.
    fn check_args_reproduce_tape(tape: &Tape, blocks: &[Block]) {
        for b in blocks {
            for i in 0..b.reps {
                for j in 0..b.len {
                    let k0 = b.start + j;
                    let k = b.start + i * b.len + j;
                    assert_eq!(tape.op_at(k), tape.op_at(k0));
                    let (u1, u2, _) = uses(tape.op_at(k0));
                    let rels = &b.args[j as usize];
                    if u1 {
                        let want = match &rels.arg1 {
                            ArgRel::Affine(t) => tape.arg1_at(k0) + i * t,
                            ArgRel::Tabled(ix) => ix[i as usize],
                        };
                        assert_eq!(
                            tape.arg1_at(k),
                            want,
                            "arg1 at block {}, i={i}, j={j}",
                            b.start
                        );
                    }
                    if u2 {
                        let want = match &rels.arg2i {
                            ArgRel::Affine(t) => tape.arg2i_at(k0) + i * t,
                            ArgRel::Tabled(ix) => ix[i as usize],
                        };
                        assert_eq!(
                            tape.arg2i_at(k),
                            want,
                            "arg2i at block {}, i={i}, j={j}",
                            b.start
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn args_predict_every_repeat() {
        let (tape, _) = linreg_tape(64);
        check_args_reproduce_tape(&tape, &detect(&tape));
    }

    /// The density block of a vectorised statement is a chain of temporaries
    /// with one accumulator: everything but the accumulator should be local.
    #[test]
    fn most_of_a_density_block_is_iteration_local() {
        let (tape, root) = linreg_tape(64);
        let blocks = detect(&tape);
        let flags = local_positions(&tape, &blocks, root);
        let biggest = blocks
            .iter()
            .enumerate()
            .max_by_key(|(_, b)| b.len)
            .map(|(i, _)| i)
            .unwrap();
        let n_local = flags[biggest].iter().filter(|f| **f).count();
        let len = blocks[biggest].len as usize;
        assert!(
            n_local >= len - 1,
            "only {n_local}/{len} positions are iteration-local"
        );
    }

    /// Whatever a gather can reach has to keep an address.
    #[test]
    fn gather_targets_are_not_local() {
        let (tape, root) = gather_tape(200);
        let blocks = detect(&tape);
        let flags = local_positions(&tape, &blocks, root);
        for (bi, b) in blocks.iter().enumerate() {
            for j in 0..b.len as usize {
                if let ArgRel::Tabled(ix) = &b.args[j].arg1 {
                    for &t in ix {
                        if t >= b.start && t < b.end() {
                            let pos = ((t - b.start) % b.len) as usize;
                            assert!(!flags[bi][pos], "a gather target stayed local");
                        }
                    }
                }
            }
        }
    }

    /// A gather (`mu[g[i]]` with irregular groups) is the case no stride
    /// describes. It must still be found, with the gather tabled.
    #[test]
    fn finds_a_gather_and_tables_it() {
        let (tape, _) = gather_tape(200);
        let blocks = detect(&tape);
        check_args_reproduce_tape(&tape, &blocks);
        let covered: u32 = blocks.iter().map(|b| b.len * b.reps).sum();
        assert!(
            covered * 100 / tape.len() as u32 > 80,
            "only {covered}/{} nodes covered",
            tape.len()
        );
        let tabled: usize = blocks
            .iter()
            .flat_map(|b| b.args.iter())
            .filter(|a| a.arg1.is_tabled() || a.arg2i.is_tabled())
            .count();
        assert!(tabled > 0, "the gather was not tabled");
    }

    /// Greedy detection takes the first offset where the opcodes repeat, and a
    /// prologue node can share the row's last opcode. Starting there splits the
    /// row, and every value straddling the split needs an address.
    #[test]
    fn a_block_starts_on_its_statement_boundary() {
        let mut tape = Tape::new();
        let a = tape.new_var(0.5);
        let b = tape.new_var(1.5);
        // A prologue whose last opcode is the one each row ends with.
        let mut acc = tape.add(a, b);
        for i in 0..40 {
            let p = tape.mul_c(a, 1.0 + i as f64);
            acc = tape.add(acc, p);
        }

        let blocks = detect(&tape);
        let found = blocks.first().expect("no block found");
        assert_eq!(tape.op_at(found.start), Op::MulC, "block starts mid-row");
        assert!(
            carried(&tape, found.start, found.len) < carried(&tape, found.start - 1, found.len),
            "the rotation did not reduce what straddles the boundary"
        );
        let flags = local_positions(&tape, &blocks, acc);
        assert_eq!(flags[0], vec![true, false], "the product should stay local");
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

/// Which positions in each block can live in a wasm local rather than in the
/// scratch buffer: written and read inside one iteration, and invisible from
/// anywhere else on the tape.
///
/// Everything else — an accumulator carrying across iterations, a vector some
/// later statement reads back, whatever a gather points at — has to stay
/// addressable, and a local is not.
pub fn local_positions(tape: &Tape, blocks: &[Block], root: u32) -> Vec<Vec<bool>> {
    let n = tape.len() as u32;
    let mut out: Vec<Vec<bool>> = blocks.iter().map(|b| vec![true; b.len as usize]).collect();

    // Which block, if any, owns a tape index, and at which position. Blocks are
    // non-overlapping and in tape order, so this bisects rather than scanning:
    // it is asked once per argument of every node on the tape.
    let owner = |k: u32| -> Option<(usize, u32, u32)> {
        let bi = blocks.partition_point(|b| b.end() <= k);
        let b = blocks.get(bi)?;
        (k >= b.start).then(|| {
            let off = k - b.start;
            (bi, off / b.len, off % b.len)
        })
    };

    let demote = |k: u32, out: &mut Vec<Vec<bool>>| {
        if let Some((bi, _, j)) = owner(k) {
            out[bi][j as usize] = false;
        }
    };

    // A contraction or a reduction reads a whole run, not one node, and the
    // emitter addresses every element of it.
    for k in 0..n {
        if !matches!(tape.op_at(k), Op::DotC | Op::Sum) {
            continue;
        }
        let e = tape.extent_at(k);
        for c in 0..e.len {
            demote(e.base + c * e.stride, &mut out);
        }
    }

    for k in 0..n {
        let (u1, u2, _) = uses(tape.op_at(k));
        for target in [(u1, tape.arg1_at(k)), (u2, tape.arg2i_at(k))] {
            let (used, t) = target;
            if !used {
                continue;
            }
            let Some((tb, ti, tj)) = owner(t) else {
                continue;
            };
            // A reference from outside the block, or from another iteration of
            // it, needs an address.
            match owner(k) {
                Some((kb, ki, _)) if kb == tb && ki == ti => {}
                _ => out[tb][tj as usize] = false,
            }
        }
    }

    // A gather reads wherever its table says, so anything it can reach stays
    // addressable.
    for b in blocks {
        for rels in &b.args {
            for rel in [&rels.arg1, &rels.arg2i] {
                if let ArgRel::Tabled(ix) = rel {
                    for &t in ix {
                        demote(t, &mut out);
                    }
                }
            }
        }
    }

    // The log density is read after the loops have run.
    demote(root, &mut out);
    out
}
