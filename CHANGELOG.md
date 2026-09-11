# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`compileTape(tape, reroll)`** — `"auto"` (the default, unchanged),
  `"always"` or `"never"`. The caller could not choose before, and the choice
  is an engine's rather than the model's: on pymcwasm's PyMC-Marketing MMM
  (9,342 nodes, straight-line under `"auto"`) a gradient took 15.7 / 44.0 /
  53.1 µs in Chromium / Firefox / WebKit straight-line and 28.2 / 36.3 / 30.6
  re-rolled, from a 319 KB module against 13 KB.
- **`AotSampler::sampleWithStats`**, returning a `SampleResult`: the draws
  `sample` returns for the same seed, with each draw's `diverging`, `tuning`,
  `stepSize`, `numSteps` and `lp` beside them — what ArviZ keeps as
  `sample_stats` — and a `chain` label for assembling several runs.
- **`AotSampler::setTargetAccept` and `setGradBasedEstimate`.** Neither could
  be set from outside: warmup aimed at nuts-rs's 0.8 acceptance, and the metric
  came from the draws alone — the reference posteriors' choice, and still the
  default. On the MMM above the sampler took about twice nutpie's gradient
  evaluations at the same target.
- `REROLL=always|never` for the `tape_from_text` example, the same choice for
  a build step that emits ahead of time rather than in the page.
- **`digamma` compiles.** The tape recorded it and `compile_tape` refused it
  with `UnsupportedOp`; the module now carries `trigamma` beside `digamma` for
  its derivative, mirroring `tapewasm_autodiff`'s to 1e-15, and the text
  format has a `digamma` instruction.

## [0.2.0] — 2026-09-11 (npm only)

crates.io still waits on a nuts-rs release that carries
[nuts-rs#76](https://github.com/pymc-devs/nuts-rs/pull/76); see 0.1.0.

### Added

- **`AotSampler::advi`, mean-field ADVI over the same AOT path `sample`
  uses.** Fits `q(θ) = N(μ, diag(σ²))` by Adam-ascending the ELBO under the
  reparameterization trick, calling `log_prob_grad` per draw rather than
  adding anything to the tape or the emitter — the primitive it needs already
  existed. `AdviResult::mu`/`sigma` average the second half of the run
  (Polyak averaging), since a constant Adam step size never settles on one
  point. Phase 1 of the ADVI plan; not gated behind `codegen`, so it also
  runs against a module compiled ahead of time. Optional `snapshot_every`
  records the raw iterate `μ` periodically through the run
  (`AdviResult::muSnapshots`/`snapshotIters`), for watching one training run
  progress rather than splicing several shorter ones together.
- **`tan`, `asin`, `acos` and `atan` in the tape text format.** The emitter has
  always been able to emit them; no front end outside this crate could ask for
  them, because the only test that reached them built the tape directly. The
  module doc now lists every instruction the format accepts.
- `every_instruction_in_the_text_format_reaches_the_emitter` runs the whole
  instruction set through both re-roll modes against the tape's reverse pass.
- **`dot_c` and `sum_run` in the tape text format** — the contraction and the
  reduction, the two ops a matrix model needs to stay compact rather than
  writing a contraction as an elementwise chain. Both take a stride measured
  in *node* indices, which a caller counting instructions has no way to
  supply directly (value numbering can merge two written instructions into
  one node), so the text instead names every element by its own instruction
  (`dot_c LEN A0 C0 A1 C1 ...`, `sum_run SEED LEN A0 A1 ...`) and `parse`
  checks the nodes behind them land evenly spaced before handing `base`/
  `stride` to the tape — an uneven run is a parse error naming the gap it
  found, not a silently wrong module.

### Changed

- **A contraction outside every re-rolled block runs as a loop** over its run,
  its coefficients staged in the constant table beside the block tables,
  rather than unrolled at ~54 B of code per element. A `dot_c` repeated too few
  times to join a block — an encoder's `X @ W` per image — made the module
  larger than the same sum written elementwise: 64 contractions of 196 went
  from 1,052 KB (688 KB elementwise) to 410 KB, and of 784 from 3,051 KB to
  452 KB.
- **A reduction can sit inside a re-rolled block.** Detection refused any
  block holding a `Sum`, so a statement that repeats one — a dense layer's
  per-output `sum_run` — fell out of every loop with everything around it: a
  decoder of 6,272 sums of 20 covered 71.5% of its tape and emitted 5,285 KB,
  against 688 KB written elementwise. It now joins its statement's block,
  unrolled once in the body (97.1%, 566 KB), with the elements it reads from
  its own iteration kept in locals: 472 µs per gradient, against 386 µs for
  the elementwise chain. A run longer than `MAX_BLOCK` keeps its own loop
  outside.

## [0.1.1] — 2026-09-10 (npm only)

### Changed

- **Dual licensed: MIT OR Apache-2.0, at your option.** 0.1.0 offered Apache-2.0
  alone. Nobody loses a right — this only adds one — but a project that is
  itself MIT no longer has to carry a second licence's notice obligations to
  use any of this. Apache-2.0 stays available for its patent grant.

## [0.1.0] — 2026-09-10 (npm only)

crates.io waits on a nuts-rs release carrying
[nuts-rs#76](https://github.com/pymc-devs/nuts-rs/pull/76): the workspace
`[patch.crates-io]` that keeps relaxed SIMD out of the bundle does not travel
into a published crate, so a crate published today would build, for anyone
depending on it, a module WebKit refuses.

First release. The tape, the emitter and the browser API were extracted from
[stanwasm](https://github.com/habakan/stanwasm), whose git history they keep.

### Changed

What changed in the move, for anyone porting a host:

- The emitted module's globals are `tapewasm_layout_id` and
  `tapewasm_abi_version`, and it imports shared memory as `tapewasm.memory`
  rather than `stan.memory`. A module built before the rename exports neither
  name, so binding one is refused rather than mixed.
- `compile(&Model, ..)` is gone. `compile_tape` is the entry point; a front end
  that has a model language lowers it to a tape first.
- `tapewasm_codegen::shapes` builds the tapes the tests and examples run on, so
  neither needs a model language.

[Unreleased]: https://github.com/habakan/tapewasm/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/habakan/tapewasm/releases/tag/v0.2.0
[0.1.1]: https://github.com/habakan/tapewasm/releases/tag/v0.1.1
[0.1.0]: https://github.com/habakan/tapewasm/releases/tag/v0.1.0
