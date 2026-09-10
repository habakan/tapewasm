# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

[Unreleased]: https://github.com/habakan/tapewasm/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/habakan/tapewasm/releases/tag/v0.1.1
[0.1.0]: https://github.com/habakan/tapewasm/releases/tag/v0.1.0
