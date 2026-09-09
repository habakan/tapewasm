# Contributing

This repository is the engine: the autodiff tape, the emitter that turns one
into a wasm module, and the sampler that runs it. Nothing here knows what a
model is.

A change to a model language belongs in the front end that has one —
[stanwasm](https://github.com/habakan/stanwasm) for Stan,
[pymcwasm](https://github.com/habakan/pymcwasm) for PyMC — not here.

## Before opening a PR

```bash
make check          # fmt, clippy, tests
make browser-test   # a compiled module in Chromium, Firefox and WebKit
```

`make check` is what CI runs. `make browser-test` is not in CI — three engines
is a large install per run, and what it checks changes with the emitter rather
than with every commit — so run it when codegen or the sampler moves.

## What a change to the emitter needs

1. **A test against the tape's own reverse pass.** That is the oracle here: it
   computes the same derivatives by a wholly different route, so the two
   agreeing is evidence about the emitter rather than about one shared
   implementation. `crates/tapewasm-codegen/tests/emitted_module.rs`.
2. **A tape to run it on.** `tapewasm_codegen::shapes` has the shapes the loop
   emitter has separate paths for — an element-wise run, an irregular gather, a
   long per-observation block, a contraction, and one that fragments. Add to it
   rather than building a one-off inline, so the examples and the other tests
   can reach the same shape.
3. **The no-wasm-gc invariant.** `crates/tapewasm-codegen/tests/no_wasm_gc.rs`
   must keep passing — plain wasm32 is the target, by design.

## Style

- Rust 2021, default `cargo fmt`, `#![forbid(unsafe_code)]` per crate
- Comments carry what you would have to dig to learn — why a constant is that
  number, what breaks without the line. Not a restatement of the code below it.
- One concern per PR, and a `CHANGELOG.md` bullet under `[Unreleased]`

Apache-2.0. By contributing you agree your work is licensed the same way.
