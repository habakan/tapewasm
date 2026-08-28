# stanwasm-codegen

Ahead-of-time compilation of a traced Stan model to a standalone wasm module.

Part of [stanwasm](https://github.com/habakan/stanwasm), which compiles and samples Stan models
entirely in the browser — no server, no cmdstan.

Takes the tape recorded by [`stanwasm-autodiff`](https://crates.io/crates/stanwasm-autodiff) and emits wasm bytes
directly via `wasm-encoder` — a fully unrolled forward and backward pass, with
no interpreter dispatch left in the inner loop. The result is instantiated
next to the sampler and shares its linear memory, so gradients cross no copy
boundary.

Emitted modules are plain wasm32 — linear memory and a manual heap, no wasm-gc.
A test in CI validates the output with `wasmparser` and the GC feature
explicitly disabled, so the target stays every browser rather than the newest
one.

You almost certainly want [`stanwasm`](https://crates.io/crates/stanwasm), or the `stanwasm` npm
package, rather than this crate directly. It is published because they depend
on it.

Stan language coverage is a documented subset — see the workspace
[README](https://github.com/habakan/stanwasm#stan-language-coverage).

Licensed under Apache-2.0.
