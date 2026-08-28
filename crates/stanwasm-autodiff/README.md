# stanwasm-autodiff

Reverse-mode automatic differentiation over a flat, struct-of-arrays tape.

Part of [stanwasm](https://github.com/habakan/stanwasm), which compiles and samples Stan models
entirely in the browser — no server, no cmdstan.

Records a model's log-density evaluation as a tape of primitive operations,
then walks it backwards for the gradient. The flat array layout exists because
this runs inside wasm, where pointer-chasing a graph of boxed nodes costs more
than the arithmetic does.

The tape is also what [`stanwasm-codegen`](https://crates.io/crates/stanwasm-codegen) consumes to emit a
standalone, fully-unrolled wasm module per model.

You almost certainly want [`stanwasm`](https://crates.io/crates/stanwasm), or the `stanwasm` npm
package, rather than this crate directly. It is published because they depend
on it.

Stan language coverage is a documented subset — see the workspace
[README](https://github.com/habakan/stanwasm#stan-language-coverage).

Licensed under Apache-2.0.
