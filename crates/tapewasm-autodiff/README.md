# tapewasm-autodiff

Reverse-mode automatic differentiation over a flat, struct-of-arrays tape.

Part of [tapewasm](https://github.com/habakan/tapewasm), which compiles a tape
to a WebAssembly module and samples it in the browser.

Records an expression as a tape of primitive operations, then walks it
backwards for the gradient. The flat array layout exists because this runs
inside wasm, where pointer-chasing a graph of boxed nodes costs more than the
arithmetic does. Equal expressions are numbered into one node as they are
recorded, so a front end that repeats itself does not pay for it.

The tape is also what [`tapewasm-codegen`](https://crates.io/crates/tapewasm-codegen)
consumes to emit a standalone wasm module.

You probably want [`tapewasm`](https://crates.io/crates/tapewasm), or the
`tapewasm` npm package, rather than this crate directly.

Licensed under Apache-2.0.
