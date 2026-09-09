# tapewasm-codegen

Emits a recorded autodiff tape as a standalone wasm module computing a value
and every partial derivative in one call.

Part of [tapewasm](https://github.com/habakan/tapewasm), which compiles a tape
to a WebAssembly module and samples it in the browser.

Takes the tape recorded by [`tapewasm-autodiff`](https://crates.io/crates/tapewasm-autodiff)
and writes wasm bytes directly via `wasm-encoder` — a forward and a backward
pass with no interpreter dispatch left in the inner loop. The module imports
the host's linear memory, so gradients cross no copy boundary. A repeated run
of instructions is re-rolled back into a loop, and one whose every slot moves
by a fixed stride runs two repeats at a time under fixed-width SIMD.

Emitted modules are plain wasm32 — linear memory and a manual heap, no wasm-gc.
A test validates the output with `wasmparser` and the GC feature explicitly
disabled, so the target stays every browser rather than the newest one.

You probably want [`tapewasm`](https://crates.io/crates/tapewasm), or the
`tapewasm` npm package, rather than this crate directly.

Licensed under Apache-2.0.
