//! Opcodes the shipped artifacts deliberately do not use, pinned by validating
//! them with that feature switched off in `wasmparser`.
//!
//! wasm-gc, because what ships is plain wasm32 — linear memory and a manual
//! heap. And relaxed SIMD, which WebKit rejects: nuts-rs reaches it through
//! pulp, and the `[patch.crates-io]` in the workspace manifest is there to keep
//! it out. That patch does not travel into a published crate, so this is the
//! check that says whether it can be dropped — as of nuts-rs 0.18.3 and pulp
//! 0.22.3, it still cannot.

use tapewasm_codegen::{compile_tape, shapes, Reroll};
use wasmparser::{Validator, WasmFeatures};

fn validates_without_gc(bytes: &[u8]) -> Result<(), String> {
    let no_gc = WasmFeatures::default() - WasmFeatures::GC;
    Validator::new_with_features(no_gc)
        .validate_all(bytes)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[test]
fn an_emitted_module_uses_no_wasm_gc() {
    // Re-rolled, so the loop emitter's own instruction selection is covered.
    let (tape, root) = shapes::linreg(2000);
    let compiled = compile_tape(&tape, 3, root, Reroll::Always).unwrap();
    validates_without_gc(&compiled.wasm).expect("emitted wasm must not use GC opcodes");
}

/// The most recently built bundle, if there is one.
fn host_wasm() -> Option<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/wasm32-unknown-unknown/release/tapewasm.wasm");
    if !path.exists() {
        eprintln!("skipping: build with `cargo build -p tapewasm --target wasm32-unknown-unknown --release` first");
        return None;
    }
    Some(std::fs::read(&path).unwrap())
}

#[test]
fn host_wasm_uses_no_wasm_gc() {
    let Some(bytes) = host_wasm() else { return };
    validates_without_gc(&bytes).expect("tapewasm wasm must not use GC opcodes");
}

/// Safari refuses a module carrying relaxed SIMD. The emitter's own `f64x2` is
/// fixed-width and stays enabled here; what this rules out is the relaxed set
/// that nuts-rs pulls in through pulp.
#[test]
fn host_wasm_uses_no_relaxed_simd() {
    let Some(bytes) = host_wasm() else { return };
    let no_relaxed = WasmFeatures::default() - WasmFeatures::RELAXED_SIMD;
    Validator::new_with_features(no_relaxed)
        .validate_all(&bytes)
        .map(|_| ())
        .map_err(|e| e.to_string())
        .expect("tapewasm wasm must not use relaxed SIMD opcodes");
}
