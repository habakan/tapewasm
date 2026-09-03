//! Opcodes the shipped artifacts deliberately do not use, pinned by validating
//! them with that feature switched off in `wasmparser`.
//!
//! wasm-gc, because what we ship is plain wasm32 — linear memory and a manual
//! heap; see ARCHITECTURE.md. And relaxed SIMD, which WebKit rejects: nuts-rs
//! reaches it through pulp, and the `[patch.crates-io]` in the workspace
//! manifest is there to keep it out. That patch does not travel into a
//! published crate, so this is the check that says whether it can be dropped —
//! as of nuts-rs 0.18.3 and pulp 0.22.3, it still cannot.

use stanwasm_codegen::compile;
use stanwasm_runtime::{Env, Model};
use wasmparser::{Validator, WasmFeatures};

fn validates_without_gc(bytes: &[u8]) -> Result<(), String> {
    let no_gc = WasmFeatures::default() - WasmFeatures::GC;
    Validator::new_with_features(no_gc)
        .validate_all(bytes)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[test]
fn aot_output_uses_no_wasm_gc() {
    let mut data = Env::new();
    data.set_scalar("N", 10.0);
    data.set_vector("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    data.set_vector(
        "y",
        &[1.1, 3.2, 5.0, 7.1, 8.9, 11.2, 13.0, 15.1, 16.8, 19.2],
    );
    let model = Model::parse_and_load(
        r#"data { int<lower=0> N; vector[N] x; vector[N] y; }
parameters { real alpha; real beta; real<lower=0> sigma; }
model {
  alpha ~ normal(0, 10); beta ~ normal(0, 10); sigma ~ exponential(1);
  y ~ normal(alpha + beta * x, sigma);
}"#,
        data,
    )
    .unwrap();
    let compiled = compile(&model, &[0.1; 3]).unwrap();
    validates_without_gc(&compiled.wasm).expect("AOT wasm must not use GC opcodes");
}

/// The most recently built bundle, if there is one.
fn host_wasm() -> Option<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/wasm32-unknown-unknown/release/stanwasm.wasm");
    if !path.exists() {
        eprintln!("skipping: build with `cargo build -p stanwasm --target wasm32-unknown-unknown --release` first");
        return None;
    }
    Some(std::fs::read(&path).unwrap())
}

#[test]
fn host_wasm_uses_no_wasm_gc() {
    let Some(bytes) = host_wasm() else { return };
    validates_without_gc(&bytes).expect("stanwasm wasm must not use GC opcodes");
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
        .expect("stanwasm wasm must not use relaxed SIMD opcodes");
}
