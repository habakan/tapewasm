//! Confirms the AOT model wasm we emit (and, when run directly, the
//! stanwasm wasm bundle) do NOT use any wasm-gc opcodes — i.e. they
//! pass `wasmparser` validation with the `GC` feature explicitly disabled.
//!
//! This pins a deliberate choice: the artifacts we ship are plain wasm32
//! (linear memory + manual heap), not wasm-gc — see ARCHITECTURE.md for why.

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

#[test]
fn host_wasm_uses_no_wasm_gc() {
    // Read the most recently-built stanwasm artifact, if it exists.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/wasm32-unknown-unknown/release/stanwasm.wasm");
    if !path.exists() {
        eprintln!("skipping: build with `cargo build -p stanwasm --target wasm32-unknown-unknown --release` first");
        return;
    }
    let bytes = std::fs::read(&path).unwrap();
    validates_without_gc(&bytes).expect("stanwasm wasm must not use GC opcodes");
}
