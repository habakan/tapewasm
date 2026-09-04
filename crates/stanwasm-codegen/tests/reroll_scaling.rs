//! Emitting a model that re-rolls into many small blocks must not cost the
//! number of blocks times the length of the tape. Deciding whether a node can
//! live in a wasm local asks which block owns each of its arguments, and that
//! search used to scan every block — quadratic on a model that fragments, which
//! is what left four posteriordb posteriors unable to compile in two minutes.
//!
//! The bound is wall clock, so it sits an order of magnitude above what the
//! linear version takes: enough to catch the quadratic return, not to measure.

use std::time::Instant;

use stanwasm_codegen::{compile_with, Reroll};
use stanwasm_runtime::{data_from_json, Model};

/// Latent Dirichlet allocation, the shape that fragments: two scattered
/// gathers per term, so most candidate blocks want more index tables than the
/// emitter allows and detection settles for small ones.
const LDA: &str = r#"
data {
  int<lower=2> V; int<lower=1> M; int<lower=1> N;
  array[N] int<lower=1, upper=V> w; array[N] int<lower=1, upper=M> doc;
  vector<lower=0>[5] alpha; vector<lower=0>[V] beta;
}
parameters { array[M] simplex[5] theta; array[5] simplex[V] phi; }
model {
  for (m in 1:M) { theta[m] ~ dirichlet(alpha); }
  for (k in 1:5) { phi[k] ~ dirichlet(beta); }
  for (n in 1:N) {
    array[5] real gamma;
    for (k in 1:5) { gamma[k] = log(theta[doc[n], k]) + log(phi[k, w[n]]); }
    target += log_sum_exp(gamma);
  }
}
"#;

#[test]
fn a_fragmented_model_compiles_in_time_proportional_to_its_tape() {
    let (v, m, n) = (60usize, 200usize, 3000usize);
    let mut seed: u64 = 12345;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as usize
    };
    let w: Vec<String> = (0..n).map(|_| format!("{}", next() % v + 1)).collect();
    let doc: Vec<String> = (0..n).map(|_| format!("{}", next() % m + 1)).collect();
    let data = format!(
        "{{\"V\":{v},\"M\":{m},\"N\":{n},\"w\":[{}],\"doc\":[{}],\"alpha\":[1,1,1,1,1],\"beta\":[{}]}}",
        w.join(","),
        doc.join(","),
        vec!["1"; v].join(",")
    );

    let model = Model::parse_and_load(LDA, data_from_json(&data).unwrap()).unwrap();
    let dummy = vec![0.1; model.n_params()];
    let started = Instant::now();
    let compiled = compile_with(&model, &dummy, Reroll::Auto).unwrap();
    let secs = started.elapsed().as_secs_f64();
    assert!(!compiled.wasm.is_empty());
    assert!(secs < 5.0, "compiled in {secs:.1}s");

    // And it has to actually re-roll. This shape's blocks each want ten index
    // tables; at a lower `MAX_TABLED` detection settles for fragments and the
    // module lands within 2.5x of the straight-line one instead of 8x under it.
    let straight = compile_with(&model, &dummy, Reroll::Never).unwrap();
    let ratio = straight.wasm.len() as f64 / compiled.wasm.len() as f64;
    assert!(
        ratio > 4.0,
        "re-rolled to only {ratio:.1}x under straight-line"
    );
}
