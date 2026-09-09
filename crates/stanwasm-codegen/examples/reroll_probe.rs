//! Which loop-form likelihoods `reroll::detect` finds, and which it gives up on.
//!
//!   cargo run -p stanwasm-codegen --example reroll_probe
//!
//! Forces `Reroll::Always`, so the only thing left to explain a module that
//! does not collapse is the detector. One repeat is about `1.2 * K * K` nodes,
//! so the `K` where the `Always` column stops being flat is where it passed
//! `reroll::MAX_BLOCK`.

use stanwasm_autodiff::Tape;
use stanwasm_codegen::{compile_with, Reroll};
use stanwasm_runtime::{Env, Model, Val};

const MULTIVARIATE_N: &str = r#"
data {
  int<lower=1> N;
  int<lower=1> K;
  array[N] vector[K] y;
}
parameters {
  vector[K] mu;
  cholesky_factor_corr[K] L;
}
model {
  mu ~ normal(0, 5);
  L  ~ lkj_corr_cholesky(2.0);
  for (n in 1:N) y[n] ~ multi_normal_cholesky(mu, L);
}
"#;

fn build(n: usize, k: usize) -> Model {
    let mut data = Env::new();
    data.set_scalar("N", n as f64);
    data.set_scalar("K", k as f64);
    // Distinct throughout: a repeated value is a common subexpression, and the
    // shared node would read as a break in the period.
    let mut seed: u64 = 987654321;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let rows: Vec<Val> = (0..n)
        .map(|_| Val::Vec((0..k).map(|_| Val::Num(rnd() * 4.0 - 2.0)).collect()))
        .collect();
    data.set("y", Val::Vec(rows));
    Model::parse_and_load(MULTIVARIATE_N, data).unwrap()
}

fn nodes(model: &Model, dummy: &[f64]) -> usize {
    let mut tape = Tape::new();
    let leaves: Vec<u32> = dummy.iter().map(|p| tape.new_var(*p)).collect();
    model.trace_forward(&mut tape, &leaves, true).unwrap();
    tape.len()
}

fn main() {
    println!(
        "{:>5} {:>4} {:>8} {:>10} {:>12} {:>10} {:>12}",
        "N", "K", "nodes", "per obs", "Always", "detect ms", "Never"
    );
    let mut prev: Option<(usize, usize)> = None;
    for (n, k) in [
        (200usize, 5usize),
        (200, 10),
        (200, 14),
        (200, 18),
        (1000, 10),
    ] {
        let model = build(n, k);
        let dummy = vec![0.1; model.n_params()];
        let total = nodes(&model, &dummy);
        let t0 = std::time::Instant::now();
        let always = compile_with(&model, &dummy, Reroll::Always)
            .unwrap()
            .wasm
            .len();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let never = compile_with(&model, &dummy, Reroll::Never)
            .unwrap()
            .wasm
            .len();
        // Two sizes of the same K differ only by the per-observation block.
        let per_obs = match prev {
            Some((pn, pt)) if pn < n => format!("{:.0}", (total - pt) as f64 / (n - pn) as f64),
            _ => "-".to_string(),
        };
        println!(
            "{n:>5} {k:>4} {total:>8} {per_obs:>10} {:>11.1}K {ms:>10.0} {:>11.1}K",
            always as f64 / 1024.0,
            never as f64 / 1024.0
        );
        prev = Some((n, total));
    }
}
