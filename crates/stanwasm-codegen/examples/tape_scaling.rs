//! How the tape, and the module emitted from it, grow with a model.
//!
//!   cargo run -p stanwasm-codegen --example tape_scaling
//!
//! The tape is scalar-level, so an array-shaped model records a node per
//! scalar. Reports nodes and module size for a covariance dimension, for the
//! same likelihood written as a loop, and for a vectorised one — which is the
//! comparison that matters, since only the last re-rolls.

use stanwasm_autodiff::Tape;
use stanwasm_codegen::compile;
use stanwasm_runtime::{Env, Model, Val};

const MULTIVARIATE_LKJ: &str = r#"
data {
  int<lower=1> K;
  vector[K] y;
}
parameters {
  vector[K] mu;
  cholesky_factor_corr[K] L;
}
model {
  mu ~ normal(0, 5);
  L  ~ lkj_corr_cholesky(2.0);
  y  ~ multi_normal_cholesky(mu, L);
}
"#;

const LINEAR_REGRESSION: &str = r#"
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 10);
  beta  ~ normal(0, 10);
  sigma ~ exponential(1);
  y ~ normal(alpha + beta * x, sigma);
}
"#;

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

fn trace_len(model: &Model, dummy: &[f64]) -> usize {
    let mut tape = Tape::new();
    let leaves: Vec<u32> = dummy.iter().map(|p| tape.new_var(*p)).collect();
    model.trace_forward(&mut tape, &leaves, true).unwrap();
    tape.len()
}

fn report(label: &str, size: usize, model: &Model) {
    let dummy = vec![0.1; model.n_params()];
    let nodes = trace_len(model, &dummy);
    let aot = match compile(model, &dummy) {
        Ok(c) => {
            let kb = c.wasm.len() as f64 / 1024.0;
            match wasmparser::Validator::new().validate_all(&c.wasm) {
                Ok(_) => format!("{kb:>8.1} KB  valid"),
                Err(e) => format!("{kb:>8.1} KB  INVALID: {e}"),
            }
        }
        Err(e) => format!("{e}"),
    };
    println!(
        "{label:>8} {size:>4}  n_params {:>5}  nodes {nodes:>7}  {aot}",
        model.n_params()
    );
}

fn main() {
    println!("-- cholesky_factor_corr[K], one observation");
    for k in [2usize, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50] {
        let mut data = Env::new();
        data.set_scalar("K", k as f64);
        let y: Vec<f64> = (0..k).map(|j| 0.37 * (j + 1) as f64).collect();
        data.set_vector("y", &y);
        match Model::parse_and_load(MULTIVARIATE_LKJ, data) {
            Ok(model) => report("K =", k, &model),
            Err(e) => println!("{:>6} {k:>4}  {e}", "K ="),
        }
    }

    println!("\n-- the same, with N observations");
    for (n, k) in [
        (1usize, 5usize),
        (10, 5),
        (50, 5),
        (100, 5),
        (100, 10),
        (500, 10),
    ] {
        let mut data = Env::new();
        data.set_scalar("N", n as f64);
        data.set_scalar("K", k as f64);
        // Distinct values throughout: identical rows are common subexpressions,
        // and a node the tape shares makes both counts unrepresentative.
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
        match Model::parse_and_load(MULTIVARIATE_N, data) {
            Ok(model) => report(&format!("N={n} K="), k, &model),
            Err(e) => println!("  N={n} K={k}  {e}"),
        }
    }

    println!("\n-- vectorised linear regression, for scale");
    for n in [10usize, 100, 1_000, 2_000, 5_000, 10_000] {
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 1.0 + 1.8 * v).collect();
        let mut data = Env::new();
        data.set_scalar("N", n as f64);
        data.set_vector("x", &x);
        data.set_vector("y", &y);
        let model = Model::parse_and_load(LINEAR_REGRESSION, data).unwrap();
        report("N =", n, &model);
    }
}
