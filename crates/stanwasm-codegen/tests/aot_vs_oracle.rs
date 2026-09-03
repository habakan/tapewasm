//! End-to-end: compile linear_regression / poisson_regression to wasm via
//! stanwasm-codegen, instantiate with wasmi, and verify that the wasm-generated
//! log_prob and gradients match the AST-evaluator oracle to floating-point
//! precision.

use stanwasm_codegen::compile;
use stanwasm_runtime::{Env, Model};
use wasmi::{Caller, Engine, Func, Linker, Memory, MemoryType, Module, Store};

fn lgamma(x: f64) -> f64 {
    stanwasm_autodiff::lgamma(x)
}
fn digamma(x: f64) -> f64 {
    stanwasm_autodiff::digamma(x)
}
fn phi(x: f64) -> f64 {
    stanwasm_autodiff::phi_cdf(x)
}

#[derive(Default)]
struct HostState;

fn install_math(linker: &mut Linker<HostState>, store: &mut Store<HostState>) {
    macro_rules! unary {
        ($name:literal, $fn:expr) => {{
            let f = Func::wrap(&mut *store, |_: Caller<'_, HostState>, x: f64| -> f64 {
                $fn(x)
            });
            linker.define("Math", $name, f).unwrap();
        }};
    }
    unary!("exp", f64::exp);
    unary!("log", f64::ln);
    unary!("sin", f64::sin);
    unary!("cos", f64::cos);
    unary!("lgamma", lgamma);
    unary!("digamma", digamma);
    unary!("phi", phi);
    let pow = Func::wrap(
        &mut *store,
        |_: Caller<'_, HostState>, x: f64, y: f64| -> f64 { x.powf(y) },
    );
    linker.define("Math", "pow", pow).unwrap();
}

fn run_aot_log_prob_grad(
    wasm: &[u8],
    n_params: usize,
    params: &[f64],
    scratch_len: usize,
    const_table: &[f64],
) -> (f64, Vec<f64>) {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm).expect("module parses");
    let mut store = Store::new(&engine, HostState);

    // Host-allocated memory shared with the AOT module: params, grads, then the
    // module's primal/adjoint scratch.
    let pages = ((n_params * 2 + scratch_len) * 8).div_ceil(65536).max(1) as u32;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();

    let mut linker: Linker<HostState> = Linker::new(&engine);
    install_math(&mut linker, &mut store);
    linker.define("stan", "memory", memory).unwrap();

    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("instantiate");

    let lpg = instance
        .get_typed_func::<(i32, i32, i32, i32), f64>(&store, "log_prob_grad")
        .unwrap();

    // Layout: params at offset 0, grads at offset n_params*8.
    let params_ptr: i32 = 0;
    let grads_ptr: i32 = (n_params * 8) as i32;
    let scratch_ptr: i32 = (n_params * 16) as i32;
    let bytes: Vec<u8> = params.iter().flat_map(|p| p.to_le_bytes()).collect();
    memory
        .write(&mut store, params_ptr as usize, &bytes)
        .unwrap();
    // Re-rolled loops read their moving constants from the tail of scratch.
    if !const_table.is_empty() {
        let at = scratch_ptr as usize + (scratch_len - const_table.len()) * 8;
        let tbl: Vec<u8> = const_table.iter().flat_map(|c| c.to_le_bytes()).collect();
        memory.write(&mut store, at, &tbl).unwrap();
    }

    let lp = lpg
        .call(&mut store, (params_ptr, grads_ptr, n_params as i32, scratch_ptr))
        .unwrap();

    let mut grad_bytes = vec![0u8; n_params * 8];
    memory
        .read(&store, grads_ptr as usize, &mut grad_bytes)
        .unwrap();
    let grads: Vec<f64> = grad_bytes
        .chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    (lp, grads)
}

fn close(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps || ((a - b) / a.abs().max(b.abs()).max(1.0)).abs() < eps
}

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

#[test]
fn linear_regression_aot_matches_oracle() {
    let mut data = Env::new();
    data.set_scalar("N", 3.0);
    data.set_vector("x", &[1.0, 2.0, 3.0]);
    data.set_vector("y", &[1.5, 3.1, 4.9]);
    let model = Model::parse_and_load(LINEAR_REGRESSION, data).unwrap();

    let dummy = vec![0.1; model.n_params()];
    let compiled = compile(&model, &dummy).unwrap();

    let test_params = vec![0.5, 1.5, -0.2];
    let (oracle_lp, oracle_grads) = model.log_prob_grad(&test_params).unwrap();
    let (aot_lp, aot_grads) =
        run_aot_log_prob_grad(&compiled.wasm, compiled.n_params, &test_params, compiled.scratch_len, &compiled.const_table);

    assert!(
        close(oracle_lp, aot_lp, 1e-12),
        "lp: oracle={oracle_lp}, aot={aot_lp}, diff={}",
        oracle_lp - aot_lp
    );
    assert_eq!(oracle_grads.len(), aot_grads.len());
    for (i, (o, a)) in oracle_grads.iter().zip(aot_grads.iter()).enumerate() {
        assert!(
            close(*o, *a, 1e-12),
            "grad[{i}]: oracle={o}, aot={a}, diff={}",
            o - a
        );
    }
}

const POISSON_REGRESSION: &str = r#"
data {
  int<lower=0> N;
  vector[N] x;
  array[N] int y;
}
parameters {
  real alpha;
  real beta;
}
model {
  alpha ~ normal(0, 5);
  beta  ~ normal(0, 1);
  for (i in 1:N) y[i] ~ poisson(exp(alpha + beta * x[i]));
}
"#;

#[test]
fn poisson_regression_aot_matches_oracle() {
    let mut data = Env::new();
    data.set_scalar("N", 5.0);
    data.set_vector("x", &[0.0, 1.0, 2.0, 3.0, 4.0]);
    data.set_vector("y", &[1.0, 2.0, 5.0, 12.0, 30.0]);
    let model = Model::parse_and_load(POISSON_REGRESSION, data).unwrap();

    let dummy = vec![0.1; model.n_params()];
    let compiled = compile(&model, &dummy).unwrap();

    let test_params = vec![0.0, 1.0];
    let (oracle_lp, oracle_grads) = model.log_prob_grad(&test_params).unwrap();
    let (aot_lp, aot_grads) =
        run_aot_log_prob_grad(&compiled.wasm, compiled.n_params, &test_params, compiled.scratch_len, &compiled.const_table);

    assert!(
        close(oracle_lp, aot_lp, 1e-12),
        "lp: oracle={oracle_lp}, aot={aot_lp}"
    );
    for (i, (o, a)) in oracle_grads.iter().zip(aot_grads.iter()).enumerate() {
        assert!(close(*o, *a, 1e-12), "grad[{i}]: oracle={o}, aot={a}");
    }
}

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

#[test]
fn multivariate_lkj_aot_matches_oracle() {
    let mut data = Env::new();
    data.set_scalar("K", 2.0);
    data.set_vector("y", &[1.0, 2.0]);
    let model = Model::parse_and_load(MULTIVARIATE_LKJ, data).unwrap();

    let dummy = vec![0.1; model.n_params()];
    let compiled = compile(&model, &dummy).unwrap();

    let test_params = vec![0.5, 1.5, 0.3];
    let (oracle_lp, oracle_grads) = model.log_prob_grad(&test_params).unwrap();
    let (aot_lp, aot_grads) =
        run_aot_log_prob_grad(&compiled.wasm, compiled.n_params, &test_params, compiled.scratch_len, &compiled.const_table);

    assert!(
        close(oracle_lp, aot_lp, 1e-12),
        "lp: oracle={oracle_lp}, aot={aot_lp}"
    );
    for (i, (o, a)) in oracle_grads.iter().zip(aot_grads.iter()).enumerate() {
        assert!(close(*o, *a, 1e-12), "grad[{i}]: oracle={o}, aot={a}");
    }
}

#[test]
fn module_validates_with_wasmparser() {
    let mut data = Env::new();
    data.set_scalar("N", 2.0);
    data.set_vector("x", &[0.0, 1.0]);
    data.set_vector("y", &[0.0, 1.0]);
    let model = Model::parse_and_load(LINEAR_REGRESSION, data).unwrap();
    let compiled = compile(&model, &[0.1; 3]).unwrap();
    let result = wasmparser::Validator::new().validate_all(&compiled.wasm);
    assert!(result.is_ok(), "wasm did not validate: {:?}", result.err());
}

#[test]
fn unsupported_op_is_reported_rather_than_trapping() {
    // The emitters have no arm for Atan, and `unimplemented!` would compile to a
    // wasm trap that takes down the module instead of reporting anything.
    let src = r#"
data { int<lower=0> N; vector[N] y; }
parameters { real a; }
model { for (n in 1:N) y[n] ~ normal(atan(a), 1.0); }
"#;
    let mut data = Env::new();
    data.set_scalar("N", 2.0);
    data.set_vector("y", &[0.1, 0.2]);
    let model = Model::parse_and_load(src, data).unwrap();

    let err = stanwasm_codegen::compile(&model, &[0.1])
        .expect_err("Atan has no AOT emitter")
        .to_string();
    assert!(err.contains("Atan") && err.contains("sample()"), "{err}");
}

/// Enough data points that the emitter re-rolls the vectorised statement into
/// a wasm loop. The small cases above stay straight-line, so without this the
/// loop emitter is never exercised.
#[test]
fn rerolled_linear_regression_matches_oracle() {
    const N: usize = 1500;
    let xs: Vec<f64> = (0..N).map(|i| -1.5 + i as f64 * 0.05).collect();
    let ys: Vec<f64> = (0..N).map(|i| 0.3 + i as f64 * 0.11).collect();
    let mut data = Env::new();
    data.set_scalar("N", N as f64);
    data.set_vector("x", &xs);
    data.set_vector("y", &ys);
    let model = Model::parse_and_load(LINEAR_REGRESSION, data).unwrap();

    let dummy = vec![0.1; model.n_params()];
    let compiled = compile(&model, &dummy).unwrap();
    assert!(
        !compiled.const_table.is_empty(),
        "expected a re-rolled loop with a moving-constant table"
    );

    for test_params in [
        vec![0.5, 1.5, -0.2],
        vec![-1.0, 0.25, 0.7],
        vec![0.0, 0.0, 0.0],
    ] {
        let (oracle_lp, oracle_grads) = model.log_prob_grad(&test_params).unwrap();
        let (aot_lp, aot_grads) = run_aot_log_prob_grad(
            &compiled.wasm,
            compiled.n_params,
            &test_params,
            compiled.scratch_len,
            &compiled.const_table,
        );
        assert!(
            close(oracle_lp, aot_lp, 1e-12),
            "lp at {test_params:?}: oracle={oracle_lp}, aot={aot_lp}"
        );
        for (i, (o, a)) in oracle_grads.iter().zip(aot_grads.iter()).enumerate() {
            assert!(
                close(*o, *a, 1e-12),
                "grad[{i}] at {test_params:?}: oracle={o}, aot={a}, diff={}",
                o - a
            );
        }
    }
}

/// Calling twice must give the same answer: the scratch buffer is reused, so a
/// stale adjoint or a clobbered constant table would only show on the second
/// call.
#[test]
fn rerolled_model_is_reentrant() {
    const N: usize = 1500;
    let mut data = Env::new();
    data.set_scalar("N", N as f64);
    data.set_vector("x", &(0..N).map(|i| i as f64 * 0.1).collect::<Vec<_>>());
    data.set_vector("y", &(0..N).map(|i| 1.0 + i as f64 * 0.2).collect::<Vec<_>>());
    let model = Model::parse_and_load(LINEAR_REGRESSION, data).unwrap();
    let compiled = compile(&model, &vec![0.1; model.n_params()]).unwrap();
    assert!(!compiled.const_table.is_empty(), "expected a re-rolled loop");

    let p = vec![0.4, 1.1, 0.3];
    let first = run_aot_log_prob_grad(
        &compiled.wasm,
        compiled.n_params,
        &p,
        compiled.scratch_len,
        &compiled.const_table,
    );
    let (oracle_lp, oracle_grads) = model.log_prob_grad(&p).unwrap();
    assert!(close(first.0, oracle_lp, 1e-12));
    for (o, a) in oracle_grads.iter().zip(first.1.iter()) {
        assert!(close(*o, *a, 1e-12));
    }
}

/// A hierarchical model whose group index is irregular. `mu[g[i]]` is the
/// gather no stride describes, so the emitter has to read its slot index from
/// a table — the one loop shape whose addresses are computed at run time.
#[test]
fn rerolled_gather_matches_oracle() {
    const N: usize = 3000;
    const G: usize = 8;
    let src = r#"data { int<lower=0> N; int<lower=1> G; array[N] int<lower=1> g; vector[N] y; }
parameters { vector[G] mu; real<lower=0> sigma; }
model {
  mu ~ normal(0, 5); sigma ~ exponential(1);
  for (i in 1:N) y[i] ~ normal(mu[g[i]], sigma);
}"#;
    let mut seed: u64 = 12345;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223) & 0xffff_ffff;
        seed as f64 / 4294967296.0
    };
    let gs: Vec<String> = (0..N).map(|_| format!("{}", 1 + (rnd() * 8.0) as u32)).collect();
    let ys: Vec<String> = (0..N).map(|i| format!("{}", (i as f64).sin() * 2.0)).collect();
    let data_json = format!(
        "{{\"N\": {N}, \"G\": {G}, \"g\": [{}], \"y\": [{}]}}",
        gs.join(","),
        ys.join(",")
    );
    let model =
        Model::parse_and_load(src, stanwasm_runtime::data_from_json(&data_json).unwrap()).unwrap();

    let compiled = compile(&model, &vec![0.1; model.n_params()]).unwrap();
    assert!(
        !compiled.const_table.is_empty(),
        "expected re-rolled loops with staged tables"
    );

    for test_params in [
        vec![0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, 0.05, 0.5],
        vec![-0.7, 0.9, 0.0, 1.2, 0.3, -1.1, 0.6, -0.4, -0.2],
    ] {
        let (oracle_lp, oracle_grads) = model.log_prob_grad(&test_params).unwrap();
        let (aot_lp, aot_grads) = run_aot_log_prob_grad(
            &compiled.wasm,
            compiled.n_params,
            &test_params,
            compiled.scratch_len,
            &compiled.const_table,
        );
        assert!(
            close(oracle_lp, aot_lp, 1e-10),
            "lp: oracle={oracle_lp}, aot={aot_lp}"
        );
        for (i, (o, a)) in oracle_grads.iter().zip(aot_grads.iter()).enumerate() {
            assert!(
                close(*o, *a, 1e-10),
                "grad[{i}]: oracle={o}, aot={a}, diff={}",
                o - a
            );
        }
    }
}

/// A matrix-vector product: one contraction node per row, in a block of its
/// own, whose result the density block reads back. That crossing is the only
/// place the scratch buffer's slot order is observable, and the contraction
/// has two emitters — unrolled in place, or inside a loop reading a staged
/// column of coefficients — which have to agree with each other and with the
/// oracle.
#[test]
fn matrix_product_matches_oracle_in_every_reroll_mode() {
    use stanwasm_codegen::{compile_with, Reroll};
    const N: usize = 2000;
    const K: usize = 4;
    let src = r#"data { int<lower=0> N; int<lower=0> K; matrix[N,K] X; vector[N] y; }
parameters { vector[K] beta; real<lower=0> sigma; }
model {
  beta ~ normal(0, 1); sigma ~ exponential(1);
  y ~ normal(X * beta, sigma);
}"#;
    let mut seed: u64 = 6789;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223) & 0xffff_ffff;
        seed as f64 / 4294967296.0 * 2.0 - 1.0
    };
    let rows: Vec<String> = (0..N)
        .map(|_| {
            let cells: Vec<String> = (0..K).map(|_| format!("{}", rnd())).collect();
            format!("[{}]", cells.join(","))
        })
        .collect();
    let ys: Vec<String> = (0..N).map(|_| format!("{}", rnd())).collect();
    let data_json = format!(
        "{{\"N\": {N}, \"K\": {K}, \"X\": [{}], \"y\": [{}]}}",
        rows.join(","),
        ys.join(",")
    );
    let model =
        Model::parse_and_load(src, stanwasm_runtime::data_from_json(&data_json).unwrap()).unwrap();

    let dummy = vec![0.1; model.n_params()];
    for mode in [Reroll::Auto, Reroll::Always, Reroll::Never] {
        let compiled = compile_with(&model, &dummy, mode).unwrap();
        if mode != Reroll::Never {
            assert!(
                !compiled.const_table.is_empty(),
                "{mode:?}: expected re-rolled loops with staged coefficients"
            );
        }
        for test_params in [
            vec![0.5, -0.3, 0.8, 0.1, -0.5],
            vec![-1.2, 0.4, 0.0, 0.9, 0.25],
        ] {
            let (oracle_lp, oracle_grads) = model.log_prob_grad(&test_params).unwrap();
            let (aot_lp, aot_grads) = run_aot_log_prob_grad(
                &compiled.wasm,
                compiled.n_params,
                &test_params,
                compiled.scratch_len,
                &compiled.const_table,
            );
            assert!(
                close(oracle_lp, aot_lp, 1e-10),
                "{mode:?}: lp oracle={oracle_lp}, aot={aot_lp}"
            );
            for (i, (o, a)) in oracle_grads.iter().zip(aot_grads.iter()).enumerate() {
                assert!(
                    close(*o, *a, 1e-10),
                    "{mode:?}: grad[{i}] oracle={o}, aot={a}, diff={}",
                    o - a
                );
            }
        }
    }
}

/// Every re-roll mode has to compute the same gradient. `Always` and `Never`
/// exercise the loop and straight-line emitters on the same trace, which is
/// the only place their outputs can be compared directly.
#[test]
fn reroll_modes_agree() {
    use stanwasm_codegen::{compile_with, Reroll};
    const N: usize = 400;
    let xs: Vec<f64> = (0..N).map(|i| -1.5 + i as f64 * 0.007).collect();
    let ys: Vec<f64> = (0..N).map(|i| 0.3 + i as f64 * 0.011).collect();
    let mut data = Env::new();
    data.set_scalar("N", N as f64);
    data.set_vector("x", &xs);
    data.set_vector("y", &ys);
    let model = Model::parse_and_load(LINEAR_REGRESSION, data).unwrap();
    let dummy = vec![0.1; model.n_params()];
    let test_params = vec![0.4, 1.3, -0.15];
    let (oracle_lp, oracle_grads) = model.log_prob_grad(&test_params).unwrap();

    for mode in [Reroll::Auto, Reroll::Always, Reroll::Never] {
        let c = compile_with(&model, &dummy, mode).unwrap();
        let (lp, grads) = run_aot_log_prob_grad(
            &c.wasm,
            c.n_params,
            &test_params,
            c.scratch_len,
            &c.const_table,
        );
        assert!(close(oracle_lp, lp, 1e-12), "{mode:?}: lp {oracle_lp} vs {lp}");
        for (i, (o, a)) in oracle_grads.iter().zip(grads.iter()).enumerate() {
            assert!(close(*o, *a, 1e-12), "{mode:?}: grad[{i}] {o} vs {a}");
        }
    }
}
