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
        run_aot_log_prob_grad(&compiled.wasm, compiled.n_params, &test_params, compiled.scratch_len);

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
        run_aot_log_prob_grad(&compiled.wasm, compiled.n_params, &test_params, compiled.scratch_len);

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
        run_aot_log_prob_grad(&compiled.wasm, compiled.n_params, &test_params, compiled.scratch_len);

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
