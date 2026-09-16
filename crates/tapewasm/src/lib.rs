//! The browser API: compile a tape to a wasm module, then sample it.
//!
//! `compileTape` reads the text form of a tape and emits a module;
//! `AotSampler` binds one and runs nuts-rs over it. The module and this one
//! share linear memory, so a draw costs one call across the boundary and no
//! copying.
//!
//! Whatever wrote the tape stays outside: a front end in JavaScript, in
//! Python under Pyodide, or in another wasm module hands over text and gets
//! draws back.

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::{E, PI};

use nuts_rs::{
    sample_sequentially, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError,
};

use rand::{rngs::ChaCha8Rng, Rng, RngExt, SeedableRng};

use thiserror::Error;

use wasm_bindgen::prelude::*;

/// The one thing a log density can go wrong by, from the sampler's side.
#[derive(Debug, Error)]
pub enum SamplerError {
    #[error("logp returned non-finite value")]
    NonFinite,
}

impl LogpError for SamplerError {
    fn is_recoverable(&self) -> bool {
        true
    }
}

/// The sampler configuration every entry point uses.
///
/// nuts-rs estimates the diagonal metric from both the draws and the gradients
/// by default. The reference implementations this project's posteriors are
/// checked against use the draws alone, so this turns the gradient term off: on a centred hierarchical model the two disagree
/// about how far down the funnel the sampler goes, which is a difference in
/// the posterior, not in the log density.
pub fn nuts_settings(num_warmup: u32, num_draws: u32) -> DiagNutsSettings {
    let mut settings = DiagNutsSettings {
        num_tune: num_warmup as u64,
        num_draws: num_draws as u64,
        ..Default::default()
    };
    settings
        .adapt_options
        .mass_matrix_options
        .use_grad_based_estimate = false;
    settings
}

/// nuts-rs asserts that its step-size adaptation has somewhere to run, and an
/// assertion inside wasm is a trap the caller cannot tell apart from any other.
pub fn no_warmup_check(num_warmup: u32) -> Result<(), JsError> {
    if num_warmup == 0 {
        return Err(JsError::new(
            "num_warmup must be at least 1: the sampler adapts its step size \
             during warmup and has no schedule to do it on with none",
        ));
    }
    Ok(())
}

/// nuts-rs refuses a starting point whose gradient has a zero component — the
/// mass matrix it adapts is scaled by that gradient — and reports only
/// "Invalid initial point", which names neither the rule nor the parameter.
pub fn init_gradient_check(names: &[String], lp: f64, grad: &[f64]) -> Result<(), String> {
    if !lp.is_finite() {
        return Err(format!(
            "log density is {lp} at the starting point; the sampler needs a \
             finite one to begin from"
        ));
    }
    let named = |predicate: fn(f64) -> bool| {
        grad.iter()
            .enumerate()
            .filter(|(_, g)| predicate(**g))
            .map(|(i, _)| names.get(i).map_or("?", String::as_str))
            .collect::<Vec<_>>()
    };
    let nan = named(f64::is_nan);
    if !nan.is_empty() {
        return Err(format!(
            "gradient is NaN for {} of the {} parameters at the starting point \
             ({}); this points to numerical arithmetic rather than a \
             structurally flat model",
            nan.len(),
            grad.len(),
            nan.join(", "),
        ));
    }
    let infinite = named(f64::is_infinite);
    if !infinite.is_empty() {
        return Err(format!(
            "gradient is infinite for {} of the {} parameters at the starting \
             point ({}); this points to numerical arithmetic rather than a \
             structurally flat model",
            infinite.len(),
            grad.len(),
            infinite.join(", "),
        ));
    }
    let bad: Vec<&str> = grad
        .iter()
        .enumerate()
        .filter(|(_, g)| **g == 0.0)
        .map(|(i, _)| names.get(i).map_or("?", String::as_str))
        .collect();
    if bad.is_empty() {
        return Ok(());
    }
    let shown = bad.iter().take(6).copied().collect::<Vec<_>>().join(", ");
    let rest = if bad.len() > 6 {
        format!(" and {} more", bad.len() - 6)
    } else {
        String::new()
    };
    Err(format!(
        "the log density does not move with {} of the {} parameters at the \
         starting point ({shown}{rest}), and the sampler cannot begin from \
         there. Start somewhere else — a random point moves every parameter \
         a flat one does not — or drop the parameters the data says nothing \
         about",
        bad.len(),
        grad.len(),
    ))
}

#[cfg(feature = "codegen")]
pub fn jserr<E: std::fmt::Display>(e: E) -> JsError {
    JsError::new(&e.to_string())
}

/// Forwards Rust panics to `console.error` with a message and backtrace rather
/// than an opaque `RuntimeError: unreachable`. Diagnostics: the instance still traps.
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen(js_name = tapewasmVersion)]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// AOT bridge: `sample_via_aot` swaps tape replay for a host-provided AOT wasm
// sharing this module's linear memory. Bind it via `setAotExports` first.

#[wasm_bindgen(module = "/js/aot_bridge.js")]
extern "C" {
    #[wasm_bindgen(js_name = aot_logp)]
    fn aot_logp(params_ptr: u32, grads_ptr: u32, n_params: u32, scratch_ptr: u32) -> f64;

    #[wasm_bindgen(js_name = set_aot_exports)]
    fn js_set_aot_exports(exports: JsValue);

    #[wasm_bindgen(js_name = clear_aot_exports)]
    fn js_clear_aot_exports();

    /// The bound module's `tapewasm_layout_id`, or NaN when nothing is bound.
    #[wasm_bindgen(js_name = aot_layout_id)]
    fn aot_layout_id() -> f64;

    /// The bound module's `tapewasm_abi_version`, or NaN when it exports none.
    #[wasm_bindgen(js_name = aot_abi_version)]
    fn aot_abi_version() -> f64;
}

#[wasm_bindgen]
extern "C" {
    /// `advi`'s progress hook, called as `(iter, mu, elbo)`.
    #[wasm_bindgen(
        typescript_type = "((iter: number, mu: Float64Array, elbo: Float64Array) => void)"
    )]
    pub type AdviSnapshotCallback;

    #[wasm_bindgen(method, catch, js_name = call)]
    fn call3(
        this: &AdviSnapshotCallback,
        ctx: &JsValue,
        iter: f64,
        mu: Vec<f64>,
        elbo: Vec<f64>,
    ) -> Result<JsValue, JsValue>;
}

/// The module shape this build knows how to run.
///
/// Kept here rather than read from `tapewasm-codegen`, which the sampler-only
/// build does not depend on. `abi_version_agrees` asserts the two are the same
/// number wherever both are present.
pub const ABI_VERSION: u32 = 1;

/// Refuse a binding that belongs to a different compilation.
///
/// `setAotExports` binds one module for the whole page, while the scratch
/// buffer belongs to a single model. Running model B's module against model
/// A's buffer writes at slot offsets A never sized for, so this compares the
/// two ids before the sampler takes the buffer.
pub fn check_aot_binding(want: u32) -> Result<(), JsError> {
    let bound = aot_layout_id();
    if bound.is_nan() {
        return Err(JsError::new(
            "no AOT module is bound: call setAotExports(instance.exports) with \
             the module this tape was compiled to. A module built by an \
             older tapewasm exports no layout id and cannot be checked, \
             so it is refused here too.",
        ));
    }
    let abi = aot_abi_version();
    if abi.is_nan() || abi as u32 != ABI_VERSION {
        // Only reachable with a module and a runtime from different releases,
        // which a page deploys together — so name both numbers and stop.
        let found = if abi.is_nan() {
            "no version at all".to_string()
        } else {
            format!("module ABI {}", abi as u32)
        };
        return Err(JsError::new(&format!(
            "the bound AOT module names {found}, and this tapewasm runs module \
             ABI {ABI_VERSION}. A precompiled module and the runtime that \
             samples it ship together, so recompile the module with this \
             version, or serve the runtime it was built with.",
        )));
    }
    if bound as u32 != want {
        return Err(JsError::new(
            "the bound AOT module was compiled for a different model. \
             setAotExports() binds one module per page, so re-bind this \
             model's own module before sampling it — the module reads and \
             writes a scratch buffer laid out for the model it was compiled \
             from.",
        ));
    }
    Ok(())
}

/// Bind a freshly-instantiated model module's exports so sampling dispatches
/// through it. Pass `instance.exports`.
///
/// One binding serves the whole page, so a page holding several models has to
/// re-bind before sampling a different one. The sampler compares the bound
/// module's `tapewasm_layout_id` against the model it belongs to and refuses
/// the pair rather than running it against the wrong scratch buffer.
#[wasm_bindgen(js_name = setAotExports)]
pub fn set_aot_exports(exports: JsValue) {
    js_set_aot_exports(exports);
}

/// Release the bound module's exports. The next draw will throw.
#[wasm_bindgen(js_name = clearAotExports)]
pub fn clear_aot_exports() {
    js_clear_aot_exports();
}

/// The linear memory backing this module. Pass as the `tapewasm.memory`
/// import when instantiating a model module so the two share buffers.
#[wasm_bindgen(js_name = sharedMemory)]
pub fn shared_memory() -> JsValue {
    wasm_bindgen::memory()
}

/// nuts-rs' log density, answered by the bound module rather than in Rust.
pub struct AotLogp {
    n_params: usize,
    /// Primal and adjoint storage the AOT module works in, two f64 per node.
    scratch_buf: Vec<f64>,
}

impl AotLogp {
    /// `scratch_buf` is what the module works in — `Compiled::scratch_len`
    /// slots, with the re-rolled loops' constants already staged at the end.
    pub fn new(n_params: usize, scratch_buf: Vec<f64>) -> Self {
        Self {
            n_params,
            scratch_buf,
        }
    }
}

impl HasDims for AotLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        let n = self.n_params as u64;
        [
            ("unconstrained_parameter".to_string(), n),
            ("dim".to_string(), n),
        ]
        .into_iter()
        .collect()
    }
}

impl CpuLogpFunc for AotLogp {
    type LogpError = SamplerError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.n_params
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, SamplerError> {
        // The caller's own slices are already in the memory the module imports,
        // and the ABI reads every parameter before it stores any gradient, so
        // handing over their addresses is what a relaying buffer would do.
        let params_ptr = position.as_ptr() as u32;
        let grads_ptr = gradient.as_mut_ptr() as u32;
        let scratch_ptr = self.scratch_buf.as_mut_ptr() as u32;
        let lp = aot_logp(params_ptr, grads_ptr, self.n_params as u32, scratch_ptr);
        if lp.is_finite() {
            Ok(lp)
        } else {
            Err(SamplerError::NonFinite)
        }
    }

    fn expand_vector<R>(&mut self, _rng: &mut R, array: &[f64]) -> Result<Vec<f64>, CpuMathError>
    where
        R: rand::Rng + ?Sized,
    {
        Ok(array.to_vec())
    }
}

/// A sampler over one compiled module.
///
/// It reaches the module alone: nothing that recorded or emitted the tape is
/// on this path, so a page that loads a module built earlier needs no
/// compiler in its bundle. Bind the module with `setAotExports` first.
#[wasm_bindgen]
pub struct AotSampler {
    n_params: usize,
    scratch_init: Vec<f64>,
    layout_id: u32,
    param_names: Vec<String>,
    target_accept: Option<f64>,
    grad_based_estimate: Option<bool>,
    /// `logProbGrad`'s evaluator, kept so a run of calls copies `scratch_init`
    /// once rather than once each. A cell rather than `&mut self`, so the
    /// method stays a shared borrow and a snapshot callback can still call it
    /// while `advi` runs.
    evaluator: RefCell<Option<AotLogp>>,
}

#[wasm_bindgen]
impl AotSampler {
    /// `scratch_init` is the buffer the module works in — zeroed primals and
    /// adjoints followed by the re-rolled loops' constant table, which is what
    /// `aotScratchInit` returns and what a compiler should record beside the
    /// module. `layout_id` is the module's `tapewasm_layout_id` global, checked
    /// against the bound exports before each run so a module and a scratch
    /// buffer built for different models cannot be used together.
    ///
    /// `param_names` only names a parameter in a rejected starting point; pass
    /// an empty array to go without.
    #[wasm_bindgen(constructor)]
    pub fn new(
        n_params: usize,
        scratch_init: Vec<f64>,
        layout_id: u32,
        param_names: Vec<String>,
    ) -> Result<AotSampler, JsError> {
        if n_params == 0 {
            return Err(JsError::new("n_params must be at least 1"));
        }
        if scratch_init.len() < 2 * n_params {
            return Err(JsError::new(&format!(
                "scratch_init has {} slots, too few for {n_params} parameters",
                scratch_init.len()
            )));
        }
        if !param_names.is_empty() && param_names.len() != n_params {
            return Err(JsError::new(&format!(
                "param_names has {} entries but n_params is {n_params}",
                param_names.len()
            )));
        }
        Ok(Self {
            n_params,
            scratch_init,
            layout_id,
            param_names,
            target_accept: None,
            grad_based_estimate: None,
            evaluator: RefCell::new(None),
        })
    }

    #[wasm_bindgen(getter, js_name = nParams)]
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Aim warmup's step-size adaptation at this acceptance rate instead of
    /// nuts-rs's 0.8. Higher adapts a smaller step: fewer divergences on a hard
    /// geometry, more gradients per draw.
    #[wasm_bindgen(js_name = setTargetAccept)]
    pub fn set_target_accept(&mut self, target: f64) -> Result<(), JsError> {
        if !(target > 0.0 && target < 1.0) {
            return Err(JsError::new(&format!(
                "target_accept must lie strictly between 0 and 1, not {target}"
            )));
        }
        self.target_accept = Some(target);
        Ok(())
    }

    /// Estimate the diagonal metric from the gradients as well as the draws, as
    /// nuts-rs does by default. Off unless set, to match the reference
    /// posteriors; see [`nuts_settings`].
    #[wasm_bindgen(js_name = setGradBasedEstimate)]
    pub fn set_grad_based_estimate(&mut self, on: bool) {
        self.grad_based_estimate = Some(on);
    }

    fn settings(&self, num_warmup: u32, num_draws: u32) -> DiagNutsSettings {
        let mut settings = nuts_settings(num_warmup, num_draws);
        if let Some(target) = self.target_accept {
            settings.adapt_options.step_size_settings.target_accept = target;
        }
        if let Some(on) = self.grad_based_estimate {
            settings
                .adapt_options
                .mass_matrix_options
                .use_grad_based_estimate = on;
        }
        settings
    }

    fn logp_fn(&self) -> AotLogp {
        AotLogp {
            n_params: self.n_params,
            scratch_buf: self.scratch_init.clone(),
        }
    }

    /// `[log_prob, d/dparam...]`.
    ///
    /// The evaluator behind it is built on the first call and reused, so
    /// calling this in a row costs one `scratch_init` copy rather than one per
    /// call — the same evaluator `sample` and `advi` keep for a whole run.
    #[wasm_bindgen(js_name = logProbGrad)]
    pub fn log_prob_grad(&self, params: &[f64]) -> Result<Vec<f64>, JsError> {
        if params.len() != self.n_params {
            return Err(JsError::new(&format!(
                "params length {} != n_params {}",
                params.len(),
                self.n_params
            )));
        }
        check_aot_binding(self.layout_id)?;
        let mut out = vec![0.0_f64; self.n_params + 1];
        let mut held = self.evaluator.borrow_mut();
        let logp_fn = held.get_or_insert_with(|| self.logp_fn());
        out[0] = logp_fn
            .logp(params, &mut out[1..])
            .map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(out)
    }

    /// `num_warmup + num_draws` draws, row-major, `n_params` wide.
    pub fn sample(
        &self,
        init: &[f64],
        num_warmup: u32,
        num_draws: u32,
        seed: u64,
    ) -> Result<Vec<f64>, JsError> {
        Ok(self.run(init, num_warmup, num_draws, seed, 0, false)?.draws)
    }

    /// [`sample`](Self::sample) with each draw's sampler statistics beside it —
    /// what ArviZ keeps as `sample_stats`. `chain` only labels the run: a
    /// different seed is what keeps two chains apart.
    #[wasm_bindgen(js_name = sampleWithStats)]
    pub fn sample_with_stats(
        &self,
        init: &[f64],
        num_warmup: u32,
        num_draws: u32,
        seed: u64,
        chain: u32,
    ) -> Result<SampleResult, JsError> {
        self.run(init, num_warmup, num_draws, seed, chain, true)
    }

    fn run(
        &self,
        init: &[f64],
        num_warmup: u32,
        num_draws: u32,
        seed: u64,
        chain: u32,
        stats: bool,
    ) -> Result<SampleResult, JsError> {
        let n = self.n_params;
        if init.len() != n {
            return Err(JsError::new(&format!(
                "init length {} != n_params {n}",
                init.len()
            )));
        }
        no_warmup_check(num_warmup)?;
        check_aot_binding(self.layout_id)?;

        let mut grad = vec![0.0_f64; n];
        let lp = self
            .logp_fn()
            .logp(init, &mut grad)
            .map_err(|e| JsError::new(&format!("{e}")))?;
        init_gradient_check(&self.param_names, lp, &grad).map_err(|e| JsError::new(&e))?;

        // Widen before adding: `u32 + u32` wraps, and a wrapped total
        // silently becomes a different (possibly enormous) run length.
        let total = num_warmup as u64 + num_draws as u64;
        let math = CpuMath::new(self.logp_fn());
        let settings = self.settings(num_warmup, num_draws);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let iter = sample_sequentially(math, settings, init, total, chain as u64, &mut rng)
            .map_err(|e| JsError::new(&format!("nuts-rs init: {e}")))?;

        let mut out = SampleResult {
            draws: vec![0.0_f64; n * total as usize],
            ..SampleResult::default()
        };
        // The log density at a draw costs one more evaluation, so only when asked for.
        let mut lp_fn = stats.then(|| self.logp_fn());
        for (i, draw) in iter.enumerate() {
            let (pos, progress) = draw.map_err(|e| JsError::new(&format!("nuts-rs draw: {e}")))?;
            out.draws[i * n..(i + 1) * n].copy_from_slice(pos.as_ref());
            if let Some(f) = lp_fn.as_mut() {
                out.diverging.push(progress.diverging as u8);
                out.tuning.push(progress.tuning as u8);
                out.step_size.push(progress.step_size);
                out.num_steps.push(progress.num_steps as u32);
                let lp = f
                    .logp(&pos, &mut grad)
                    .map_err(|e| JsError::new(&format!("{e}")))?;
                out.lp.push(lp);
            }
        }
        Ok(out)
    }

    /// Mean-field ADVI: fits `q(θ) = N(μ, diag(σ²))` to the log density by
    /// Adam-ascending the ELBO under the reparameterization trick
    /// `θ = μ + σ·η`, `η ~ N(0, I)`. `init` seeds `μ`; `ω = log σ` starts at
    /// `0` (`σ = 1`).
    ///
    /// The ELBO gradient combines the tape's `∂logp/∂θ` (from `log_prob_grad`)
    /// with the reparameterization's own Jacobian and the mean-field Gaussian
    /// entropy's gradient, both exact rather than estimated:
    /// `∂ELBO/∂μ_i = ∂logp/∂θ_i`, `∂ELBO/∂ω_i = ∂logp/∂θ_i · σ_i · η_i + 1`.
    /// Each is averaged over `mc_samples` reparameterized draws per iteration.
    ///
    /// `AdviResult::mu`/`sigma` average `(μ, ω)` over the second half of
    /// `num_iters` (Polyak averaging): a constant Adam step size never settles
    /// on one point, so the plain last iterate keeps wandering a "noise ball"
    /// around the optimum. `elbo_trace` still reports the raw iterate's ELBO
    /// per step, for diagnosing convergence rather than for the fitted result.
    ///
    /// `snapshot_every` (0 disables it) additionally copies the raw iterate
    /// `μ` every that-many iterations into `AdviResult::muSnapshots` — a film
    /// strip of the one training run, rather than several shorter ones
    /// spliced together: restarting Adam's moment estimates partway through
    /// measurably converges to a worse optimum, so this is the only way to
    /// watch a run progress without paying for that.
    ///
    /// `on_snapshot`, if given, is also called at each snapshot with the
    /// iteration, a copy of that `μ`, and the ELBO trace since the previous
    /// call — so a run inside a Worker can report as it goes rather than only
    /// once it returns.
    #[allow(clippy::too_many_arguments)] // positional, as JS calls it
    pub fn advi(
        &self,
        init: &[f64],
        num_iters: u32,
        mc_samples: u32,
        learning_rate: f64,
        seed: u64,
        snapshot_every: u32,
        on_snapshot: Option<AdviSnapshotCallback>,
    ) -> Result<AdviResult, JsError> {
        let n = self.n_params;
        if init.len() != n {
            return Err(JsError::new(&format!(
                "init length {} != n_params {n}",
                init.len()
            )));
        }
        if num_iters == 0 {
            return Err(JsError::new("num_iters must be at least 1"));
        }
        if mc_samples == 0 {
            return Err(JsError::new("mc_samples must be at least 1"));
        }
        if learning_rate.is_nan() || learning_rate <= 0.0 {
            return Err(JsError::new("learning_rate must be positive"));
        }
        check_aot_binding(self.layout_id)?;

        const BETA1: f64 = 0.9;
        const BETA2: f64 = 0.999;
        const EPS: f64 = 1e-8;

        let mut logp_fn = self.logp_fn();
        let mut mu = init.to_vec();
        let mut omega = vec![0.0_f64; n];
        let (mut m_mu, mut v_mu) = (vec![0.0_f64; n], vec![0.0_f64; n]);
        let (mut m_omega, mut v_omega) = (vec![0.0_f64; n], vec![0.0_f64; n]);

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut elbo_trace = Vec::with_capacity(num_iters as usize);

        let mut eta = vec![0.0_f64; n];
        let mut sigma = vec![0.0_f64; n];
        let mut theta = vec![0.0_f64; n];
        let mut grad = vec![0.0_f64; n];
        let mut grad_mu = vec![0.0_f64; n];
        let mut grad_omega = vec![0.0_f64; n];

        let burn_in = num_iters / 2;
        let mut sum_mu = vec![0.0_f64; n];
        let mut sum_omega = vec![0.0_f64; n];
        let mut avg_count: u32 = 0;

        let mut mu_snapshots = Vec::new();
        let mut snapshot_iters = Vec::new();
        let mut elbo_reported = 0;

        for t in 1..=num_iters {
            grad_mu.iter_mut().for_each(|g| *g = 0.0);
            grad_omega.iter_mut().for_each(|g| *g = 0.0);
            let mut lp_sum = 0.0;
            // `omega` only moves with Adam, below, so every MC sample in this
            // iteration shares one exponential per parameter.
            for i in 0..n {
                sigma[i] = omega[i].exp();
            }

            for _ in 0..mc_samples {
                fill_standard_normal(&mut rng, &mut eta);
                for i in 0..n {
                    theta[i] = mu[i] + sigma[i] * eta[i];
                }
                let lp = logp_fn
                    .logp(&theta, &mut grad)
                    .map_err(|e| JsError::new(&format!("advi: {e} at iteration {t}")))?;
                if !lp.is_finite() {
                    return Err(JsError::new(&format!(
                        "advi: log density is non-finite at iteration {t}; \
                         lower the learning rate or check the model at \
                         extreme unconstrained values"
                    )));
                }
                lp_sum += lp;
                for i in 0..n {
                    grad_mu[i] += grad[i];
                    grad_omega[i] += grad[i] * sigma[i] * eta[i];
                }
            }

            let s = mc_samples as f64;
            let bias1 = 1.0 - BETA1.powi(t as i32);
            let bias2 = 1.0 - BETA2.powi(t as i32);
            for i in 0..n {
                let g_mu = grad_mu[i] / s;
                let g_omega = grad_omega[i] / s + 1.0;

                m_mu[i] = BETA1 * m_mu[i] + (1.0 - BETA1) * g_mu;
                v_mu[i] = BETA2 * v_mu[i] + (1.0 - BETA2) * g_mu * g_mu;
                mu[i] += learning_rate * (m_mu[i] / bias1) / ((v_mu[i] / bias2).sqrt() + EPS);

                m_omega[i] = BETA1 * m_omega[i] + (1.0 - BETA1) * g_omega;
                v_omega[i] = BETA2 * v_omega[i] + (1.0 - BETA2) * g_omega * g_omega;
                omega[i] +=
                    learning_rate * (m_omega[i] / bias1) / ((v_omega[i] / bias2).sqrt() + EPS);
            }

            let entropy = omega.iter().sum::<f64>() + 0.5 * n as f64 * (2.0 * PI * E).ln();
            elbo_trace.push(lp_sum / s + entropy);

            if t > burn_in {
                avg_count += 1;
                for i in 0..n {
                    sum_mu[i] += mu[i];
                    sum_omega[i] += omega[i];
                }
            }

            if snapshot_every > 0 && (t % snapshot_every == 0 || t == num_iters) {
                mu_snapshots.extend_from_slice(&mu);
                snapshot_iters.push(t as f64);
                if let Some(f) = &on_snapshot {
                    let elbo = elbo_trace[elbo_reported..].to_vec();
                    elbo_reported = elbo_trace.len();
                    f.call3(&JsValue::UNDEFINED, t as f64, mu.clone(), elbo)
                        .map_err(|e| JsError::new(&format!("advi: on_snapshot threw: {e:?}")))?;
                }
            }
        }

        let count = avg_count as f64;
        Ok(AdviResult {
            mu: sum_mu.iter().map(|s| s / count).collect(),
            sigma: sum_omega.iter().map(|s| (s / count).exp()).collect(),
            elbo_trace,
            mu_snapshots,
            snapshot_iters,
        })
    }
}

/// Draws and, beside each, the sampler's statistics — warmup first, as
/// [`AotSampler::sample`] returns them.
#[wasm_bindgen]
#[derive(Default)]
pub struct SampleResult {
    draws: Vec<f64>,
    diverging: Vec<u8>,
    tuning: Vec<u8>,
    step_size: Vec<f64>,
    num_steps: Vec<u32>,
    lp: Vec<f64>,
}

#[wasm_bindgen]
impl SampleResult {
    /// `num_warmup + num_draws` draws, row-major, `n_params` wide.
    #[wasm_bindgen(getter)]
    pub fn draws(&self) -> Vec<f64> {
        self.draws.clone()
    }

    /// 1 where the draw's trajectory diverged.
    #[wasm_bindgen(getter)]
    pub fn diverging(&self) -> Vec<u8> {
        self.diverging.clone()
    }

    /// 1 for the warmup draws.
    #[wasm_bindgen(getter)]
    pub fn tuning(&self) -> Vec<u8> {
        self.tuning.clone()
    }

    #[wasm_bindgen(getter, js_name = stepSize)]
    pub fn step_size(&self) -> Vec<f64> {
        self.step_size.clone()
    }

    /// Leapfrog steps the draw's trajectory took.
    #[wasm_bindgen(getter, js_name = numSteps)]
    pub fn num_steps(&self) -> Vec<u32> {
        self.num_steps.clone()
    }

    /// The log density at each draw.
    #[wasm_bindgen(getter)]
    pub fn lp(&self) -> Vec<f64> {
        self.lp.clone()
    }
}

/// The fitted mean-field variational distribution and its ELBO trajectory.
#[wasm_bindgen]
pub struct AdviResult {
    mu: Vec<f64>,
    sigma: Vec<f64>,
    elbo_trace: Vec<f64>,
    mu_snapshots: Vec<f64>,
    snapshot_iters: Vec<f64>,
}

#[wasm_bindgen]
impl AdviResult {
    #[wasm_bindgen(getter)]
    pub fn mu(&self) -> Vec<f64> {
        self.mu.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn sigma(&self) -> Vec<f64> {
        self.sigma.clone()
    }

    /// The ELBO estimate after each iteration, in order.
    #[wasm_bindgen(getter, js_name = elboTrace)]
    pub fn elbo_trace(&self) -> Vec<f64> {
        self.elbo_trace.clone()
    }

    /// The raw (pre-averaging) iterate `μ` every `snapshot_every` iterations,
    /// one snapshot's `n_params` values after another. Empty unless `advi`
    /// was called with `snapshot_every > 0`.
    #[wasm_bindgen(getter, js_name = muSnapshots)]
    pub fn mu_snapshots(&self) -> Vec<f64> {
        self.mu_snapshots.clone()
    }

    /// The iteration number each entry of `muSnapshots` was taken at.
    #[wasm_bindgen(getter, js_name = snapshotIters)]
    pub fn snapshot_iters(&self) -> Vec<f64> {
        self.snapshot_iters.clone()
    }
}

/// Standard normal draws by Box-Muller, so one distribution does not pull in
/// `rand_distr`; seeded like `sample()`, a run is reproducible.
fn fill_standard_normal<R: Rng>(rng: &mut R, out: &mut [f64]) {
    let (pairs, rest) = out.as_chunks_mut::<2>();
    for pair in pairs {
        let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2: f64 = rng.random();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        pair[0] = r * theta.cos();
        pair[1] = r * theta.sin();
    }
    if let Some(last) = rest.first_mut() {
        let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2: f64 = rng.random();
        *last = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    }
}

/// What [`compile_tape`] produced, and everything [`AotSampler`] needs from it.
#[cfg(feature = "codegen")]
#[wasm_bindgen]
pub struct CompiledTape {
    wasm: Vec<u8>,
    n_params: usize,
    scratch_init: Vec<f64>,
    layout_id: u32,
}

#[cfg(feature = "codegen")]
#[wasm_bindgen]
impl CompiledTape {
    /// The module. Instantiate it against [`shared_memory`] and the `Math`
    /// imports, then bind it with `setAotExports`.
    #[wasm_bindgen(getter)]
    pub fn wasm(&self) -> Vec<u8> {
        self.wasm.clone()
    }

    #[wasm_bindgen(getter, js_name = nParams)]
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    #[wasm_bindgen(getter, js_name = scratchInit)]
    pub fn scratch_init(&self) -> Vec<f64> {
        self.scratch_init.clone()
    }

    #[wasm_bindgen(getter, js_name = layoutId)]
    pub fn layout_id(&self) -> u32 {
        self.layout_id
    }
}

/// Compile a tape written by another front end.
///
/// `tape` is the text format `tapewasm_codegen::tape_text` documents: one
/// instruction per line, operands naming instructions rather than nodes. A
/// front end that can build a tape reaches the emitter this way — including
/// one that is not Rust and not in this module.
///
/// The format is not an artifact and carries no compatibility promise: a tape
/// is written and consumed inside one call.
///
/// `reroll` says when a vectorised statement becomes a wasm loop: `"auto"`
/// (the default, straight-line below a size threshold), `"always"`, `"never"`,
/// or a node count to re-roll past, written as a number.
///
/// **Which is faster is an engine's preference, not the model's.** Measured
/// across three engines, straight-line and re-rolled cross over around 60,000
/// nodes in V8 and around 2,000 in SpiderMonkey and JavaScriptCore — thirty
/// times apart, so no single threshold serves all three. `"auto"` takes the
/// lower one: near-optimal for two of the three, and up to 7.6x off on V8 for
/// a trace between them. A caller that knows its engine passes the number.
///
/// `"always"` is also the smallest module, often by an order of magnitude.
#[cfg(feature = "codegen")]
#[wasm_bindgen(js_name = compileTape)]
pub fn compile_tape(tape: &str, reroll: Option<String>) -> Result<CompiledTape, JsError> {
    use tapewasm_codegen::Reroll;
    // A number is a node count to re-roll past — what a caller uses when it
    // knows which engine will run the module. The three named modes stay.
    let reroll = match reroll.as_deref() {
        None | Some("auto") => Reroll::Auto,
        Some("always") => Reroll::Always,
        Some("never") => Reroll::Never,
        Some(other) => match other.parse::<usize>() {
            Ok(n) => Reroll::Above(n),
            Err(_) => {
                return Err(JsError::new(&format!(
                    "reroll must be \"auto\", \"always\", \"never\" or a node count, \
                     not {other:?}"
                )))
            }
        },
    };
    let program = tapewasm_codegen::tape_text::parse(tape).map_err(jserr)?;
    let compiled =
        tapewasm_codegen::compile_tape(&program.tape, program.n_params, program.root, reroll)
            .map_err(jserr)?;

    let mut scratch_init = vec![0.0_f64; compiled.scratch_len];
    let at = compiled.scratch_len - compiled.const_table.len();
    scratch_init[at..].copy_from_slice(&compiled.const_table);

    Ok(CompiledTape {
        wasm: compiled.wasm,
        n_params: compiled.n_params,
        scratch_init,
        layout_id: compiled.layout_id,
    })
}
