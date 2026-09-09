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

use std::collections::HashMap;

use nuts_rs::{
    sample_sequentially, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError,
};

use rand::{rngs::ChaCha8Rng, SeedableRng};

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
         there. `randomInit(seed)` finds one, or drop the parameters the data \
         says nothing about",
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
    /// Persistent scratch buffer for params (params_ptr) inside our memory.
    params_buf: Vec<f64>,
    /// Persistent scratch buffer for grads (grads_ptr) inside our memory.
    grads_buf: Vec<f64>,
    /// Primal and adjoint storage the AOT module works in, two f64 per node.
    scratch_buf: Vec<f64>,
}

impl AotLogp {
    /// `scratch_buf` is what the module works in — `Compiled::scratch_len`
    /// slots, with the re-rolled loops' constants already staged at the end.
    pub fn new(n_params: usize, scratch_buf: Vec<f64>) -> Self {
        Self {
            n_params,
            params_buf: vec![0.0; n_params],
            grads_buf: vec![0.0; n_params],
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
        // Copy position into the persistent params buffer; capture pointers.
        self.params_buf.copy_from_slice(position);
        let params_ptr = self.params_buf.as_ptr() as u32;
        let grads_ptr = self.grads_buf.as_mut_ptr() as u32;
        let scratch_ptr = self.scratch_buf.as_mut_ptr() as u32;
        let lp = aot_logp(params_ptr, grads_ptr, self.n_params as u32, scratch_ptr);
        gradient.copy_from_slice(&self.grads_buf);
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
        })
    }

    #[wasm_bindgen(getter, js_name = nParams)]
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    fn logp_fn(&self) -> AotLogp {
        AotLogp {
            n_params: self.n_params,
            params_buf: vec![0.0; self.n_params],
            grads_buf: vec![0.0; self.n_params],
            scratch_buf: self.scratch_init.clone(),
        }
    }

    /// `[log_prob, d/dparam...]`.
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
        let lp = self
            .logp_fn()
            .logp(params, &mut out[1..])
            .map_err(|e| JsError::new(&format!("{e}")))?;
        out[0] = lp;
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
        let settings = nuts_settings(num_warmup, num_draws);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let iter = sample_sequentially(math, settings, init, total, 0, &mut rng)
            .map_err(|e| JsError::new(&format!("nuts-rs init: {e}")))?;

        let mut out = vec![0.0_f64; n * total as usize];
        for (i, draw) in iter.enumerate() {
            let (pos, _progress) = draw.map_err(|e| JsError::new(&format!("nuts-rs draw: {e}")))?;
            out[i * n..(i + 1) * n].copy_from_slice(pos.as_ref());
        }
        Ok(out)
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
#[cfg(feature = "codegen")]
#[wasm_bindgen(js_name = compileTape)]
pub fn compile_tape(tape: &str) -> Result<CompiledTape, JsError> {
    let program = tapewasm_codegen::tape_text::parse(tape).map_err(jserr)?;
    let compiled = tapewasm_codegen::compile_tape(
        &program.tape,
        program.n_params,
        program.root,
        tapewasm_codegen::Reroll::default(),
    )
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
