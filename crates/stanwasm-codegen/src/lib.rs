//! AOT compilation: trace a Stan model on the autodiff tape, then emit a
//! self-contained wasm module that computes log_prob and gradients in one call.
//!
//! Emits wasm binary directly via `wasm-encoder`, with no WAT step and no
//! browser-side `wabt` dependency.
//!
//! Generated module ABI (zero-copy variant: memory is imported, not exported,
//! so the AOT module shares the host's linear memory for parameter and
//! gradient buffers — no inter-wasm memcpy):
//!
//!   (import "stan" "memory" (memory 1))           — shared linear memory
//!   log_prob_grad(params_ptr: i32, grads_ptr: i32, n_params: i32) -> f64
//!     reads params_ptr..params_ptr+n_params*8 and writes
//!     grads_ptr..grads_ptr+n_params*8 in shared memory; returns log_prob.
//!
//! Required imports (host-provided):
//!   ("stan","memory")  — shared memory
//!   ("Math","exp"|"log"|"sin"|"cos"|"pow"|"lgamma"|"digamma"|"phi")
//!   Only the math imports actually needed by the recorded tape are emitted.

#![forbid(unsafe_code)]

use stanwasm_autodiff::{Op, Tape};
use stanwasm_runtime::Model;
use thiserror::Error;
use wasm_encoder::{
    CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection, ImportSection,
    Instruction, MemoryType, Module, TypeSection, ValType,
};

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("trace produced empty tape — no log_prob computation recorded")]
    EmptyTape,
    #[error("internal: {0}")]
    Internal(String),
    #[error(
        "model too large for the AOT path: the forward trace has {tape_len} \
         nodes, which needs {locals} wasm locals — over the {limit} a browser \
         engine accepts per function. Sample this model with `sample()` \
         (tape replay), which has no such limit, or reduce N"
    )]
    TooManyLocals {
        tape_len: usize,
        locals: usize,
        limit: usize,
    },
    #[error(transparent)]
    Eval(#[from] stanwasm_runtime::EvalError),
}

/// V8's per-function local limit. Other engines allow more, but the browser is
/// the target, so this is the binding constraint.
pub const MAX_WASM_LOCALS: usize = 50_000;

#[derive(Debug, Clone)]
pub struct Compiled {
    pub wasm: Vec<u8>,
    pub n_params: usize,
}

/// Trace `model` on a fresh tape using `dummy_params` (all 0.1 is a reasonable
/// default — eight_schools needs non-zero seeds), then emit a model-specific
/// wasm module.
pub fn compile(model: &Model, dummy_params: &[f64]) -> Result<Compiled, CodegenError> {
    if dummy_params.len() != model.n_params() {
        return Err(CodegenError::Internal(format!(
            "dummy_params len {} != model n_params {}",
            dummy_params.len(),
            model.n_params()
        )));
    }
    let mut tape = Tape::new();
    let leaves: Vec<u32> = dummy_params.iter().map(|p| tape.new_var(*p)).collect();
    let root = model.trace_forward(&mut tape, &leaves, true)?;
    if tape.is_empty() {
        return Err(CodegenError::EmptyTape);
    }
    // `build_log_prob_grad` declares one primal and one adjoint local per tape
    // node. V8 rejects a function with more than 50,000 locals, so past this
    // point the emitted module fails to *instantiate* in the browser with an
    // opaque `CompileError: local count too large`. Fail here, where the
    // message can say what to do instead.
    let locals = 2 * tape.len();
    if locals > MAX_WASM_LOCALS {
        return Err(CodegenError::TooManyLocals {
            tape_len: tape.len(),
            locals,
            limit: MAX_WASM_LOCALS,
        });
    }
    let wasm = emit(&tape, dummy_params.len(), root);
    Ok(Compiled {
        wasm,
        n_params: dummy_params.len(),
    })
}

/// Lower the recorded tape to a wasm module.
fn emit(tape: &Tape, n_params: usize, root: u32) -> Vec<u8> {
    let n = tape.len() as u32;
    let needs = scan_imports(tape);

    // ---- type section ------------------------------------------------------
    // type 0: (i32, i32, i32) -> f64    (log_prob_grad: params_ptr, grads_ptr, n_params)
    // type 1: (f64) -> f64              (unary math: exp/log/sin/cos/lgamma/digamma/phi)
    // type 2: (f64, f64) -> f64         (pow)
    let mut types = TypeSection::new();
    types
        .ty()
        .function([ValType::I32, ValType::I32, ValType::I32], [ValType::F64]);
    types.ty().function([ValType::F64], [ValType::F64]);
    types
        .ty()
        .function([ValType::F64, ValType::F64], [ValType::F64]);

    // ---- import section ----------------------------------------------------
    let mut imports = ImportSection::new();
    // Memory is imported from "stan" — shared with the host wasm.
    imports.import(
        "stan",
        "memory",
        EntityType::Memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        }),
    );

    let mut math_idx = MathImportIndex::default();
    if needs.exp {
        math_idx.exp = Some(math_idx_add_unary(&mut math_idx, &mut imports, "exp"));
    }
    if needs.log {
        math_idx.log = Some(math_idx_add_unary(&mut math_idx, &mut imports, "log"));
    }
    if needs.sin {
        math_idx.sin = Some(math_idx_add_unary(&mut math_idx, &mut imports, "sin"));
    }
    if needs.cos {
        math_idx.cos = Some(math_idx_add_unary(&mut math_idx, &mut imports, "cos"));
    }
    if needs.lgamma {
        math_idx.lgamma = Some(math_idx_add_unary(&mut math_idx, &mut imports, "lgamma"));
    }
    if needs.digamma {
        math_idx.digamma = Some(math_idx_add_unary(&mut math_idx, &mut imports, "digamma"));
    }
    if needs.phi {
        math_idx.phi = Some(math_idx_add_unary(&mut math_idx, &mut imports, "phi"));
    }
    if needs.pow {
        math_idx.pow = Some(math_idx_add_binary(&mut math_idx, &mut imports, "pow"));
    }

    // Function index space: imported funcs first, then defined funcs. The
    // memory import does NOT take a function-index slot (it's a separate
    // import kind), so n_func_imports counts only the math imports.
    let n_func_imports = math_idx.count();

    // ---- function section --------------------------------------------------
    let mut functions = FunctionSection::new();
    functions.function(0); // log_prob_grad: type 0

    let log_prob_grad_idx = n_func_imports;

    // ---- export section ----------------------------------------------------
    let mut exports = ExportSection::new();
    exports.export("log_prob_grad", ExportKind::Func, log_prob_grad_idx);

    // ---- code section ------------------------------------------------------
    let mut codes = CodeSection::new();

    let _ = n_params;
    let lpg = build_log_prob_grad(tape, root, n, &math_idx);
    codes.function(&lpg);

    // ---- assemble ----------------------------------------------------------
    let mut module = Module::new();
    module.section(&types);
    module.section(&imports);
    module.section(&functions);
    module.section(&exports);
    module.section(&codes);
    module.finish()
}

#[derive(Default, Debug)]
struct ImportNeeds {
    exp: bool,
    log: bool,
    sin: bool,
    cos: bool,
    pow: bool,
    lgamma: bool,
    digamma: bool,
    phi: bool,
}

#[derive(Default)]
struct MathImportIndex {
    exp: Option<u32>,
    log: Option<u32>,
    sin: Option<u32>,
    cos: Option<u32>,
    pow: Option<u32>,
    lgamma: Option<u32>,
    digamma: Option<u32>,
    phi: Option<u32>,
    /// Running counter of function imports added so far. Memory imports
    /// occupy a different index space and do NOT advance this.
    next: u32,
}

impl MathImportIndex {
    fn count(&self) -> u32 {
        self.next
    }

    fn add_unary(&mut self, imports: &mut ImportSection, name: &str) -> u32 {
        let idx = self.next;
        imports.import("Math", name, EntityType::Function(1));
        self.next += 1;
        idx
    }

    fn add_binary(&mut self, imports: &mut ImportSection, name: &str) -> u32 {
        let idx = self.next;
        imports.import("Math", name, EntityType::Function(2));
        self.next += 1;
        idx
    }
}

fn scan_imports(tape: &Tape) -> ImportNeeds {
    let mut needs = ImportNeeds::default();
    for k in 0..tape.len() {
        match tape.op_at(k as u32) {
            Op::Exp => needs.exp = true,
            Op::Log => needs.log = true,
            Op::Sin => {
                needs.sin = true;
                needs.cos = true; // backward of sin uses cos
            }
            Op::Cos => {
                needs.cos = true;
                needs.sin = true; // backward of cos uses sin
            }
            Op::Pow => needs.pow = true,
            Op::Lgamma => {
                needs.lgamma = true;
                needs.digamma = true; // backward
            }
            Op::Phi => {
                needs.phi = true;
                needs.exp = true; // backward uses exp
            }
            // Tan/Asin/Acos/Atan/Erf/Erfc/Digamma/Sqrt/Abs and arithmetic ops
            // are handled inline (sqrt/abs as f64 instructions; others not yet
            // emitted because the runtime currently does not produce them).
            _ => {}
        }
    }
    needs
}

fn math_idx_add_unary(m: &mut MathImportIndex, imports: &mut ImportSection, name: &str) -> u32 {
    m.add_unary(imports, name)
}

fn math_idx_add_binary(m: &mut MathImportIndex, imports: &mut ImportSection, name: &str) -> u32 {
    m.add_binary(imports, name)
}

fn build_log_prob_grad(tape: &Tape, root: u32, n: u32, m: &MathImportIndex) -> Function {
    // Function has 3 i32 params (params_ptr, grads_ptr, n_params).
    // Locals: 2*n f64 (primals at 3..3+n, adjoints at 3+n..3+2n).
    // Param slots occupy local indices 0..3.
    const PARAMS_PTR: u32 = 0;
    const GRADS_PTR: u32 = 1;
    // n_params (param 2) is unused — the recorded tape already encodes it.
    const PRIMAL_BASE: u32 = 3;
    let adjoint_base = PRIMAL_BASE + n;
    let total_locals = 2 * n;
    let mut f = Function::new([(total_locals, ValType::F64)]);

    // ---- forward pass ------------------------------------------------------
    for k in 0..n {
        emit_forward(&mut f, tape, k, m);
        f.instruction(&Instruction::LocalSet(PRIMAL_BASE + k));
    }

    // ---- initialize root adjoint = 1.0 ------------------------------------
    f.instruction(&Instruction::F64Const(1.0));
    f.instruction(&Instruction::LocalSet(adjoint_base + root));

    // ---- backward pass (reverse order) ------------------------------------
    for k_rev in (0..n).rev() {
        emit_backward(&mut f, tape, k_rev, PRIMAL_BASE, adjoint_base, m);
    }

    // ---- store gradients to memory at grads_ptr + i*8 --------------------
    // We don't know n_params here; we use the leaf-prefix convention (the
    // first leaves in the tape are the params). Gradients for those leaves
    // live at adjoint_base..adjoint_base+n_params. We loop over the first
    // n_params primal slots, but n_params is a runtime parameter, so we
    // unroll based on the leaf-count we observed during tracing.
    let n_params_observed = leaf_count(tape);
    for pi in 0..n_params_observed {
        // grads_ptr + pi * 8
        f.instruction(&Instruction::LocalGet(GRADS_PTR));
        f.instruction(&Instruction::I32Const((pi * 8) as i32));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalGet(adjoint_base + pi));
        f.instruction(&Instruction::F64Store(wasm_encoder::MemArg {
            offset: 0,
            align: 3,
            memory_index: 0,
        }));
    }
    let _ = PARAMS_PTR; // referenced by emit_forward

    // ---- return log_prob ---------------------------------------------------
    f.instruction(&Instruction::LocalGet(PRIMAL_BASE + root));
    f.instruction(&Instruction::End);
    f
}

fn leaf_count(tape: &Tape) -> u32 {
    let mut n = 0u32;
    for k in 0..tape.len() {
        if tape.op_at(k as u32) == Op::Leaf {
            n += 1;
        } else {
            break;
        }
    }
    n
}

fn is_param_leaf(tape: &Tape, k: u32) -> bool {
    if tape.op_at(k) != Op::Leaf {
        return false;
    }
    // Leaves before the first non-leaf op are parameters; later leaves (rare
    // — only created if Val::to_tape forces a constant) are not.
    for j in 0..k {
        if tape.op_at(j) != Op::Leaf {
            return false;
        }
    }
    true
}

fn emit_forward(f: &mut Function, tape: &Tape, k: u32, m: &MathImportIndex) {
    let op = tape.op_at(k);
    let a1 = tape.arg1_at(k);
    let a2i = tape.arg2i_at(k);
    let a2f = tape.arg2f_at(k);
    // Local indices: params_ptr=0, grads_ptr=1, n_params(unused)=2,
    // primals = 3..3+n, adjoints = 3+n..3+2n
    const PARAMS_PTR: u32 = 0;
    let pb = 3u32; // primal_base
    let p_a1 = pb + a1;
    let p_a2 = pb + a2i;
    match op {
        Op::Leaf => {
            if is_param_leaf(tape, k) {
                f.instruction(&Instruction::LocalGet(PARAMS_PTR));
                f.instruction(&Instruction::I32Const((k * 8) as i32));
                f.instruction(&Instruction::I32Add);
                f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }));
            } else {
                f.instruction(&Instruction::F64Const(tape.value(k)));
            }
        }
        Op::Add => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::LocalGet(p_a2));
            f.instruction(&Instruction::F64Add);
        }
        Op::Sub => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::LocalGet(p_a2));
            f.instruction(&Instruction::F64Sub);
        }
        Op::Mul => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::LocalGet(p_a2));
            f.instruction(&Instruction::F64Mul);
        }
        Op::Div => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::LocalGet(p_a2));
            f.instruction(&Instruction::F64Div);
        }
        Op::Neg => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::F64Neg);
        }
        Op::Exp => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::Call(m.exp.expect("exp import missing")));
        }
        Op::Log => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::Call(m.log.expect("log import missing")));
        }
        Op::Sin => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::Call(m.sin.expect("sin import missing")));
        }
        Op::Cos => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::Call(m.cos.expect("cos import missing")));
        }
        Op::Sqrt => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::F64Sqrt);
        }
        Op::Pow => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::F64Const(a2f));
            f.instruction(&Instruction::Call(m.pow.expect("pow import missing")));
        }
        Op::Abs => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::F64Abs);
        }
        Op::Lgamma => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::Call(m.lgamma.expect("lgamma import missing")));
        }
        Op::AddC => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::F64Const(a2f));
            f.instruction(&Instruction::F64Add);
        }
        Op::SubC => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::F64Const(a2f));
            f.instruction(&Instruction::F64Sub);
        }
        Op::RsubC => {
            f.instruction(&Instruction::F64Const(a2f));
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::F64Sub);
        }
        Op::MulC => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::F64Const(a2f));
            f.instruction(&Instruction::F64Mul);
        }
        Op::DivC => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::F64Const(a2f));
            f.instruction(&Instruction::F64Div);
        }
        Op::RdivC => {
            f.instruction(&Instruction::F64Const(a2f));
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::F64Div);
        }
        Op::Phi => {
            f.instruction(&Instruction::LocalGet(p_a1));
            f.instruction(&Instruction::Call(m.phi.expect("phi import missing")));
        }
        Op::Erf | Op::Erfc | Op::Tan | Op::Asin | Op::Acos | Op::Atan | Op::Digamma => {
            unimplemented!("codegen for op {op:?}");
        }
    }
}

fn emit_backward(
    f: &mut Function,
    tape: &Tape,
    k: u32,
    primal_base: u32,
    adjoint_base: u32,
    m: &MathImportIndex,
) {
    let op = tape.op_at(k);
    let a1 = tape.arg1_at(k);
    let a2i = tape.arg2i_at(k);
    let a2f = tape.arg2f_at(k);
    let dk = adjoint_base + k;
    let da1 = adjoint_base + a1;
    let da2 = adjoint_base + a2i;
    let pa1 = primal_base + a1;
    let pa2 = primal_base + a2i;
    let pk = primal_base + k;

    match op {
        Op::Leaf => {} // adjoint already accumulated by callers
        Op::Add => {
            adj_incr(f, da1, dk);
            adj_incr(f, da2, dk);
        }
        Op::Sub => {
            adj_incr(f, da1, dk);
            adj_decr(f, da2, dk);
        }
        Op::Mul => {
            adj_incr_mul(f, da1, dk, pa2);
            adj_incr_mul(f, da2, dk, pa1);
        }
        Op::Div => {
            adj_incr_div(f, da1, dk, pa2);
            adj_decr_mul_div2(f, da2, dk, pa1, pa2);
        }
        Op::Neg => {
            adj_decr(f, da1, dk);
        }
        Op::Exp => {
            // d/dx exp(x) = exp(x) = primal[k]
            adj_incr_mul(f, da1, dk, pk);
        }
        Op::Log => {
            adj_incr_div(f, da1, dk, pa1);
        }
        Op::Sin => {
            adj_incr_fn1(f, da1, dk, pa1, m.cos.unwrap());
        }
        Op::Cos => {
            adj_decr_fn1(f, da1, dk, pa1, m.sin.unwrap());
        }
        Op::Sqrt => {
            adj_incr_div2(f, da1, dk, pk);
        }
        Op::Pow => {
            adj_incr_pow(f, da1, dk, pa1, a2f, m.pow.unwrap());
        }
        Op::Abs => {
            adj_incr_sign(f, da1, dk, pa1);
        }
        Op::Lgamma => {
            adj_incr_fn1(f, da1, dk, pa1, m.digamma.unwrap());
        }
        Op::AddC | Op::SubC => {
            adj_incr(f, da1, dk);
        }
        Op::RsubC => {
            adj_decr(f, da1, dk);
        }
        Op::MulC => {
            adj_incr_mulc(f, da1, dk, a2f);
        }
        Op::DivC => {
            adj_incr_divc(f, da1, dk, a2f);
        }
        Op::RdivC => {
            adj_decr_rdivc(f, da1, dk, a2f, pa1);
        }
        Op::Phi => {
            adj_incr_phi(f, da1, dk, pa1, m.exp.unwrap());
        }
        Op::Erf | Op::Erfc | Op::Tan | Op::Asin | Op::Acos | Op::Atan | Op::Digamma => {
            unimplemented!("backward for op {op:?}");
        }
    }
}

// ---- adjoint update emitters ----
// Each takes the local indices for adjoint `da` and source adjoint `dk`, plus
// (optionally) primal locals `tv`, `ta`, `tb`, and writes the wasm sequence
// that performs `da += <expression involving dk and primals>`.

// d[da] += d[dk]
fn adj_incr(f: &mut Function, da: u32, dk: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] -= d[dk]
fn adj_decr(f: &mut Function, da: u32, dk: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] * t[tv]
fn adj_incr_mul(f: &mut Function, da: u32, dk: u32, tv: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::LocalGet(tv));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] / t[tv]
fn adj_incr_div(f: &mut Function, da: u32, dk: u32, tv: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::LocalGet(tv));
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] -= d[dk] * t[ta] / (t[tb] * t[tb])
fn adj_decr_mul_div2(f: &mut Function, da: u32, dk: u32, ta: u32, tb: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::LocalGet(ta));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(tb));
    f.instruction(&Instruction::LocalGet(tb));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] / (2 * t[tk])
fn adj_incr_div2(f: &mut Function, da: u32, dk: u32, tk: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Const(2.0));
    f.instruction(&Instruction::LocalGet(tk));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] * fn(t[tv])
fn adj_incr_fn1(f: &mut Function, da: u32, dk: u32, tv: u32, fn_idx: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::LocalGet(tv));
    f.instruction(&Instruction::Call(fn_idx));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] -= d[dk] * fn(t[tv])
fn adj_decr_fn1(f: &mut Function, da: u32, dk: u32, tv: u32, fn_idx: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::LocalGet(tv));
    f.instruction(&Instruction::Call(fn_idx));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] * c
fn adj_incr_mulc(f: &mut Function, da: u32, dk: u32, c: f64) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Const(c));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] / c
fn adj_incr_divc(f: &mut Function, da: u32, dk: u32, c: f64) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Const(c));
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] -= d[dk] * c / (t[ta] * t[ta])
fn adj_decr_rdivc(f: &mut Function, da: u32, dk: u32, c: f64, ta: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Const(c));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(ta));
    f.instruction(&Instruction::LocalGet(ta));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] * exp * pow(t[tv], exp - 1)
fn adj_incr_pow(f: &mut Function, da: u32, dk: u32, tv: u32, exponent: f64, pow_idx: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Const(exponent));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::LocalGet(tv));
    f.instruction(&Instruction::F64Const(exponent - 1.0));
    f.instruction(&Instruction::Call(pow_idx));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] * copysign(1.0, t[ta])  (ABS backward)
fn adj_incr_sign(f: &mut Function, da: u32, dk: u32, ta: u32) {
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Const(1.0));
    f.instruction(&Instruction::LocalGet(ta));
    f.instruction(&Instruction::F64Copysign);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}

// d[da] += d[dk] * (1/sqrt(2π)) * exp(-0.5 * t[ta]²)  (Phi backward)
fn adj_incr_phi(f: &mut Function, da: u32, dk: u32, ta: u32, exp_idx: u32) {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    f.instruction(&Instruction::LocalGet(da));
    f.instruction(&Instruction::LocalGet(dk));
    f.instruction(&Instruction::F64Const(-0.5));
    f.instruction(&Instruction::LocalGet(ta));
    f.instruction(&Instruction::LocalGet(ta));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::Call(exp_idx));
    f.instruction(&Instruction::F64Const(INV_SQRT_2PI));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::LocalSet(da));
}
