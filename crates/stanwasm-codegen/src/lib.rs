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

mod reroll;

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
        "`{op}` is not supported on the AOT path yet — the emitter has no \
         instruction sequence for it. Sample this model with `sample()` \
         (tape replay), which supports every op the runtime produces."
    )]
    UnsupportedOp { op: String },
    #[error(transparent)]
    Eval(#[from] stanwasm_runtime::EvalError),
}

/// Function parameter holding the scratch base address, used only by
/// [`Layout::Memory`]. Primals and adjoints live there, two f64 per tape node.
const SCRATCH_PTR: u32 = 3;

/// First local index available for slots: the four i32 parameters come first.
const FIRST_SLOT_LOCAL: u32 = 4;

#[derive(Debug, Clone)]
pub struct Compiled {
    pub wasm: Vec<u8>,
    pub n_params: usize,
    /// f64 slots the caller must pass as the module's scratch buffer.
    pub scratch_len: usize,
    /// Loop constants the caller stages at slot `scratch_len - const_table.len()`
    /// before the first call. Empty when nothing was re-rolled.
    pub const_table: Vec<f64>,
}

/// Trace `model` on a fresh tape at `dummy_params` (0.1 throughout; eight_schools
/// needs non-zero seeds), then emit a model-specific wasm module.
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
    // The emitters have no arm for these, and an `unimplemented!` would compile to a
    // wasm trap that takes the whole module down rather than reporting anything.
    for k in 0..tape.len() {
        let op = tape.op_at(k as u32);
        if matches!(
            op,
            Op::Erf | Op::Erfc | Op::Tan | Op::Asin | Op::Acos | Op::Atan | Op::Digamma
        ) {
            return Err(CodegenError::UnsupportedOp {
                op: format!("{op:?}"),
            });
        }
    }
    let (wasm, const_table) = emit(&tape, dummy_params.len(), root);
    Ok(Compiled {
        wasm,
        n_params: dummy_params.len(),
        scratch_len: 2 * tape.len() + const_table.len(),
        const_table,
    })
}

/// Lower the recorded tape to a wasm module.
fn emit(tape: &Tape, n_params: usize, root: u32) -> (Vec<u8>, Vec<f64>) {
    let n = tape.len() as u32;
    // Straight-line code keeps every value in a register-allocated local and
    // wins while V8 still optimises the function. Past roughly this many nodes
    // it stops, and the unrolled body collapses to well under tape-replay
    // speed; a re-rolled loop is flat from there on. Measured on linear
    // regression: straight-line 7.4 vs looped 16.2 us/gradient at 12k nodes,
    // 82.0 vs 31.5 at 24k.
    const RE_ROLL_ABOVE: usize = 12_000;
    let blocks = if tape.len() > RE_ROLL_ABOVE {
        reroll::detect(tape)
    } else {
        Vec::new()
    };
    // Constants that move with the loop index live past the adjoints, in block
    // then node order; the caller stages them once.
    let const_base = 2 * n;
    let mut const_table: Vec<f64> = Vec::new();
    for b in &blocks {
        for c in b.consts.iter().flatten() {
            const_table.extend_from_slice(c);
        }
    }
    let needs = scan_imports(tape);

    // ---- type section: 0 = (i32,i32,i32,i32)->f64 (log_prob_grad: params_ptr,
    // grads_ptr, n_params, scratch_ptr), 1 = (f64)->f64 (unary math),
    // 2 = (f64,f64)->f64 (pow)
    let mut types = TypeSection::new();
    types.ty().function(
        [ValType::I32, ValType::I32, ValType::I32, ValType::I32],
        [ValType::F64],
    );
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

    // Imported funcs first, then defined. The memory import takes no function-index
    // slot, so n_func_imports counts only the math imports.
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
    let lpg = build_log_prob_grad(tape, root, n, &math_idx, &blocks, const_base);
    codes.function(&lpg);

    // ---- assemble ----------------------------------------------------------
    let mut module = Module::new();
    module.section(&types);
    module.section(&imports);
    module.section(&functions);
    module.section(&exports);
    module.section(&codes);
    (module.finish(), const_table)
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
            // Tan/Asin/Acos/Atan/Erf/Erfc/Digamma/Sqrt/Abs and arithmetic are inline;
            // the rest are not emitted because the runtime does not produce them.
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

/// Per-iteration base pointers for a re-rolled block: one i32 local per
/// distinct non-zero stride, holding `scratch + i * stride * 8`.
struct StridePtrs {
    iv: u32,
    strides: Vec<u32>,
    first_local: u32,
}

impl StridePtrs {
    fn addr(&self, base_slot: u32, stride: u32) -> Addr {
        if stride == 0 {
            return Addr::Mem {
                ptr: SCRATCH_PTR,
                slot: base_slot,
            };
        }
        let idx = self
            .strides
            .iter()
            .position(|&t| t == stride)
            .expect("stride was registered");
        Addr::Mem {
            ptr: self.first_local + idx as u32,
            slot: base_slot,
        }
    }

    fn emit_setup(&self, f: &mut Function) {
        for (idx, &t) in self.strides.iter().enumerate() {
            f.instruction(&Instruction::LocalGet(SCRATCH_PTR));
            f.instruction(&Instruction::LocalGet(self.iv));
            f.instruction(&Instruction::I32Const((t * 8) as i32));
            f.instruction(&Instruction::I32Mul);
            f.instruction(&Instruction::I32Add);
            f.instruction(&Instruction::LocalSet(self.first_local + idx as u32));
        }
    }
}

/// Every non-zero stride a block's reads and writes use, deduplicated.
fn block_strides(tape: &Tape, b: &reroll::Block) -> Vec<u32> {
    let mut out: Vec<u32> = vec![b.len];
    for j in 0..b.len {
        let k0 = b.start + j;
        let st = b.strides[j as usize];
        for (used, stride) in [
            (reroll::uses(tape.op_at(k0)).0, st.arg1),
            (reroll::uses(tape.op_at(k0)).1, st.arg2i),
        ] {
            if used && stride != 0 && !out.contains(&stride) {
                out.push(stride);
            }
        }
        // A moving constant walks its table one f64 per repeat.
        if b.consts[j as usize].is_some() && !out.contains(&1) {
            out.push(1);
        }
    }
    out
}

/// The f64 argument of block node `j` at the current iteration.
fn block_cst(tape: &Tape, b: &reroll::Block, j: u32, tbl: &[Option<u32>], sp: &StridePtrs) -> Cst {
    match tbl[j as usize] {
        Some(base) => Cst::At(sp.addr(base, 1)),
        None => {
            let k0 = b.start + j;
            Cst::Imm(if tape.op_at(k0) == Op::Leaf {
                tape.value(k0)
            } else {
                tape.arg2f_at(k0)
            })
        }
    }
}

fn build_log_prob_grad(
    tape: &Tape,
    root: u32,
    n: u32,
    m: &MathImportIndex,
    blocks: &[reroll::Block],
    const_base: u32,
) -> Function {
    // 4 i32 params at local indices 0..4: params_ptr, grads_ptr, n_params
    // (unused — the recorded tape already encodes it), scratch_ptr.
    const GRADS_PTR: u32 = 1;
    let adj = n; // adjoint slots follow the primals
    let lay = Layout::for_tape(2 * n, !blocks.is_empty());
    let f64_locals = lay.local_count(2 * n);
    let widest = blocks
        .iter()
        .map(|b| block_strides(tape, b).len() as u32)
        .max()
        .unwrap_or(0);
    // One induction variable, reused by every block, then the base pointers.
    let i32_locals = if blocks.is_empty() { 0 } else { 1 + widest };
    let i32_base = FIRST_SLOT_LOCAL + f64_locals;
    let iv = i32_base;

    let mut decl: Vec<(u32, ValType)> = Vec::new();
    if f64_locals > 0 {
        decl.push((f64_locals, ValType::F64));
    }
    if i32_locals > 0 {
        decl.push((i32_locals, ValType::I32));
    }
    let mut f = Function::new(decl);

    // ---- zero the adjoint half --------------------------------------------
    // Locals start at zero; a caller-owned scratch buffer is reused across
    // calls, so its adjoint half has to be cleared. Primals are all written
    // before they are read either way.
    if let Layout::Memory = lay {
        f.instruction(&Instruction::LocalGet(SCRATCH_PTR));
        f.instruction(&Instruction::I32Const((n * 8) as i32));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::I32Const(0));
        f.instruction(&Instruction::I32Const((n * 8) as i32));
        f.instruction(&Instruction::MemoryFill(0));
    }

    // Moving-constant table offsets, assigned in block then node order to
    // match the table the caller stages at `const_base`.
    let mut tables: Vec<Vec<Option<u32>>> = Vec::with_capacity(blocks.len());
    let mut next = const_base;
    for b in blocks {
        let mut t = Vec::with_capacity(b.len as usize);
        for j in 0..b.len {
            match &b.consts[j as usize] {
                Some(v) => {
                    t.push(Some(next));
                    next += v.len() as u32;
                }
                None => t.push(None),
            }
        }
        tables.push(t);
    }

    let straight_fwd = |f: &mut Function, k: u32| {
        let a1 = lay.at(tape.arg1_at(k));
        let a2 = lay.at(tape.arg2i_at(k));
        let cst = Cst::Imm(if tape.op_at(k) == Op::Leaf {
            tape.value(k)
        } else {
            tape.arg2f_at(k)
        });
        astore_addr(f, lay.at(k));
        emit_forward(f, tape, k, m, a1, a2, cst);
        astore_end(f, lay.at(k));
    };

    // ---- forward pass ------------------------------------------------------
    let mut k = 0u32;
    let mut bi = 0usize;
    while k < n {
        if bi < blocks.len() && blocks[bi].start == k {
            let b = &blocks[bi];
            let sp = StridePtrs {
                iv,
                strides: block_strides(tape, b),
                first_local: iv + 1,
            };
            f.instruction(&Instruction::I32Const(0));
            f.instruction(&Instruction::LocalSet(iv));
            f.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty));
            sp.emit_setup(&mut f);
            for j in 0..b.len {
                let k0 = b.start + j;
                let st = b.strides[j as usize];
                let a1 = sp.addr(tape.arg1_at(k0), st.arg1);
                let a2 = sp.addr(tape.arg2i_at(k0), st.arg2i);
                let cst = block_cst(tape, b, j, &tables[bi], &sp);
                let w = sp.addr(k0, b.len);
                astore_addr(&mut f, w);
                emit_forward(&mut f, tape, k0, m, a1, a2, cst);
                astore_end(&mut f, w);
            }
            f.instruction(&Instruction::LocalGet(iv));
            f.instruction(&Instruction::I32Const(1));
            f.instruction(&Instruction::I32Add);
            f.instruction(&Instruction::LocalTee(iv));
            f.instruction(&Instruction::I32Const(b.reps as i32));
            f.instruction(&Instruction::I32LtU);
            f.instruction(&Instruction::BrIf(0));
            f.instruction(&Instruction::End);
            k = b.end();
            bi += 1;
        } else {
            straight_fwd(&mut f, k);
            k += 1;
        }
    }

    // ---- initialize root adjoint = 1.0 ------------------------------------
    astore_addr(&mut f, lay.at(adj + root));
    f.instruction(&Instruction::F64Const(1.0.into()));
    astore_end(&mut f, lay.at(adj + root));

    // ---- backward pass (reverse order) ------------------------------------
    let mut k = n;
    let mut bi = blocks.len();
    while k > 0 {
        if bi > 0 && blocks[bi - 1].end() == k {
            bi -= 1;
            let b = &blocks[bi];
            let sp = StridePtrs {
                iv,
                strides: block_strides(tape, b),
                first_local: iv + 1,
            };
            f.instruction(&Instruction::I32Const((b.reps - 1) as i32));
            f.instruction(&Instruction::LocalSet(iv));
            f.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty));
            sp.emit_setup(&mut f);
            for j in (0..b.len).rev() {
                let k0 = b.start + j;
                let st = b.strides[j as usize];
                emit_backward(
                    &mut f,
                    tape,
                    k0,
                    m,
                    Back {
                        dk: sp.addr(adj + k0, b.len),
                        da1: sp.addr(adj + tape.arg1_at(k0), st.arg1),
                        da2: sp.addr(adj + tape.arg2i_at(k0), st.arg2i),
                        pa1: sp.addr(tape.arg1_at(k0), st.arg1),
                        pa2: sp.addr(tape.arg2i_at(k0), st.arg2i),
                        pk: sp.addr(k0, b.len),
                        cst: block_cst(tape, b, j, &tables[bi], &sp),
                    },
                );
            }
            f.instruction(&Instruction::LocalGet(iv));
            f.instruction(&Instruction::I32Const(1));
            f.instruction(&Instruction::I32Sub);
            f.instruction(&Instruction::LocalTee(iv));
            f.instruction(&Instruction::I32Const(0));
            f.instruction(&Instruction::I32GeS);
            f.instruction(&Instruction::BrIf(0));
            f.instruction(&Instruction::End);
            k = b.start;
        } else {
            k -= 1;
            emit_backward(
                &mut f,
                tape,
                k,
                m,
                Back {
                    dk: lay.at(adj + k),
                    da1: lay.at(adj + tape.arg1_at(k)),
                    da2: lay.at(adj + tape.arg2i_at(k)),
                    pa1: lay.at(tape.arg1_at(k)),
                    pa2: lay.at(tape.arg2i_at(k)),
                    pk: lay.at(k),
                    cst: Cst::Imm(tape.arg2f_at(k)),
                },
            );
        }
    }

    // ---- store gradients at grads_ptr + i*8. n_params is a runtime parameter,
    // so unroll over the leaf-prefix count observed during tracing instead. ---
    let n_params_observed = leaf_count(tape);
    for pi in 0..n_params_observed {
        f.instruction(&Instruction::LocalGet(GRADS_PTR));
        f.instruction(&Instruction::I32Const((pi * 8) as i32));
        f.instruction(&Instruction::I32Add);
        aload(&mut f, lay.at(adj + pi));
        f.instruction(&Instruction::F64Store(wasm_encoder::MemArg {
            offset: 0,
            align: 3,
            memory_index: 0,
        }));
    }

    // ---- return log_prob ---------------------------------------------------
    aload(&mut f, lay.at(root));
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

fn emit_forward(
    f: &mut Function,
    tape: &Tape,
    k: u32,
    m: &MathImportIndex,
    a1: Addr,
    a2: Addr,
    cst: Cst,
) {
    let op = tape.op_at(k);
    const PARAMS_PTR: u32 = 0;
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
                f.instruction(&Instruction::F64Const((tape.value(k)).into()));
            }
        }
        Op::Add => {
            aload(f, a1);
            aload(f, a2);
            f.instruction(&Instruction::F64Add);
        }
        Op::Sub => {
            aload(f, a1);
            aload(f, a2);
            f.instruction(&Instruction::F64Sub);
        }
        Op::Mul => {
            aload(f, a1);
            aload(f, a2);
            f.instruction(&Instruction::F64Mul);
        }
        Op::Div => {
            aload(f, a1);
            aload(f, a2);
            f.instruction(&Instruction::F64Div);
        }
        Op::Neg => {
            aload(f, a1);
            f.instruction(&Instruction::F64Neg);
        }
        Op::Exp => {
            aload(f, a1);
            f.instruction(&Instruction::Call(m.exp.expect("exp import missing")));
        }
        Op::Log => {
            aload(f, a1);
            f.instruction(&Instruction::Call(m.log.expect("log import missing")));
        }
        Op::Sin => {
            aload(f, a1);
            f.instruction(&Instruction::Call(m.sin.expect("sin import missing")));
        }
        Op::Cos => {
            aload(f, a1);
            f.instruction(&Instruction::Call(m.cos.expect("cos import missing")));
        }
        Op::Sqrt => {
            aload(f, a1);
            f.instruction(&Instruction::F64Sqrt);
        }
        Op::Pow => {
            aload(f, a1);
            cload(f, cst);
            f.instruction(&Instruction::Call(m.pow.expect("pow import missing")));
        }
        Op::Abs => {
            aload(f, a1);
            f.instruction(&Instruction::F64Abs);
        }
        Op::Lgamma => {
            aload(f, a1);
            f.instruction(&Instruction::Call(m.lgamma.expect("lgamma import missing")));
        }
        Op::AddC => {
            aload(f, a1);
            cload(f, cst);
            f.instruction(&Instruction::F64Add);
        }
        Op::SubC => {
            aload(f, a1);
            cload(f, cst);
            f.instruction(&Instruction::F64Sub);
        }
        Op::RsubC => {
            cload(f, cst);
            aload(f, a1);
            f.instruction(&Instruction::F64Sub);
        }
        Op::MulC => {
            aload(f, a1);
            cload(f, cst);
            f.instruction(&Instruction::F64Mul);
        }
        Op::DivC => {
            aload(f, a1);
            cload(f, cst);
            f.instruction(&Instruction::F64Div);
        }
        Op::RdivC => {
            cload(f, cst);
            aload(f, a1);
            f.instruction(&Instruction::F64Div);
        }
        Op::Phi => {
            aload(f, a1);
            f.instruction(&Instruction::Call(m.phi.expect("phi import missing")));
        }
        Op::Erf | Op::Erfc | Op::Tan | Op::Asin | Op::Acos | Op::Atan | Op::Digamma => {
            unimplemented!("codegen for op {op:?}");
        }
    }
}

/// The six values one backward step touches, plus its f64 argument.
#[derive(Clone, Copy)]
struct Back {
    dk: Addr,
    da1: Addr,
    da2: Addr,
    pa1: Addr,
    pa2: Addr,
    pk: Addr,
    cst: Cst,
}

fn emit_backward(f: &mut Function, tape: &Tape, k: u32, m: &MathImportIndex, b: Back) {
    let op = tape.op_at(k);
    let Back {
        dk,
        da1,
        da2,
        pa1,
        pa2,
        pk,
        cst,
    } = b;

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
            {
                // `reroll` rejects a block whose Pow exponent moves, so this
                // is always an immediate.
                let Cst::Imm(e) = cst else {
                    unreachable!("Pow exponent is never table-backed")
                };
                adj_incr_pow(f, da1, dk, pa1, e, m.pow.unwrap());
            }
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
            adj_incr_mulc(f, da1, dk, cst);
        }
        Op::DivC => {
            adj_incr_divc(f, da1, dk, cst);
        }
        Op::RdivC => {
            adj_decr_rdivc(f, da1, dk, cst, pa1);
        }
        Op::Phi => {
            adj_incr_phi(f, da1, dk, pa1, m.exp.unwrap());
        }
        Op::Erf | Op::Erfc | Op::Tan | Op::Asin | Op::Acos | Op::Atan | Op::Digamma => {
            unimplemented!("backward for op {op:?}");
        }
    }
}

// ---- adjoint update emitters: given adjoint `da`, source adjoint `dk` and any
// primal locals, emit `da += <expression in dk and primals>`. ----


// ---- value storage -------------------------------------------------------
//
// A tape node's primal and adjoint each occupy one "slot": primals are slots
// 0..n, adjoints n..2n, and a re-rolled loop's moving constants follow at 2n.
// Locals are register-allocated and much faster, but a function is capped at
// 50,000 of them and a loop body cannot hold a whole vector in locals, so
// slots otherwise live in a caller-owned scratch buffer.

/// Wasm locals cost nothing to address but are capped per function. V8's limit
/// is the binding one; other engines allow more.
const MAX_WASM_LOCALS: u32 = 50_000;

/// Where one value lives. Inside a loop body the slot index moves with the
/// iteration, so the varying part sits in `ptr` and only the constant part
/// folds into the load/store immediate.
#[derive(Clone, Copy)]
enum Addr {
    Local(u32),
    /// `ptr` is an i32 local holding a byte address; `slot` is added as an
    /// f64-indexed immediate offset.
    Mem { ptr: u32, slot: u32 },
}

/// How the straight-line emitter turns a slot number into an [`Addr`].
#[derive(Clone, Copy)]
enum Layout {
    Locals { first_local: u32 },
    Memory,
}

impl Layout {
    fn for_tape(n_slots: u32, has_loops: bool) -> Self {
        if !has_loops && n_slots <= MAX_WASM_LOCALS {
            Layout::Locals {
                first_local: FIRST_SLOT_LOCAL,
            }
        } else {
            Layout::Memory
        }
    }

    fn local_count(&self, n_slots: u32) -> u32 {
        match self {
            Layout::Locals { .. } => n_slots,
            Layout::Memory => 0,
        }
    }

    fn at(&self, slot: u32) -> Addr {
        match *self {
            Layout::Locals { first_local } => Addr::Local(first_local + slot),
            Layout::Memory => Addr::Mem {
                ptr: SCRATCH_PTR,
                slot,
            },
        }
    }
}

fn memarg(slot: u32) -> wasm_encoder::MemArg {
    wasm_encoder::MemArg {
        offset: slot as u64 * 8,
        align: 3,
        memory_index: 0,
    }
}

/// A node's f64 argument. Inside a re-rolled loop it moves with the iteration
/// and has to be read from the table the caller stages in scratch.
#[derive(Clone, Copy)]
enum Cst {
    Imm(f64),
    At(Addr),
}

fn cload(f: &mut Function, c: Cst) {
    match c {
        Cst::Imm(v) => {
            f.instruction(&Instruction::F64Const(v.into()));
        }
        Cst::At(a) => aload(f, a),
    }
}

fn aload(f: &mut Function, a: Addr) {
    match a {
        Addr::Local(i) => {
            f.instruction(&Instruction::LocalGet(i));
        }
        Addr::Mem { ptr, slot } => {
            f.instruction(&Instruction::LocalGet(ptr));
            f.instruction(&Instruction::F64Load(memarg(slot)));
        }
    }
}

/// Push whatever the matching [`astore_end`] needs underneath the value.
fn astore_addr(f: &mut Function, a: Addr) {
    if let Addr::Mem { ptr, .. } = a {
        f.instruction(&Instruction::LocalGet(ptr));
    }
}

fn astore_end(f: &mut Function, a: Addr) {
    match a {
        Addr::Local(i) => {
            f.instruction(&Instruction::LocalSet(i));
        }
        Addr::Mem { slot, .. } => {
            f.instruction(&Instruction::F64Store(memarg(slot)));
        }
    }
}

// d[da] += d[dk]
fn adj_incr(f: &mut Function, da: Addr, dk: Addr) {
    astore_addr(f, da);
    aload(f, da);
    aload(f, dk);
    f.instruction(&Instruction::F64Add);
    astore_end(f, da);
}

// d[da] -= d[dk]
fn adj_decr(f: &mut Function, da: Addr, dk: Addr) {
    astore_addr(f, da);
    aload(f, da);
    aload(f, dk);
    f.instruction(&Instruction::F64Sub);
    astore_end(f, da);
}

// d[da] += d[dk] * t[tv]
fn adj_incr_mul(f: &mut Function, da: Addr, dk: Addr, tv: Addr) {
    astore_addr(f, da);
    aload(f, da);
    aload(f, dk);
    aload(f, tv);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    astore_end(f, da);
}

// d[da] += d[dk] / t[tv]
fn adj_incr_div(f: &mut Function, da: Addr, dk: Addr, tv: Addr) {
    astore_addr(f, da);
    aload(f, da);
    aload(f, dk);
    aload(f, tv);
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Add);
    astore_end(f, da);
}

// d[da] -= d[dk] * t[ta] / (t[tb] * t[tb])
fn adj_decr_mul_div2(f: &mut Function, da: Addr, dk: Addr, ta: Addr, tb: Addr) {
    astore_addr(f, da);
    aload(f, da);
    aload(f, dk);
    aload(f, ta);
    f.instruction(&Instruction::F64Mul);
    aload(f, tb);
    aload(f, tb);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Sub);
    astore_end(f, da);
}

// d[da] += d[dk] / (2 * t[tk])
fn adj_incr_div2(f: &mut Function, da: Addr, dk: Addr, tk: Addr) {
    astore_addr(f, da);
    aload(f, da);
    aload(f, dk);
    f.instruction(&Instruction::F64Const(2.0.into()));
    aload(f, tk);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Add);
    astore_end(f, da);
}

// d[da] += d[dk] * fn(t[tv])
fn adj_incr_fn1(f: &mut Function, da: Addr, dk: Addr, tv: Addr, fn_idx: u32) {
    astore_addr(f, da);
    aload(f, da);
    aload(f, dk);
    aload(f, tv);
    f.instruction(&Instruction::Call(fn_idx));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    astore_end(f, da);
}

// d[da] -= d[dk] * fn(t[tv])
fn adj_decr_fn1(f: &mut Function, da: Addr, dk: Addr, tv: Addr, fn_idx: u32) {
    astore_addr(f, da);
    aload(f, da);
    aload(f, dk);
    aload(f, tv);
    f.instruction(&Instruction::Call(fn_idx));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Sub);
    astore_end(f, da);
}

// d[da] += d[dk] * c
fn adj_incr_mulc(f: &mut Function, da: Addr, dk: Addr, c: Cst) {
    astore_addr(f, da);
    aload(f, da);
    aload(f, dk);
    cload(f, c);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    astore_end(f, da);
}

// d[da] += d[dk] / c
fn adj_incr_divc(f: &mut Function, da: Addr, dk: Addr, c: Cst) {
    astore_addr(f, da);
    aload(f, da);
    aload(f, dk);
    cload(f, c);
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Add);
    astore_end(f, da);
}

// d[da] -= d[dk] * c / (t[ta] * t[ta])
fn adj_decr_rdivc(f: &mut Function, da: Addr, dk: Addr, c: Cst, ta: Addr) {
    astore_addr(f, da);
    aload(f, da);
    aload(f, dk);
    cload(f, c);
    f.instruction(&Instruction::F64Mul);
    aload(f, ta);
    aload(f, ta);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Div);
    f.instruction(&Instruction::F64Sub);
    astore_end(f, da);
}

// d[da] += d[dk] * exp * pow(t[tv], exp - 1)
fn adj_incr_pow(f: &mut Function, da: Addr, dk: Addr, tv: Addr, exponent: f64, pow_idx: u32) {
    astore_addr(f, da);
    aload(f, da);
    aload(f, dk);
    f.instruction(&Instruction::F64Const(exponent.into()));
    f.instruction(&Instruction::F64Mul);
    aload(f, tv);
    f.instruction(&Instruction::F64Const((exponent - 1.0).into()));
    f.instruction(&Instruction::Call(pow_idx));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    astore_end(f, da);
}

// d[da] += d[dk] * copysign(1.0, t[ta])  (ABS backward)
fn adj_incr_sign(f: &mut Function, da: Addr, dk: Addr, ta: Addr) {
    astore_addr(f, da);
    aload(f, da);
    aload(f, dk);
    f.instruction(&Instruction::F64Const(1.0.into()));
    aload(f, ta);
    f.instruction(&Instruction::F64Copysign);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    astore_end(f, da);
}

// d[da] += d[dk] * (1/sqrt(2π)) * exp(-0.5 * t[ta]²)  (Phi backward)
fn adj_incr_phi(f: &mut Function, da: Addr, dk: Addr, ta: Addr, exp_idx: u32) {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    astore_addr(f, da);
    aload(f, da);
    aload(f, dk);
    f.instruction(&Instruction::F64Const((-0.5).into()));
    aload(f, ta);
    aload(f, ta);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::Call(exp_idx));
    f.instruction(&Instruction::F64Const(INV_SQRT_2PI.into()));
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    astore_end(f, da);
}
