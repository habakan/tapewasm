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
//! The module uses the fixed-width SIMD proposal: a re-rolled loop whose every
//! slot moves by one or not at all runs two repeats at a time. Every engine
//! that ships wasm today has it (Safari since 16.4), but an embedder that
//! disables the proposal will reject the module.
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

/// When to re-roll a vectorised statement into a wasm loop.
///
/// Which is faster is an engine preference, not a property of the model.
/// Measured per gradient at N=200 (linear regression, then the same for
/// logistic and student_t): V8 2.20 straight-line vs 2.63 looped, SpiderMonkey
/// 14.0 vs 20.3, JavaScriptCore 4.33 vs 2.33. The first two keep optimising a
/// large straight-line function; the third gives up on it. Past a few tens of
/// thousands of nodes every engine prefers the loop, which is what `Auto`
/// falls back on.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Reroll {
    /// Straight-line until the trace is large enough that no engine keeps
    /// optimising it.
    #[default]
    Auto,
    /// Always loop: smallest module, and what JavaScriptCore prefers.
    Always,
    /// Never loop. Diagnostic only — a large model exceeds what an engine will
    /// optimise, and eventually what it will hold in locals.
    Never,
}

/// Node count past which `Auto` re-rolls. Straight-line wins below it on V8 and
/// SpiderMonkey by keeping every value in a register-allocated local.
const RE_ROLL_ABOVE: usize = 12_000;

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
    compile_with(model, dummy_params, Reroll::default())
}

/// [`compile`], choosing when to re-roll vectorised statements.
pub fn compile_with(
    model: &Model,
    dummy_params: &[f64],
    reroll: Reroll,
) -> Result<Compiled, CodegenError> {
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
    let (wasm, const_table) = emit(&tape, dummy_params.len(), root, reroll);
    Ok(Compiled {
        wasm,
        n_params: dummy_params.len(),
        scratch_len: 2 * tape.len() + const_table.len(),
        const_table,
    })
}

/// Lower the recorded tape to a wasm module.
fn emit(tape: &Tape, n_params: usize, root: u32, reroll: Reroll) -> (Vec<u8>, Vec<f64>) {
    let n = tape.len() as u32;
    let blocks = match reroll {
        Reroll::Never => Vec::new(),
        Reroll::Always => reroll::detect(tape),
        Reroll::Auto if tape.len() > RE_ROLL_ABOVE => reroll::detect(tape),
        Reroll::Auto => Vec::new(),
    };
    // Constants that move with the loop index live past the adjoints, in block
    // then node order; the caller stages them once.
    let const_base = 2 * n;
    let slots = Slots::plan(tape, &blocks);
    let (const_table, _) = stage_tables(tape, &blocks, const_base, &slots);
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
    let lpg = build_log_prob_grad(tape, root, n, &math_idx, &blocks, const_base, &slots);
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

/// Number the positions a block keeps in locals, densely, so every block can
/// reuse the same pool.
fn block_locals_of(flags: &[bool], len: u32, base: u32, stride: u32) -> BlockLocals {
    let mut idx = Vec::with_capacity(len as usize);
    let mut next = 0;
    for &keep in flags.iter().take(len as usize) {
        if keep {
            idx.push(Some(next));
            next += 1;
        } else {
            idx.push(None);
        }
    }
    BlockLocals { idx, base, stride }
}

/// Which of a block's positions are kept in wasm locals, and where.
struct BlockLocals {
    /// Position -> dense local index, `None` when the value needs an address.
    idx: Vec<Option<u32>>,
    /// First f64 local; the adjoint half starts `stride` above it.
    base: u32,
    stride: u32,
}

impl BlockLocals {
    fn prim(&self, j: u32) -> Option<Addr> {
        self.idx[j as usize].map(|i| Addr::Local(self.base + i))
    }

    fn adj(&self, j: u32) -> Option<Addr> {
        self.idx[j as usize].map(|i| Addr::Local(self.base + self.stride + i))
    }

    /// A same-iteration read of a position this block keeps in a local.
    fn arg(&self, b: &reroll::Block, rel: &reroll::ArgRel, arg0: u32) -> Option<u32> {
        match rel {
            reroll::ArgRel::Affine(t)
                if *t == b.len && arg0 >= b.start && arg0 < b.start + b.len =>
            {
                let j = arg0 - b.start;
                self.idx[j as usize].map(|_| j)
            }
            _ => None,
        }
    }
}

/// Where a block node's per-iteration tables start, in scratch slots.
#[derive(Clone, Copy, Default)]
struct NodeTables {
    cst: Option<u32>,
    arg1: Option<u32>,
    arg2i: Option<u32>,
    /// A contraction's coefficients, one column of `reps` per element.
    dot: Option<u32>,
}

/// Lay out every re-rolled block's tables in one buffer the caller stages at
/// `const_base`. Emission and staging must agree on the order, so both come
/// from here.
fn stage_tables(
    tape: &Tape,
    blocks: &[reroll::Block],
    const_base: u32,
    slots: &Slots,
) -> (Vec<f64>, Vec<Vec<NodeTables>>) {
    let mut buf: Vec<f64> = Vec::new();
    let mut maps = Vec::with_capacity(blocks.len());
    let push = |buf: &mut Vec<f64>, vals: &[f64]| {
        let at = const_base + buf.len() as u32;
        buf.extend_from_slice(vals);
        at
    };
    for b in blocks {
        let mut m = Vec::with_capacity(b.len as usize);
        for j in 0..b.len as usize {
            let mut t = NodeTables::default();
            if let Some(v) = &b.consts[j] {
                t.cst = Some(push(&mut buf, v));
            }
            // A table holds slots, not tape indices: what the emitted load
            // adds to the scratch pointer.
            if let reroll::ArgRel::Tabled(ix) = &b.args[j].arg1 {
                let f: Vec<f64> = ix.iter().map(|&i| slots.at(i) as f64).collect();
                t.arg1 = Some(push(&mut buf, &f));
            }
            if let reroll::ArgRel::Tabled(ix) = &b.args[j].arg2i {
                let f: Vec<f64> = ix.iter().map(|&i| slots.at(i) as f64).collect();
                t.arg2i = Some(push(&mut buf, &f));
            }
            let k0 = b.start + j as u32;
            if tape.op_at(k0) == Op::DotC {
                let e = tape.extent_at(k0);
                let mut f = Vec::with_capacity((e.len * b.reps) as usize);
                for c in 0..e.len as usize {
                    for i in 0..b.reps {
                        let at = b.start + i * b.len + j as u32;
                        f.push(tape.coeffs(tape.extent_at(at))[c]);
                    }
                }
                t.dot = Some(push(&mut buf, &f));
            }
            m.push(t);
        }
        maps.push(m);
    }
    (buf, maps)
}

/// Where one of a block's operands sits in the scratch buffer: slot
/// `base + i * stride` at repeat `i`.
#[derive(Clone, Copy)]
struct SlotRel {
    base: u32,
    stride: u32,
}

/// One block position's slot relations: the value it writes, and each integer
/// argument — `None` where the argument is tabled and its address comes from
/// the table instead.
struct PosRel {
    out: SlotRel,
    arg1: Option<SlotRel>,
    arg2i: Option<SlotRel>,
    /// Every element of a contraction's operand run; empty for other opcodes.
    elems: Vec<SlotRel>,
}

/// Which scratch slot each tape node's primal occupies; its adjoint sits `n`
/// slots above that.
///
/// Position-major inside a re-rolled block — repeat `i` of position `j` lands
/// at `base + j * reps + i` — so consecutive repeats of one value are
/// adjacent. The tape's own order interleaves a block's positions instead, and
/// a later block reading a value the earlier one produced then walks by the
/// producing block's length rather than by one slot.
///
/// A block keeps the tape's order when an argument would stop being affine
/// under the permutation, which the emitter has no form for. That happens to
/// the accumulator of a vectorised density: its chain starts at a node outside
/// the block, one the tape order puts exactly one repeat before the first, so
/// a single stride describes it only by adjacency. Leaving that block alone
/// still lets the block it reads from be permuted, which is the crossing that
/// matters.
struct Slots(Option<Vec<u32>>);

impl Slots {
    fn at(&self, k: u32) -> u32 {
        match &self.0 {
            Some(m) => m[k as usize],
            None => k,
        }
    }

    fn plan(tape: &Tape, blocks: &[reroll::Block]) -> Self {
        if blocks.is_empty() {
            return Slots(None);
        }
        let mut packed = vec![true; blocks.len()];
        loop {
            let slots = Slots(Some(Slots::assign(tape, blocks, &packed)));
            let bad: Vec<usize> = (0..blocks.len())
                .filter(|&i| slots.rels(tape, &blocks[i]).is_none())
                .collect();
            if bad.is_empty() {
                return slots;
            }
            // Un-permuting a block only moves its slots back towards the
            // tape's own order, so this settles; a round that changes nothing
            // means the failure is elsewhere and the tape's order has to do.
            if !bad.iter().any(|&i| packed[i]) {
                return Slots(None);
            }
            for i in bad {
                packed[i] = false;
            }
        }
    }

    /// Slots in tape order, except that a permuted block's repeats of one
    /// position are laid out consecutively.
    fn assign(tape: &Tape, blocks: &[reroll::Block], packed: &[bool]) -> Vec<u32> {
        let n = tape.len() as u32;
        let mut map = vec![0u32; n as usize];
        let (mut next, mut k, mut bi) = (0u32, 0u32, 0usize);
        while k < n {
            match blocks.get(bi) {
                Some(b) if b.start == k => {
                    for i in 0..b.reps {
                        for j in 0..b.len {
                            map[(b.start + i * b.len + j) as usize] = next
                                + if packed[bi] {
                                    j * b.reps + i
                                } else {
                                    i * b.len + j
                                };
                        }
                    }
                    next += b.len * b.reps;
                    k = b.end();
                    bi += 1;
                }
                _ => {
                    map[k as usize] = next;
                    next += 1;
                    k += 1;
                }
            }
        }
        map
    }

    /// `slot(arg0 + i * t)` as an affine function of `i`, or `None` when the
    /// permutation does not leave it one.
    fn rel(&self, arg0: u32, t: u32, reps: u32) -> Option<SlotRel> {
        let base = self.at(arg0);
        if t == 0 || reps < 2 {
            return Some(SlotRel { base, stride: 0 });
        }
        let stride = self.at(arg0 + t).checked_sub(base)?;
        (2..reps)
            .all(|i| self.at(arg0 + i * t) == base + i * stride)
            .then_some(SlotRel { base, stride })
    }

    fn rels(&self, tape: &Tape, b: &reroll::Block) -> Option<Vec<PosRel>> {
        (0..b.len)
            .map(|j| {
                let k0 = b.start + j;
                let arg = |rel: &reroll::ArgRel, arg0: u32| match rel {
                    reroll::ArgRel::Affine(t) => self.rel(arg0, *t, b.reps).map(Some),
                    reroll::ArgRel::Tabled(_) => Some(None),
                };
                let mut elems = Vec::new();
                if tape.op_at(k0) == Op::DotC {
                    let e = tape.extent_at(k0);
                    let reroll::ArgRel::Affine(t) = b.args[j as usize].arg1 else {
                        return None;
                    };
                    for c in 0..e.len {
                        elems.push(self.rel(tape.arg1_at(k0) + c * e.stride, t, b.reps)?);
                    }
                }
                Some(PosRel {
                    out: self.rel(k0, b.len, b.reps)?,
                    arg1: arg(&b.args[j as usize].arg1, tape.arg1_at(k0))?,
                    arg2i: arg(&b.args[j as usize].arg2i, tape.arg2i_at(k0))?,
                    elems,
                })
            })
            .collect()
    }
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

/// Every non-zero slot stride a block's reads and writes use, deduplicated.
/// Any table walks one f64 per repeat, so reading one needs stride 1.
fn block_strides(b: &reroll::Block, rels: &[PosRel]) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    let want = |t: u32, out: &mut Vec<u32>| {
        if t != 0 && !out.contains(&t) {
            out.push(t);
        }
    };
    for (j, r) in rels.iter().enumerate() {
        want(r.out.stride, &mut out);
        for rel in [r.arg1, r.arg2i] {
            match rel {
                Some(sr) => want(sr.stride, &mut out),
                None => want(1, &mut out),
            }
        }
        for e in &r.elems {
            want(e.stride, &mut out);
        }
        // A contraction's coefficients are staged one column per element, so
        // reading this repeat's walks by one.
        if b.consts[j].is_some() || !r.elems.is_empty() {
            want(1, &mut out);
        }
    }
    out
}

/// The f64 argument of block node `j` at the current iteration.
fn block_cst(tape: &Tape, b: &reroll::Block, j: u32, tbl: &[NodeTables], sp: &StridePtrs) -> Cst {
    match tbl[j as usize].cst {
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

/// Resolve block node `j`'s argument to a primal address, emitting the index
/// arithmetic first when the argument is a gather.
fn block_arg(
    f: &mut Function,
    sp: &StridePtrs,
    rel: Option<SlotRel>,
    tbl: Option<u32>,
    tmp: u32,
) -> Addr {
    match rel {
        Some(sr) => sp.addr(sr.base, sr.stride),
        None => {
            let at = tbl.expect("a tabled argument has a table");
            // tmp = scratch + table[at + i] * 8
            f.instruction(&Instruction::LocalGet(SCRATCH_PTR));
            aload(f, sp.addr(at, 1));
            f.instruction(&Instruction::I32TruncF64U);
            f.instruction(&Instruction::I32Const(8));
            f.instruction(&Instruction::I32Mul);
            f.instruction(&Instruction::I32Add);
            f.instruction(&Instruction::LocalSet(tmp));
            Addr::Mem { ptr: tmp, slot: 0 }
        }
    }
}

/// The adjoint address matching a primal one. A gather's adjoint sits `adj`
/// slots above its primal, which is an immediate on the same base pointer.
fn adj_of(rel: Option<SlotRel>, pa: Addr, adj: u32, sp: &StridePtrs) -> Addr {
    match (rel, pa) {
        (Some(sr), _) => sp.addr(adj + sr.base, sr.stride),
        (None, Addr::Mem { ptr, .. }) => Addr::Mem { ptr, slot: adj },
        _ => unreachable!("a tabled argument resolves to a memory address"),
    }
}

/// A contraction's operand run, paired with its coefficients: the addresses
/// the unrolled sum reads, in element order.
type DotOps = Vec<(Addr, Cst)>;

/// `∑ elem * coeff`, left on the stack.
fn emit_dot_forward(f: &mut Function, ops: &DotOps) {
    for (c, (a, k)) in ops.iter().enumerate() {
        aload(f, *a);
        cload(f, *k);
        f.instruction(&Instruction::F64Mul);
        if c > 0 {
            f.instruction(&Instruction::F64Add);
        }
    }
}

/// Its backward step: each element's adjoint takes `dk * coeff`. The
/// coefficients are constants, so nothing needs the primal.
fn emit_dot_backward(f: &mut Function, dk: Addr, ops: &DotOps) {
    for (da, k) in ops {
        adj_incr_mulc(f, *da, dk, *k);
    }
}

/// The addresses a block-resident contraction reads: its operand elements at
/// this repeat, against the column of coefficients staged for each.
/// `off` is 0 for the primals and the adjoint offset for the adjoints, which
/// sit that many slots above them.
fn block_dot_ops(
    tape: &Tape,
    k0: u32,
    r: &PosRel,
    reps: u32,
    at: u32,
    sp: &StridePtrs,
    off: u32,
) -> DotOps {
    let e = tape.extent_at(k0);
    (0..e.len as usize)
        .map(|c| {
            (
                sp.addr(off + r.elems[c].base, r.elems[c].stride),
                Cst::At(sp.addr(at + c as u32 * reps, 1)),
            )
        })
        .collect()
}

/// The same for a contraction the emitter left straight-line, whose
/// coefficients fold into immediates rather than a staged column.
fn straight_dot_ops(tape: &Tape, k: u32, slots: &Slots, lay: Layout, off: u32) -> DotOps {
    let e = tape.extent_at(k);
    let base = tape.arg1_at(k);
    tape.coeffs(e)
        .iter()
        .enumerate()
        .map(|(c, v)| {
            (
                lay.at(off + slots.at(base + c as u32 * e.stride)),
                Cst::Imm(*v),
            )
        })
        .collect()
}

/// Opcodes with a lane-wise `f64x2` form. The host math imports have none —
/// `exp` and `log` are calls — so a density that uses one stays scalar.
fn lane_wise(op: Op) -> bool {
    matches!(
        op,
        Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::Neg
            | Op::AddC
            | Op::SubC
            | Op::RsubC
            | Op::MulC
            | Op::DivC
            | Op::RdivC
            | Op::DotC
    )
}

/// Whether a block's loop can run two repeats at once.
///
/// Two lanes have to be one `v128` apart, which means every slot a repeat
/// touches moves by one or not at all; a value carried from the previous
/// repeat would need the lanes to run in sequence, and a gather would need
/// each lane addressed separately. Positions kept in wasm locals are excluded
/// for now — widening those needs a second local pool.
fn widenable(tape: &Tape, b: &reroll::Block, rels: &[PosRel], flags: &[bool]) -> bool {
    if !b.reps.is_multiple_of(2) || flags.iter().any(|keep| *keep) {
        return false;
    }
    (0..b.len).all(|j| {
        let k0 = b.start + j;
        let r = &rels[j as usize];
        let carried = [&b.args[j as usize].arg1, &b.args[j as usize].arg2i]
            .into_iter()
            .zip([tape.arg1_at(k0), tape.arg2i_at(k0)])
            .any(|(rel, arg0)| matches!(rel, reroll::ArgRel::Affine(t) if *t == b.len && arg0 < b.start));
        lane_wise(tape.op_at(k0))
            && !carried
            && r.out.stride == 1
            && r.arg1.is_some_and(|sr| sr.stride <= 1)
            && r.arg2i.is_some_and(|sr| sr.stride <= 1)
            && r.elems.iter().all(|sr| sr.stride <= 1)
    })
}

/// Where a widened body reads one of its operands: a slot pair, or a
/// loop-invariant scalar broadcast to both lanes.
fn wide_addr(sr: SlotRel, sp: &StridePtrs, off: u32) -> Addr {
    match sp.addr(off + sr.base, sr.stride) {
        Addr::Mem { ptr, slot } if sr.stride == 0 => Addr::Splat { ptr, slot },
        a => a,
    }
}

/// The adjoint a widened body accumulates into. A loop-invariant target is the
/// same slot for both lanes, so it cannot be read-modify-written in the loop;
/// it gets a lane pair of its own, summed into the slot afterwards.
fn wide_adj(sr: SlotRel, sp: &StridePtrs, adj: u32, priv_of: &[(u32, u32)]) -> Addr {
    if sr.stride == 0 {
        let slot = adj + sr.base;
        let (_, local) = priv_of
            .iter()
            .find(|(s, _)| *s == slot)
            .expect("every loop-invariant adjoint target was given a lane pair");
        return Addr::Local(*local);
    }
    sp.addr(adj + sr.base, sr.stride)
}

/// Every loop-invariant adjoint slot a widened block accumulates into.
fn priv_targets(tape: &Tape, b: &reroll::Block, rels: &[PosRel], adj: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let want = |sr: &SlotRel, out: &mut Vec<u32>| {
        if sr.stride == 0 && !out.contains(&(adj + sr.base)) {
            out.push(adj + sr.base);
        }
    };
    for (j, r) in rels.iter().enumerate() {
        let (u1, u2, _) = reroll::uses(tape.op_at(b.start + j as u32));
        if u1 {
            if let Some(sr) = &r.arg1 {
                want(sr, &mut out);
            }
        }
        if u2 {
            if let Some(sr) = &r.arg2i {
                want(sr, &mut out);
            }
        }
        for sr in &r.elems {
            want(sr, &mut out);
        }
    }
    out
}

/// One widened pass over a block body, forward or backward: two repeats per
/// iteration, every value a lane pair.
#[allow(clippy::too_many_arguments)]
fn emit_wide_body(
    f: &mut Function,
    tape: &Tape,
    b: &reroll::Block,
    sp: &StridePtrs,
    tbl: &[NodeTables],
    rels: &[PosRel],
    adj: u32,
    privs: &[(u32, u32)],
    forward: bool,
) {
    let order: Vec<u32> = if forward {
        (0..b.len).collect()
    } else {
        (0..b.len).rev().collect()
    };
    for j in order {
        let k0 = b.start + j;
        let r = &rels[j as usize];
        let out = wide_addr(r.out, sp, 0);
        let dout = wide_addr(r.out, sp, adj);
        let cst = block_cst(tape, b, j, tbl, sp);

        if tape.op_at(k0) == Op::DotC {
            let at = tbl[j as usize].dot.expect("staged");
            let col = |c: usize| sp.addr(at + c as u32 * b.reps, 1);
            let e = tape.extent_at(k0);
            if forward {
                wstore_addr(f, out);
                for c in 0..e.len as usize {
                    wload(f, wide_addr(r.elems[c], sp, 0));
                    wload(f, col(c));
                    f.instruction(&Instruction::F64x2Mul);
                    if c > 0 {
                        f.instruction(&Instruction::F64x2Add);
                    }
                }
                wstore_end(f, out);
            } else {
                for c in 0..e.len as usize {
                    let da = wide_adj(r.elems[c], sp, adj, privs);
                    wadj(f, da, false, |f| {
                        wload(f, dout);
                        wload(f, col(c));
                        f.instruction(&Instruction::F64x2Mul);
                    });
                }
            }
            continue;
        }

        let (s1, s2) = (r.arg1.expect("checked"), r.arg2i.expect("checked"));
        let (pa1, pa2) = (wide_addr(s1, sp, 0), wide_addr(s2, sp, 0));
        let op = tape.op_at(k0);
        if forward {
            wstore_addr(f, out);
            match op {
                Op::Add | Op::Sub | Op::Mul | Op::Div => {
                    wload(f, pa1);
                    wload(f, pa2);
                }
                Op::RsubC | Op::RdivC => {
                    wcload(f, cst);
                    wload(f, pa1);
                }
                Op::Neg => wload(f, pa1),
                _ => {
                    wload(f, pa1);
                    wcload(f, cst);
                }
            }
            f.instruction(&match op {
                Op::Add | Op::AddC => Instruction::F64x2Add,
                Op::Sub | Op::SubC | Op::RsubC => Instruction::F64x2Sub,
                Op::Mul | Op::MulC => Instruction::F64x2Mul,
                Op::Div | Op::DivC | Op::RdivC => Instruction::F64x2Div,
                Op::Neg => Instruction::F64x2Neg,
                _ => unreachable!("checked by `lane_wise`"),
            });
            wstore_end(f, out);
            continue;
        }

        // Only the arguments the opcode actually reads get resolved: an unused
        // `arg2i` holds a stale index, and a lane pair was never set aside for
        // whatever slot that lands on.
        let da1 = wide_adj(s1, sp, adj, privs);
        let pass = |f: &mut Function| wload(f, dout);
        match op {
            Op::Add => {
                let da2 = wide_adj(s2, sp, adj, privs);
                wadj(f, da1, false, pass);
                wadj(f, da2, false, pass);
            }
            Op::Sub => {
                let da2 = wide_adj(s2, sp, adj, privs);
                wadj(f, da1, false, pass);
                wadj(f, da2, true, pass);
            }
            Op::Mul => {
                let da2 = wide_adj(s2, sp, adj, privs);
                wadj(f, da1, false, |f| {
                    wload(f, dout);
                    wload(f, pa2);
                    f.instruction(&Instruction::F64x2Mul);
                });
                wadj(f, da2, false, |f| {
                    wload(f, dout);
                    wload(f, pa1);
                    f.instruction(&Instruction::F64x2Mul);
                });
            }
            Op::Div => {
                let da2 = wide_adj(s2, sp, adj, privs);
                wadj(f, da1, false, |f| {
                    wload(f, dout);
                    wload(f, pa2);
                    f.instruction(&Instruction::F64x2Div);
                });
                wadj(f, da2, true, |f| {
                    wload(f, dout);
                    wload(f, pa1);
                    f.instruction(&Instruction::F64x2Mul);
                    wload(f, pa2);
                    wload(f, pa2);
                    f.instruction(&Instruction::F64x2Mul);
                    f.instruction(&Instruction::F64x2Div);
                });
            }
            Op::Neg | Op::RsubC => wadj(f, da1, true, pass),
            Op::AddC | Op::SubC => wadj(f, da1, false, pass),
            Op::MulC => wadj(f, da1, false, |f| {
                wload(f, dout);
                wcload(f, cst);
                f.instruction(&Instruction::F64x2Mul);
            }),
            Op::DivC => wadj(f, da1, false, |f| {
                wload(f, dout);
                wcload(f, cst);
                f.instruction(&Instruction::F64x2Div);
            }),
            Op::RdivC => wadj(f, da1, true, |f| {
                wload(f, dout);
                wcload(f, cst);
                f.instruction(&Instruction::F64x2Mul);
                wload(f, pa1);
                wload(f, pa1);
                f.instruction(&Instruction::F64x2Mul);
                f.instruction(&Instruction::F64x2Div);
            }),
            _ => unreachable!("checked by `lane_wise`"),
        }
    }
}

/// Sum a widened block's lane pairs back into the slots they stand for.
fn emit_priv_reduce(f: &mut Function, privs: &[(u32, u32)]) {
    for (slot, local) in privs {
        f.instruction(&Instruction::LocalGet(SCRATCH_PTR));
        f.instruction(&Instruction::LocalGet(SCRATCH_PTR));
        f.instruction(&Instruction::F64Load(memarg(*slot)));
        for lane in 0..2 {
            f.instruction(&Instruction::LocalGet(*local));
            f.instruction(&Instruction::F64x2ExtractLane(lane));
            f.instruction(&Instruction::F64Add);
        }
        f.instruction(&Instruction::F64Store(memarg(*slot)));
    }
}

/// One forward pass over a block body. `locals_only` limits it to the
/// positions kept in locals: the backward loop reconstructs those, because a
/// local cannot carry a value from the forward loop to here.
#[allow(clippy::too_many_arguments)]
fn emit_block_forward(
    f: &mut Function,
    tape: &Tape,
    m: &MathImportIndex,
    b: &reroll::Block,
    sp: &StridePtrs,
    bl: &BlockLocals,
    tbl: &[NodeTables],
    rels: &[PosRel],
    tmp1: u32,
    tmp2: u32,
    locals_only: bool,
) {
    for j in 0..b.len {
        if locals_only && bl.prim(j).is_none() {
            continue;
        }
        let k0 = b.start + j;
        let nt = tbl[j as usize];
        let ar = &b.args[j as usize];
        // Gather addresses are computed before the value is pushed: the store
        // address has to stay on top of the stack.
        let r = &rels[j as usize];
        if tape.op_at(k0) == Op::DotC {
            let ops = block_dot_ops(tape, k0, r, b.reps, nt.dot.expect("staged"), sp, 0);
            let w = bl.prim(j).unwrap_or(sp.addr(r.out.base, r.out.stride));
            astore_addr(f, w);
            emit_dot_forward(f, &ops);
            astore_end(f, w);
            continue;
        }
        let a1 = match bl.arg(b, &ar.arg1, tape.arg1_at(k0)) {
            Some(t) => bl.prim(t).expect("checked local"),
            None => block_arg(f, sp, r.arg1, nt.arg1, tmp1),
        };
        let a2 = match bl.arg(b, &ar.arg2i, tape.arg2i_at(k0)) {
            Some(t) => bl.prim(t).expect("checked local"),
            None => block_arg(f, sp, r.arg2i, nt.arg2i, tmp2),
        };
        let cst = block_cst(tape, b, j, tbl, sp);
        let w = bl.prim(j).unwrap_or(sp.addr(r.out.base, r.out.stride));
        astore_addr(f, w);
        emit_forward(f, tape, k0, m, a1, a2, cst);
        astore_end(f, w);
    }
}

/// One backward pass over a block body: recompute the iteration-local primals
/// the forward loop left in locals, clear the adjoints they accumulate into,
/// then one step per node in reverse.
#[allow(clippy::too_many_arguments)]
fn emit_block_backward(
    f: &mut Function,
    tape: &Tape,
    m: &MathImportIndex,
    b: &reroll::Block,
    sp: &StridePtrs,
    bl: &BlockLocals,
    tbl: &[NodeTables],
    rels: &[PosRel],
    adj: u32,
    tmp1: u32,
    tmp2: u32,
) {
    // Forward left iteration-local values in locals, which do not
    // survive to here: recompute this iteration's, then clear the
    // adjoints they accumulate into.
    emit_block_forward(f, tape, m, b, sp, bl, tbl, rels, tmp1, tmp2, true);
    for j in 0..b.len {
        if let Some(a) = bl.adj(j) {
            astore_addr(f, a);
            f.instruction(&Instruction::F64Const(0.0.into()));
            astore_end(f, a);
        }
    }
    for j in (0..b.len).rev() {
        let k0 = b.start + j;
        let nt = tbl[j as usize];
        let ar = &b.args[j as usize];
        let r = &rels[j as usize];
        if tape.op_at(k0) == Op::DotC {
            let at = nt.dot.expect("staged");
            let ops = block_dot_ops(tape, k0, r, b.reps, at, sp, adj);
            let dk = bl.adj(j).unwrap_or(sp.addr(adj + r.out.base, r.out.stride));
            emit_dot_backward(f, dk, &ops);
            continue;
        }
        let pa1 = match bl.arg(b, &ar.arg1, tape.arg1_at(k0)) {
            Some(t) => bl.prim(t).expect("checked local"),
            None => block_arg(f, sp, r.arg1, nt.arg1, tmp1),
        };
        let pa2 = match bl.arg(b, &ar.arg2i, tape.arg2i_at(k0)) {
            Some(t) => bl.prim(t).expect("checked local"),
            None => block_arg(f, sp, r.arg2i, nt.arg2i, tmp2),
        };
        let da1 = match bl.arg(b, &ar.arg1, tape.arg1_at(k0)) {
            Some(t) => bl.adj(t).expect("checked local"),
            None => adj_of(r.arg1, pa1, adj, sp),
        };
        let da2 = match bl.arg(b, &ar.arg2i, tape.arg2i_at(k0)) {
            Some(t) => bl.adj(t).expect("checked local"),
            None => adj_of(r.arg2i, pa2, adj, sp),
        };
        let pout = sp.addr(r.out.base, r.out.stride);
        let dout = sp.addr(adj + r.out.base, r.out.stride);
        emit_backward(
            f,
            tape,
            k0,
            m,
            Back {
                dk: bl.adj(j).unwrap_or(dout),
                da1,
                da2,
                pa1,
                pa2,
                pk: bl.prim(j).unwrap_or(pout),
                cst: block_cst(tape, b, j, tbl, sp),
            },
        );
    }
}

fn build_log_prob_grad(
    tape: &Tape,
    root: u32,
    n: u32,
    m: &MathImportIndex,
    blocks: &[reroll::Block],
    const_base: u32,
    slots: &Slots,
) -> Function {
    // 4 i32 params at local indices 0..4: params_ptr, grads_ptr, n_params
    // (unused — the recorded tape already encodes it), scratch_ptr.
    const GRADS_PTR: u32 = 1;
    let adj = n; // adjoint slots follow the primals
    let lay = Layout::for_tape(2 * n, !blocks.is_empty());
    // Positions written and read inside one iteration go in locals instead of
    // scratch; the widest block sizes the pool, and every block reuses it.
    let block_local = reroll::local_positions(tape, blocks, root);
    let widest_locals = block_local
        .iter()
        .map(|f| f.iter().filter(|x| **x).count() as u32)
        .max()
        .unwrap_or(0);
    let f64_locals = match lay {
        Layout::Locals { .. } => lay.local_count(2 * n),
        Layout::Memory => 2 * widest_locals,
    };
    let block_rels: Vec<Vec<PosRel>> = blocks
        .iter()
        .map(|b| slots.rels(tape, b).expect("checked by the plan"))
        .collect();
    let widest = blocks
        .iter()
        .zip(&block_rels)
        .map(|(b, r)| block_strides(b, r).len() as u32)
        .max()
        .unwrap_or(0);
    // One induction variable, reused by every block, then the base pointers,
    // then a scratch address per possible gather.
    let i32_locals = if blocks.is_empty() {
        0
    } else {
        1 + widest + reroll::MAX_TABLED as u32
    };
    let locals_base = FIRST_SLOT_LOCAL;
    let i32_base = FIRST_SLOT_LOCAL + f64_locals;
    let iv = i32_base;
    let tmp1 = iv + 1 + widest;
    let tmp2 = tmp1 + 1;

    // Which blocks run two repeats at a time, and where each one's
    // loop-invariant adjoints accumulate while they do.
    let wide: Vec<bool> = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| widenable(tape, b, &block_rels[i], &block_local[i]))
        .collect();
    let v128_base = i32_base + i32_locals;
    let privs: Vec<Vec<(u32, u32)>> = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| {
            if !wide[i] {
                return Vec::new();
            }
            priv_targets(tape, b, &block_rels[i], adj)
                .into_iter()
                .enumerate()
                .map(|(n, slot)| (slot, v128_base + n as u32))
                .collect()
        })
        .collect();
    let v128_locals = privs.iter().map(|p| p.len() as u32).max().unwrap_or(0);

    let mut decl: Vec<(u32, ValType)> = Vec::new();
    if f64_locals > 0 {
        decl.push((f64_locals, ValType::F64));
    }
    if i32_locals > 0 {
        decl.push((i32_locals, ValType::I32));
    }
    if v128_locals > 0 {
        decl.push((v128_locals, ValType::V128));
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

    let (_, tables) = stage_tables(tape, blocks, const_base, slots);

    let straight_fwd = |f: &mut Function, k: u32| {
        let w = lay.at(slots.at(k));
        if tape.op_at(k) == Op::DotC {
            astore_addr(f, w);
            emit_dot_forward(f, &straight_dot_ops(tape, k, slots, lay, 0));
            astore_end(f, w);
            return;
        }
        let a1 = lay.at(slots.at(tape.arg1_at(k)));
        let a2 = lay.at(slots.at(tape.arg2i_at(k)));
        let cst = Cst::Imm(if tape.op_at(k) == Op::Leaf {
            tape.value(k)
        } else {
            tape.arg2f_at(k)
        });
        astore_addr(f, w);
        emit_forward(f, tape, k, m, a1, a2, cst);
        astore_end(f, w);
    };

    // ---- forward pass ------------------------------------------------------
    let mut k = 0u32;
    let mut bi = 0usize;
    while k < n {
        if bi < blocks.len() && blocks[bi].start == k {
            let b = &blocks[bi];
            let sp = StridePtrs {
                iv,
                strides: block_strides(b, &block_rels[bi]),
                first_local: iv + 1,
            };
            let bl = block_locals_of(&block_local[bi], b.len, locals_base, widest_locals);
            let step = if wide[bi] { 2 } else { 1 };
            f.instruction(&Instruction::I32Const(0));
            f.instruction(&Instruction::LocalSet(iv));
            f.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty));
            sp.emit_setup(&mut f);
            if wide[bi] {
                emit_wide_body(
                    &mut f,
                    tape,
                    b,
                    &sp,
                    &tables[bi],
                    &block_rels[bi],
                    adj,
                    &privs[bi],
                    true,
                );
            } else {
                emit_block_forward(
                    &mut f,
                    tape,
                    m,
                    b,
                    &sp,
                    &bl,
                    &tables[bi],
                    &block_rels[bi],
                    tmp1,
                    tmp2,
                    false,
                );
            }
            f.instruction(&Instruction::LocalGet(iv));
            f.instruction(&Instruction::I32Const(step));
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
    astore_addr(&mut f, lay.at(adj + slots.at(root)));
    f.instruction(&Instruction::F64Const(1.0.into()));
    astore_end(&mut f, lay.at(adj + slots.at(root)));

    // ---- backward pass (reverse order) ------------------------------------
    let mut k = n;
    let mut bi = blocks.len();
    while k > 0 {
        if bi > 0 && blocks[bi - 1].end() == k {
            bi -= 1;
            let b = &blocks[bi];
            let sp = StridePtrs {
                iv,
                strides: block_strides(b, &block_rels[bi]),
                first_local: iv + 1,
            };
            let bl = block_locals_of(&block_local[bi], b.len, locals_base, widest_locals);
            let step: i32 = if wide[bi] { 2 } else { 1 };
            if wide[bi] {
                for (_, local) in &privs[bi] {
                    f.instruction(&Instruction::V128Const(0));
                    f.instruction(&Instruction::LocalSet(*local));
                }
            }
            f.instruction(&Instruction::I32Const(b.reps as i32 - step));
            f.instruction(&Instruction::LocalSet(iv));
            f.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty));
            sp.emit_setup(&mut f);
            if wide[bi] {
                emit_wide_body(
                    &mut f,
                    tape,
                    b,
                    &sp,
                    &tables[bi],
                    &block_rels[bi],
                    adj,
                    &privs[bi],
                    false,
                );
                f.instruction(&Instruction::LocalGet(iv));
                f.instruction(&Instruction::I32Const(step));
                f.instruction(&Instruction::I32Sub);
                f.instruction(&Instruction::LocalTee(iv));
                f.instruction(&Instruction::I32Const(0));
                f.instruction(&Instruction::I32GeS);
                f.instruction(&Instruction::BrIf(0));
                f.instruction(&Instruction::End);
                emit_priv_reduce(&mut f, &privs[bi]);
                k = b.start;
                continue;
            }
            emit_block_backward(
                &mut f,
                tape,
                m,
                b,
                &sp,
                &bl,
                &tables[bi],
                &block_rels[bi],
                adj,
                tmp1,
                tmp2,
            );
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
            if tape.op_at(k) == Op::DotC {
                let ops = straight_dot_ops(tape, k, slots, lay, adj);
                emit_dot_backward(&mut f, lay.at(adj + slots.at(k)), &ops);
                continue;
            }
            emit_backward(
                &mut f,
                tape,
                k,
                m,
                Back {
                    dk: lay.at(adj + slots.at(k)),
                    da1: lay.at(adj + slots.at(tape.arg1_at(k))),
                    da2: lay.at(adj + slots.at(tape.arg2i_at(k))),
                    pa1: lay.at(slots.at(tape.arg1_at(k))),
                    pa2: lay.at(slots.at(tape.arg2i_at(k))),
                    pk: lay.at(slots.at(k)),
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
        aload(&mut f, lay.at(adj + slots.at(pi)));
        f.instruction(&Instruction::F64Store(wasm_encoder::MemArg {
            offset: 0,
            align: 3,
            memory_index: 0,
        }));
    }

    // ---- return log_prob ---------------------------------------------------
    aload(&mut f, lay.at(slots.at(root)));
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
        Op::DotC => unreachable!("a contraction is emitted from its own path"),
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
        Op::DotC => unreachable!("a contraction is emitted from its own path"),
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
    /// In a widened body: one scalar read, broadcast to both lanes. What a
    /// loop-invariant operand becomes when two repeats run at once.
    Splat { ptr: u32, slot: u32 },
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
        Addr::Splat { .. } => unreachable!("a broadcast is read by the widened emitter"),
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
        Addr::Splat { .. } => unreachable!("a broadcast is never written"),
    }
}

// ---- widened access ------------------------------------------------------
//
// Two lanes at a time. A `v128` here is always two consecutive repeats of one
// value, so the only alignment a slot pair can promise is the 8 bytes a slot
// has: `iv` steps by two but the slot itself may sit at either parity.

fn wmemarg(slot: u32) -> wasm_encoder::MemArg {
    wasm_encoder::MemArg {
        offset: slot as u64 * 8,
        align: 3,
        memory_index: 0,
    }
}

fn wload(f: &mut Function, a: Addr) {
    match a {
        Addr::Local(i) => {
            f.instruction(&Instruction::LocalGet(i));
        }
        Addr::Mem { ptr, slot } => {
            f.instruction(&Instruction::LocalGet(ptr));
            f.instruction(&Instruction::V128Load(wmemarg(slot)));
        }
        Addr::Splat { ptr, slot } => {
            f.instruction(&Instruction::LocalGet(ptr));
            f.instruction(&Instruction::F64Load(memarg(slot)));
            f.instruction(&Instruction::F64x2Splat);
        }
    }
}

fn wstore_addr(f: &mut Function, a: Addr) {
    if let Addr::Mem { ptr, .. } = a {
        f.instruction(&Instruction::LocalGet(ptr));
    }
}

fn wstore_end(f: &mut Function, a: Addr) {
    match a {
        Addr::Local(i) => {
            f.instruction(&Instruction::LocalSet(i));
        }
        Addr::Mem { slot, .. } => {
            f.instruction(&Instruction::V128Store(wmemarg(slot)));
        }
        Addr::Splat { .. } => unreachable!("a broadcast is never written"),
    }
}

/// A node's f64 argument in a widened body: a moving constant is a slot pair
/// like any other value, a fixed one is the same in both lanes.
fn wcload(f: &mut Function, c: Cst) {
    match c {
        Cst::Imm(v) => {
            let bits = v.to_bits() as u128;
            f.instruction(&Instruction::V128Const(((bits << 64) | bits) as i128));
        }
        Cst::At(a) => wload(f, a),
    }
}

/// `d[da] += <two lanes on the stack>`, for whichever accumulation the caller
/// pushed. Splitting it this way keeps one shape for every backward step.
fn wadj(f: &mut Function, da: Addr, sub: bool, push: impl FnOnce(&mut Function)) {
    wstore_addr(f, da);
    wload(f, da);
    push(f);
    f.instruction(if sub {
        &Instruction::F64x2Sub
    } else {
        &Instruction::F64x2Add
    });
    wstore_end(f, da);
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
