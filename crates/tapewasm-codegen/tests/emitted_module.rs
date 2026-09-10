//! End-to-end: emit a tape as wasm, run it under wasmi, and check the result
//! against the tape's own reverse pass.
//!
//! The tape is the oracle here — it computes the same derivatives by a wholly
//! different route (interpreted, node by node) than the module does, so the
//! two agreeing is evidence about the emitter rather than about one shared
//! implementation.

use tapewasm_codegen::shapes;
use tapewasm_codegen::{compile_tape, Reroll};
use wasmi::{Caller, Engine, Func, Linker, Memory, MemoryType, Module, Store};

fn lgamma(x: f64) -> f64 {
    tapewasm_autodiff::lgamma(x)
}
fn digamma(x: f64) -> f64 {
    tapewasm_autodiff::digamma(x)
}
fn phi(x: f64) -> f64 {
    tapewasm_autodiff::phi_cdf(x)
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
    unary!("tan", f64::tan);
    unary!("asin", f64::asin);
    unary!("acos", f64::acos);
    unary!("atan", f64::atan);
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
    // The emitter widens a re-rolled loop to `f64x2` where it can, which wasmi
    // parses only with the proposal enabled.
    let mut config = wasmi::Config::default();
    config.wasm_simd(true);
    let engine = Engine::new(&config);
    let module = Module::new(&engine, wasm).expect("module parses");
    let mut store = Store::new(&engine, HostState);

    // Host-allocated memory shared with the AOT module: params, grads, then the
    // module's primal/adjoint scratch.
    let pages = ((n_params * 2 + scratch_len) * 8).div_ceil(65536).max(1) as u32;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();

    let mut linker: Linker<HostState> = Linker::new(&engine);
    install_math(&mut linker, &mut store);
    linker.define("tapewasm", "memory", memory).unwrap();

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
        .call(
            &mut store,
            (params_ptr, grads_ptr, n_params as i32, scratch_ptr),
        )
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

/// What the tape computes at `params`, by its own forward replay and reverse
/// pass.
fn tape_oracle(tape: &mut tapewasm_autodiff::Tape, root: u32, params: &[f64]) -> (f64, Vec<f64>) {
    tape.forward_replay(params);
    tape.reset_grads();
    tape.backward(root);
    let grads = (0..params.len() as u32).map(|i| tape.grad_at(i)).collect();
    (tape.value(root), grads)
}

/// Emit `tape`, run the module at `params`, and check both against the tape.
fn agrees(tape: &mut tapewasm_autodiff::Tape, root: u32, params: &[f64], mode: Reroll, eps: f64) {
    let c = compile_tape(tape, params.len(), root, mode).unwrap();
    let (lp, grads) =
        run_aot_log_prob_grad(&c.wasm, c.n_params, params, c.scratch_len, &c.const_table);
    let (want_lp, want_grads) = tape_oracle(tape, root, params);
    assert!(close(lp, want_lp, eps), "{mode:?}: lp {lp} vs {want_lp}");
    for (i, (a, w)) in grads.iter().zip(want_grads.iter()).enumerate() {
        assert!(close(*a, *w, eps), "{mode:?}: grad[{i}] {a} vs {w}");
    }
}

#[test]
fn a_straight_line_module_matches_the_tape() {
    let (mut tape, root) = shapes::linreg(20);
    agrees(&mut tape, root, &[0.4, 1.1, -0.3], Reroll::Auto, 1e-12);
}

/// Enough points that `Auto` re-rolls the element-wise runs into wasm loops.
/// The small case above stays straight-line, so without this the loop emitter
/// is never exercised.
#[test]
fn a_rerolled_module_matches_the_tape() {
    let (mut tape, root) = shapes::linreg(2000);
    let c = compile_tape(&tape, 3, root, Reroll::Auto).unwrap();
    assert!(!c.const_table.is_empty(), "expected a re-rolled loop");
    agrees(&mut tape, root, &[0.4, 1.1, 0.3], Reroll::Auto, 1e-12);
}

/// Calling twice must give the same answer: the scratch buffer is reused, so a
/// stale adjoint or a clobbered constant table would only show on the second
/// call.
#[test]
fn a_rerolled_module_is_reentrant() {
    let (tape, root) = shapes::linreg(2000);
    let c = compile_tape(&tape, 3, root, Reroll::Auto).unwrap();
    let p = [0.4, 1.1, 0.3];
    let first = run_aot_log_prob_grad(&c.wasm, c.n_params, &p, c.scratch_len, &c.const_table);
    let again = run_aot_log_prob_grad(&c.wasm, c.n_params, &p, c.scratch_len, &c.const_table);
    assert_eq!(first, again);
}

/// A gather (`mu[g[i]]` with irregular groups) is the case no stride
/// describes, so the emitter reads its slot index from a table — the one loop
/// shape whose addresses are computed at run time.
#[test]
fn a_gather_matches_the_tape() {
    let (mut tape, root) = shapes::gather(400);
    let p: Vec<f64> = (0..9).map(|i| 0.3 - i as f64 * 0.05).collect();
    agrees(&mut tape, root, &p, Reroll::Always, 1e-12);
}

/// A block long enough that the emitter has to keep most of it in the scratch
/// buffer rather than in locals.
#[test]
fn a_long_block_matches_the_tape() {
    let (mut tape, root) = shapes::mvn_cholesky(120, 8);
    let n = 8 + 8 * 9 / 2;
    let p: Vec<f64> = (0..n)
        .map(|i| 0.1 + (i as f64 * 0.37).sin() * 0.2)
        .collect();
    agrees(&mut tape, root, &p, Reroll::Auto, 1e-10);
}

/// A contraction against constant coefficients, whose result a later block
/// reads back. That crossing is the only place the scratch buffer's slot order
/// is observable, and the contraction has two emitters — unrolled in place, or
/// inside a loop reading a staged column — which have to agree.
#[test]
fn a_contraction_matches_the_tape_in_every_reroll_mode() {
    let (mut tape, root) = shapes::matvec(2000, 4);
    for mode in [Reroll::Auto, Reroll::Always, Reroll::Never] {
        agrees(&mut tape, root, &[0.5, -0.3, 0.8, 0.1, -0.5], mode, 1e-10);
        agrees(&mut tape, root, &[-1.2, 0.4, 0.0, 0.9, 0.25], mode, 1e-10);
    }
}

/// A contraction repeated too few times to join a block, in a module that
/// re-rolls something else, walks its run in a loop against a staged column.
#[test]
fn a_contraction_outside_every_block_matches_the_tape() {
    let mut tape = tapewasm_autodiff::Tape::new();
    let p: Vec<u32> = (0..40).map(|i| tape.new_var(0.1 * i as f64)).collect();
    let c0: Vec<f64> = (0..40).map(|i| (i as f64 * 0.7).sin()).collect();
    let c1: Vec<f64> = (0..20).map(|i| 1.0 - i as f64 * 0.03).collect();
    let d0 = tape.dot_c(p[0], 1, &c0);
    let d1 = tape.dot_c(p[1], 2, &c1);
    let mut acc = tape.mul(d0, d1);
    for &v in &p {
        let sq = tape.mul(v, v);
        acc = tape.add(acc, sq);
    }
    let c = compile_tape(&tape, 40, acc, Reroll::Always).unwrap();
    assert!(
        c.const_table.windows(40).any(|w| w == c0.as_slice()),
        "the straight-line contraction's coefficients were not staged"
    );
    let params: Vec<f64> = (0..40).map(|i| 0.3 - i as f64 * 0.02).collect();
    for mode in [Reroll::Auto, Reroll::Always, Reroll::Never] {
        agrees(&mut tape, acc, &params, mode, 1e-12);
    }
}

/// A reduction inside a repeating statement — a dense layer's per-output sum —
/// is re-rolled with the rest of its statement, seed and run both moving.
#[test]
fn a_reduction_inside_a_block_matches_the_tape() {
    let (h, k) = (5, 30);
    let mut tape = tapewasm_autodiff::Tape::new();
    let x: Vec<u32> = (0..h).map(|i| tape.new_var(0.2 * i as f64 - 0.3)).collect();
    let w: Vec<u32> = (0..h * k)
        .map(|i| tape.new_var((i as f64 * 0.37).sin()))
        .collect();
    let b: Vec<u32> = (0..k).map(|j| tape.new_var(0.1 * j as f64)).collect();
    let mut acc = tape.mul_c(x[0], 0.5);
    for j in 0..k {
        let p: Vec<u32> = (0..h).map(|i| tape.mul(x[i], w[j * h + i])).collect();
        let s = tape.sum_run(b[j], p[0], 1, h as u32);
        let e = tape.exp(s);
        let sq = tape.mul(e, s);
        acc = tape.add(acc, sq);
    }
    let params: Vec<f64> = (0..h + h * k + k)
        .map(|i| 0.1 + (i as f64 * 0.53).cos() * 0.3)
        .collect();
    for mode in [Reroll::Auto, Reroll::Always, Reroll::Never] {
        agrees(&mut tape, acc, &params, mode, 1e-12);
    }
}

/// `Always` and `Never` on one tape is the only place the loop and
/// straight-line emitters can be compared directly.
#[test]
fn the_reroll_modes_agree() {
    let (mut tape, root) = shapes::linreg(400);
    for mode in [Reroll::Auto, Reroll::Always, Reroll::Never] {
        agrees(&mut tape, root, &[0.4, 1.3, -0.15], mode, 1e-12);
    }
}

/// `tan`, `asin`, `acos` and `atan` are imports the emitter has to ask for by
/// name, and nothing else in these shapes reaches them.
#[test]
fn the_inverse_trig_functions_match_the_tape() {
    let mut tape = tapewasm_autodiff::Tape::new();
    let a = tape.new_var(0.1);
    let b = tape.new_var(0.1);
    let mut acc = tape.mul(a, b);
    for i in 0..4 {
        let x = i as f64 * 0.5 - 0.75;
        let t = tape.mul_c(a, 0.3);
        let t = tape.add_c(t, 0.1 * x);
        let t = tape.tan(t);
        acc = tape.add(acc, t);
        let s = tape.mul_c(b, 0.2);
        let s = tape.asin(s);
        acc = tape.add(acc, s);
        let c = tape.mul_c(a, 0.15);
        let c = tape.acos(c);
        acc = tape.add(acc, c);
        let d = tape.mul(a, b);
        let d = tape.add_c(d, x);
        let d = tape.atan(d);
        acc = tape.add(acc, d);
    }
    // Two points, because `acos` and `asin` bend hardest away from zero.
    agrees(&mut tape, acc, &[0.4, -0.6], Reroll::Auto, 1e-12);
    agrees(&mut tape, acc, &[-1.1, 0.9], Reroll::Auto, 1e-12);
}

#[test]
fn the_module_validates_with_wasmparser() {
    let (tape, root) = shapes::linreg(20);
    let c = compile_tape(&tape, 3, root, Reroll::Auto).unwrap();
    let result = wasmparser::Validator::new().validate_all(&c.wasm);
    assert!(result.is_ok(), "wasm did not validate: {:?}", result.err());
}

/// No emitter arm for the Student-t tail, so the refusal has to be a message
/// rather than a wasm trap.
#[test]
fn an_op_without_an_emitter_arm_is_reported_rather_than_trapping() {
    let mut tape = tapewasm_autodiff::Tape::new();
    let a = tape.new_var(0.1);
    let root = tape.student_t_lccdf(a, 4.0);
    let err = compile_tape(&tape, 1, root, Reroll::Auto)
        .expect_err("the Student-t tail has no emitter")
        .to_string();
    assert!(err.contains("StudentTLccdf"), "{err}");
}

/// The id a host compares before handing a module someone else's scratch
/// buffer, so it has to travel with the module and it has to differ whenever
/// the buffers do.
#[test]
fn the_layout_id_is_exported_and_identifies_the_buffers() {
    let (t2, r2) = shapes::linreg(2);
    let two = compile_tape(&t2, 3, r2, Reroll::Auto).unwrap();
    let two_again = compile_tape(&t2, 3, r2, Reroll::Auto).unwrap();
    let (t4, r4) = shapes::linreg(4);
    let four = compile_tape(&t4, 3, r4, Reroll::Auto).unwrap();

    assert_eq!(
        two.layout_id, two_again.layout_id,
        "recompiling one tape must not move its id"
    );
    assert_ne!(two.layout_id, four.layout_id);
    assert_ne!(two.scratch_len, four.scratch_len);

    let mut found = None;
    for payload in wasmparser::Parser::new(0).parse_all(&two.wasm) {
        if let wasmparser::Payload::ExportSection(section) = payload.unwrap() {
            for export in section {
                let export = export.unwrap();
                if export.name == "tapewasm_layout_id" {
                    found = Some(export.kind);
                }
            }
        }
    }
    assert_eq!(
        found,
        Some(wasmparser::ExternalKind::Global),
        "the module exports no layout id global"
    );
}

#[test]
fn compile_tape_rejects_a_param_count_the_tape_does_not_open_with() {
    let mut tape = tapewasm_autodiff::Tape::new();
    let a = tape.new_var(0.5);
    let root = tape.exp(a);
    let err = compile_tape(&tape, 2, root, Reroll::default()).unwrap_err();
    assert!(err.to_string().contains("n_params is 2"), "{err}");
}

/// Every instruction the text format accepts, through the emitter, against the
/// tape's reverse pass.
///
/// The hole this closes: `tan`, `asin`, `acos` and `atan` were emittable and
/// unwritable for as long as the format existed, because the only test that
/// reached them built the tape directly. Adding an op to the emitter without
/// adding it here now leaves a listed instruction untested rather than an
/// unreachable one unnoticed. `dot_c`/`sum_run` close the same hole for the
/// contraction and the reduction — this exercises them through the emitter
/// (both re-roll modes), where `tape_text`'s own tests only check parsing.
#[test]
fn every_instruction_in_the_text_format_reaches_the_emitter() {
    // One line per instruction, arranged so nothing lands outside a domain:
    // `asin`/`acos` want |x| <= 1, `log`/`lgamma`/`sqrt` want x > 0.
    let src = "\
n_params 2
new_var 0.4
new_var 0.7
add 0 1
sub 0 1
mul 0 1
div 0 1
neg 0
exp 0
mul_c 7 0.1
log 8
sin 0
cos 0
tan 0
asin 0
acos 0
atan 0
sqrt 8
abs 3
lgamma 8
phi 0
pow 0 3.0
add_c 0 1.5
sub_c 0 0.25
rsub_c 0 2.0
mul_c 0 1.5
div_c 0 2.0
rdiv_c 8 1.0
add 2 3
add 27 4
add 28 5
add 29 6
add 30 9
add 31 10
add 32 11
add 33 12
add 34 13
add 35 14
add 36 15
add 37 16
add 38 17
add 39 18
add 40 19
add 41 20
add 42 21
add 43 22
add 44 23
add 45 24
add 46 25
add 47 26
dot_c 3 2 1.0 3 2.0 4 3.0
sum_run 0 3 5 6 7
add 48 49
add 51 50
root 52
";
    let program = tapewasm_codegen::tape_text::parse(src).expect("every instruction parses");
    let mut tape = program.tape;
    // Both re-roll modes, because the straight-line and loop emitters select
    // instructions separately and only the tape is shared.
    for mode in [Reroll::Never, Reroll::Always] {
        let c = compile_tape(&tape, 2, program.root, mode).unwrap();
        let params = [0.4, 0.7];
        let (lp, grads) =
            run_aot_log_prob_grad(&c.wasm, c.n_params, &params, c.scratch_len, &c.const_table);
        let (want_lp, want_grads) = tape_oracle(&mut tape, program.root, &params);
        assert!(close(lp, want_lp, 1e-12), "{mode:?}: lp {lp} vs {want_lp}");
        for (i, (a, w)) in grads.iter().zip(want_grads.iter()).enumerate() {
            assert!(close(*a, *w, 1e-12), "{mode:?}: grad[{i}] {a} vs {w}");
        }
    }
}
