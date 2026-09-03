//! `lgamma` and `digamma` as functions inside the emitted module.
//!
//! What the host does well stays imported. `exp` and `log` were written out
//! here too and measured: a logistic regression at N=5000 went from 132 to 255
//! µs per gradient, so V8's builtin beats a reduce-and-polynomial in the
//! module even with the call across the boundary. They went back.
//!
//! What the host does not have at all is defined here. JavaScript has no
//! `lgamma` or `digamma`, so every embedder was left to write its own, and the
//! series in this project's own benchmark harness was accurate to 2e-9 — which
//! is what a gradient through `student_t` was worth. Against CmdStan at the
//! same point it now agrees to 3e-14 rather than 8e-9.
//!
//! These mirror `stanwasm_autodiff`'s versions operation for operation, so the
//! two paths agree, and that crate's tests hold the shared algorithm to 1e-14
//! against a reference computed at 60 decimal digits.

use wasm_encoder::{BlockType, Function, Instruction, ValType};

/// Where the recurrence stops shifting and the asymptotic series takes over.
/// The same value `stanwasm_autodiff` uses.
const SHIFT_TO: f64 = 12.0;

fn f64c(v: f64) -> Instruction<'static> {
    Instruction::F64Const(v.into())
}

/// `while arg < SHIFT_TO { body; arg += 1 }`.
///
/// The test is "go round again while below", not "leave once at or above", so
/// a NaN argument leaves rather than sits here forever.
fn shift_loop(f: &mut Function, arg: u32, body: &[Instruction]) {
    f.instruction(&Instruction::Block(BlockType::Empty));
    f.instruction(&Instruction::Loop(BlockType::Empty));
    for i in [
        Instruction::LocalGet(arg),
        f64c(SHIFT_TO),
        Instruction::F64Lt,
        Instruction::I32Eqz,
        Instruction::BrIf(1),
    ] {
        f.instruction(&i);
    }
    for i in body {
        f.instruction(i);
    }
    for i in [
        Instruction::LocalGet(arg),
        f64c(1.0),
        Instruction::F64Add,
        Instruction::LocalSet(arg),
        Instruction::Br(0),
    ] {
        f.instruction(&i);
    }
    f.instruction(&Instruction::End);
    f.instruction(&Instruction::End);
}

/// An alternating Horner over `sq`, innermost coefficient first: the series
/// both functions carry are `c0 - sq(c1 - sq(c2 - ...))`.
fn alternating_horner(f: &mut Function, sq: u32, coeffs: &[f64]) {
    f.instruction(&f64c(coeffs[0]));
    for c in &coeffs[1..] {
        for i in [
            Instruction::LocalGet(sq),
            Instruction::F64Mul,
            f64c(*c),
            Instruction::F64Sub,
            Instruction::F64Neg,
        ] {
            f.instruction(&i);
        }
    }
}

/// Stirling-series log-Gamma. `log_idx` is the host's `log`.
pub fn lgamma(log_idx: u32) -> Function {
    const X: u32 = 0;
    const Z: u32 = 1;
    const R: u32 = 2;
    const INV: u32 = 3;
    const SQ: u32 = 4;
    let mut f = Function::new([(4, ValType::F64)]);

    for i in [
        Instruction::LocalGet(X),
        Instruction::LocalSet(Z),
        f64c(0.0),
        Instruction::LocalSet(R),
    ] {
        f.instruction(&i);
    }
    shift_loop(
        &mut f,
        Z,
        &[
            Instruction::LocalGet(R),
            Instruction::LocalGet(Z),
            Instruction::Call(log_idx),
            Instruction::F64Sub,
            Instruction::LocalSet(R),
        ],
    );

    for i in [
        f64c(1.0),
        Instruction::LocalGet(Z),
        Instruction::F64Div,
        Instruction::LocalTee(INV),
        Instruction::LocalGet(INV),
        Instruction::F64Mul,
        Instruction::LocalSet(SQ),
        // r + (z - 0.5) log z - z + log(2π)/2
        Instruction::LocalGet(R),
        Instruction::LocalGet(Z),
        f64c(0.5),
        Instruction::F64Sub,
        Instruction::LocalGet(Z),
        Instruction::Call(log_idx),
        Instruction::F64Mul,
        Instruction::F64Add,
        Instruction::LocalGet(Z),
        Instruction::F64Sub,
        f64c(0.5 * (2.0 * std::f64::consts::PI).ln()),
        Instruction::F64Add,
        Instruction::LocalGet(INV),
    ] {
        f.instruction(&i);
    }
    alternating_horner(
        &mut f,
        SQ,
        &[
            691.0 / 360360.0,
            1.0 / 1188.0,
            1.0 / 1680.0,
            1.0 / 1260.0,
            1.0 / 360.0,
            1.0 / 12.0,
        ],
    );
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Add);
    f.instruction(&Instruction::End);
    f
}

/// Asymptotic-series digamma. `log_idx` is the host's `log`.
pub fn digamma(log_idx: u32) -> Function {
    const X: u32 = 0;
    const Z: u32 = 1;
    const R: u32 = 2;
    const INV: u32 = 3;
    const SQ: u32 = 4;
    let mut f = Function::new([(4, ValType::F64)]);

    for i in [
        Instruction::LocalGet(X),
        Instruction::LocalSet(Z),
        f64c(0.0),
        Instruction::LocalSet(R),
    ] {
        f.instruction(&i);
    }
    shift_loop(
        &mut f,
        Z,
        &[
            Instruction::LocalGet(R),
            f64c(1.0),
            Instruction::LocalGet(Z),
            Instruction::F64Div,
            Instruction::F64Sub,
            Instruction::LocalSet(R),
        ],
    );

    for i in [
        f64c(1.0),
        Instruction::LocalGet(Z),
        Instruction::F64Div,
        Instruction::LocalTee(INV),
        Instruction::LocalGet(INV),
        Instruction::F64Mul,
        Instruction::LocalSet(SQ),
        // r + log z - 1/(2z) - z⁻² · series
        Instruction::LocalGet(R),
        Instruction::LocalGet(Z),
        Instruction::Call(log_idx),
        Instruction::F64Add,
        f64c(0.5),
        Instruction::LocalGet(INV),
        Instruction::F64Mul,
        Instruction::F64Sub,
        Instruction::LocalGet(SQ),
    ] {
        f.instruction(&i);
    }
    alternating_horner(
        &mut f,
        SQ,
        &[
            691.0 / 32760.0,
            1.0 / 132.0,
            1.0 / 240.0,
            1.0 / 252.0,
            1.0 / 120.0,
            1.0 / 12.0,
        ],
    );
    f.instruction(&Instruction::F64Mul);
    f.instruction(&Instruction::F64Sub);
    f.instruction(&Instruction::End);
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_encoder::{
        CodeSection, EntityType, ExportKind, ExportSection, FunctionSection, ImportSection, Module,
        TypeSection,
    };
    use wasmi::{Caller, Engine, Func, Linker, Store};

    /// A module exporting just the two, over an imported `log`.
    fn instantiate() -> (Store<()>, wasmi::Instance) {
        let mut types = TypeSection::new();
        types.ty().function([ValType::F64], [ValType::F64]);
        let mut imports = ImportSection::new();
        imports.import("Math", "log", EntityType::Function(0));
        let mut functions = FunctionSection::new();
        functions.function(0);
        functions.function(0);
        let mut exports = ExportSection::new();
        exports.export("lgamma", ExportKind::Func, 1);
        exports.export("digamma", ExportKind::Func, 2);
        let mut codes = CodeSection::new();
        codes.function(&lgamma(0));
        codes.function(&digamma(0));
        let mut m = Module::new();
        m.section(&types);
        m.section(&imports);
        m.section(&functions);
        m.section(&exports);
        m.section(&codes);

        let engine = Engine::default();
        let module = wasmi::Module::new(&engine, m.finish()).expect("module parses");
        let mut store = Store::new(&engine, ());
        let mut linker: Linker<()> = Linker::new(&engine);
        let log = Func::wrap(&mut store, |_: Caller<'_, ()>, x: f64| -> f64 { x.ln() });
        linker.define("Math", "log", log).unwrap();
        let instance = linker
            .instantiate_and_start(&mut store, &module)
            .expect("instantiate");
        (store, instance)
    }

    fn check(name: &str, host: fn(f64) -> f64) {
        let (mut store, instance) = instantiate();
        let f = instance
            .get_typed_func::<f64, f64>(&store, name)
            .expect("exported");
        let (mut worst, mut at) = (0.0_f64, 0.0);
        for i in 1..4000 {
            let x = i as f64 * 0.01;
            let got = f.call(&mut store, x).unwrap();
            let want = host(x);
            let e = (got - want).abs() / (1.0 + want.abs());
            if e > worst {
                worst = e;
                at = x;
            }
        }
        assert!(worst < 1e-15, "{name}: {worst:.2e} at x = {at}");
    }

    /// Against the same algorithm in `stanwasm_autodiff`, which is what the
    /// tape-replay path runs and what that crate holds to a reference. This
    /// catches a transcription slip, which is how these go wrong.
    #[test]
    fn lgamma_matches_the_recorded_tape() {
        check("lgamma", stanwasm_autodiff::lgamma);
    }

    #[test]
    fn digamma_matches_the_recorded_tape() {
        check("digamma", stanwasm_autodiff::digamma);
    }

    /// A NaN has to leave the shifting loop rather than sit in it.
    #[test]
    fn a_nan_terminates() {
        let (mut store, instance) = instantiate();
        for name in ["lgamma", "digamma"] {
            let f = instance.get_typed_func::<f64, f64>(&store, name).unwrap();
            assert!(f.call(&mut store, f64::NAN).unwrap().is_nan(), "{name}");
        }
    }
}
