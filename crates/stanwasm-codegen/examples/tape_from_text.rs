//! Replay a tape written by another front end, compile it, and run it.
//!
//!   cargo run -p stanwasm-codegen --example tape_from_text -- <file>
//!
//! The file is one instruction per line, each naming a `Tape` method. Operands
//! are instruction numbers, not node indices: the tape numbers equal
//! expressions into one node, so the two run apart.
//!
//!   n_params 3
//!   test_params 0.5 1.2 0.1
//!   new_var 0.1          <- node 0
//!   add_c 0 1.0          <- node 3, say
//!   root 41
//!
//! Prints `lp` and the gradient the emitted module computes at `test_params`,
//! for a caller that wants to check them against its own. Given a second
//! argument it also writes the module there and prints the buffer sizes a host
//! needs to call it, so the caller can drive the module itself.

use stanwasm_autodiff::Tape;
use stanwasm_codegen::{compile_tape, Reroll};
use wasmi::{Caller, Engine, Func, Linker, Memory, MemoryType, Module, Store};

fn replay(src: &str) -> (Tape, usize, u32, Vec<f64>) {
    let mut tape = Tape::new();
    let mut n_params = 0usize;
    let mut root = 0u32;
    let mut test_params = Vec::new();
    let mut ids: Vec<u32> = Vec::new();

    for (lineno, line) in src.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let i = |k: usize| ids[f[k].parse::<usize>().expect("instruction index")];
        let c = |k: usize| f[k].parse::<f64>().expect("constant");
        match f[0] {
            "n_params" => {
                n_params = f[1].parse().unwrap();
                continue;
            }
            "test_params" => {
                test_params = f[1..].iter().map(|s| s.parse().unwrap()).collect();
                continue;
            }
            "root" => {
                root = ids[f[1].parse::<usize>().unwrap()];
                continue;
            }
            _ => {}
        }
        let node = match f[0] {
            "new_var" => tape.new_var(c(1)),
            "add" => tape.add(i(1), i(2)),
            "sub" => tape.sub(i(1), i(2)),
            "mul" => tape.mul(i(1), i(2)),
            "div" => tape.div(i(1), i(2)),
            "neg" => tape.neg(i(1)),
            "exp" => tape.exp(i(1)),
            "log" => tape.log(i(1)),
            "sin" => tape.sin(i(1)),
            "cos" => tape.cos(i(1)),
            "sqrt" => tape.sqrt(i(1)),
            "abs" => tape.abs(i(1)),
            "lgamma" => tape.lgamma(i(1)),
            "phi" => tape.phi(i(1)),
            "pow" => tape.pow(i(1), c(2)),
            "add_c" => tape.add_c(i(1), c(2)),
            "sub_c" => tape.sub_c(i(1), c(2)),
            "rsub_c" => tape.rsub_c(i(1), c(2)),
            "mul_c" => tape.mul_c(i(1), c(2)),
            "div_c" => tape.div_c(i(1), c(2)),
            "rdiv_c" => tape.rdiv_c(i(1), c(2)),
            other => panic!("line {}: unknown instruction {other:?}", lineno + 1),
        };
        ids.push(node);
    }
    (tape, n_params, root, test_params)
}

#[derive(Default)]
struct HostState;

fn run(
    wasm: &[u8],
    n_params: usize,
    params: &[f64],
    scratch_len: usize,
    consts: &[f64],
) -> (f64, Vec<f64>) {
    let mut config = wasmi::Config::default();
    config.wasm_simd(true);
    let engine = Engine::new(&config);
    let module = Module::new(&engine, wasm).expect("module parses");
    let mut store = Store::new(&engine, HostState);
    let pages = ((n_params * 2 + scratch_len) * 8).div_ceil(65536).max(1) as u32;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();

    let mut linker: Linker<HostState> = Linker::new(&engine);
    macro_rules! unary {
        ($name:literal, $f:expr) => {{
            let f = Func::wrap(&mut store, |_: Caller<'_, HostState>, x: f64| -> f64 {
                $f(x)
            });
            linker.define("Math", $name, f).unwrap();
        }};
    }
    unary!("exp", f64::exp);
    unary!("log", f64::ln);
    unary!("sin", f64::sin);
    unary!("cos", f64::cos);
    unary!("tan", f64::tan);
    unary!("asin", f64::asin);
    unary!("acos", f64::acos);
    unary!("atan", f64::atan);
    unary!("lgamma", stanwasm_autodiff::lgamma);
    unary!("digamma", stanwasm_autodiff::digamma);
    unary!("phi", stanwasm_autodiff::phi_cdf);
    let pow = Func::wrap(
        &mut store,
        |_: Caller<'_, HostState>, x: f64, y: f64| -> f64 { x.powf(y) },
    );
    linker.define("Math", "pow", pow).unwrap();
    linker.define("stan", "memory", memory).unwrap();

    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("instantiate");
    let lpg = instance
        .get_typed_func::<(i32, i32, i32, i32), f64>(&store, "log_prob_grad")
        .unwrap();

    let grads_ptr = (n_params * 8) as i32;
    let scratch_ptr = (n_params * 16) as i32;
    let bytes: Vec<u8> = params.iter().flat_map(|p| p.to_le_bytes()).collect();
    memory.write(&mut store, 0, &bytes).unwrap();
    if !consts.is_empty() {
        let at = scratch_ptr as usize + (scratch_len - consts.len()) * 8;
        let tbl: Vec<u8> = consts.iter().flat_map(|c| c.to_le_bytes()).collect();
        memory.write(&mut store, at, &tbl).unwrap();
    }
    let lp = lpg
        .call(&mut store, (0, grads_ptr, n_params as i32, scratch_ptr))
        .unwrap();
    let mut buf = vec![0u8; n_params * 8];
    memory.read(&store, grads_ptr as usize, &mut buf).unwrap();
    let grads = buf
        .chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (lp, grads)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: tape_from_text <file>");
    let src = std::fs::read_to_string(&path).expect("read tape");
    let (tape, n_params, root, test_params) = replay(&src);
    eprintln!("replayed {} nodes, {n_params} params", tape.len());

    let compiled = compile_tape(&tape, n_params, root, Reroll::default()).expect("compile");
    eprintln!("emitted {} bytes", compiled.wasm.len());

    let params = if test_params.is_empty() {
        vec![0.1; n_params]
    } else {
        test_params
    };
    let (lp, grads) = run(
        &compiled.wasm,
        compiled.n_params,
        &params,
        compiled.scratch_len,
        &compiled.const_table,
    );
    println!("lp {lp:.15e}");
    for g in &grads {
        println!("grad {g:.15e}");
    }

    if let Some(out) = std::env::args().nth(2) {
        std::fs::write(&out, &compiled.wasm).expect("write module");
        println!("wasm {out}");
        println!("n_params {}", compiled.n_params);
        println!("scratch_len {}", compiled.scratch_len);
        println!("layout_id {}", compiled.layout_id);
        for c in &compiled.const_table {
            println!("const {c:.17e}");
        }
    }
}
