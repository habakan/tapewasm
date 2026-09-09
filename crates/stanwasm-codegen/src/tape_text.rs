//! A tape written as text, for a front end that is not in this process.
//!
//! [`compile_tape`](crate::compile_tape) needs a [`Tape`], and a front end that
//! builds one somewhere else — another language, another wasm module — needs a
//! way to hand it over. This is that way: one instruction per line, each naming
//! a `Tape` method.
//!
//! ```text
//! n_params 3
//! test_params 0.5 1.2 0.1     # optional: a point a caller wants evaluated
//! new_var 0.1                 # instruction 0
//! new_var 0.9                 # instruction 1
//! mul 0 1                     # instruction 2
//! add_c 2 4.0                 # instruction 3
//! root 3
//! ```
//!
//! Blank lines and `#` comments are skipped. `n_params` is required and names
//! the leading run of `new_var`s that are the parameters; a `new_var` after any
//! other instruction is a constant. `root` names the instruction whose value
//! the module returns.
//!
//! **Operands are instruction numbers, not node indices.** The tape numbers
//! equal expressions into one node, so the k-th instruction is not generally
//! the k-th node, and a front end that assumes otherwise produces a module that
//! runs and is wrong. Refer to instructions and this module keeps the mapping.
//!
//! The format is not an artifact: a tape is written and consumed in one go,
//! nothing outlives the call, and so nothing here is promised across versions.

use stanwasm_autodiff::Tape;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TapeTextError {
    #[error("line {line}: {what}")]
    Line { line: usize, what: String },
    #[error("no `n_params` line")]
    NoNParams,
    #[error("no `root` line")]
    NoRoot,
}

/// A tape read back from text, with what [`compile_tape`](crate::compile_tape)
/// needs beside it.
pub struct Program {
    pub tape: Tape,
    pub n_params: usize,
    pub root: u32,
    /// Whatever `test_params` named, empty when the line was absent.
    pub test_params: Vec<f64>,
}

pub fn parse(src: &str) -> Result<Program, TapeTextError> {
    let mut tape = Tape::new();
    let mut n_params = None;
    let mut root = None;
    let mut test_params = Vec::new();
    let mut ids: Vec<u32> = Vec::new();

    for (k, raw) in src.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let no = k + 1;
        let bad = |what: String| TapeTextError::Line { line: no, what };
        let f: Vec<&str> = line.split_whitespace().collect();

        let want = |k: usize| -> Result<&str, TapeTextError> {
            f.get(k)
                .copied()
                .ok_or_else(|| bad(format!("`{}` wants {} operands", f[0], k)))
        };
        let idx = |k: usize| -> Result<u32, TapeTextError> {
            let at: usize = want(k)?
                .parse()
                .map_err(|_| bad(format!("`{}` is not an instruction number", f[k])))?;
            ids.get(at)
                .copied()
                .ok_or_else(|| bad(format!("instruction {at} has not been written yet")))
        };
        let num = |k: usize| -> Result<f64, TapeTextError> {
            want(k)?
                .parse()
                .map_err(|_| bad(format!("`{}` is not a number", f[k])))
        };

        match f[0] {
            "n_params" => {
                n_params = Some(
                    want(1)?
                        .parse()
                        .map_err(|_| bad("`n_params` is not a count".into()))?,
                );
                continue;
            }
            "test_params" => {
                test_params = f[1..]
                    .iter()
                    .map(|s| s.parse().map_err(|_| bad(format!("`{s}` is not a number"))))
                    .collect::<Result<_, _>>()?;
                continue;
            }
            "root" => {
                root = Some(idx(1)?);
                continue;
            }
            _ => {}
        }

        let node = match f[0] {
            "new_var" => tape.new_var(num(1)?),
            "add" => tape.add(idx(1)?, idx(2)?),
            "sub" => tape.sub(idx(1)?, idx(2)?),
            "mul" => tape.mul(idx(1)?, idx(2)?),
            "div" => tape.div(idx(1)?, idx(2)?),
            "neg" => tape.neg(idx(1)?),
            "exp" => tape.exp(idx(1)?),
            "log" => tape.log(idx(1)?),
            "sin" => tape.sin(idx(1)?),
            "cos" => tape.cos(idx(1)?),
            "sqrt" => tape.sqrt(idx(1)?),
            "abs" => tape.abs(idx(1)?),
            "lgamma" => tape.lgamma(idx(1)?),
            "phi" => tape.phi(idx(1)?),
            "pow" => tape.pow(idx(1)?, num(2)?),
            "add_c" => tape.add_c(idx(1)?, num(2)?),
            "sub_c" => tape.sub_c(idx(1)?, num(2)?),
            "rsub_c" => tape.rsub_c(idx(1)?, num(2)?),
            "mul_c" => tape.mul_c(idx(1)?, num(2)?),
            "div_c" => tape.div_c(idx(1)?, num(2)?),
            "rdiv_c" => tape.rdiv_c(idx(1)?, num(2)?),
            other => return Err(bad(format!("unknown instruction `{other}`"))),
        };
        ids.push(node);
    }

    Ok(Program {
        tape,
        n_params: n_params.ok_or(TapeTextError::NoNParams)?,
        root: root.ok_or(TapeTextError::NoRoot)?,
        test_params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_numbers_survive_the_tape_sharing_a_node() {
        // `mul 0 1` twice is one node, so instruction 3 and instruction 2 name
        // the same one — and instruction 4 still has to mean what it says.
        let p =
            parse("n_params 2\nnew_var 0.5\nnew_var 2.0\nmul 0 1\nmul 0 1\nadd_c 3 1.0\nroot 4")
                .unwrap();
        assert_eq!(p.n_params, 2);
        assert!((p.tape.value(p.root) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn a_forward_reference_is_an_error_rather_than_a_wrong_node() {
        let e = match parse("n_params 1\nnew_var 0.5\nadd 0 9\nroot 1") {
            Err(e) => e,
            Ok(_) => panic!("a forward reference parsed"),
        };
        assert!(e.to_string().contains("has not been written yet"), "{e}");
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let p = parse("# a tape\nn_params 1\n\nnew_var 3.0  # the only parameter\nroot 0").unwrap();
        assert_eq!(p.tape.len(), 1);
        assert_eq!(p.n_params, 1);
    }
}
