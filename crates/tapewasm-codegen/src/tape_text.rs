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
//! The instructions, all of them:
//!
//! ```text
//! new_var C                             a leaf
//! add A B    sub A B    mul A B         two operands
//! div A B
//! neg A      exp A      log A           one operand
//! sin A      cos A      tan A
//! asin A     acos A     atan A
//! sqrt A     abs A      lgamma A
//! phi A
//! pow A C                               one operand and a constant
//! add_c A C  sub_c A C  rsub_c A C      `rsub_c` is `C - A`
//! mul_c A C  div_c A C  rdiv_c A C      `rdiv_c` is `C / A`
//! dot_c LEN A0 C0 A1 C1 ... A(LEN-1) C(LEN-1)   a contraction (see below)
//! sum_run SEED LEN A0 A1 ... A(LEN-1)           a reduction (see below)
//! ```
//!
//! `Tape` records four more — `student_t_lccdf`, `erf`, `erfc` and `digamma` —
//! which the emitter has no instruction sequence for, so writing them here
//! would only move where the refusal happens.
//!
//! **Operands are instruction numbers, not node indices.** The tape numbers
//! equal expressions into one node, so the k-th instruction is not generally
//! the k-th node, and a front end that assumes otherwise produces a module that
//! runs and is wrong. Refer to instructions and this module keeps the mapping.
//!
//! **`dot_c` and `sum_run` name every element rather than a stride.** Both
//! [`Tape`] methods take a *node*-index stride, which a caller counting
//! instructions has no way to supply directly — value numbering can merge two
//! written instructions into one node, so instruction spacing and node spacing
//! are not the same thing. Instead the text names each element's own
//! instruction, and this module looks up the node behind each one and checks
//! that they land evenly spaced (an increasing, constant gap) before handing
//! `base`/`stride` to the tape; an uneven run — most often two elements that
//! turned out to be the same subexpression — is a parse error naming the gap
//! it found, not a wrong module. `dot_c` pairs each named instruction with its
//! own coefficient; `sum_run`'s `SEED` is the instruction its run is added to.
//!
//! The format is not an artifact: a tape is written and consumed in one go,
//! nothing outlives the call, and so nothing here is promised across versions.

use tapewasm_autodiff::Tape;
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
        let node_for = |tok: &str| -> Result<u32, TapeTextError> {
            let at: usize = tok
                .parse()
                .map_err(|_| bad(format!("`{tok}` is not an instruction number")))?;
            ids.get(at)
                .copied()
                .ok_or_else(|| bad(format!("instruction {at} has not been written yet")))
        };
        let idx = |k: usize| -> Result<u32, TapeTextError> { node_for(want(k)?) };
        let num = |k: usize| -> Result<f64, TapeTextError> {
            want(k)?
                .parse()
                .map_err(|_| bad(format!("`{}` is not a number", f[k])))
        };
        // Named elements' nodes to the `(base, stride)` a run needs; uneven spacing
        // usually means value numbering merged two of them (see the module doc).
        let evenly_spaced = |nodes: &[u32]| -> Result<(u32, u32), TapeTextError> {
            if nodes.len() == 1 {
                return Ok((nodes[0], 1));
            }
            let mut stride = None;
            for w in nodes.windows(2) {
                let gap = w[1].checked_sub(w[0]).ok_or_else(|| {
                    bad("the named instructions' nodes must come in increasing order".to_string())
                })?;
                if gap == 0 {
                    return Err(bad(
                        "two of the named instructions are the same node — value \
                         numbering must have merged them into one subexpression"
                            .to_string(),
                    ));
                }
                match stride {
                    None => stride = Some(gap),
                    Some(s) if s == gap => {}
                    Some(s) => {
                        return Err(bad(format!(
                            "the named instructions are not evenly spaced in the tape \
                             (a gap of {gap} where the run so far was {s} apart) — value \
                             numbering may have merged one of them with an earlier node"
                        )))
                    }
                }
            }
            Ok((nodes[0], stride.unwrap()))
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
            "tan" => tape.tan(idx(1)?),
            "asin" => tape.asin(idx(1)?),
            "acos" => tape.acos(idx(1)?),
            "atan" => tape.atan(idx(1)?),
            "lgamma" => tape.lgamma(idx(1)?),
            "phi" => tape.phi(idx(1)?),
            "pow" => tape.pow(idx(1)?, num(2)?),
            "add_c" => tape.add_c(idx(1)?, num(2)?),
            "sub_c" => tape.sub_c(idx(1)?, num(2)?),
            "rsub_c" => tape.rsub_c(idx(1)?, num(2)?),
            "mul_c" => tape.mul_c(idx(1)?, num(2)?),
            "div_c" => tape.div_c(idx(1)?, num(2)?),
            "rdiv_c" => tape.rdiv_c(idx(1)?, num(2)?),
            "dot_c" => {
                let len: usize = want(1)?
                    .parse()
                    .map_err(|_| bad("`dot_c` length is not a count".into()))?;
                if len == 0 {
                    return Err(bad("`dot_c` needs at least one element".into()));
                }
                if f.len() != 2 + 2 * len {
                    return Err(bad(format!(
                        "`dot_c` with length {len} wants {} operands ({len} instruction/\
                         coefficient pairs), found {}",
                        2 * len,
                        f.len() - 2
                    )));
                }
                let mut nodes = Vec::with_capacity(len);
                let mut coeffs = Vec::with_capacity(len);
                for c in 0..len {
                    nodes.push(node_for(f[2 + 2 * c])?);
                    let tok = f[3 + 2 * c];
                    coeffs.push(
                        tok.parse::<f64>()
                            .map_err(|_| bad(format!("`{tok}` is not a number")))?,
                    );
                }
                let (base, stride) = evenly_spaced(&nodes)?;
                tape.dot_c(base, stride, &coeffs)
            }
            "sum_run" => {
                let seed = idx(1)?;
                let len: usize = want(2)?
                    .parse()
                    .map_err(|_| bad("`sum_run` length is not a count".into()))?;
                if len == 0 {
                    return Err(bad("`sum_run` needs at least one element".into()));
                }
                if f.len() != 3 + len {
                    return Err(bad(format!(
                        "`sum_run` with length {len} wants {len} instruction operands, found {}",
                        f.len() - 3
                    )));
                }
                let nodes = (0..len)
                    .map(|c| node_for(f[3 + c]))
                    .collect::<Result<Vec<_>, _>>()?;
                let (base, stride) = evenly_spaced(&nodes)?;
                tape.sum_run(seed, base, stride, len as u32)
            }
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

    #[test]
    fn dot_c_names_each_element_and_matches_the_direct_api() {
        let mut p = parse(
            "n_params 3\nnew_var 2.0\nnew_var 3.0\nnew_var 4.0\n\
             dot_c 3 0 10.0 1 20.0 2 30.0\nroot 3",
        )
        .unwrap();
        // 2*10 + 3*20 + 4*30 = 200
        assert!((p.tape.value(p.root) - 200.0).abs() < 1e-12);
        p.tape.backward(p.root);
        assert_eq!(p.tape.grad_at(0), 10.0);
        assert_eq!(p.tape.grad_at(1), 20.0);
        assert_eq!(p.tape.grad_at(2), 30.0);
    }

    #[test]
    fn sum_run_names_each_element_and_matches_the_direct_api() {
        let mut p = parse(
            "n_params 4\nnew_var 1.0\nnew_var 2.0\nnew_var 3.0\nnew_var 4.0\n\
             sum_run 0 3 1 2 3\nroot 4",
        )
        .unwrap();
        // seed 1.0 + (2.0 + 3.0 + 4.0) = 10.0
        assert!((p.tape.value(p.root) - 10.0).abs() < 1e-12);
        p.tape.backward(p.root);
        assert_eq!(p.tape.grad_at(0), 1.0);
        assert_eq!(p.tape.grad_at(1), 1.0);
        assert_eq!(p.tape.grad_at(2), 1.0);
        assert_eq!(p.tape.grad_at(3), 1.0);
    }

    #[test]
    fn a_single_element_run_needs_no_stride() {
        let p = parse("n_params 1\nnew_var 5.0\ndot_c 1 0 3.0\nroot 1").unwrap();
        assert!((p.tape.value(p.root) - 15.0).abs() < 1e-12);
    }

    #[test]
    fn an_empty_run_is_a_parse_error() {
        let e = match parse("n_params 1\nnew_var 5.0\ndot_c 0\nroot 1") {
            Err(e) => e,
            Ok(_) => panic!("an empty dot_c parsed"),
        };
        assert!(e.to_string().contains("at least one element"), "{e}");
    }

    #[test]
    fn an_unevenly_spaced_run_is_a_parse_error_naming_the_gap() {
        // Instructions 0, 2, 5 — gaps of 2 then 3 — name nodes 0, 2, 5 since
        // none of these leaves are shared, so the run itself is the uneven one.
        let e = match parse(
            "n_params 6\nnew_var 1.0\nnew_var 2.0\nnew_var 3.0\nnew_var 4.0\nnew_var 5.0\n\
             new_var 6.0\ndot_c 3 0 1.0 2 1.0 5 1.0\nroot 6",
        ) {
            Err(e) => e,
            Ok(_) => panic!("an unevenly spaced dot_c parsed"),
        };
        assert!(e.to_string().contains("not evenly spaced"), "{e}");
    }

    #[test]
    fn value_numbering_merging_two_named_instructions_is_a_parse_error() {
        // `mul 0 1` twice is one node, so instructions 2 and 3 name it twice —
        // the case this format has to reject rather than emit a wrong stride for.
        let e = match parse(
            "n_params 2\nnew_var 0.5\nnew_var 2.0\nmul 0 1\nmul 0 1\n\
             dot_c 2 2 1.0 3 1.0\nroot 2",
        ) {
            Err(e) => e,
            Ok(_) => panic!("a dot_c over a merged node parsed"),
        };
        assert!(e.to_string().contains("same node"), "{e}");
    }

    #[test]
    fn a_decreasing_run_is_a_parse_error() {
        let e = match parse(
            "n_params 3\nnew_var 1.0\nnew_var 2.0\nnew_var 3.0\n\
             dot_c 2 2 1.0 0 1.0\nroot 2",
        ) {
            Err(e) => e,
            Ok(_) => panic!("a decreasing dot_c parsed"),
        };
        assert!(e.to_string().contains("increasing order"), "{e}");
    }
}
