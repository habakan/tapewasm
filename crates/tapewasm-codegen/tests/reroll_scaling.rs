//! Emitting a tape that re-rolls into many small blocks must not cost the
//! number of blocks times the length of the tape. Deciding whether a node can
//! live in a wasm local asks which block owns each of its arguments, and that
//! search used to scan every block — quadratic on a tape that fragments, which
//! is what left four posteriordb posteriors unable to compile in two minutes.
//!
//! The bound is wall clock, so it sits an order of magnitude above what the
//! linear version takes: enough to catch the quadratic return, not to measure.

use std::time::Instant;

use tapewasm_codegen::{compile_tape, shapes, Reroll};

#[test]
fn a_fragmented_tape_compiles_in_time_proportional_to_its_length() {
    let (tape, root) = shapes::scattered(3000, 200, 60, 5);
    let n_params = 200 * 5 + 5 * 60;

    let started = Instant::now();
    let compiled = compile_tape(&tape, n_params, root, Reroll::Auto).unwrap();
    let secs = started.elapsed().as_secs_f64();
    assert!(!compiled.wasm.is_empty());
    assert!(secs < 5.0, "compiled in {secs:.1}s");

    // And it has to actually re-roll: these blocks each want ten index tables,
    // and a lower `MAX_TABLED` leaves detection settling for fragments.
    let straight = compile_tape(&tape, n_params, root, Reroll::Never).unwrap();
    let ratio = straight.wasm.len() as f64 / compiled.wasm.len() as f64;
    assert!(
        ratio > 4.0,
        "re-rolled to only {ratio:.1}x under straight-line"
    );
}
