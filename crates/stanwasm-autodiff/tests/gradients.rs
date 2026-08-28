//! Verify forward + backward against analytical derivatives.

use stanwasm_autodiff::{log_prob_grad, Tape};

fn close(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

#[test]
fn linear_combination() {
    // f(x, y) = 2x + 3y at (4, 5) → 23, grad (2, 3)
    let (v, g) = log_prob_grad(&[4.0, 5.0], |t, xs| {
        let two_x = t.mul_c(xs[0], 2.0);
        let three_y = t.mul_c(xs[1], 3.0);
        t.add(two_x, three_y)
    });
    assert!(close(v, 23.0, 1e-12));
    assert!(close(g[0], 2.0, 1e-12));
    assert!(close(g[1], 3.0, 1e-12));
}

#[test]
fn product_rule() {
    // f(x, y) = x * y at (3, 4) → 12, grad (4, 3)
    let (v, g) = log_prob_grad(&[3.0, 4.0], |t, xs| t.mul(xs[0], xs[1]));
    assert!(close(v, 12.0, 1e-12));
    assert_eq!(g, vec![4.0, 3.0]);
}

#[test]
fn quotient_rule() {
    // f(x, y) = x / y at (6, 2) → 3, df/dx = 1/y = 0.5, df/dy = -x/y² = -1.5
    let (v, g) = log_prob_grad(&[6.0, 2.0], |t, xs| t.div(xs[0], xs[1]));
    assert!(close(v, 3.0, 1e-12));
    assert!(close(g[0], 0.5, 1e-12));
    assert!(close(g[1], -1.5, 1e-12));
}

#[test]
fn exp_log_chain() {
    // f(x) = log(exp(x) + 1) at x=2 → log(e^2+1)≈2.1269; df/dx = e^x/(e^x+1)
    let x0 = 2.0_f64;
    let expected = (x0.exp() + 1.0).ln();
    let expected_g = x0.exp() / (x0.exp() + 1.0);
    let (v, g) = log_prob_grad(&[x0], |t, xs| {
        let e = t.exp(xs[0]);
        let s = t.add_c(e, 1.0);
        t.log(s)
    });
    assert!(close(v, expected, 1e-12));
    assert!(close(g[0], expected_g, 1e-12));
}

#[test]
fn pow_rule() {
    // f(x) = x^3 at x=2 → 8, grad = 3x² = 12
    let (v, g) = log_prob_grad(&[2.0], |t, xs| t.pow(xs[0], 3.0));
    assert!(close(v, 8.0, 1e-12));
    assert!(close(g[0], 12.0, 1e-12));
}

#[test]
fn negative_log_likelihood_normal() {
    // Negative log-likelihood of N(x | mu, sigma) for one observation.
    // -0.5 * ((x - mu)/sigma)^2 - log(sigma)  (drop constants)
    // Verify gradient wrt mu and sigma at x=2, mu=0, sigma=1
    //   d/dmu = (x-mu)/sigma² = 2
    //   d/dsigma = (x-mu)²/sigma³ - 1/sigma = 4 - 1 = 3
    let (v, g) = log_prob_grad(&[0.0, 1.0], |t, xs| {
        let mu = xs[0];
        let sigma = xs[1];
        let x = t.new_var(2.0);
        let diff = t.sub(x, mu);
        let z = t.div(diff, sigma);
        let z2 = t.mul(z, z);
        let half_z2 = t.mul_c(z2, -0.5);
        let log_sigma = t.log(sigma);
        t.sub(half_z2, log_sigma)
    });
    let expected = -0.5 * 4.0 - 0.0;
    assert!(close(v, expected, 1e-12));
    assert!(close(g[0], 2.0, 1e-12));
    assert!(close(g[1], 3.0, 1e-12));
}

#[test]
fn cse_dedups_log() {
    // Two log(x) calls on the same input should reuse the same tape node.
    let mut tape = Tape::new();
    let x = tape.new_var(2.0);
    let len_before = tape.len();
    let a = tape.log(x);
    let mid = tape.len();
    let b = tape.log(x);
    let after = tape.len();
    assert_eq!(a, b);
    assert_eq!(after - mid, 0, "second log should hit cache");
    assert_eq!(mid - len_before, 1);
}

#[test]
fn cse_invalidates_on_reset() {
    let mut tape = Tape::new();
    let x = tape.new_var(2.0);
    let _ = tape.log(x);
    tape.reset();
    let y = tape.new_var(2.0);
    let len_before = tape.len();
    let _ = tape.log(y);
    assert_eq!(
        tape.len() - len_before,
        1,
        "after reset, log must recompute"
    );
}

#[test]
fn forward_replay_matches_fresh_trace() {
    // Build a trace with known params, then replay with different params and
    // verify the result matches what a fresh trace would produce.
    use stanwasm_autodiff::Tape;
    let mut tape = Tape::new();
    let x = tape.new_var(2.0);
    let y = tape.new_var(3.0);
    let xy = tape.mul(x, y);
    let exp_xy = tape.exp(xy);
    let plus = tape.add_c(exp_xy, 1.0);
    let root = tape.log(plus);

    // Replay with new values
    tape.forward_replay(&[1.5, 2.0]);
    tape.reset_grads();
    tape.backward(root);
    let lp_replay = tape.value(root);
    let g_replay: Vec<f64> = (0..2).map(|i| tape.grad_at(i)).collect();

    // Fresh trace for comparison
    let (lp_fresh, g_fresh) = log_prob_grad(&[1.5, 2.0], |t, xs| {
        let xy = t.mul(xs[0], xs[1]);
        let exp_xy = t.exp(xy);
        let plus = t.add_c(exp_xy, 1.0);
        t.log(plus)
    });

    assert!(
        (lp_replay - lp_fresh).abs() < 1e-12,
        "{lp_replay} vs {lp_fresh}"
    );
    for i in 0..2 {
        assert!(
            (g_replay[i] - g_fresh[i]).abs() < 1e-12,
            "grad[{i}]: {} vs {}",
            g_replay[i],
            g_fresh[i]
        );
    }
}

#[test]
fn special_functions_known_points() {
    use stanwasm_autodiff::{digamma, lgamma, phi_cdf};
    // lgamma(1) = 0, lgamma(2) = 0
    assert!(close(lgamma(1.0), 0.0, 1e-9));
    assert!(close(lgamma(2.0), 0.0, 1e-9));
    // digamma(1) = -γ ≈ -0.5772156649
    assert!(close(digamma(1.0), -0.577_215_664_9, 1e-6));
    // Phi(0) = 0.5
    assert!(close(phi_cdf(0.0), 0.5, 1e-7));
    // Phi(1.96) ≈ 0.975
    assert!(close(phi_cdf(1.96), 0.975, 5e-4));
}
