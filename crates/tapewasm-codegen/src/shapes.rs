//! Tapes with the shapes the emitter cares about, built directly.
//!
//! Re-rolling looks for repeated runs of opcodes, so what it does depends on
//! the shape of the tape and not on what recorded it. These build the three
//! shapes the loop emitter has separate paths for — an element-wise run, an
//! irregular gather, and a long per-observation block — so the tests and
//! examples here need no model language to produce one.
//!
//! Values are arbitrary but distinct: the tape numbers equal expressions into
//! one node, and a repeated constant would break the very period being tested.

use tapewasm_autodiff::Tape;

/// A deterministic stream, so a failing test is the same run every time.
fn rnd(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64
}

/// `y ~ normal(alpha + beta * x, sigma)` over `n` points.
///
/// Two element-wise runs for the mean, then one block per point for the
/// density. Parameters are `alpha`, `beta` and `log sigma`.
pub fn linreg(n: usize) -> (Tape, u32) {
    let mut tape = Tape::new();
    let alpha = tape.new_var(0.1);
    let beta = tape.new_var(0.9);
    let log_sigma = tape.new_var(0.2);

    let sigma = tape.exp(log_sigma);
    let inv_sigma = tape.rdiv_c(sigma, 1.0);
    let a2 = tape.mul(alpha, alpha);
    let b2 = tape.mul(beta, beta);
    let priors = tape.add(a2, b2);
    let priors = tape.mul_c(priors, -0.005);
    let mut acc = tape.sub(priors, sigma);

    let mus: Vec<u32> = (0..n).map(|i| tape.mul_c(beta, i as f64 * 0.1)).collect();
    let mus: Vec<u32> = mus.into_iter().map(|m| tape.add(alpha, m)).collect();
    for (i, mu) in mus.into_iter().enumerate() {
        let d = tape.sub_c(mu, 1.0 + i as f64 * 0.2);
        let z = tape.mul(d, inv_sigma);
        let z2 = tape.mul(z, z);
        acc = tape.sub(acc, z2);
    }
    let root = tape.sub(acc, log_sigma);
    (tape, root)
}

/// `y[i] ~ normal(mu[g[i]], sigma)` with `g` irregular, over `n` points.
///
/// `mu[g[i]]` is the read no stride describes, so the emitter has to table the
/// slot index. Parameters are eight group means and `log sigma`.
pub fn gather(n: usize) -> (Tape, u32) {
    const G: usize = 8;
    let mut tape = Tape::new();
    let mu: Vec<u32> = (0..G)
        .map(|i| tape.new_var(0.1 + i as f64 * 0.05))
        .collect();
    let log_sigma = tape.new_var(0.5);

    let sigma = tape.exp(log_sigma);
    let inv_sigma = tape.rdiv_c(sigma, 1.0);
    let mut acc = tape.neg(sigma);
    for &m in &mu {
        let m2 = tape.mul(m, m);
        acc = tape.sub(acc, m2);
    }

    let mut seed = 12345_u64;
    for i in 0..n {
        let g = (rnd(&mut seed) * G as f64) as usize;
        let d = tape.sub_c(mu[g.min(G - 1)], (i as f64).sin() * 2.0);
        let z = tape.mul(d, inv_sigma);
        let z2 = tape.mul(z, z);
        acc = tape.sub(acc, z2);
    }
    let root = tape.sub(acc, log_sigma);
    (tape, root)
}

/// `y[n] ~ multi_normal_cholesky(mu, L)` over `n` observations in `k`
/// dimensions: forward substitution per observation, so the block is roughly
/// `k²` long and `MAX_BLOCK` has to reach past it.
///
/// Parameters are `mu` then `L` in row-major lower-triangular order, the
/// diagonal held positive by an exponential.
pub fn mvn_cholesky(n: usize, k: usize) -> (Tape, u32) {
    let mut tape = Tape::new();
    let mu: Vec<u32> = (0..k)
        .map(|i| tape.new_var(0.1 + i as f64 * 0.01))
        .collect();
    let mut l = vec![vec![0_u32; k]; k];
    for (i, row) in l.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().take(i + 1).enumerate() {
            *cell = tape.new_var(if i == j { 0.2 } else { 0.05 * (i + j) as f64 });
        }
    }
    for (i, row) in l.iter_mut().enumerate() {
        row[i] = tape.exp(row[i]);
    }

    let mut acc = tape.new_var(0.0);
    let mut seed = 987654321_u64;
    for _ in 0..n {
        let y: Vec<f64> = (0..k).map(|_| rnd(&mut seed) * 4.0 - 2.0).collect();
        let mut z: Vec<u32> = Vec::with_capacity(k);
        for i in 0..k {
            let mut s = tape.rsub_c(mu[i], y[i]);
            for (j, &zj) in z.iter().enumerate() {
                let t = tape.mul(l[i][j], zj);
                s = tape.sub(s, t);
            }
            let zi = tape.div(s, l[i][i]);
            let z2 = tape.mul(zi, zi);
            acc = tape.sub(acc, z2);
            z.push(zi);
        }
    }
    for (i, row) in l.iter().enumerate() {
        let ld = tape.log(row[i]);
        acc = tape.sub(acc, ld);
    }
    (tape, acc)
}

/// `y ~ normal(X * beta, sigma)` with `X` constant: one contraction node per
/// row, whose result the density block reads back.
///
/// That crossing is the only place the scratch buffer's slot order is
/// observable, and the contraction has two emitters — unrolled in place, or
/// inside a loop reading a staged column of coefficients.
pub fn matvec(n: usize, k: usize) -> (Tape, u32) {
    let mut tape = Tape::new();
    let beta: Vec<u32> = (0..k)
        .map(|i| tape.new_var(0.1 + i as f64 * 0.02))
        .collect();
    let log_sigma = tape.new_var(0.1);

    let sigma = tape.exp(log_sigma);
    let inv_sigma = tape.rdiv_c(sigma, 1.0);
    let mut acc = tape.neg(sigma);
    for &b in &beta {
        let b2 = tape.mul(b, b);
        acc = tape.sub(acc, b2);
    }

    let mut seed = 6789_u64;
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..k).map(|_| rnd(&mut seed) * 2.0 - 1.0).collect())
        .collect();
    let mus: Vec<u32> = rows.iter().map(|r| tape.dot_c(beta[0], 1, r)).collect();
    for mu in mus {
        let d = tape.sub_c(mu, rnd(&mut seed) * 2.0 - 1.0);
        let z = tape.mul(d, inv_sigma);
        let z2 = tape.mul(z, z);
        acc = tape.sub(acc, z2);
    }
    let root = tape.sub(acc, log_sigma);
    (tape, root)
}

/// The shape that fragments: two scattered gathers per term, so most candidate
/// blocks want more index tables than the emitter allows and detection settles
/// for small ones.
///
/// `docs` groups of `terms` weights and `terms` groups of `vocab`, mixed per
/// item the way a topic model mixes them. Deciding whether a node can live in a
/// wasm local asks which block owns each argument, and on a tape like this
/// there are thousands of blocks to ask.
pub fn scattered(n: usize, docs: usize, vocab: usize, terms: usize) -> (Tape, u32) {
    let mut tape = Tape::new();
    let theta: Vec<u32> = (0..docs * terms)
        .map(|i| tape.new_var(0.2 + (i % 7) as f64 * 0.01))
        .collect();
    let phi: Vec<u32> = (0..terms * vocab)
        .map(|i| tape.new_var(0.02 + (i % 11) as f64 * 0.001))
        .collect();

    let log_theta: Vec<u32> = theta.iter().map(|&t| tape.log(t)).collect();
    let log_phi: Vec<u32> = phi.iter().map(|&p| tape.log(p)).collect();

    let mut seed = 12345_u64;
    let mut acc = tape.new_var(0.0);
    for _ in 0..n {
        let d = (rnd(&mut seed) * docs as f64) as usize % docs;
        let w = (rnd(&mut seed) * vocab as f64) as usize % vocab;
        let gs: Vec<u32> = (0..terms)
            .map(|k| {
                let g = tape.add(log_theta[d * terms + k], log_phi[k * vocab + w]);
                tape.exp(g)
            })
            .collect();
        let mut s = gs[0];
        for &g in &gs[1..] {
            s = tape.add(s, g);
        }
        let l = tape.log(s);
        acc = tape.add(acc, l);
    }
    (tape, acc)
}
