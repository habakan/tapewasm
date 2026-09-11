// Tape text for the shapes `bench.mjs` times, sized by their data. Deterministic,
// so two runs time the same tapes.

function writer() {
  const lines = [];
  let next = 0;
  return { lines, op: (text) => (lines.push(text), next++) };
}

const x = (i) => 2 * Math.sin(i * 0.7);
const y = (i) => 0.5 + 1.3 * x(i) + 0.2 * Math.cos(i * 1.3);

/** `y ~ normal(alpha + beta * x, sigma)` elementwise: one block per point. */
export function linreg(n) {
  const { lines, op } = writer();
  lines.push("n_params 3");
  const alpha = op("new_var 0.1");
  const beta = op("new_var 0.2");
  const logSigma = op("new_var 0.0");
  const inv = op(`rdiv_c ${op(`exp ${logSigma}`)} 1.0`);
  let acc = op(`mul_c ${logSigma} ${-n}`);
  for (let i = 0; i < n; i++) {
    const mu = op(`add ${alpha} ${op(`mul_c ${beta} ${x(i)}`)}`);
    const z = op(`mul ${op(`rsub_c ${mu} ${y(i)}`)} ${inv}`);
    acc = op(`add ${acc} ${op(`mul_c ${op(`mul ${z} ${z}`)} -0.5`)}`);
  }
  lines.push(`root ${acc}`);
  return lines.join("\n");
}

/** `y ~ normal(X beta, 1)` with `X` constant: a `dot_c` per row, or the same sum elementwise. */
export function matvec(n, k, contraction) {
  const { lines, op } = writer();
  lines.push(`n_params ${k}`);
  const beta = Array.from({ length: k }, (_, j) => op(`new_var ${0.1 * (j + 1)}`));
  const X = (i, j) => Math.sin(i * 0.37 + j * 1.1);
  let acc = null;
  for (let i = 0; i < n; i++) {
    let mu;
    if (contraction) {
      mu = op(`dot_c ${k} ${beta.map((b, j) => `${b} ${X(i, j)}`).join(" ")}`);
    } else {
      mu = op(`mul_c ${beta[0]} ${X(i, 0)}`);
      for (let j = 1; j < k; j++) mu = op(`add ${mu} ${op(`mul_c ${beta[j]} ${X(i, j)}`)}`);
    }
    const r = op(`rsub_c ${mu} ${y(i)}`);
    const term = op(`mul_c ${op(`mul ${r} ${r}`)} -0.5`);
    acc = acc === null ? term : op(`add ${acc} ${term}`);
  }
  lines.push(`root ${acc}`);
  return lines.join("\n");
}
