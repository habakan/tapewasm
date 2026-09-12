# tapewasm

Compile a log density to a WebAssembly module and draw from it in the browser.
No server does the sampling.

You hand over an expression as a tape — a flat list of arithmetic operations —
and get back a wasm module that computes the value and every partial derivative
in one call, plus a sampler ([nuts-rs](https://github.com/pymc-devs/nuts-rs))
that runs it. The two share linear memory, so a draw costs one call across the
boundary and no copying.

Nothing here knows what a model is. What writes the tape stays outside, and has
been, so far, a Stan parser and a PyTensor graph walker.

## What a front end does

```js
import init, { compileTape, AotSampler, setAotExports, sharedMemory } from "tapewasm";

await init();

// The tape: one instruction per line, operands naming earlier instructions.
const built = compileTape(`
n_params 2
new_var 0.1      # 0: mu
new_var 0.1      # 1: log sigma
exp 1            # 2: sigma
rdiv_c 2 1.0     # 3: 1 / sigma
rsub_c 0 1.7     # 4: y - mu, for one observation
mul 4 3          # 5
mul 5 5          # 6
mul_c 6 -0.5     # 7
sub 7 2          # 8: an exponential(1) prior on sigma
root 8
`);

const module = await WebAssembly.instantiate(built.wasm, {
  tapewasm: { memory: sharedMemory() },
  Math,                     // plus lgamma, digamma and phi if the tape reaches them
});
setAotExports(module.instance.exports);

const sampler = new AotSampler(
  built.nParams, built.scratchInit, built.layoutId, ["mu", "log_sigma"],
);
const draws = sampler.sample(new Float64Array([0.1, 0.1]), 1000, 1000, 42n);
```

`draws` is draws-major and `nParams` wide, warmup first.

`browser-tests/prepare.mjs` writes a linear regression this way and checks the
result in Chromium, Firefox and WebKit.

## Two shapes

The emitter and the sampler are separate, so a page can carry either or both.

| | **compile in the page** | **compile beforehand** |
| --- | --- | --- |
| the page loads | the full bundle, 368 KB | the sampler alone, 176 KB, and a module |
| the model | anything, recompiled live | fixed when the page was built |
| build with | `make wasm` | `make wasm-sampler` |

A module compiled beforehand is 5–35 KB for the models tried so far. It carries
its data with it: the numbers are constants on the tape, so a module answers for
one model and one dataset.

## What is in here

```
crates/tapewasm-autodiff/   the tape, and its own reverse pass
crates/tapewasm-codegen/    the emitter: tape in, wasm module out
crates/tapewasm/            the browser API over both, and the sampler
browser-tests/              a compiled module run in three engines
```

`tapewasm_codegen::tape_text` documents the text format. It is not an artifact
format: a tape is written and consumed inside one call, and nothing here is
promised across versions.

## Front ends

- [stanwasm](https://github.com/habakan/stanwasm) — models written in a subset
  of the Stan language, parsed and traced in the browser.
- [pymcwasm](https://github.com/habakan/pymcwasm) — a PyMC model's log density,
  lowered from its PyTensor graph.

Both reach this same emitter and this same sampler.

## Is it right?

`cargo test` checks each emitted module against the tape's own reverse pass,
which computes the same derivatives by a wholly different route — interpreted,
node by node — so the two agreeing is evidence about the emitter rather than
about one shared implementation. The shapes tested are an element-wise run, an
irregular gather, a long per-observation block and a contraction, which are the
cases the loop emitter has separate paths for.

`tapewasm-autodiff`'s own tests hold `lgamma`, `digamma` and the normal CDF to
1e-14 against references computed at 60 decimal digits.

## Building

```
make check          # fmt, clippy, tests
make wasm           # the bundle, into ts/pkg/
make browser-test   # a compiled module in Chromium, Firefox and WebKit
```

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT),
at your option. Unless you state otherwise, any contribution you submit shall be
dual licensed as above, with no additional terms.
