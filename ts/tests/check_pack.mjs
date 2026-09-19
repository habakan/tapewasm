// What `npm pack` would actually ship, asserted rather than assumed — by
// packing it, unpacking it somewhere else, and importing what comes out.
//
//   node ts/tests/check_pack.mjs
//
// Three failures a published version cannot be taken back from. `npm pack`
// collects only files under `ts/`, so the repo-root licences reach no tarball
// on their own. `wasm-pack` writes a `.gitignore` of `*` into `ts/pkg/` that
// npm honours, which ships a package carrying no wasm at all. And `files` names
// what travels one entry at a time, so a module the entry point imports can be
// left behind — `calibrate.js` was, and `tapewasm@0.3.2` cannot be imported at
// all because of it. The list said nothing; importing the tarball says it in
// one line.

import { execFileSync } from "node:child_process";
import { mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ts = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const out = mkdtempSync(resolve(tmpdir(), "tapewasm-pack-"));
const fail = (msg) => {
  console.error(`error: ${msg}`);
  rmSync(out, { recursive: true, force: true });
  process.exit(1);
};

try {
  const packed = JSON.parse(
    execFileSync("npm", ["pack", "--json", "--pack-destination", out], { cwd: ts }),
  )[0];
  const files = packed.files.map((f) => f.path);

  for (const l of ["LICENSE-APACHE", "LICENSE-MIT"]) {
    if (!files.includes(l)) fail(`npm tarball ships no ${l}:\n${files.join("\n")}`);
  }
  const wasm = files.filter((p) => p.endsWith(".wasm"));
  if (wasm.length !== 1) {
    fail(`expected exactly one .wasm in the npm tarball, got ${wasm.length}`);
  }

  // The tarball unpacks to `package/`, and importing it resolves every path the
  // entry point names — which no list of file names can check.
  const tgz = readdirSync(out).find((f) => f.endsWith(".tgz"));
  execFileSync("tar", ["xzf", resolve(out, tgz), "-C", out]);
  await import(pathToFileURL(resolve(out, "package", "index.js")).href);

  console.log(`npm tarball ok: ${files.length} files, both licences, ${wasm[0]}, imports`);
} catch (e) {
  fail(String(e.message ?? e).split("\n").slice(0, 3).join("\n"));
} finally {
  rmSync(out, { recursive: true, force: true });
}
