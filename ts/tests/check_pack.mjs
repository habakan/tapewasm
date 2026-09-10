// What `npm pack` would actually ship, asserted rather than assumed.
//
//   npm pack --dry-run --json > pack.json && node tests/check_pack.mjs pack.json
//
// Two failures a published version cannot be taken back from: `npm pack` collects
// only files under `ts/`, so the repo-root licences reach no tarball on their own,
// and `wasm-pack` writes a `.gitignore` of `*` into `ts/pkg/` that npm honours.

import { readFileSync } from "node:fs";

const path = process.argv[2];
if (!path) {
  console.error("usage: check_pack.mjs <npm pack --json output>");
  process.exit(2);
}

const files = JSON.parse(readFileSync(path, "utf8"))[0].files.map((f) => f.path);
const fail = (msg) => {
  console.error(`error: ${msg}:\n${files.join("\n")}`);
  process.exit(1);
};

for (const l of ["LICENSE-APACHE", "LICENSE-MIT"]) {
  if (!files.some((p) => p === l)) fail(`npm tarball ships no ${l}`);
}

const wasm = files.filter((p) => p.endsWith(".wasm"));
if (wasm.length !== 1) {
  fail(`expected exactly one .wasm in the npm tarball, got ${wasm.length}`);
}

console.log(`npm tarball ok: ${files.length} files, both licences, ${wasm[0]}`);
