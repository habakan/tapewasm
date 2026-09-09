// Static files for the page: the fixtures a build step wrote, and ts/pkg/
// beside them. No framework — a page embedding a precompiled model needs none.

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, dirname, extname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..");
const types = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".wasm": "application/wasm", ".json": "application/json",
};

createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  const path = url.pathname === "/" ? "/page.html" : url.pathname;
  const root = path.startsWith("/pkg/") ? resolve(repo, "ts") : here;
  try {
    const body = await readFile(resolve(root, "." + path));
    res.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
}).listen(8123, "127.0.0.1");
