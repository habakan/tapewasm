# Task runner for tapewasm.
#
# Thin wrappers over cargo / wasm-pack / node, so that what CONTRIBUTING.md and
# CI would each spell out in prose lives in one discoverable place and the
# thing you run locally is literally the thing CI runs.
#
# Written for GNU Make 3.81 — the version macOS still ships.

.PHONY: help
help:
	@grep -hE '^[a-z][a-z-]*:.*##' $(MAKEFILE_LIST) \
	  | sed 's/:[^#]*##/|/' \
	  | awk -F'|' '{ printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 }'

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

TESTFLAGS ?=

.PHONY: fmt
fmt: ## Format all Rust code
	cargo fmt --all

.PHONY: fmt-check
fmt-check: ## Fail if anything is unformatted (what CI runs)
	cargo fmt --all -- --check

.PHONY: clippy
clippy: ## Lint the workspace with warnings denied
	cargo clippy --workspace --all-targets -- -D warnings
	cargo clippy -p tapewasm --no-default-features --all-targets -- -D warnings

.PHONY: test
test: ## Native test suite, all crates
	cargo test --workspace $(TESTFLAGS)

.PHONY: check
check: fmt-check clippy test ## fmt-check + clippy + test

# The bundle is a real file, not a phony target, so `make browser-test` after an
# unrelated edit does not pay for a wasm-pack rebuild.
WASM_OUT := ts/pkg/tapewasm_bg.wasm
WASM_SRC := $(shell find crates -name '*.rs') Cargo.toml Cargo.lock

.PHONY: wasm
wasm: $(WASM_OUT) ## Build the wasm bundle into ts/pkg/

$(WASM_OUT): $(WASM_SRC)
	@command -v wasm-pack >/dev/null 2>&1 \
	  || { echo "error: wasm-pack not found. Install with: cargo install wasm-pack" >&2; exit 1; }
# wasm-pack writes into the out-dir without clearing it, so anything it no
# longer emits just stays — and `"files": ["pkg/"]` would ship it.
	rm -rf $(ROOT)/ts/pkg
	wasm-pack build crates/tapewasm --target web --out-dir $(ROOT)/ts/pkg --release
# wasm-pack drops its own `.gitignore` (just `*`) into ts/pkg/. npm honors a
# nested .gitignore with no matching .npmignore, so leaving it publishes an
# empty pkg/ despite `files` saying to include it.
	@rm -f $(ROOT)/ts/pkg/.gitignore
	@printf '  %-12s %8s bytes  %8s gzip\n' "full" \
	  "$$(wc -c < $(WASM_OUT) | tr -d ' ')" \
	  "$$(gzip -c $(WASM_OUT) | wc -c | tr -d ' ')"

# The same crate without its emitter: `AotSampler` and the sampler, for a page
# that ships a module compiled when the page was written. Separate out-dir so it
# never overwrites the bundle the npm package publishes.
SAMPLER_OUT := ts/pkg-sampler/tapewasm_bg.wasm

.PHONY: wasm-sampler
wasm-sampler: $(SAMPLER_OUT) ## Build the sampler-only bundle into ts/pkg-sampler/

$(SAMPLER_OUT): $(WASM_SRC)
	@command -v wasm-pack >/dev/null 2>&1 \
	  || { echo "error: wasm-pack not found. Install with: cargo install wasm-pack" >&2; exit 1; }
	rm -rf $(ROOT)/ts/pkg-sampler
	wasm-pack build crates/tapewasm --target web --out-dir $(ROOT)/ts/pkg-sampler \
	  --release -- --no-default-features
	@rm -f $(ROOT)/ts/pkg-sampler/.gitignore
# The point of this bundle is its size, so say it rather than leaving it to be
# discovered.
	@printf '  %-12s %8s bytes  %8s gzip\n' "sampler" \
	  "$$(wc -c < $(SAMPLER_OUT) | tr -d ' ')" \
	  "$$(gzip -c $(SAMPLER_OUT) | wc -c | tr -d ' ')"

.PHONY: browser-test
browser-test: wasm ## Run a compiled module in Chromium, Firefox and WebKit
	cd browser-tests && npm test

.PHONY: package
package: wasm ## Dry-run packaging every crate + the npm tarball
	cargo package --workspace --no-verify
# Both licence texts have to travel with the artifact — the offer is either one,
# so shipping half of it is not the offer — and `cargo package` and `npm pack`
# each collect only files inside their own directory, so the copies at the repo
# root reach no tarball. The npm tarball has a second invisible failure:
# `wasm-pack` writes its own `.gitignore` (containing `*`) into `ts/pkg/`, which
# npm honours when no `.npmignore` sits beside it, and that ships a package
# carrying no wasm at all. A published version cannot be taken back, so all of
# this is asserted rather than assumed.
	@list=$$(mktemp); \
	for f in target/package/*.crate; do \
	  tar tzf "$$f" > "$$list"; \
	  for l in LICENSE-APACHE LICENSE-MIT; do \
	    grep -q "/$$l\$$" "$$list" \
	      || { echo "error: $$f ships no $$l" >&2; rm -f "$$list"; exit 1; }; \
	  done; \
	done; \
	rm -f "$$list"
	@echo "both licences present in every .crate"
# A script rather than a `node -e`: make 4.3 and make 3.81 disagree about a
# backslash-continued line inside a recipe, and the older one is what macOS
# ships, so the inline form passes locally and never runs on CI.
	@pack=$$(mktemp); \
	(cd ts && npm pack --dry-run --json) > "$$pack" \
	  && node ts/tests/check_pack.mjs "$$pack"; \
	status=$$?; rm -f "$$pack"; exit $$status

.PHONY: clean
clean: ## Remove build output
	cargo clean
	rm -rf $(ROOT)/ts/pkg $(ROOT)/ts/pkg-sampler $(ROOT)/browser-tests/fixtures
