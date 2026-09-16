# Releasing

Maintainer notes. `.github/workflows/release.yml` runs on the tag: it verifies
the tagged tree and creates the GitHub release. Nothing there publishes — both
registries are published by hand from a local checkout, and no job holds a
credential for either.

**A published version is permanent.** crates.io can yank and npm can
deprecate, but neither frees the version number or removes the code. Every
check below exists because something here is not reversible.

## 1. Set the version

It appears in three places, and nothing keeps them in sync automatically:

| File | Field |
|---|---|
| `Cargo.toml` | `workspace.package.version` |
| `Cargo.toml` | `version = "…"` on both internal deps under `[workspace.dependencies]` |
| `ts/package.json` | `version` |

Both `LICENSE-APACHE` and `LICENSE-MIT` have to sit beside every crate manifest
and in `ts/`; `make package` fails if either is missing from a tarball.

Cargo does not accept `version.workspace` inside `[workspace.dependencies]`,
which is why the two requirements are written out by hand.

Then move `CHANGELOG.md`'s `[Unreleased]` section under a `## [X.Y.Z] — DATE`
heading. The date has to be the day you actually tag: the GitHub release body
is extracted from this section by heading match, and the `guard` job fails the
tag if no section matches.

## 2. Check locally

```bash
make check TESTFLAGS=--release   # fmt + clippy + the release test suite
make browser-test                # a compiled module in three engines
make package                     # what both registries would actually receive
```

`make package` runs `cargo package --workspace`, not a per-crate loop. That
matters before the first release: `cargo package -p tapewasm-codegen` on its
own resolves `tapewasm-autodiff` from the crates.io index and fails until
0.1.0 is really published there, while the workspace form resolves siblings
locally and can check all three manifests today.

It then asserts two things that are invisible until someone installs the
result. That the Apache-2.0 licence text is inside every artifact — `cargo
package` and `npm pack` each collect only files under their own directory, so
the repo-root `LICENSE` reaches no tarball on its own. And that `pkg/` carries
the wasm: `wasm-pack` writes its own `.gitignore` (containing `*`) into
`ts/pkg/`, and npm honours a nested `.gitignore` when no `.npmignore` sits
beside it, which publishes a package whose `pkg/` is empty — no wasm, no glue
JS — despite `package.json`'s `files` saying to include it. The `wasm` target
deletes that file after every build; this check is what catches it coming back.

## 3. Tag, and let CI check the tree

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Nothing becomes public from this. A tag that fails `guard` or `verify` can be
deleted with `git push --delete origin vX.Y.Z`, and nothing is staged, because
`stage-npm` waits on both. Once they pass, the tag does spend the npm version
number — `npm stage reject` is what frees it again.

Check that `release.yml` is enabled before tagging (`gh workflow list --all`).
A disabled workflow does not fail on its trigger — the tag lands and nothing
runs at all.

## 4. Approve what CI staged

Pushing the tag stages the release: `release.yml`'s `stage-npm` job builds the
bundle from the tagged tree and runs `npm stage publish`. Staging needs no 2FA,
which is why it can run unattended — and nothing is public until the maintainer
approves with 2FA. Needs npm 11.15 or later, which is why the job installs its
own npm rather than using the one Node ships.

The job holds no stored token. It exchanges GitHub's OIDC identity for a
short-lived npm one, so npm has to know which workflow is allowed to speak for
the package: on npmjs.com, **Settings → Trusted publisher → GitHub Actions**,
with this repository and the workflow file `release.yml`. Without that the job
fails with `ENEEDAUTH`, having published nothing.

Building in CI also gets the release something a laptop cannot produce: npm
enables provenance automatically over a trusted-publisher token, so the tarball
carries an attestation tying it to this commit and this workflow run.

Then check what was staged and approve it:

```bash
npm stage list tapewasm
npm stage view <stage-id>       # version, file list, integrity
npm stage approve <stage-id>    # 2FA; `npm stage reject <stage-id>` drops it instead
```

**A staged version already holds its number** — the registry refuses a
publish of a version that is staged. Approval waits on the registry's malware
scan, and the version shows up a few minutes after; confirm with
`npm view tapewasm version --prefer-online`.

To stage by hand instead — the job is broken, or the tag is already spent — the
steps it replaced still work from a clean checkout of the tag, where `make wasm`
bakes the working tree into the bundle:

```bash
git status --short          # must be clean
git rev-parse HEAD          # must be the tagged commit
make wasm
cd ts && npm stage publish --access public
```

Pass no `--provenance` there: a tarball staged from a laptop cannot carry an
attestation, and the flag fails rather than being ignored.

## 5. Publish to crates.io

**Blocked as of 0.1.0.** The workspace `[patch.crates-io]` takes nuts-rs from
their main branch for a wasm fix that no published version carries, and a patch
does not travel into a published crate — so a crate published today builds, for
anyone who depends on it, a module WebKit refuses. `cargo publish` succeeds at
this, and a version cannot be taken back. Wait for a nuts-rs release, drop the
patch, and let `cargo test -p tapewasm-codegen --test no_wasm_gc` confirm it.
Skip this step until then and mark the CHANGELOG heading "(npm only)" with the
reason.

Strictly in this order. Each manifest resolves the ones before it from the
registry rather than from its path, so a crate cannot go up before its
dependencies:

```bash
cargo publish -p tapewasm-autodiff
cargo publish -p tapewasm-codegen
cargo publish -p tapewasm
```

Between each, wait for the index to catch up — the next `cargo publish` cannot
resolve a dependency the registry has not indexed yet.

Downstream, [stanwasm](https://github.com/habakan/stanwasm) pins these by git
rev until they are on crates.io, so a release here means updating that rev
there.
