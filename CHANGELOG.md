# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First release. The tape, the emitter and the browser API were extracted from
[stanwasm](https://github.com/habakan/stanwasm), whose git history they keep.

### Changed

What changed in the move, for anyone porting a host:

- The emitted module's globals are `tapewasm_layout_id` and
  `tapewasm_abi_version`, and it imports shared memory as `tapewasm.memory`
  rather than `stan.memory`. A module built before the rename exports neither
  name, so binding one is refused rather than mixed.
- `compile(&Model, ..)` is gone. `compile_tape` is the entry point; a front end
  that has a model language lowers it to a tape first.
- `tapewasm_codegen::shapes` builds the tapes the tests and examples run on, so
  neither needs a model language.
