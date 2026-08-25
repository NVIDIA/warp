<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Classification Helpers

This reference is loaded during Phases 3, 4, and 5 of the skill. It defines
reusable classification rules for commits and CHANGELOG entries.

## Compatibility and risk record

Record these fields before deciding how prominently to report a change:

| Field | Question |
|---|---|
| Surface | What documented API, behavior, format, or platform changed? |
| Reachability | How does a user reach it through a supported entry point? |
| Deprecation | Is it stable, experimental, scheduled for removal, or unsupported? |
| Action | Dismiss, document, label planned/experimental, or raise an alarm? |

Use the repository's compatibility and deprecation policies as the authority.
Source visibility, a header declaration, or a downstream-looking name does not
create a support promise. Keep a compatibility fact separate from its release
risk: a call can stop compiling while still being a planned removal whose
documented window was satisfied.

## Public API surface (Phase 4a, 4e)

Used to decide whether a symbol is "genuinely new" vs. "pre-existed and got extended":

Phase 4e should run `scripts/diff_public_api.py` first and consume its JSON
facts for runtime signatures and public stub removals. Use the path rules below
to choose modules for that helper and to guide manual fallback when the helper
emits warnings.

- `warp/__init__.py` — top-level re-exports (the user-facing Python surface).
- `warp/__init__.pyi` — top-level public stubs used by type checkers and IDEs.
- `warp/_src/builtins.py` — kernel-scope builtins (user-facing within `@wp.kernel`).
- Any module reached from `warp/__init__.py` via `from warp._src.<mod> import X as X`.
- `warp/config.py` — user-facing configuration flags (e.g., `wp.config.track_memory`).
- Public submodules named in CHANGELOG entries, such as `warp.fem`,
  `warp.optim.linear`, `warp.jax`, and `warp.jax_experimental`. Resolve the
  public package/module first (e.g., `warp/fem/__init__.py`,
  `warp/optim/linear.py`), then follow public re-exports from that namespace to
  the real source modules and matching public `.pyi` stubs. Include
  backing-commit source modules when the public package re-exports a broad
  namespace and the CHANGELOG entry names the package rather than a specific
  symbol.

**To determine if `wp.X` existed at base**:
- For top-level symbols: grep `git show <base>:warp/__init__.py` for `import X as X`.
- For submodule attributes: read the submodule source at base (e.g., `git show <base>:warp/_src/utils.py` and look for the attribute definition).
- For kernel builtins: grep `git show <base>:warp/_src/builtins.py` for `add_builtin("<name>"`.
- For public submodule APIs: parse public functions, public classes, class
  `__init__` methods, and public methods (non-leading-underscore names) in the
  resolved modules at base and HEAD.
- For public stubs: parse `.pyi` files and compare public functions, classes,
  methods, overload variants, module-level annotated attributes, and exported
  aliases. If a stub-only public symbol exists at base and disappears at HEAD,
  report a public stub removal even when runtime source did not change.

**Breaking signature-shape signals**:
- New required parameter without an overload/wrapper preserving the old call form.
- Removed, renamed, or reordered existing parameter.
- Existing positional-or-keyword parameter moved behind `*` (a
  positional-to-keyword-only move).
- New positional-or-keyword parameter inserted before any existing
  positional-or-keyword parameter, even when defaulted. Old positional calls now
  bind differently.
- Public stub symbol, overload, method, or attribute removed from `.pyi`.

**Usually non-breaking signature-shape signals**:
- New defaulted positional-or-keyword parameter appended after all existing
  positional-or-keyword parameters.
- New defaulted keyword-only parameter.

## Deprecated compatibility path signals (Phase 4f)

Candidate paths include files or packages whose name contains `deprecated`,
`deprecation`, `compat`, `compatibility`, or `backcompat`; public modules named
in `Deprecated`, `Removed`, or migration-style CHANGELOG entries; and modules
that warn and forward from an old namespace to a replacement namespace.

Breaking signals:
- New `raise` statements in the deprecated path.
- Changed exception type or message before the path reaches the replacement API.
- Stricter guard conditions before forwarding/delegation.
- Removed `try`/`except` fallback around an old import or compatibility alias.
- New import-time error in a path that previously warned and delegated.

## Semantic-change discovery hints

The following are high-risk hints, not an exhaustive scope. Inspect other paths
when a pending entry or diff plausibly changes supported, observable behavior.
Not every change in a high-risk path is breaking:

- `warp/_src/codegen.py` — Python code generation; changes can alter emitted CUDA/CPU code.
- `warp/native/**` — C++/CUDA source; changes can alter runtime numerical or ABI behavior.

Typical NOT-BREAKING signals in these paths:
- Pure internal refactors, renames of internal identifiers.
- Comment or formatting changes.
- Performance optimizations that preserve observable output.
- Build-system touches (CMake flags, compile deps).
- Test-only changes.
- Bug fixes where the previous behavior was demonstrably wrong.

Typical BREAKING signals, when the affected surface is supported and reachable:
- Changes to emitted instructions or register allocation affecting numerics.
- Default compiler-flag changes that alter semantics.
- Struct layout changes that affect an ABI covered by the compatibility policy.
- Algorithm swaps that produce different numerical results.
- Control-flow changes that alter when or how often user code is invoked.

## Heuristic paths commonly relevant (reference only)

These are noted here so Claude can pattern-match when reading commits, but the skill does NOT use them to produce a "commits without CHANGELOG entries" audit trace. Purpose is recognition, not bucketing:

- `.github/**`, `.gitlab-ci.yml`, `.gitlab/**`, `tools/**`, `.pre-commit-config.yaml`, `uv.lock`, `.python-version` — infrastructure, not user-facing.
- `asv.conf.json`, `benchmarks/**` — benchmark harness.
- `docs/**`, root-level `*.md` — documentation.
- `warp/_src/**` other than `codegen.py` — internal Python implementation.
- `warp/native/**` — native code (see semantic-break analysis above).
