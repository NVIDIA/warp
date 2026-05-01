# Classification Helpers

This reference is loaded during Phases 3, 4, and 5 of the skill. It defines concrete path and naming rules Claude relies on when analyzing commits and CHANGELOG entries.

## Public API surface (Phase 4a, 4e)

Used to decide whether a symbol is "genuinely new" vs. "pre-existed and got extended":

- `warp/__init__.py` — top-level re-exports (the user-facing Python surface).
- `warp/_src/builtins.py` — kernel-scope builtins (user-facing within `@wp.kernel`).
- Any module reached from `warp/__init__.py` via `from warp._src.<mod> import X as X`.
- `warp/config.py` — user-facing configuration flags (e.g., `wp.config.track_memory`).

**To determine if `wp.X` existed at base**:
- For top-level symbols: grep `git show <base>:warp/__init__.py` for `import X as X`.
- For submodule attributes: read the submodule source at base (e.g., `git show <base>:warp/_src/utils.py` and look for the attribute definition).
- For kernel builtins: grep `git show <base>:warp/_src/builtins.py` for `add_builtin("<name>"`.

## Paths that trigger Phase 4f semantic-breaking analysis

Commits touching these paths get per-commit judgment (Phase 4f). Not every change in them is breaking — Claude reads the diff and decides:

- `warp/_src/codegen.py` — Python code generation; changes can alter emitted CUDA/CPU code.
- `warp/native/**` — C++/CUDA source; changes can alter runtime numerical or ABI behavior.

Typical NOT-BREAKING signals in these paths:
- Pure internal refactors, renames of internal identifiers.
- Comment or formatting changes.
- Performance optimizations that preserve observable output.
- Build-system touches (CMake flags, compile deps).
- Test-only changes.
- Bug fixes where the previous behavior was demonstrably wrong.

Typical BREAKING signals:
- Changes to emitted instructions or register allocation affecting numerics.
- Default compiler-flag changes that alter semantics.
- Struct layout changes that affect ABI for user-accessible types.
- Algorithm swaps that produce different numerical results.
- Control-flow changes that alter when or how often user code is invoked.

## Heuristic paths commonly relevant (reference only)

These are noted here so Claude can pattern-match when reading commits, but the skill does NOT use them to produce a "commits without CHANGELOG entries" audit trace. Purpose is recognition, not bucketing:

- `.github/**`, `.gitlab-ci.yml`, `.gitlab/**`, `tools/**`, `.pre-commit-config.yaml`, `uv.lock`, `.python-version` — infrastructure, not user-facing.
- `asv.conf.json`, `benchmarks/**` — benchmark harness.
- `docs/**`, root-level `*.md` — documentation.
- `warp/_src/**` other than `codegen.py` — internal Python implementation.
- `warp/native/**` — native code (see semantic-break analysis above).
