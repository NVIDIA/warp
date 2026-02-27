# Warp Development Guidelines

- Prefer `uv run` to run Python (`uv run script.py`, `uv run -m module_name`, `uv run python -c "..."`), or activate `.venv` first. Do not use the system Python directly.
  - With extras: `uv run --extra dev -m warp.tests`
  - With additional packages: `uv run --with rich script.py`
- If `warp/bin/` is empty, build first with `build_lib.py` (or `--quick`).
- Never use `python -c "..."` to run temporary scripts that define `@wp.kernel` functions—always write to a `.py` file because Warp's codegen calls `inspect.getsourcelines()`, which fails for code not in a file.
- Always capitalize proper names in docstrings and error messages (NumPy, not numpy; Warp, not warp).
- Create a feature branch before committing—never commit directly to `main`. Use `username/short-description`.
- Always use imperative mood in commit messages ("Fix X", not "Fixed X"), ~50 char subject, reference issues as `(GH-XXX)`. Body explains *why*, not what.
- Always use `git commit --signoff` (or `-s`) to add a `Signed-off-by` line (DCO).
- CI lives in both GitLab (`.gitlab-ci.yml`) and GitHub (`.github/workflows/`). Lightweight jobs (linting, docs, packaging) may exist in both—keep them in sync. GPU-dependent jobs differ by platform.
- Pin GitHub Actions to commit hashes, not tags. Good: `uses: astral-sh/setup-uv@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86`. Bad: `uses: astral-sh/setup-uv@v4`.
- Let uv infer Python version from `.python-version`. Use `uv python install` without arguments.

Run `uvx pre-commit run --files <files>` to format changed files, or `-a` for all files. Rebuild native libraries only after changes to `warp/native/` C++/CUDA code:

- Standard build: `uv run build_lib.py` (~5 min)
- Quick build: `uv run build_lib.py --quick` (~2-4 min)
  - Only use if CUDA driver version ≥ Toolkit version
  - Check: `nvidia-smi` (driver) vs Toolkit (set via WARP_CUDA_PATH, CUDA_HOME, CUDA_PATH, or `which nvcc`)

Build docs (~1 min) with `uv run --extra docs build_docs.py 2>&1 | tee /tmp/build_docs.log` so you can inspect warnings without re-running.

## PR Instructions

- If opening a pull request on GitHub, use the template in `.github/PULL_REQUEST_TEMPLATE.md`.
- If opening a merge request on GitLab, use the template in `.gitlab/merge_request_templates/Default.md`. If a GitHub issue exists for the change, end the MR title with a reference (e.g., `[GH-123]`).
- If a change modifies user-facing behavior, append an entry to the end of the `Unreleased` section of CHANGELOG.md. Use imperative present tense ("Add X"), include issue refs `([GH-XXX](https://github.com/NVIDIA/warp/issues/XXX))`, and avoid internal implementation details.
- For complex features, consider adding a design doc in `design/`. See `design/README.md` for guidelines.

## Tests

Always use `unittest`, not pytest.

- Run all tests (~10-20 min): `uv run --extra dev -m warp.tests -s autodetect`
- Run all tests in a specific file: `uv run warp/tests/test_modules_lite.py`
- Prefer running multiple test classes in parallel over running test files one at a time in a loop: `uv run --extra dev -m warp.tests -s autodetect -k TestArray -k TestCodeGen -k TestFunc`
- Always add new test modules to `default_suite` in `warp/tests/unittest_suites.py`.
- Use standard `unittest.TestCase` methods when tests target a fixed device (e.g., CPU-only). Use `add_function_test()` only when tests need to run across multiple devices via `get_test_devices()`.
- Avoid timing-based assertions (e.g., asserting speedup ratios)—they are flaky in parallel CI environments with variable CPU load.
- Never call `wp.clear_kernel_cache()` or `wp.clear_lto_cache()` in test files—not in `__main__` blocks, test methods, or module scope. Cache clearing is not multi-process-safe; concurrent clears cause LLVM crashes. The test suite runner and `build_lib.py` already handle this.
- Use `np.testing.assert_allclose()` instead of `np.allclose()` for detailed error messages on failure.
- Explicit synchronization is rarely needed for correct behavior—most operations (e.g., `.numpy()`) implicitly synchronize. The exception is unit tests where a test function doesn't call an implicit sync before returning; add `wp.synchronize_device()` there due to the async nature of the CUDA API. Prefer `wp.synchronize_device()` over `wp.synchronize()` (the latter syncs all devices).

## Docstrings

Follow Google-style docstrings. Use doctest for code examples where practical.

- Document `__init__` parameters in the class docstring, not the `__init__` method
- Don't repeat default values from signatures—Sphinx autodoc shows them automatically
- Use double backticks for code elements and parameter cross-references (RST syntax): ``` ``data`` ```, ``` ``.nvdb`` ```—not italics (`*data*`)
- Use attribute docstrings (`"""..."""` after members) for enum/class constant docs, not `#:` comments—Sphinx supports both but only attribute docstrings work in VSCode/Pylance
- Use Sphinx roles for cross-references: `:class:`warp.array``, `:func:`warp.launch``, `:mod:`warp.render``, `:attr:`FILTER_LINEAR``
- In `builtins.py`, use `Args:` and `Returns:` (Google style), not `:param:` and `:returns:` (RST style)

## Codebase Internals

- The native library (`build_lib.py`) statically embeds: NVRTC (CUDA runtime compiler, Linux/Windows only), LLVM/Clang (CPU kernel compiler), and libmathdx (cuBLASDx/cuFFTDx for tile operations, Linux/Windows only).
- JIT compilation artifacts (`.cpp`/`.cu` from `codegen.py`, `.cubin`/`.ptx`/`.o`) live in a kernel cache directory (see `warp/_src/build.py:init_kernel_cache`).
- Refer to `warp/examples/` for reference patterns for kernels, launches, and array usage.
- Always import from `warp` in public-facing code, not `warp._src`. In internal code, import directly from `warp/_src/` modules. Public API is re-exported through `warp/__init__.py`.
- `warp._src.utils` imports `warp._src.context` at module level—importing from `utils` in early-loaded modules (e.g., `texture.py`) causes circular imports. Use lazy imports (`from warp._src.utils import ... # noqa: PLC0415` inside functions) when needed.
- Use `warp._src.utils.warn()` instead of `warnings.warn()`—it routes warnings to stdout (some applications don't want Warp writing to stderr).
- Use `DeviceLike` type annotation (from `warp._src.context`) for `device` parameters. Import under `TYPE_CHECKING` to avoid circular imports.
- Native bindings use ctypes; function signatures are registered in `Runtime.__init__` in `warp/_src/context.py`.
- If you modify `warp/_src/builtins.py`, run `build_docs.py` to regenerate `warp/__init__.pyi`.
