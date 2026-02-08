# AGENTS.md

## Project Overview

Warp is a Python framework for writing high-performance simulation and graphics code. It JIT compiles Python functions to efficient kernel code that runs on CPU or CUDA GPUs.

### Architecture

The native library (`build_lib.py`) statically embeds:

- NVRTC — CUDA runtime compiler for GPU kernels (Linux/Windows only)
- LLVM/Clang — Compiler for CPU kernels (warp-clang library)
- libmathdx — NVIDIA math library (cuBLASDx/cuFFTDx) for tile operations (Linux/Windows only)

### Kernel Cache

JIT compilation artifacts live in a cache directory (see `warp/_src/build.py:init_kernel_cache`). This includes generated `.cpp`/`.cu` files from `codegen.py` and compiled binaries (`.cubin`, `.ptx`, `.o`).

## Core Concepts

Key decorators: `@wp.kernel` (parallel GPU/CPU code), `@wp.func` (reusable functions), `@wp.struct` (composite types). Kernels use `wp.tid()` for thread index, cannot `return` values, and are launched via `wp.launch()`. Arrays (`wp.array`) wrap device memory and interoperate with NumPy/PyTorch/JAX. See `warp/examples/` for patterns.

## Python Execution

- This project uses `uv` (matches CI/CD): `uv run script.py`, `uv run -m module_name`, or `uv run python -c "..."`
- With extras: `uv run --extra dev -m warp.tests`
- With additional packages: `uv run --with rich script.py`
- Use `uv run python` (runs with project dependencies) or activate `.venv` first, not the system Python.
- If `warp/bin/` is empty, build first with `build_lib.py` (or `--quick`, see Testing section for requirements).

## Testing

IMPORTANT: Warp uses `unittest`, not pytest.

- Rebuild native libraries (only needed after changes to `warp/native/` C++/CUDA code) with `build_lib.py`
  - Standard build: `uv run build_lib.py` (~10-20 minutes)
  - Quick build: `uv run build_lib.py --quick` for rapid iteration (~2-4 minutes)
    - Compiles for minimal GPU architectures and disables CUDA forward compatibility
    - IMPORTANT: Only use if your CUDA driver version ≥ CUDA Toolkit version being used for build
    - Example: Toolkit 13.1 with Driver 13.0 will NOT work; Driver 13.1+ required
    - Check versions: `nvidia-smi` (driver) and CUDA Toolkit being used (set via WARP_CUDA_PATH, CUDA_HOME, CUDA_PATH env vars, or `which nvcc` if unset)
- Run all tests (~10-20 min): `uv run --extra dev -m warp.tests -s autodetect`
- Run all tests in a specific file (preferred for targeted fixes): `uv run warp/tests/test_modules_lite.py`
- New test modules should be added to `default_suite` in `warp/tests/unittest_suites.py`.
- Use standard `unittest.TestCase` methods when tests target a fixed device (e.g., CPU-only). Use `add_function_test()` only when tests need to run across multiple devices via `get_test_devices()`.
- Avoid timing-based assertions (e.g., asserting speedup ratios) in tests—they are flaky in parallel CI environments with variable CPU load.
- Test file and class naming: group by feature prefix (e.g., `test_module_parallel_load.py` / `TestModuleParallelLoad` alongside `test_module_hashing.py` / `TestModuleHashing`).
- NEVER call `wp.clear_kernel_cache()` or `wp.clear_lto_cache()` in test files—not in `__main__` blocks, test methods, or module scope. Cache clearing is not multi-process-safe; concurrent clears cause LLVM crashes ("IO failure on output stream: Bad file descriptor"). The test suite runner (`unittest_parallel.py`) and `build_lib.py` already clear the cache from a single process at the right times.
- Use `np.testing.assert_allclose()` instead of `np.allclose()` for array comparisons—it provides detailed error messages on failure.

### Synchronization Patterns

- Use `wp.synchronize_device()` inside `wp.ScopedDevice()` contexts, not `wp.synchronize()`. The latter synchronizes *all* devices, which is rarely what you want.
- Don't call `wp.synchronize()` or `wp.synchronize_device()` before `.numpy()`—`.numpy()` implicitly synchronizes.

### Kernel Definition

- NEVER define `@wp.kernel` functions in code passed to `python -c "..."`. Warp's codegen uses `inspect.getsourcelines()` to read kernel source, which fails for code not in a file. Write kernels to proper `.py` files instead.

## Code Style

### Formatting

Run `uvx pre-commit run -a` to format code (config in `.pre-commit-config.yaml`).

### Docstrings

Follow Google-style docstrings with these Warp-specific guidelines:

- Document `__init__` parameters in the class docstring, not the `__init__` method
- Don't repeat default values from signatures—Sphinx autodoc shows them automatically
- Use double backticks for code elements (RST syntax): ``` ``.nvdb`` ```, ``` ``"none"`` ```
- Use double backticks for parameter cross-references in docstrings: ``` ``data`` ```, ``` ``device`` ```—not italics (`*data*`)
- Use attribute docstrings (`"""..."""` after members) for enum/class constant docs, not `#:` comments—Sphinx supports both but only attribute docstrings work in VSCode/Pylance
- Use Sphinx roles for cross-references: `:class:`warp.array``, `:func:`warp.launch``, `:mod:`warp.render``
- Use `:attr:` to cross-reference class constants: `:attr:`FILTER_LINEAR``
- In `builtins.py`, use `Args:` and `Returns:` (Google style), not `:param:` and `:returns:` (RST style)
- Capitalize product names in docstrings and error messages: "NumPy" not "numpy", "Warp" not "warp"

## CHANGELOG.md

- Use imperative present tense ("Add X", not "Added X" or "This adds X").
- Include issue refs: `([GH-XXX](https://github.com/NVIDIA/warp/issues/XXX))`.
- Avoid internal implementation details users wouldn't understand.

## Commit Messages

- IMPORTANT: Create a feature branch before committing—never commit directly to `main`. Use a descriptive branch name like `username/short-description`.
- Use imperative mood ("Fix X", not "Fixed X"), ~50 char subject, reference issues as `(GH-XXX)`. Body explains *why*, not what.
- ALWAYS use `git commit --signoff` (or `-s`) to add a `Signed-off-by` line (DCO).

## Documentation

- Build docs with `uv run --extra docs build_docs.py` (not `make`/`sphinx-build`).
- Use doctest for code examples where practical.

## Codebase Internals

- `warp/_src/` contains internal implementation, re-exported through `warp/__init__.py`. Public-facing code should import from `warp`, not `warp._src`. Internal code should import directly from `warp/_src/` modules.
- `warp._src.utils` imports `warp._src.context` at module level—importing from `utils` in early-loaded modules (e.g., `texture.py`) causes circular imports. Use lazy imports (`from warp._src.utils import ... # noqa: PLC0415` inside functions) when needed.
- Use `warp._src.utils.warn()` instead of `warnings.warn()`—it routes warnings to stdout (some applications don't want Warp writing to stderr).
- Use `DeviceLike` type annotation (from `warp._src.context`) for `device` parameters. Import under `TYPE_CHECKING` to avoid circular imports.
- Native bindings use ctypes; function signatures are registered in `Runtime.__init__` in `warp/_src/context.py`.
- `warp/_src/builtins.py` defines kernel-callable functions. After modifying, run `build_docs.py` to regenerate `warp/__init__.pyi`.

## CI/CD and GitHub

Dual pipelines exist for GitLab (`.gitlab-ci.yml`) and GitHub (`.github/workflows/`)—changes may need updating in both. Follow templates in `.github/` for issues and PRs.

### GitHub Actions Security

- IMPORTANT: Pin actions to commit hashes, not tags or semantic versions. Tags can be moved or compromised; commit hashes are immutable.
  - Good: `uses: astral-sh/setup-uv@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86`
  - Bad: `uses: astral-sh/setup-uv@v4` or `uses: astral-sh/setup-uv@v4.0.0`
- Let uv infer the Python version from `.python-version` instead of specifying it explicitly in workflows. Use `uv python install` without arguments.
