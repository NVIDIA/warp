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

- Use `uv run` for all Python execution (matches CI/CD): `uv run script.py` or `uv run -m module_name`
- With extras: `uv run --extra dev -m warp.tests`
- With additional packages: `uv run --with rich script.py`
- Never run bare `python`/`python3`—always use `uv run` or activate `.venv` first.

## Testing

IMPORTANT: Warp uses `unittest`, not pytest.

- Rebuild native libraries (only needed after changes to `warp/native/` C++/CUDA code) with `build_lib.py`
- Run all tests (~10-20 min): `uv run --extra dev -m warp.tests -s autodetect`
- Run specific test file (preferred): `uv run warp/tests/test_modules_lite.py`
- New test modules should be added to `default_suite` in `warp/tests/unittest_suites.py`.
- NEVER call `wp.clear_kernel_cache()` outside `if __name__ == "__main__":` blocks—parallel test runners will conflict.

## Code Style

### Formatting

Run `uvx pre-commit run -a` to format code (config in `.pre-commit-config.yaml`).

### Docstrings

Follow Google-style docstrings with these Warp-specific guidelines:

- Document `__init__` parameters in the class docstring, not the `__init__` method
- Don't repeat default values from signatures—Sphinx autodoc shows them automatically
- Use double backticks for code elements (RST syntax): ``` ``.nvdb`` ```, ``` ``"none"`` ```
- Use Sphinx roles for cross-references: `:class:`warp.array``, `:func:`warp.launch``, `:mod:`warp.render``
- In `builtins.py`, use `Args:` and `Returns:` (Google style), not `:param:` and `:returns:` (RST style)

## CHANGELOG.md

- Use imperative present tense ("Add X", not "Added X" or "This adds X").
- Include issue refs: `([GH-XXX](https://github.com/NVIDIA/warp/issues/XXX))`.
- Avoid internal implementation details users wouldn't understand.

## Commit Messages

Use imperative mood ("Fix X", not "Fixed X"), ~50 char subject, reference issues as `(GH-XXX)`. Body explains *why*, not what.

## Documentation

- Build docs with `uv run --extra docs build_docs.py` (not `make`/`sphinx-build`).
- Use doctest for code examples where practical.

## Codebase Internals

- `warp/_src/` contains internal implementation, re-exported through `warp/__init__.py`. Public-facing code should import from `warp`, not `warp._src`. Internal code should import directly from `warp/_src/` modules.
- Native bindings use ctypes; function signatures are registered in `Runtime.__init__` in `warp/_src/context.py`.
- `warp/_src/builtins.py` defines kernel-callable functions. After modifying, run `build_docs.py` to regenerate `warp/__init__.pyi`.

## CI/CD and GitHub

Dual pipelines exist for GitLab (`.gitlab-ci.yml`) and GitHub (`.github/workflows/`)—changes may need updating in both. Follow templates in `.github/` for issues and PRs.
