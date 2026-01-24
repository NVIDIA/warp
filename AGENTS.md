# AGENTS.md

Instructions for AI coding agents working on the Warp codebase.

## Project Overview

Warp is a Python framework for writing high-performance simulation and graphics code. It JIT compiles Python functions to efficient kernel code that runs on CPU or CUDA GPUs. The codebase includes Python (`warp/`), C++/CUDA native code (`warp/native/`), and documentation (`docs/`).

**Architecture:** The native library (`build_lib.py`) statically embeds:

- **NVRTC** — CUDA runtime compiler for GPU kernels (Linux/Windows only)
- **LLVM/Clang** — Compiler for CPU kernels (warp-clang library)
- **libmathdx** — NVIDIA math library (cuBLASDx/cuFFTDx) for tile operations (Linux/Windows only)

**Kernel cache:** JIT compilation artifacts live in a cache directory (see `warp/_src/build.py:init_kernel_cache`). This includes generated `.cpp`/`.cu` files from `codegen.py` and compiled binaries (`.cubin`, `.ptx`, `.o`).

## Core Concepts

**Kernels:** Computational kernels are Python functions decorated with `@wp.kernel`. They are JIT-compiled to C++/CUDA and executed in parallel across threads. Kernels must use typed arguments, cannot `return` values, and use only a subset of Python. Use `wp.tid()` to get the thread index.

```python
@wp.kernel
def example(data: wp.array(dtype=wp.float32)):
    i = wp.tid()  # Thread index
    data[i] = wp.float32(i)
```

**Launching kernels:** Use `wp.launch(kernel, dim=..., inputs=[...])` to execute a kernel. The `dim` specifies the number of parallel threads. Kernel results are written to arrays passed as arguments.

**Arrays:** `wp.array` wraps GPU/CPU memory allocations. Create with `wp.zeros()`, `wp.ones()`, `wp.empty()`, or `wp.array(numpy_data)`. Arrays have a `dtype` (scalar or composite like `wp.vec3`) and `device` (`"cuda:0"` or `"cpu"`).

**User functions:** Use `@wp.func` for reusable code callable from kernels. Unlike kernels, functions can return values.

**Structs:** Use `@wp.struct` for user-defined composite types that can be passed to kernels or used as array dtypes.

**Automatic differentiation:** Warp supports reverse-mode autodiff. Create arrays with `requires_grad=True`, wrap kernel launches in `with wp.Tape() as tape:`, then call `tape.backward(loss)` to compute gradients via `array.grad`.

**Interoperability:** Warp arrays interoperate with NumPy (`wp.array(numpy_arr)`, `warp_arr.numpy()`), PyTorch, and JAX.

## Python Execution

**Preferred:** Use `uv run` for all Python execution (matches CI/CD):

```bash
uv run script.py
uv run -m module_name
uv run -m warp.tests
```

**With extras or additional packages:**

```bash
# Install dev extras (linting, testing tools)
uv run --extra dev -m warp.tests

# Add a package not in pyproject.toml extras
uv run --with rich script.py
uv run --with pandas analyze.py
```

**Fallback:** If `uv` is not installed, activate a virtual environment first:

```bash
source .venv/bin/activate
python script.py
```

**Never:** Run bare `python`/`python3` without environment isolation—this uses system Python with potentially wrong dependencies.

## Testing

Warp uses **unittest**, not pytest.

**Run all tests** (~10-20 min):

```bash
uv run --extra dev -m warp.tests -s autodetect
```

**Run specific test file:**

```bash
uv run warp/tests/test_modules_lite.py
```

**Common options** (see `uv run -m warp.tests --help` or `warp/_src/thirdparty/unittest_parallel.py`):

- `-s autodetect` — Auto-discover all test files
- `-p "test_*.py"` — Match specific file pattern
- `-k "cuda"` — Run tests matching substring
- `--serial-fallback` — Single-process mode for debugging

**Adding tests:** New test modules should be added to `default_suite` in `warp/tests/unittest_suites.py`.

## Code Style

### Formatting

Code is formatted automatically via pre-commit hooks:

- **Python**: Ruff (linting + formatting)
- **C++/CUDA**: clang-format

Run `uvx pre-commit run --all-files` to format (or `pre-commit run --all-files` if installed).

### Type Hints

- Put type hints in function signatures, not docstrings
- Use `wp.DeviceLike` for flexible device input, `wp.Device` when only Device objects accepted

### Docstrings

Follow Google-style docstring conventions with these Warp-specific guidelines:

- Document `__init__` parameters in the **class docstring**, not the `__init__` method
- Don't repeat default values from signatures—Sphinx autodoc shows them automatically
- Use double backticks for code elements in docstrings (RST syntax): ``` ``.nvdb`` ```, ``` ``"none"`` ```

### Cross-References

Use Sphinx roles: `:class:`warp.array``, `:func:`warp.launch``, `:mod:`warp.render``

## CHANGELOG.md

When adding entries to CHANGELOG.md:

**Entry format:**

- Start with imperative verb: Add, Remove, Fix, Change, Deprecate, etc.
- Use present tense
- Include issue refs: `([GH-XXX](https://github.com/NVIDIA/warp/issues/XXX))`
- Use backticks for code: `` `wp.function_name()` ``

**Sections:**

- **Added** — New features
- **Removed** — Deleted functionality
- **Deprecated** — Marked for future removal
- **Changed** — Modified behavior
- **Fixed** — Bug fixes
- **Documentation** — Doc improvements

**Avoid:**

- Starting with "The" or "This"
- Past tense ("added", "fixed")
- Vague descriptions without context
- Referencing internal implementation details users wouldn't understand

## Commit Messages

**Subject line:**

- Start with imperative verb: Fix, Add, Remove, Refactor, Update, etc.
- Limit to ~50 characters (72 max)
- Capitalize first word
- No trailing period
- Include issue reference when applicable: `(GH-XXX)`

**Test:** A good subject completes the sentence: "If applied, this commit will ___"

**Body (for non-trivial changes):**

- Blank line after subject
- Wrap at 72 characters
- Explain *why*, not just what changed (the code explains *how*)
- Use `-` bullet points for multiple items

## Building

Rebuild native libraries (only needed after changes to `warp/native/` C++/CUDA code):

```bash
uv run build_lib.py
```

See [docs/user_guide/installation.rst](docs/user_guide/installation.rst) for build requirements (CUDA Toolkit, compilers) and Docker builds.

**Build documentation:**

```bash
uv run --extra docs build_docs.py           # Build HTML docs
uv run --extra docs build_docs.py --doctest # Run doctest validation
```

Use `build_docs.py` instead of `make` or `sphinx-build` directly—it runs `docs/generate_reference.py` as a prerequisite. New documentation with code blocks should use doctest where practical (note: output from `wp.print()`/`wp.printf()` is not captured by doctest).

## Project Structure

```
.                   # Repository root
├── warp/           # Python package
│   ├── _src/       # Internal implementation (private)
│   ├── native/     # C++/CUDA source
│   ├── tests/      # Test files (test_*.py)
│   └── examples/   # Example scripts
├── asv/            # ASV benchmarks
├── docs/           # Sphinx documentation
├── CHANGELOG.md    # Release notes
└── pyproject.toml  # Package configuration
```

**Public API:** Internal implementation lives in `warp/_src/` and is re-exported through `warp/__init__.py`. Public-facing code (examples, documentation) should import from `warp`, not `warp._src`. Internal code in `warp/_src/` should import directly from other `warp/_src/` modules:

```python
# Public-facing code (examples, docs)
import warp as wp

# Internal code (warp/_src/*.py)
from warp._src.types import float32
```

**Native bindings:** Warp uses ctypes to interface with the native library. Function signatures are registered in the `Runtime` class's `__init__` method in `warp/_src/context.py`.

**Built-Ins:** `warp/_src/builtins.py` defines functions callable from Warp kernels (and sometimes Python). Key attributes:

- `export=True` — Exposed in `warp` namespace (registered via `add_builtin()`)
- `hidden=True` — Not documented
- `is_differentiable=False` — No adjoint defined

Built-ins are registered at runtime on `import warp`. For type checkers/IDE autocomplete, `warp/__init__.pyi` provides stubs (generated by `build_docs.py`).

**CI/CD:** The project is hosted on both GitLab and GitHub with separate pipelines:

- **GitLab**: `.gitlab-ci.yml` and `.gitlab/ci/`
- **GitHub**: `.github/workflows/`

## GitHub Issues and Pull Requests

**Issues:** Should generally follow one of the templates in `.github/ISSUE_TEMPLATE/` (bug report, feature request, question, documentation), but additional sections can be added.

**Pull requests:** Follow `.github/PULL_REQUEST_TEMPLATE.md` structure, but additional sections can be added to provide more context.

## Additional Resources

- [Contribution Guide](docs/user_guide/contribution_guide.rst)
- [CONTRIBUTING.md](CONTRIBUTING.md) — DCO sign-off requirements for external contributors
- [docs/user_guide/compatibility.rst](docs/user_guide/compatibility.rst) — Support policy and support matrix
- [README.md](README.md) — Installation, examples, and build requirements
