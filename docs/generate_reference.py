# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate concise API .rst files for selected modules.

This helper scans the *top-level* modules and writes one reStructuredText file
per module with an `autosummary` directive. When Sphinx later builds
the documentation (with `autosummary_generate = True`), individual stub pages
will be created automatically for every listed symbol.

The generated files live in `docs/reference/` (git-ignored by default).

Usage (from the repository root):

    python docs/generate_reference.py
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import json
import operator
import pkgutil
import shutil
import subprocess
import sys
from bisect import bisect
from collections.abc import Mapping, Sequence
from enum import IntEnum
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from string import digits
from types import ModuleType, NoneType
from typing import Callable, TypeVar, get_origin

import warp as wp

# Configuration
# -----------------------------------------------------------------------------

# Repo root.
ROOT_DIR = Path(__file__).resolve().parent.parent

# Module containing Warp's built-ins functions and requiring special handling.
BUILTINS_MODULE = "warp._src.lang"

# Fallback category for items not explicitly categorized.
DEFAULT_CATEGORY = "Misc"

# Symbols representing modules have a dedicated category displayed at the top.
SUBMODULES_CATEGORY = "Submodules"

# Category for listing deprecated symbols.
DEPRECATED_CATEGORY = "Deprecated"

# Output directory (relative to ROOT_DIR).
OUTPUT_DIR = ROOT_DIR / "docs"

# Where autosummary should place generated stub pages (relative to each .rst
# file). Keeping them alongside the .rst files avoids clutter elsewhere.
TOCTREE_DIR = "_generated"  # Sub-folder inside OUTPUT_DIR.

# Where this script generates the .rst file for each module.
API_REF_DIR = "api_reference"
BUILTINS_REF_DIR = "language_reference"


# Mock Dependencies
# -----------------------------------------------------------------------------

MOCK_PACKAGES = ("jax", "paddle", "pxr", "torch")


class MockAttr:
    """Helper for mocking attributes from a module."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


class MockModule(ModuleType):
    """Helper for mocking a module."""

    def __init__(self, name: str):
        super().__init__(name)
        self.__name__ = name
        self.__file__ = None
        self.__path__ = []

    def __getattr__(self, name: str):
        return MockAttr(f"{self.__name__}.{name}")


class MockLoader(Loader):
    """Helper for mocking a ModuleSpec loader."""

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        """Create and return a MockModule for the given spec."""
        return MockModule(spec.name)

    def exec_module(self, module: ModuleType) -> None:
        """Execute the module."""
        pass


class MockFinder(MetaPathFinder):
    """A meta path finder that intercepts imports for mocked packages."""

    def __init__(self, packages: Sequence[str]):
        self._packages = packages

    def find_spec(self, fullname: str, path, target=None) -> ModuleSpec | None:
        """Return a ModuleSpec if the module should be mocked."""
        for package in self._packages:
            if fullname == package or fullname.startswith(f"{package}."):
                return ModuleSpec(fullname, MockLoader())

        return None


def install_mock_modules() -> None:
    """Install mock modules for dependencies not present in the environment."""
    mods = tuple(x for x in MOCK_PACKAGES if x not in sys.modules)
    if mods:
        sys.meta_path.insert(0, MockFinder(mods))


# Helpers
# -----------------------------------------------------------------------------

VALUE_TYPES = (
    bool,
    int,
    float,
    set,
    Mapping,
    NoneType,
    Sequence,
    *wp._src.types.scalar_types,
    *wp._src.types.vector_types,
)


def get_isolated_dir(
    module_name: str,
) -> tuple[str]:
    """Import a module in a fresh Python process and return its dir()."""
    code = f'''
import importlib
import json
import docs.generate_reference
module = importlib.import_module("{module_name}")
print(json.dumps(dir(module)))
'''
    result = subprocess.run((sys.executable, "-c", code), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to import {module_name}: {result.stderr}")

    return json.loads(result.stdout.split("\n")[-2])


def is_symbol_public(
    name: str,
) -> bool:
    """Check whether a symbol name is public."""
    return not name.startswith("_")


@functools.cache
def get_public_symbols(
    module_name: str,
    run_isolated: bool = False,
) -> tuple[str, ...]:
    """Return the list of public names for a given module name."""
    if run_isolated:
        return tuple(sorted(filter(is_symbol_public, get_isolated_dir(module_name))))

    module = importlib.import_module(module_name)
    return tuple(sorted(filter(is_symbol_public, dir(module))))


class SymbolType(IntEnum):
    MODULE = 0
    ANNOTATION = 1
    CLASS = 2
    FUNCTION = 3
    VALUE = 4


def get_symbol_type(
    module: ModuleType,
    symbol: str,
) -> SymbolType:
    """Determine the type of a symbol (module, class, function, etc.)."""
    attr = getattr(module, symbol)

    if inspect.ismodule(attr):
        return SymbolType.MODULE

    if isinstance(attr, TypeVar) or get_origin(attr) is not None:
        return SymbolType.ANNOTATION

    if inspect.isclass(attr) or isinstance(attr, wp._src.codegen.Struct):
        return SymbolType.CLASS

    if isinstance(attr, (Callable, wp._src.context.Function)) or hasattr(attr, "func"):
        return SymbolType.FUNCTION

    if isinstance(attr, VALUE_TYPES):
        return SymbolType.VALUE

    return NotImplementedError


def split_trailing_digits(
    symbol: str,
) -> tuple[str, int | str]:
    """Split a symbol name into its prefix and trailing digits."""
    base = symbol.rstrip(digits)
    if len(base) == len(symbol):
        return (symbol, -1)

    return (base, int(symbol[len(base) :]))


def sort_symbols(
    module: ModuleType,
    symbols: Sequence[str],
) -> tuple[str, ...]:
    """Sort symbols based on their type and name."""
    try:
        return sorted(
            symbols,
            key=lambda symbol: (get_symbol_type(module, symbol), split_trailing_digits(symbol)),
        )
    except TypeError:
        invalid_symbols = []
        for symbol in symbols:
            if get_symbol_type(module, symbol) is NotImplementedError:
                invalid_symbols.append(symbol)

        if invalid_symbols:
            invalid_symbols_fmt = ", ".join(f"`{x}`" for x in invalid_symbols)
            raise RuntimeError(
                f"Found symbols in the module `{module.__name__}` that couldn't be associated to a type: "
                f"{invalid_symbols_fmt}"
            ) from None


def get_builtin_symbols_per_category(
    symbols: Sequence[str],
) -> Mapping[str, tuple[str, ...]]:
    """Group built-in symbols by their category."""
    symbols_per_category = {}
    for symbol in symbols:
        attr = wp._src.context.builtin_functions[symbol]
        symbols_per_category.setdefault(attr.group, []).append(symbol)

    return symbols_per_category


def get_symbols_per_category(
    module: ModuleType,
    symbols: Sequence[str],
) -> Mapping[str, tuple[str, ...]]:
    """Group module symbols by their category based on source comments."""
    # Support proxy modules.
    underlying_module = getattr(module.__class__, "_underlying_module_", module)

    # In order to categorize the public symbols found, we check whether they
    # are defined under a comment representing a category section.
    # To map the public symbols to the line were they are defined, we use
    # Python's `ast.parse()`. However this ignores comments, so we need
    # to process them in a separate pass.

    with open(underlying_module.__file__) as f:
        code = f.read()

    # Find the category sections.
    # Store their corresponding location as a separate sorted array
    # to facilitate indexing with bisect.
    # The default category is set to `Misc`.
    categories = [DEFAULT_CATEGORY]
    categories_loc = [-1]
    for i, line in enumerate(code.split("\n")):
        if line.startswith("# category: "):
            category = line[12:].strip()
            categories.append(category)
            categories_loc.append(i)

    # Parse the code to find the line where each public symbol is defined.
    # Only module-level statements are considered.
    symbols_line_map = {}
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if len(node.names) == 1 and node.names[0].name == "*":
                parsed_symbols = get_public_symbols(node.module)
            else:
                parsed_symbols = tuple(x.name if x.asname is None else x.asname for x in node.names)
        elif isinstance(node, ast.Assign):
            parsed_symbols = tuple(x.id for x in node.targets)
        elif isinstance(node, ast.AnnAssign):
            parsed_symbols = (node.target.id,)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            parsed_symbols = (node.name,)
        else:
            continue

        symbols_line_map.update({x: node.lineno for x in parsed_symbols if x in symbols})

    # Check if there's any public symbol that we missed while parsing the AST.
    missing_symbols = tuple(x for x in symbols if x not in symbols_line_map)
    if missing_symbols:
        missing_symbols_fmt = ", ".join(f"`{x}`" for x in missing_symbols)
        raise RuntimeError(
            f"Public symbols that were not found when parsing the AST for the module `{module.__name__}`: {missing_symbols_fmt}"
        )

    # Group the public symbols by their category.
    # Initialize it with the submodules category to make sure it's listed first.
    symbols_per_category = {}
    for symbol, line in sorted(symbols_line_map.items(), key=operator.itemgetter(1)):
        if inspect.ismodule(getattr(module, symbol)):
            continue

        category_idx = bisect(categories_loc, line) - 1
        category = categories[category_idx]
        symbols_per_category.setdefault(category, []).append(symbol)

    if DEFAULT_CATEGORY in symbols_per_category and len(symbols_per_category) == 1:
        symbols_per_category["API"] = symbols_per_category[DEFAULT_CATEGORY]
        del symbols_per_category[DEFAULT_CATEGORY]

    return symbols_per_category


def write_module_page(
    name: str,
    symbols: Sequence[str],
    aliased_symbols: Mapping[str, str],
    submodules: Sequence[str],
    output_dir: Path,
) -> None:
    """Create an .rst file for the given module name."""
    module = importlib.import_module(name)

    if name == BUILTINS_MODULE:
        # Create a dummy module for the rest of the code to run smoothly.
        title = "Built-Ins"
        current_module = "warp._src.lang"
        output_file_name = "builtins"
        symbols_per_category = get_builtin_symbols_per_category(symbols)
    else:
        title = name
        current_module = name
        output_file_name = name.replace(".", "_")
        symbols_per_category = get_symbols_per_category(module, symbols)

    symbols_per_category = {k: sort_symbols(module, v) for k, v in symbols_per_category.items() if v}

    lines: list[str] = []

    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    doc = (module.__doc__ or "").strip()
    if doc:
        lines.append(doc)
        lines.append("")

    lines.append(f".. currentmodule:: {current_module}")
    lines.append("")

    if submodules:
        lines.append(SUBMODULES_CATEGORY)
        lines.append("-" * len(SUBMODULES_CATEGORY))
        lines.append("")

        if name == "warp":
            # Use :doc: references for top-level warp submodules to avoid duplicate
            # toctree warnings since these are also listed in index.rst.
            for submodule in submodules:
                submodule_link = submodule.replace(".", "_")
                lines.append(f"- :doc:`{submodule} <{submodule_link}>`")
        else:
            # Use a toctree for nested submodules so they're indexed'.
            lines.append(".. toctree::")
            lines.append("   :maxdepth: 1")
            lines.append("")

            for submodule in submodules:
                submodule_link = submodule.replace(".", "_")
                lines.append(f"   {submodule} <{submodule_link}>")

        lines.append("")

    for category, symbols in symbols_per_category.items():
        lines.append(category)
        lines.append("-" * len(category))
        lines.append("")

        category_non_aliased_symbols = tuple(x for x in symbols if x not in aliased_symbols)
        if category_non_aliased_symbols:
            lines.append(".. autosummary::")
            lines.append("   :nosignatures:")

            lines.append(f"   :toctree: {TOCTREE_DIR}")

            if name == BUILTINS_MODULE:
                lines.append("   :template: builtins.rst")

            lines.append("")

            lines.extend(f"   {x}" for x in category_non_aliased_symbols)
            lines.append("")

        category_aliased_symbols = tuple(x for x in symbols if x in aliased_symbols)
        if category_aliased_symbols:
            for symbol in category_aliased_symbols:
                source_module = aliased_symbols[symbol]
                lines.append(f"- :obj:`{symbol} <{source_module}.{symbol}>`")

            lines.append("")

    file = output_dir / f"{output_file_name}.rst"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("\n".join(lines), encoding="utf-8")

    symbol_count = sum(len(x) for x in symbols_per_category.values())
    print(f"Wrote {file.relative_to(ROOT_DIR)} ({symbol_count} symbols)")


SKIP = (
    "warp._src",
    "warp.examples",
    "warp.tests",
)


def run():
    """Execute the documentation generation process."""
    install_mock_modules()

    output_api_dir = OUTPUT_DIR / API_REF_DIR
    output_language_dir = OUTPUT_DIR / BUILTINS_REF_DIR

    shutil.rmtree(output_api_dir, ignore_errors=True)
    shutil.rmtree(output_language_dir, ignore_errors=True)

    # First pass: collect all modules that will be documented and their symbols.
    symbols_per_module = {}
    packages = pkgutil.walk_packages(path=wp.__path__, prefix=wp.__name__ + ".")
    for _, module_name, _ in ((None, "warp", None), *packages):
        if any(module_name.startswith(x) for x in SKIP):
            continue

        if module_name == "warp":
            # Using `pkgutil.walk_packages()` causes additional symbols being added to the namespace
            # so we need to call `import module; dir(module)` in an isolated subprocess to have a clean
            # listing of the public API. It's quite slower so we only use it where necessary,
            # which is for the root module warp.
            symbols = tuple(x for x in get_public_symbols(module_name, run_isolated=True))
            # Filter out builtin functions (documented separately in builtins reference),
            # but only if the symbol in the warp module IS the builtin function object.
            # This keeps functions like `zeros` which share a name with a builtin but are
            # different functions (array creation vs kernel builtin).
            symbols = tuple(x for x in symbols if getattr(wp, x) is not wp._src.context.builtin_functions.get(x, None))
        else:
            symbols = tuple(x for x in get_public_symbols(module_name))

        if symbols:
            symbols_per_module[module_name] = symbols

    # Second pass: generate documentation with re-export detection
    for module_name, symbols in symbols_per_module.items():
        module = importlib.import_module(module_name)

        aliased_symbols = {}
        submodules = []
        for other_module_name, other_symbols in symbols_per_module.items():
            if not other_module_name.startswith(f"{module_name}."):
                continue

            other_module = importlib.import_module(other_module_name)
            aliased_symbols.update(
                {
                    x: other_module_name
                    for x in symbols
                    if x in other_symbols and getattr(module, x) is getattr(other_module, x)
                }
            )

            submodule_rel_name = other_module_name[len(module_name) + 1 :]
            if "." not in submodule_rel_name:
                submodules.append(other_module_name)

        write_module_page(module_name, symbols, aliased_symbols, submodules, output_api_dir)

    # Third pass: handle the built-ins symbols.
    builtins_symbols = tuple(k for k, v in wp._src.context.builtin_functions.items() if not v.hidden)
    write_module_page(BUILTINS_MODULE, builtins_symbols, {}, (), output_language_dir)


# Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run()
    print("Done.")
