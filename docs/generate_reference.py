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
import logging
import operator
import pkgutil
import shutil
import subprocess
import sys
from bisect import bisect
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from string import digits
from types import ModuleType
from typing import Callable, TypeVar, get_origin

import warp as wp

logger = logging.getLogger(__name__)

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

# Module prefixes to skip during documentation generation.
SKIP = (
    "warp._src",
    "warp.examples",
    "warp.tests",
)


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
    type(None),  # NoneType is only available in types module from Python 3.10+
    Sequence,
    *wp._src.types.scalar_types,
    *wp._src.types.vector_types,
)


def get_isolated_dir(
    module_name: str,
) -> tuple[str]:
    """Import a module in a fresh Python process and return its dir().

    Uses delimited output to robustly parse the JSON result regardless of
    any other stdout content (warnings, logging, etc.).
    """
    code = f'''
import importlib
import json
import docs.generate_reference
module = importlib.import_module("{module_name}")
# Write to stdout with clear delimiters
print("__SYMBOLS_START__")
print(json.dumps(dir(module)))
print("__SYMBOLS_END__")
'''
    result = subprocess.run((sys.executable, "-c", code), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to import {module_name} in subprocess:\n{result.stderr}")

    # Parse between delimiters
    stdout = result.stdout
    start_marker = "__SYMBOLS_START__\n"
    end_marker = "\n__SYMBOLS_END__"
    start = stdout.find(start_marker)
    end = stdout.find(end_marker)
    if start < 0 or end < 0:
        raise RuntimeError(f"Unexpected output format from subprocess (missing delimiters):\n{stdout}")

    json_str = stdout[start + len(start_marker) : end]
    return json.loads(json_str)


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
    """Determine the type of a symbol (module, class, function, etc.).

    Raises:
        NotImplementedError: If the symbol type cannot be determined.
    """
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

    raise NotImplementedError(f"Unknown symbol type for '{symbol}' in module '{module.__name__}': {type(attr)}")


def split_trailing_digits(
    symbol: str,
) -> tuple[str, int | str]:
    """Split a symbol name into its prefix and trailing digits."""
    base = symbol.rstrip(digits)
    if len(base) == len(symbol):
        return (symbol, -1)

    return (base, int(symbol.removeprefix(base)))


def sort_symbols(
    module: ModuleType,
    symbols: Sequence[str],
) -> tuple[str, ...]:
    """Sort symbols based on their type and name."""
    return sorted(
        symbols,
        key=lambda symbol: (get_symbol_type(module, symbol), split_trailing_digits(symbol)),
    )


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

    with open(underlying_module.__file__, encoding="utf-8") as f:
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
            # Validate that we don't have more than one level of nesting
            if category.count(">") > 1:
                raise ValueError(
                    f"Category '{category}' in {underlying_module.__file__} has multiple levels of nesting. "
                    f"Only single-level nesting is supported (e.g., 'Parent > Child')."
                )
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


@dataclass
class Category:
    """Represents a documentation category with its symbols and hierarchy information."""

    full_name: str  # Full category name (e.g., "Data Types > Scalars")
    display_name: str  # Display name (e.g., "Scalars")
    parent: str | None  # Parent name (e.g., "Data Types") or None for top-level
    symbols: tuple[str, ...]  # Symbols in this category
    is_subcategory: bool  # Whether this is a subcategory


@dataclass
class ParentHeader:
    """Represents a parent section header (without content, just the title)."""

    name: str  # Parent name (e.g., "Data Types")


def build_category_structure(
    symbols_per_category: Mapping[str, tuple[str, ...]],
) -> dict[str | None, list[Category]]:
    """Build a hierarchical category structure from flat category mapping.

    Returns a dict mapping parent names (or None for top-level) to lists of Category objects.
    """
    structure: dict[str | None, list[Category]] = defaultdict(list)

    for full_name, symbols in symbols_per_category.items():
        if ">" in full_name:
            # Subcategory: "Parent > Child"
            parent, child = full_name.split(">", 1)
            parent = parent.strip()
            child = child.strip()
            structure[parent].append(
                Category(
                    full_name=full_name,
                    display_name=child,
                    parent=parent,
                    symbols=symbols,
                    is_subcategory=True,
                )
            )
        else:
            # Top-level category
            structure[None].append(
                Category(
                    full_name=full_name,
                    display_name=full_name,
                    parent=None,
                    symbols=symbols,
                    is_subcategory=False,
                )
            )

    return structure


def determine_render_order(
    symbols_per_category: Mapping[str, tuple[str, ...]],
    category_structure: dict[str | None, list[Category]],
) -> list[Category | ParentHeader]:
    """Determine the order in which to render categories and parent headers.

    Returns a list of Category objects and ParentHeader objects in the order they
    should appear in the documentation.
    """
    result: list[Category | ParentHeader] = []
    rendered = set()

    for full_name in symbols_per_category.keys():
        if full_name in rendered:
            continue

        if ">" in full_name:
            # Subcategory: render parent header first (if not already done)
            parent_name = full_name.split(">")[0].strip()
            if parent_name not in rendered:
                # Parent doesn't exist as standalone category, so add parent header
                result.append(ParentHeader(name=parent_name))
                rendered.add(parent_name)

            # Add all subcategories of this parent
            for cat in category_structure[parent_name]:
                if cat.full_name not in rendered:
                    result.append(cat)
                    rendered.add(cat.full_name)
        else:
            # Top-level category
            for cat in category_structure[None]:
                if cat.full_name == full_name and cat.full_name not in rendered:
                    result.append(cat)
                    rendered.add(cat.full_name)

                    # Add subcategories if any
                    if cat.display_name in category_structure:
                        for subcat in category_structure[cat.display_name]:
                            if subcat.full_name not in rendered:
                                result.append(subcat)
                                rendered.add(subcat.full_name)
                    break

    return result


def render_category_to_rst(
    cat: Category,
    aliased_symbols: Mapping[str, str],
    lines: list[str],
    module_name: str,
) -> None:
    """Render a single category with its symbols to RST format.

    Args:
        cat: Category to render
        aliased_symbols: Mapping of symbols that are re-exported from submodules
        lines: List to append RST lines to
        module_name: Name of the module being documented (for special handling)
    """
    # Render section header
    underline = "^" * len(cat.display_name) if cat.is_subcategory else "-" * len(cat.display_name)
    lines.append(cat.display_name)
    lines.append(underline)
    lines.append("")

    # Render non-aliased symbols
    non_aliased = tuple(s for s in cat.symbols if s not in aliased_symbols)
    if non_aliased:
        lines.append(".. autosummary::")
        lines.append("   :nosignatures:")
        lines.append(f"   :toctree: {TOCTREE_DIR}")
        if module_name == BUILTINS_MODULE:
            lines.append("   :template: builtins.rst")
        lines.append("")
        lines.extend(f"   {s}" for s in non_aliased)
        lines.append("")

    # Render aliased symbols
    aliased = tuple(s for s in cat.symbols if s in aliased_symbols)
    if aliased:
        for symbol in aliased:
            source_module = aliased_symbols[symbol]
            lines.append(f"- :obj:`{symbol} <{source_module}.{symbol}>`")
        lines.append("")


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

    # Use automodule directive to create cross-reference target and render docstring.
    # This enables :mod: cross-references to work and properly processes Sphinx
    # directives in the module docstring (e.g., :class:, admonitions, etc.).
    lines.append(f".. automodule:: {name}")
    lines.append("   :no-members:")
    lines.append("")

    lines.append(f".. currentmodule:: {current_module}")
    lines.append("")

    if submodules:
        # Separate submodules into auto-available and explicit-import categories.
        # A submodule is auto-available if it's imported in the parent module's namespace.
        auto_available = []
        explicit_import = []
        for submodule in submodules:
            # Extract the relative name (e.g., "types" from "warp.types")
            submodule_rel_name = submodule.split(".")[-1]
            # Check if this submodule name exists as a symbol in the parent module
            if submodule_rel_name in symbols:
                auto_available.append(submodule)
            else:
                explicit_import.append(submodule)

        # For non-warp modules, add a hidden toctree to include submodules in the
        # documentation tree (avoids Sphinx "not included in any toctree" warnings).
        # Top-level warp submodules are already in a toctree in index.rst.
        if name != "warp" and submodules:
            lines.append(".. toctree::")
            lines.append("   :hidden:")
            lines.append("")
            for submodule in submodules:
                submodule_link = submodule.replace(".", "_")
                lines.append(f"   {submodule_link}")
            lines.append("")

        # Generate "Submodules" section for auto-available modules
        if auto_available:
            lines.append(SUBMODULES_CATEGORY)
            lines.append("-" * len(SUBMODULES_CATEGORY))
            lines.append("")
            lines.append(f"These modules are automatically available when you ``import {name}``.")
            lines.append("")

            for submodule in auto_available:
                lines.append(f"- :mod:`{submodule}`")

            lines.append("")

        # Generate "Additional Submodules" section only if there are explicit-import modules
        if explicit_import:
            additional_category = "Additional Submodules"
            lines.append(additional_category)
            lines.append("-" * len(additional_category))
            lines.append("")
            # Use first module as concrete example
            first_module = explicit_import[0]
            lines.append(f"These modules must be explicitly imported (e.g., ``import {first_module}``).")
            lines.append("")

            for submodule in explicit_import:
                lines.append(f"- :mod:`{submodule}`")

            lines.append("")

    # Build category structure and determine render order
    category_structure = build_category_structure(symbols_per_category)
    render_order = determine_render_order(symbols_per_category, category_structure)

    # Render categories and headers in order
    for item in render_order:
        if isinstance(item, ParentHeader):
            # Render parent header (section without content)
            lines.append(item.name)
            lines.append("-" * len(item.name))
            lines.append("")
        elif isinstance(item, Category):
            # Render category with its symbols
            render_category_to_rst(item, aliased_symbols, lines, name)
        else:
            raise TypeError(f"Unexpected item type in render order: {type(item)}")

    file = output_dir / f"{output_file_name}.rst"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("\n".join(lines), encoding="utf-8")

    symbol_count = sum(len(x) for x in symbols_per_category.values())
    logger.info(f"Wrote {file.relative_to(ROOT_DIR)} ({symbol_count} symbols)")


def run():
    """Execute the documentation generation process."""
    logger.info("Generating API reference stubs...")

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
            all_symbols = get_public_symbols(module_name, run_isolated=True)
            # Filter out builtin functions (documented separately in builtins reference),
            # but only if the symbol in the warp module IS the builtin function object.
            # This keeps functions like `zeros` which share a name with a builtin but are
            # different functions (array creation vs kernel builtin).
            symbols = tuple(
                x for x in all_symbols if getattr(wp, x) is not wp._src.context.builtin_functions.get(x, None)
            )
            if (filtered_count := len(all_symbols) - len(symbols)) > 0:
                logger.debug(f"Filtered {filtered_count} builtin functions from warp module (documented separately)")
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

            submodule_rel_name = other_module_name.removeprefix(f"{module_name}.")
            if "." not in submodule_rel_name:
                submodules.append(other_module_name)

        write_module_page(module_name, symbols, aliased_symbols, submodules, output_api_dir)

    # Third pass: handle the built-ins symbols.
    # Include a builtin if ANY of its overloads are visible (not hidden).
    # This matches the behavior of the old export_functions_rst system.
    def has_visible_overload(func):
        """Check if a builtin function has at least one non-hidden overload."""
        if hasattr(func, "overloads"):
            return any(not overload.hidden for overload in func.overloads)
        return not func.hidden

    builtins_symbols = tuple(k for k, v in wp._src.context.builtin_functions.items() if has_visible_overload(v))
    write_module_page(BUILTINS_MODULE, builtins_symbols, {}, (), output_language_dir)

    # Log summary statistics
    total_modules = len(symbols_per_module) + 1  # +1 for builtins
    total_symbols = sum(len(syms) for syms in symbols_per_module.values()) + len(builtins_symbols)
    logger.info(f"Generated documentation for {total_modules} modules with {total_symbols} total symbols")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    run()
    logger.info("API reference stub generation completed successfully")
