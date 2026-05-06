# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import ast
import importlib
import inspect
import operator
import os
import pkgutil
import re
import sys

import docutils
import sphinx
from sphinx import addnodes
from sphinx.environment.adapters.toctree import note_toctree
from sphinx.ext.autosummary import autosummary_toc
from sphinx.ext.autosummary.generate import AutosummaryRenderer
from sphinx.ext.napoleon.docstring import GoogleDocstring

_RE_WP_DOT = re.compile(r"\bwp\.")

# -- Path setup --------------------------------------------------------------

HERE = os.path.dirname(__file__)
WARP_PATH = os.path.realpath(os.path.join(HERE, ".."))

sys.path.insert(0, WARP_PATH)

try:
    import warp as wp
    import warp._src
except ImportError as e:
    raise ImportError(
        f"Could not import warp module: {e}. "
        "Warp must be importable to build documentation. "
        "Run build_lib.py first, then run build_docs.py with the docs extra installed."
    ) from e

import docs.generate_reference  # noqa: E402  # must come after sys.path setup (imports warp)

# Determine the Git version/tag from CI environment variables.
# 1. Check for GitHub Actions' variable.
# 2. Check for GitLab CI's variable.
# 3. Fallback to 'main' for local builds.
GITHUB_VERSION = os.environ.get("GITHUB_REF_NAME") or os.environ.get("CI_COMMIT_REF_NAME") or "main"


# -- Project information -----------------------------------------------------

project = "Warp"
version = wp.__version__
release = version


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # Expands `auto*` directives with their corresponding docstrings.
    "sphinx.ext.autosummary",  # Builds stub .rst files and summary tables.
    "sphinx.ext.doctest",  # Tests code snippets in docs.
    "sphinx.ext.extlinks",  # Generates markups to shorten external links.
    "sphinx.ext.githubpages",  # Makes the doc compatible with GitHub Pages.
    "sphinx.ext.intersphinx",  # Allows linking to external projects' documentations.
    "sphinx.ext.linkcode",  # Adds GitHub source code links.
    "sphinx.ext.napoleon",  # Supports Google and NumPy style docstrings..
    # Third-party extensions.
    "myst_parser",  # Parses markdown files.
    "sphinx_copybutton",  # Adds a copy button to code blocks.
]

# Enable nitpicky mode to warn about unresolved references.
nitpicky = True

# Suppress warnings for types that Sphinx cannot resolve due to structural
# limitations (dynamic ctypes types, py:data/py:class role mismatches,
# mocked dependencies, runtime repr leaking into signatures, etc.).
nitpick_ignore_regex = [
    # Public type aliases indexed as py:data but referenced as py:class (Sphinx limitation)
    (r"py:class", r"(warp\.|wp\.)?(Scalar|Int|Float|DeviceLike)"),
    # Internal meta-types used in builtin function signatures (not exported)
    (
        r"py:class",
        r"(Vector|Quaternion|Matrix|Array|Transformation|Tile|TileStack|IndexedArray|IndexedFabricArray|FabricArray|Shape|DType|Any)",
    ),
    # Array type parameters from warp.array() annotations (e.g., "dtype=warp.float32", "ndim=3")
    # Sphinx splits "warp.array(dtype=float, ndim=3)" and tries to resolve each part as a class.
    (r"py:class", r"(ndim|dtype)=.*"),
    # Bare integer Literal values (e.g., Literal[3]) that leak into builtin function signatures
    (r"py:class", r"\d+"),
    # Internal _src paths that leak into annotations via from __future__ import annotations
    # (411 warnings across ~65 files; fixing requires project-wide annotation refactor)
    (r"py:class", r"(warp|wp)\._src\..*"),
    # Ctypes-based geometric types that can't be documented as classes (vec*, mat*, quat, etc.)
    (r"py:class", r"(warp\.)?(vec\d*[ihfd]?|mat\d+[ihfd]?|quat[hfd]?|spatial_(vector|matrix)[hfd]?|transform[hfd]?)"),
    # Type names used in FEM and internal annotations (e.g., Sample, Coords)
    (
        r"py:class",
        r"(Struct|BlockType|Rows|Cols|Sample|Coords|ElementIndex|"
        r"ElementArg|ElementEvalArg|ElementIndexArg|TopologyArg|EvalArg|"
        r"BsrMatrixOrExpression|_Var|_FuncParams|FunctionMetadata|KernelHooks|"
        r"launch_bounds_t|FieldRestriction|scalar)",
    ),
    # FEM nested type annotations (e.g., Geometry.CellArg, FunctionSpace.dof_dtype)
    (r"py:class", r"\w+\.(\w*Arg|\w*dtype|LocalValueMap)"),
    # External/mocked types and builtins.bool (Warp shadows bool, forcing builtins.bool in annotations)
    (r"py:class", r"(paddle\.Tensor|Usd\.Stage|_ctypes\.Structure|builtins\.bool)"),
    # numpy internal type annotations
    (r"py:class", r"numpy\..*"),
    # Stringified property/cached_property objects leaking into type annotations
    (r"py:class", r"<(property|functools\.cached_property) object at .*>"),
    # Autosummary-generated member stubs for Warp classes (Texture*, fem.*, etc.)
    (r"py:obj", r"warp\.(Texture\w+|fixedarray|indexedarray|indexedfabricarray|fabricarray|fem\.).*"),
    # Internal C++/Python interop methods on geometric types
    (
        r"py:obj",
        r"warp\.(vec|mat|quat|transform|spatial_vector|spatial_matrix)\w*\.(from_ptr|scalar_export|scalar_import)",
    ),
    # Matrix accessor methods
    (r"py:obj", r"warp\.(mat|spatial_matrix)\w+\.(get_col|get_row|set_col|set_row)"),
    # Built-in numeric methods/properties inherited by enums/int/float subclasses
    (
        r"py:obj",
        r".*\.(conjugate|bit_length|bit_count|to_bytes|from_bytes|as_integer_ratio|is_integer|real|imag|numerator|denominator)",
    ),
    # jax_callable lives in warp.jax_experimental (jax itself is mocked)
    (r"py:func", r"warp\.jax_experimental\.jax_callable"),
]


# -- Options for source files ------------------------------------------------

exclude_patterns = [".DS_Store", "Thumbs.db", "_build", "_src", "superpowers"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


# -- Options for templating --------------------------------------------------

templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------

html_theme = "nvidia_sphinx_theme"
html_theme_options = {
    "announcement": "Warp v1.13.0 is now available. See the <a href='https://github.com/NVIDIA/warp/releases/tag/v1.13.0'>release notes</a>.",
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "article_header_end": ["view-page-source.html"],
    "use_edit_page_button": True,
    "copyright_override": {"start": 2022},
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "github_url": "https://github.com/NVIDIA/warp",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/warp-lang",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "navigation_depth": 2,
    "sidebar_includehidden": False,
}
html_title = f"Warp {version}"
html_context = {
    "github_user": "NVIDIA",
    "github_repo": "warp",
    "github_version": GITHUB_VERSION,
    "doc_path": "docs",
}
html_css_files = ["custom.css"]
html_static_path = ["_static"]
html_show_sphinx = False


# -- sphinx.ext.autodoc ------------------------------------------------------

autodoc_default_options = {
    "members": True,  # Includes all public members, not just the class' docstring.
    "member-order": "bysource",  # Keep members in the order they appear in the source code.
    "undoc-members": False,  # Skips documenting members without a docstring.
    "exclude-members": "__weakref__",  # Skips these names even if they have a docstring.
    "autosummary": True,  # Generate summary tables for members.
}

# Mock external dependencies that might not be installed.
autodoc_mock_imports = ["jax", "paddle", "pxr", "torch"]

# Show typehints as content of the function or method instead of in the signature.
autodoc_typehints = "description"

# Show the literal expression for default arguments instead of evaluating them.
autodoc_preserve_defaults = True

# Prevent docstrings from being inherited from parent classes or methods.
autodoc_inherit_docstrings = False


# -- sphinx.ext.autosummary --------------------------------------------------

# Document imported classes and functions.
autosummary_imported_members = True

# Only document symbols defined in `__all__` if present.
autosummary_ignore_module_all = False

# Map object names to unique filenames to avoid collisions on case-insensitive
# file systems (e.g., Windows NTFS).
autosummary_filename_map = {
    "warp.e": "warp.e_constant",
    "warp.half_pi": "warp.half_pi_constant",
    "warp.inf": "warp.inf_constant",
    "warp.ln2": "warp.ln2_constant",
    "warp.ln10": "warp.ln10_constant",
    "warp.log10e": "warp.log10e_constant",
    "warp.log2e": "warp.log2e_constant",
    "warp.nan": "warp.nan_constant",
    "warp.phi": "warp.phi_constant",
    "warp.pi": "warp.pi_constant",
    "warp.tau": "warp.tau_constant",
    "warp.kernel": "warp.kernel_decorator",
    "warp.launch": "warp.launch_function",
    "warp.fem.cells": "warp.fem.cells_function",
    "warp.fem.integrand": "warp.fem.integrand_decorator",
    # Linear solver functions share names with their functor classes (cg vs CG, etc.);
    # suffix the function stubs so they don't collide on case-insensitive filesystems.
    "warp.optim.linear.cg": "warp.optim.linear.cg_function",
    "warp.optim.linear.cr": "warp.optim.linear.cr_function",
    "warp.optim.linear.bicgstab": "warp.optim.linear.bicgstab_function",
    "warp.optim.linear.gmres": "warp.optim.linear.gmres_function",
}

# Map internal builtin function paths to public names for output filenames.
# Autosummary generates stubs under ``warp._src.lang.<func>`` because that is
# where builtins are defined; this mapping rewrites filenames to ``warp.<func>``
# so that published URLs use the public API path.
for _builtin_name in wp._src.context.builtin_functions:
    _internal = f"warp._src.lang.{_builtin_name}"
    _public = f"warp.{_builtin_name}"
    autosummary_filename_map.setdefault(_internal, _public)


def normalize_docstring(doc: str) -> str:
    """Normalize docstrings for consistent RST indentation and formatting."""
    if not doc:
        return ""
    cleaned = inspect.cleandoc(doc)
    if not cleaned:
        return ""
    rst = str(GoogleDocstring(cleaned))
    # Rewrite ``wp.`` aliases in :type:/:rtype: fields so Sphinx cross-references
    # resolve correctly.  Only target field-list lines to avoid mangling code
    # examples that legitimately use ``import warp as wp``.
    return re.sub(r"^(:(rtype|type\s+\w+):.*)\bwp\.", r"\1warp.", rst, flags=re.MULTILINE) if "wp." in rst else rst


class AutosummaryRenderer(AutosummaryRenderer):
    # Module containing Warp's built-ins functions and requiring special handling.
    BUILTINS_TEMPLATE_FILE = "builtins.rst"

    def render(self, template_name, context):
        if template_name == self.BUILTINS_TEMPLATE_FILE:
            fullname = context["fullname"]
            symbol = fullname.split(".")[-1]

            head = wp._src.context.builtin_functions[symbol]

            # Collect all overloads, filtering out hidden ones.
            # Note: head.overloads already includes the head itself (see Function.__init__)
            all_funcs = head.overloads if hasattr(head, "overloads") else [head]
            visible_overloads = [f for f in all_funcs if not f.hidden]

            # Build overload info for each visible overload
            overloads_info = []
            for func in visible_overloads:
                args = {k: wp._src.context.type_str(v) for k, v in func.input_types.items()}

                try:
                    return_type = wp._src.context.type_str(func.value_func(None, None))
                except Exception:
                    return_type = "None"

                if hasattr(func, "overloads"):
                    sig = wp._src.context.resolve_exported_function_sig(func)
                    is_exported = sig is not None
                else:
                    is_exported = False

                overloads_info.append(
                    {
                        "args": ", ".join(f"{k}: {v}" for k, v in args.items()),
                        "return_type": return_type,
                        "is_exported": is_exported,
                        "is_differentiable": func.is_differentiable,
                        "doc": normalize_docstring(func.doc),
                    }
                )

            # Insert metadata that can be accessed from the template.
            context.update(
                {
                    "wp_display_name": f"warp.{symbol}",
                    "wp_overloads": overloads_info,
                }
            )

        return super().render(template_name, context)


sphinx.ext.autosummary.generate.AutosummaryRenderer = AutosummaryRenderer


# -- sphinx.ext.doctest ------------------------------------------------------

# Code to run for every doctest block.
doctest_global_setup = """
from typing import Any
import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
"""


# -- sphinx.ext.extlinks -----------------------------------------------------

extlinks = {
    # Short-hand for :github:`path/to/file.py#L123`.
    "github": (f"https://github.com/NVIDIA/warp/blob/{GITHUB_VERSION}/%s", "%s"),
}


# -- sphinx.ext.intersphinx --------------------------------------------------

# Mapping to external documentation to enable cross-linking (e.g., :class:`numpy.ndarray`)
intersphinx_mapping = {
    "jax": ("https://docs.jax.dev/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "pytorch": ("https://pytorch.org/docs/stable", None),
}


# -- sphinx.ext.linkcode -----------------------------------------------------


def linkcode_resolve(domain, info):
    """Tries to generate external links to code hosted on the Warp GitHub

    This is used for the sphinx.ext.linkcode extension.

    References:
        https://github.com/google/jax/blob/main/docs/conf.py
        https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html
    """
    if domain != "py":
        return None
    if not info["module"]:
        return None
    if not info["fullname"]:
        return None

    try:
        mod = sys.modules.get(info["module"])
        obj = operator.attrgetter(info["fullname"])(mod)
        if isinstance(obj, property):
            obj = obj.fget
        filename = inspect.getsourcefile(obj)
        source, linenum = inspect.getsourcelines(obj)
    except Exception:
        return None

    filename = os.path.relpath(filename, start=os.path.dirname(wp.__file__))
    lines = f"#L{linenum}-L{linenum + len(source)}" if linenum else ""

    return f"https://github.com/NVIDIA/warp/blob/{GITHUB_VERSION}/warp/{filename}{lines}"


# -- sphinx_copybutton -------------------------------------------------------

copybutton_prompt_is_regexp = True  # Tell copybutton the text variable below is a regex.
copybutton_prompt_text = r">>> |\.\.\. |\$ "  # Prompts (>>>, ..., $) to strip when copying code.


# -- Hooks -------------------------------------------------------------------


BUILTIN_DOCSTRINGS = (
    "Convert a string or number to a floating-point number",
    "Return the integer represented by the given array of bytes",
    "bool(x) -> bool",
    "int([x]) -> integer",
)


def filter_builtin_docstrings(app, what, name, obj, options, lines):
    """Remove docstrings inherited from Python built-in types."""
    if not lines:
        return

    for prefix in BUILTIN_DOCSTRINGS:
        if lines[0].startswith(prefix):
            lines.clear()
            return


def rewrite_wp_in_docstrings(app, what, name, obj, options, lines):
    """Rewrite ``wp.`` import aliases to ``warp.`` in docstrings.

    This complements rewrite_wp_aliases (which handles signatures) by also
    fixing docstring content that references ``wp.`` types.  Lines inside
    doctest blocks (``>>>``, ``...``) and RST literal blocks (introduced by
    ``::`` and indented) are skipped to preserve copy-pasteable code examples
    that use ``import warp as wp``.
    """
    if not any("wp." in line for line in lines):
        return

    in_code_block = False
    code_block_indent = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        # Skip doctest lines before checking for ``::`` so that a doctest
        # line ending with ``::`` doesn't falsely trigger code-block mode.
        if stripped.startswith(">>>") or stripped.startswith("..."):
            continue  # preserve doctest examples

        # Detect RST literal code blocks: a line ending with ``::``
        # starts a block after the next blank line.
        if stripped.endswith("::") and not in_code_block:
            in_code_block = True
            code_block_indent = -1  # will be set on first non-blank line
            if "wp." in line:
                lines[i] = _RE_WP_DOT.sub("warp.", line)
            continue

        if in_code_block:
            if code_block_indent == -1:
                # Still looking for the first indented line of the block
                if stripped == "":
                    continue  # blank line between ``::`` and block body
                code_block_indent = len(line) - len(stripped)
                continue  # skip this code line
            # Inside the block: any non-blank line with sufficient indent
            if stripped == "" or (len(line) - len(stripped)) >= code_block_indent:
                continue  # skip code block content
            # Dedented non-blank line ends the block
            in_code_block = False

        if "wp." in line:
            lines[i] = _RE_WP_DOT.sub("warp.", line)


def build_constant_docs_cache():
    """Build a cache of constant docstrings from all warp._src modules."""
    out = {}

    for _, modname, _ in pkgutil.walk_packages(warp._src.__path__, prefix=warp._src.__name__ + "."):
        try:
            module = importlib.import_module(modname)
            source = inspect.getsource(module)
        except Exception:
            continue

        tree = ast.parse(source)

        for idx, node in enumerate(tree.body[:-1]):
            docstring = ""
            next_node = tree.body[idx + 1]
            if isinstance(next_node, ast.Expr):
                value = next_node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    docstring = inspect.cleandoc(value.value)
            if not docstring:
                continue

            # Handle regular assignments (ast.Assign)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        out[target.id] = docstring
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                out[elt.id] = docstring
            # Handle annotated assignments (ast.AnnAssign), e.g. `x: int = 1`
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                out[node.target.id] = docstring

    return out


CONSTANT_DOCS_CACHE = build_constant_docs_cache()


def populate_reexported_docstrings(app, what, name, obj, options, lines):
    """Populate docstrings for re-exported module-level constants.

    When a constant is re-exported via `from module import X as X`, Sphinx's
    autodoc cannot find its docstring. This function looks up the docstring
    from the source module using string literals that follow constant
    assignments.
    """
    if lines:
        # Already has a docstring, nothing to do
        return

    if what != "data":
        return

    # Extract the simple name from the full qualified name
    simple_name = name.split(".")[-1]

    if simple_name in CONSTANT_DOCS_CACHE:
        lines.append(CONSTANT_DOCS_CACHE[simple_name])


def rewrite_internal_module_paths(app, doctree, docname):
    """Replace internal module paths with public paths in the rendered output.

    Replaces all occurrences of '._src.lang.' with '.' in text nodes.
    """
    for node in doctree.traverse(docutils.nodes.Text):
        if "._src.lang." in node.astext():
            new_text = node.astext().replace("._src.lang.", ".")
            node.parent.replace(node, docutils.nodes.Text(new_text))


def rewrite_wp_aliases(app, what, name, obj, options, signature, return_annotation):
    """Rewrite ``wp.`` import aliases to ``warp.`` in autodoc signatures.

    Modules that use ``from __future__ import annotations`` with ``import warp
    as wp`` produce stringified annotations like ``"wp.array"`` instead of
    ``"warp.array"``.  Sphinx cannot resolve the ``wp`` alias, so this hook
    normalises them before cross-reference resolution.
    """

    def _fix(text):
        if text is None:
            return None
        return _RE_WP_DOT.sub("warp.", text)

    return _fix(signature), _fix(return_annotation)


def resolve_wp_aliases(app, env, node, contnode):
    """Resolve ``wp.*`` cross-references by retrying as ``warp.*``.

    When ``from __future__ import annotations`` is active and a module uses
    ``import warp as wp``, stringified annotations like ``wp.array`` end up in
    Sphinx's type-description output.  The signature and docstring hooks cannot
    intercept every code path (e.g. ``autodoc_typehints = "description"``), so
    this ``missing-reference`` handler rewrites the target at resolution time.
    """
    reftarget = node.get("reftarget", "")
    if not reftarget.startswith("wp."):
        return None  # not a wp.* reference, let Sphinx handle it

    new_target = "warp." + reftarget[3:]

    # Save originals so we can restore on failed resolution
    orig_target = reftarget
    orig_text = None
    text_node = None
    if contnode and hasattr(contnode, "children") and contnode.children:
        text_node = contnode.children[0]
        if hasattr(text_node, "astext") and text_node.astext().startswith("wp."):
            orig_text = text_node.astext()

    # Rewrite target and display text
    node["reftarget"] = new_target
    if orig_text is not None:
        text_node.parent.replace(text_node, docutils.nodes.Text("warp." + orig_text[3:]))

    # Re-resolve using Sphinx's domain
    domain = env.get_domain("py")
    result = domain.resolve_xref(
        env, node.get("refdoc", ""), app.builder, node.get("reftype", "class"), new_target, node, contnode
    )

    if result is not None:
        return result

    # Resolution failed — restore original values so Sphinx reports the
    # correct target name in any warning
    node["reftarget"] = orig_target
    if orig_text is not None and text_node is not None:
        new_text_node = contnode.children[0] if contnode.children else None
        if new_text_node is not None:
            new_text_node.parent.replace(new_text_node, docutils.nodes.Text(orig_text))

    return None


def generate_reference_docs(app):
    """Generate API and language reference .rst files before Sphinx reads sources."""
    docs.generate_reference.run()


def drop_autosummary_toctrees(app, doctree):
    # autosummary's `:toctree:` wraps its generated toctree in
    # `autosummary_toc`, a `nodes.comment` subclass. HTML writers skip the
    # wrapper, but Sphinx's TocTreeCollector still descends into it and
    # copies the inner toctree into `env.tocs`, which is what populates the
    # left sidebar. With navigation_depth=2 that exposes every generated
    # stub under its parent API/Language reference page. Remove the wrapper
    # so the toctree never reaches `env.tocs`; call `note_toctree` first so
    # `env.toctree_includes` still records the stubs and they aren't flagged
    # as orphan documents.
    #
    # Must run before TocTreeCollector (doctree-read, default priority 500;
    # lower priority runs first), hence priority=400 in setup().
    docname = app.env.docname
    for asum in list(doctree.findall(autosummary_toc)):
        for tocnode in asum.findall(addnodes.toctree):
            note_toctree(app.env, docname, tocnode)
        asum.parent.remove(asum)


def setup(app):
    """Sphinx extension setup."""
    # Priority must be lower than autosummary's default (500) so that the
    # reference .rst files exist before autosummary scans for stub directives.
    app.connect("builder-inited", generate_reference_docs, priority=400)
    app.connect("autodoc-process-docstring", filter_builtin_docstrings)
    app.connect("autodoc-process-docstring", rewrite_wp_in_docstrings)
    app.connect("autodoc-process-docstring", populate_reexported_docstrings)
    app.connect("autodoc-process-signature", rewrite_wp_aliases)
    app.connect("missing-reference", resolve_wp_aliases)
    # Lower priority runs first; this must precede TocTreeCollector
    # (default 500) so the autosummary wrappers are gone before it
    # populates `env.tocs`.
    app.connect("doctree-read", drop_autosummary_toctrees, priority=400)
    app.connect("doctree-resolved", rewrite_internal_module_paths)
