# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import ast
import enum
import functools
import importlib
import importlib.util
import inspect
import json
import operator
import os
import pkgutil
import re
import sys
from typing import TypeVar

import docutils
import sphinx
import sphinx.util.logging
from sphinx import addnodes
from sphinx.environment.adapters.toctree import note_toctree
from sphinx.ext.autodoc.preserve_defaults import update_defvalue
from sphinx.ext.autosummary import autosummary_toc
from sphinx.ext.autosummary.generate import AutosummaryRenderer
from sphinx.util.inspect import signature as sphinx_signature
from sphinx.util.inspect import stringify_signature
from sphinx.util.typing import stringify_annotation

_RE_WP_DOT = re.compile(r"\bwp\.")

# -- Path setup --------------------------------------------------------------

HERE = os.path.dirname(__file__)
WARP_PATH = os.path.realpath(os.path.join(HERE, ".."))
sys.path.insert(0, WARP_PATH)
sys.path.insert(0, os.path.join(HERE, "_ext"))

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
    # Local extensions, from `docs/_ext`.
    "wp_function_tags",  # Renders the property tags and source links of Warp functions.
    "wp_doctest_shard",  # Runs selected doctest documents in isolated subprocess shards.
]

# Generate targets for Markdown headings through level 2 so standard fragment
# links in included GitHub-flavored Markdown resolve as Sphinx references.
myst_heading_anchors = 2

# Enable nitpicky mode to warn about unresolved references.
nitpicky = True

# Suppress warnings for types that Sphinx cannot resolve due to structural
# limitations (dynamic ctypes types, py:data/py:class role mismatches,
# mocked dependencies, runtime repr leaking into signatures, etc.).
nitpick_ignore_regex = [
    # Public type aliases indexed as py:data but referenced as py:class (Sphinx limitation)
    (r"py:class", r"(warp\.|wp\.)?(Scalar|Int|Float|DeviceLike)"),
    # numpy.typing aliases used in annotations but imported only under TYPE_CHECKING
    (r"py:class", r"npt\.ArrayLike"),
    # Internal meta-types used in builtin function signatures (not exported)
    (
        r"py:class",
        r"(Vector|Quaternion|Matrix|Array|Transformation|Tile|TileStack|IndexedArray|IndexedFabricArray|FabricArray|Shape|DType|Any)",
    ),
    # Private codegen query subtypes — visible in Sphinx RST (generated from Python source)
    # but not in __init__.pyi (rewritten by export_stubs) or the public API.
    (r"py:class", r"(_BvhQueryAabb|_BvhQueryRay|_BvhQueryCapsule|_BvhQuerySphere|_MeshQuerySphere)"),
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
        r"LaunchBounds|launch_bounds_t|FieldRestriction|scalar)",
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
    # Members inherited from IsoSurfaceBase, listed in member tables but documented on the base class
    (r"py:obj", r"warp\.geometry\.IsoSurface(MarchingCubes|Nets)\.(resize|surface|extract)"),
    # KrylovSolve inherits reset_stats from a private base; the user guide documents it.
    (r"py:obj", r"warp\.optim\.linear\.KrylovSolve\.reset_stats"),
    # MultiRHSOperator inherits these documented LinearOperator attributes.
    (
        "py:obj",
        r"warp\.optim\.linear\.MultiRHSOperator\.(batch_count|batch_offsets|device|dtype|matvec|max_batch_length|scalar_type|shape)",
    ),
    # FSAI inherits these attributes; their documentation is on LinearOperator.
    (
        r"py:obj",
        r"warp\.optim\.linear\.FSAI\.(batch_count|batch_offsets|device|dtype|matvec|max_batch_length|scalar_type|shape)",
    ),
    # Everything on the deprecated `wp.MarchingCubes` alias is inherited from, and
    # documented on, `warp.geometry.IsoSurfaceMarchingCubes`
    (r"py:(obj|attr)", r"warp\.MarchingCubes\..*"),
    (
        r"py:attr",
        r"(CUBE_CORNER_OFFSETS|EDGE_TO_CORNERS|CASE_TO_TRI_RANGE|TRI_LOCAL_INDICES)",
    ),
    # Internal C++/Python interop methods on geometric types
    (
        r"py:obj",
        r"warp\.(vec|mat|quat|transform|spatial_vector|spatial_matrix)\w*\.(from_ptr|scalar_export|scalar_import)",
    ),
    # Matrix accessor methods
    (r"py:obj", r"warp\.(mat|spatial_matrix)\w*\.(get_col|get_row|set_col|set_row)"),
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

# Enable the version switcher when DOC_VERSION is set (CI builds only).
# Local builds without DOC_VERSION will not render the switcher.
# nvidia_sphinx_theme places the switcher in navbar_center by default once
# `switcher` is configured; explicitly setting `navbar_end` would put a
# second copy there and render two "Choose Version" dropdowns side by side.
#
# `show_version_warning_banner` reads versions.json at runtime and renders
# the "this is an older version" banner whenever the page's `version_match`
# differs from the `preferred` entry. Bumping `preferred` in versions.json
# (a one-line JSON edit during the next release) makes every archived
# version's docs surface the banner without a rebuild.
doc_version = os.environ.get("DOC_VERSION", "")
if doc_version:
    html_theme_options.update(
        {
            "check_switcher": False,
            "switcher": {
                "json_url": "https://nvidia.github.io/warp/versions.json",
                "version_match": doc_version,
            },
            "show_version_warning_banner": True,
        }
    )

    # Override Sphinx's ``release`` (which feeds ``DOCUMENTATION_OPTIONS.VERSION``,
    # used by the PyData warning banner) for release-doc builds. The docs collapse
    # patch releases into ``/vMAJOR.MINOR/`` and ``versions.json`` records the
    # preferred entry as ``"version": "MAJOR.MINOR"``; without this override, a
    # patch like ``1.12.1`` semver-compares strictly greater than the preferred
    # ``1.12``, so the banner classifies the stable docs as "an unstable
    # development version" and links "Switch to stable version" back to itself.
    # We leave ``version`` (used in ``html_title``) at ``wp.__version__`` so the
    # navbar still shows the precise patch version.
    if re.fullmatch(r"\d+\.\d+", doc_version):
        release = doc_version

html_title = f"Warp {version}"
html_context = {
    "github_user": "NVIDIA",
    "github_repo": "warp",
    "github_version": GITHUB_VERSION,
    "doc_path": "docs",
}
html_css_files = ["custom.css"]
html_js_files = ["copy-page-source.js"]
html_static_path = ["_static"]
html_copy_source = True
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

# Merge the ``__init__`` docstring into the class description. Many Warp classes
# (e.g. ``warp.array``, ``warp.Mesh``, ``warp.fem`` geometries) document their
# constructor arguments on ``__init__`` rather than the class docstring. The
# autosummary class template intentionally omits ``.. automethod:: __init__``
# (it fails to resolve for the dynamically generated ctypes vec/mat/quat types),
# so this setting is what keeps those constructor arguments in the rendered docs.
autoclass_content = "both"

# Show typehints as content of the function or method instead of in the signature.
autodoc_typehints = "description"

# Show the literal expression for default arguments instead of evaluating them.
autodoc_preserve_defaults = True

# Prevent docstrings from being inherited from parent classes or methods.
autodoc_inherit_docstrings = False


# Work around a Sphinx 9 formatting bug: when a class is documented as an
# attribute alias (``doc_as_attr``), ``sphinx.ext.autodoc._generate`` appends
# the "alias of ..." note directly after the directive's content without a
# separating blank line. Our public vec/mat/quat/transform and ``warp.fem``
# index-type aliases carry autosummary "Methods"/"Attributes" tables, so the
# unindented "alias of" line runs into the preceding ``.. autosummary::``
# block and docutils reports "Explicit markup ends without a blank line".
# Prepend a blank line so the generated reST stays well-formed. The private
# ``_generate`` module was added in Sphinx 9. Keep the guard for environments
# using an older Sphinx release.
try:
    import sphinx.ext.autodoc._generate as _autodoc_generate
except ModuleNotFoundError as e:
    if e.name != "sphinx.ext.autodoc._generate":
        raise
else:
    if hasattr(_autodoc_generate, "_body_alias_lines"):
        _orig_body_alias_lines = _autodoc_generate._body_alias_lines

        def _body_alias_lines_with_blank(**kwargs):
            emitted = False
            for line in _orig_body_alias_lines(**kwargs):
                if not emitted:
                    yield ""  # separate the alias note from any preceding directive content
                    emitted = True
                yield line

        _autodoc_generate._body_alias_lines = _body_alias_lines_with_blank


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

AUTOSUMMARY_ANNOTATION_OVERRIDES = {
    "warp.config.enable_mempools_at_init": ": bool = True",
    "warp.config.launch_array_access_mode": (
        ": warp.config.LaunchArrayAccessMode = warp.config.LaunchArrayAccessMode.RELAXED"
    ),
    "warp.config.log_level": ": int = warp.LOG_INFO",
    # Hide the internal module path that ``repr()`` exposes.
    "warp.DeviceLike": "= warp.Device | str | None",
}

# Map internal builtin function paths to public names for output filenames.
# Autosummary generates stubs under ``warp._src.lang.<func>`` because that is
# where builtins are defined; this mapping rewrites filenames to ``warp.<func>``
# so that published URLs use the public API path.
for _builtin_name in wp._src.context.builtin_functions:
    _internal = f"warp._src.lang.{_builtin_name}"
    _public = f"warp.{_builtin_name}"
    autosummary_filename_map.setdefault(_internal, _public)


from docs._ext.warp_function_reference import describe_function, source_url  # noqa: E402


def _is_warp_module(module_name):
    return module_name == "warp" or module_name.startswith("warp.")


def _is_foreign_member(cls, member):
    """Return whether ``member`` is first defined outside Warp, such as by ``int`` or ``enum.Enum``."""
    for owner in inspect.getmro(cls):
        if member in vars(owner):
            return not _is_warp_module(owner.__module__)

    # Instance attributes that the source analyzer found belong to Warp.
    return False


def _get_documented_class(context):
    module_name = context.get("module")
    object_name = context.get("objname")
    if not module_name or not object_name:
        return None

    try:
        module = importlib.import_module(module_name)
        cls = operator.attrgetter(object_name)(module)
    except (AttributeError, ImportError):
        return None

    return cls if inspect.isclass(cls) else None


def _get_class_alias_doc(context, cls):
    """Return the docstring of a class documented under another name, such as ``warp.fem.ElementIndex``.

    Autodoc renders such an alias as "alias of int", and looks for its docstring
    only in the module that re-exports it. The docstring that follows the
    assignment in ``warp._src`` never reaches the page, so the template adds it.
    """
    name = context["objname"].rpartition(".")[2]
    if cls.__name__ == name:
        return None

    return find_constant_doc(context["fullname"])


def _filter_class_members(context, cls):
    if cls is wp.fixedarray:
        # fixedarray is a compiler-only type descriptor. Its inherited array API
        # and internal vars property are not part of its public interface.
        context["methods"] = []
        context["attributes"] = []
        return

    for member_group in ("methods", "attributes"):
        context[member_group] = [
            member for member in context.get(member_group, ()) if not _is_foreign_member(cls, member)
        ]


def _is_type_var(fullname):
    """Return whether ``fullname`` resolves to a :class:`typing.TypeVar`."""
    module_name, _, attr_name = fullname.rpartition(".")
    try:
        obj = getattr(importlib.import_module(module_name), attr_name)
    except (ImportError, AttributeError, ValueError):
        return False

    return isinstance(obj, TypeVar)


class AutosummaryRenderer(AutosummaryRenderer):
    # Template documenting built-ins and other Warp functions, requiring special handling.
    WARP_FUNCTION_TEMPLATE_FILE = "warp_function.rst"

    def render(self, template_name, context):
        context["wp_annotation_override"] = AUTOSUMMARY_ANNOTATION_OVERRIDES.get(context.get("fullname"))
        context["wp_no_value"] = _is_type_var(context.get("fullname", ""))

        if context.get("objtype") == "class" and (cls := _get_documented_class(context)) is not None:
            _filter_class_members(context, cls)
            context["wp_alias_doc"] = _get_class_alias_doc(context, cls)

        # Autosummary classifies a `warp._src.context.Function` as data, which renders
        # its `repr()` in place of a signature.  Route the registered ones through the
        # function template instead so they document like the built-ins, without
        # having to split the module pages into per-template blocks.
        if context.get("fullname") in wp._src.context.api_functions:
            template_name = self.WARP_FUNCTION_TEMPLATE_FILE

        if template_name == self.WARP_FUNCTION_TEMPLATE_FILE:
            fullname = context["fullname"]
            symbol = fullname.split(".")[-1]
            module = context["module"]

            # Module-scoped API functions keep their own namespace; built-ins are
            # documented from the `warp._src.lang` placeholder module and published
            # under the flat `warp.` namespace instead.
            publication = wp._src.context.api_functions.get(f"{module}.{symbol}")
            if publication is None and module == "warp._src.lang":
                publication = wp._src.context.api_functions.get(f"warp.{symbol}")
            if publication is None:
                func = wp._src.context.builtin_functions[symbol]
                display_name = f"warp.{symbol}"
            else:
                func = publication.function
                display_name = publication.public_name

            # Insert metadata that can be accessed from the template.
            context.update(
                {
                    "wp_display_name": display_name,
                    "wp_module": module,
                    "wp_overloads": describe_function(func, publication, GITHUB_VERSION),
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

wp.config.log_level = wp.LOG_WARNING
wp.init()
"""


# -- sphinx.ext.extlinks -----------------------------------------------------

extlinks = {
    # Short-hand for :github:`path/to/file.py#L123`.
    "github": (f"https://github.com/NVIDIA/warp/blob/{GITHUB_VERSION}/%s", "%s"),
}


# -- sphinx.ext.intersphinx --------------------------------------------------

# Intersphinx resolves external cross-references (e.g. :class:`numpy.ndarray`)
# by fetching each project's objects.inv over the network. Builds are lenient by
# default so an unreachable inventory doesn't hard-fail a local build; CI passes
# --warnings-as-errors to promote stale/unreachable URLs to errors.
_intersphinx_mapping = {
    "jax": ("https://docs.jax.dev/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "pytorch": ("https://docs.pytorch.org/docs/stable", None),
}

# Fail fast on unreachable inventories instead of hanging on the default socket
# timeout (seconds).
intersphinx_timeout = 2

# PyTorch's ``stable`` documentation URLs are client-side redirect stubs that
# preserve fragments in browsers. The linkcheck builder cannot follow those
# redirects before validating anchors, so only skip anchor checks for them;
# URL availability is still checked.
linkcheck_anchors_ignore_for_url = [r"https://docs\.pytorch\.org/docs/stable/.*"]

_sphinx_logger = sphinx.util.logging.getLogger(__name__)


def _with_fallback_inventories(mapping, fallback_dir):
    """Add cached inventories from ``fallback_dir`` as second locations.

    CI can provide cached copies of the inventories, named ``<name>.inv``, so
    that a slow or unreachable docs site does not fail a warnings-as-errors
    build. Sphinx tries the URL first and reads the cached copy only if that
    fails, which it reports at INFO level. The names must match the files CI
    provides (``fetch-intersphinx-fallback`` in ``.gitlab/ci/common.yml``).
    """
    if not fallback_dir:
        return mapping
    result = {}
    for name, (target_uri, inventory) in mapping.items():
        cached = os.path.join(fallback_dir, f"{name}.inv")
        result[name] = (target_uri, (inventory, cached)) if os.path.isfile(cached) else (target_uri, inventory)
    with_fallback = sorted(name for name, (_, inventory) in result.items() if isinstance(inventory, tuple))
    _sphinx_logger.info(f"intersphinx: cached fallback inventories for {', '.join(with_fallback) or 'none'}")
    return result


# WARP_DOCS_OFFLINE=1 skips external resolution entirely for known-offline builds.
if os.environ.get("WARP_DOCS_OFFLINE") == "1":
    _sphinx_logger.info("intersphinx: WARP_DOCS_OFFLINE=1 set; skipping external cross-reference resolution.")
    intersphinx_mapping = {}
else:
    intersphinx_mapping = _with_fallback_inventories(
        _intersphinx_mapping, os.environ.get("WARP_INTERSPHINX_FALLBACK_DIR")
    )


# -- sphinx.ext.linkcode -----------------------------------------------------


def linkcode_resolve(domain, info):
    """Tries to generate external links to code hosted on the Warp GitHub

    This is used for the sphinx.ext.linkcode extension.

    References:
        https://github.com/jax-ml/jax/blob/main/docs/conf.py
        https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html
    """
    if domain != "py":
        return None
    if not info["module"]:
        return None
    if not info["fullname"]:
        return None

    if info["module"] == "warp._src.lang" or f"{info['module']}.{info['fullname']}" in wp._src.context.api_functions:
        return None  # The Warp function template links each overload to its own source.

    try:
        mod = sys.modules.get(info["module"])
        obj = operator.attrgetter(info["fullname"])(mod)
    except Exception:
        return None
    if isinstance(obj, property):
        obj = obj.fget

    return source_url(obj, GITHUB_VERSION)


# -- sphinx_copybutton -------------------------------------------------------

copybutton_prompt_is_regexp = True  # Tell copybutton the text variable below is a regex.
copybutton_prompt_text = r">>> |\.\.\. |\$ "  # Prompts (>>>, ..., $) to strip when copying code.


# -- Hooks -------------------------------------------------------------------


def drop_value_type_docstrings(app, what, name, obj, options, lines):
    """Drop a data docstring that autodoc read from the type of the value.

    Without a docstring next to a re-export, autodoc falls back to
    ``obj.__doc__``, which for values such as ``warp.PI``, ``warp.LOG_INFO``,
    or ``warp.Scalar`` is the docstring of ``float``, ``int``, or
    ``typing.TypeVar``. Clearing it lets ``populate_reexported_docstrings``
    add the docstring of the Warp constant instead.
    """
    if what not in ("data", "attribute") or inspect.isclass(obj) or inspect.isroutine(obj):
        return

    # Compare with ``==``: built-in types build a new ``__doc__`` string on each access.
    type_doc = getattr(type(obj), "__doc__", None)
    if not isinstance(type_doc, str) or getattr(obj, "__doc__", None) != type_doc:
        return

    # A docstring that the source analyzer found for the attribute stays.
    if "\n".join(lines).strip() == inspect.cleandoc(type_doc).strip():
        lines.clear()


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
    """Map each documented constant in ``warp._src`` to its docstring, keyed by its qualified name.

    A key names the module that assigns the constant, such as
    ``warp._src.fem.types.Coords``, so that constants with the same name in
    different modules keep their own docstrings.
    """
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
                        out[f"{modname}.{target.id}"] = docstring
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                out[f"{modname}.{elt.id}"] = docstring
            # Handle annotated assignments (ast.AnnAssign), e.g. `x: int = 1`
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                out[f"{modname}.{node.target.id}"] = docstring

    return out


CONSTANT_DOCS_CACHE = build_constant_docs_cache()


def _find_import_source(module_name, name):
    """Return the ``(module, name)`` that ``module_name`` imports ``name`` from, or ``None``."""
    module = importlib.import_module(module_name)
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError):
        return None

    # The last import that binds the name wins, as at runtime.
    for node in reversed(tree.body):
        if not isinstance(node, ast.ImportFrom):
            continue

        source = importlib.util.resolve_name("." * node.level + (node.module or ""), module.__package__)
        for alias in node.names:
            if alias.name == "*":
                source_module = importlib.import_module(source)
                # Without ``__all__``, a star import binds every name that doesn't start with an underscore.
                exported_names = getattr(source_module, "__all__", None)
                if exported_names is None:
                    exported_names = [x for x in vars(source_module) if not x.startswith("_")]

                if name in exported_names:
                    return source, name
            elif (alias.asname or alias.name) == name:
                return source, alias.name

    return None


@functools.cache
def find_constant_doc(fullname):
    """Return the docstring of the constant that a public name, such as ``warp.fem.Coords``, refers to.

    Follows the imports from the public module to the ``warp._src`` module
    that assigns the constant. Warns when the imports stop outside
    ``warp._src`` while a ``warp._src`` constant with the same name has a
    docstring, since the page would otherwise lose it without notice.
    """
    module_name, _, name = fullname.rpartition(".")
    while (qualname := f"{module_name}.{name}") not in CONSTANT_DOCS_CACHE:
        source = _find_import_source(module_name, name)
        if source is None:
            # Inside ``warp._src``, the imports stop where the constant is defined, so it has no docstring.
            # Elsewhere, a re-export that they can't follow, such as ``FOO = _x.FOO``, hides the docstring.
            if f"{module_name}.".startswith("warp._src."):
                return None

            if candidates := sorted(x for x in CONSTANT_DOCS_CACHE if x.rpartition(".")[2] == name):
                _sphinx_logger.warning(
                    "cannot find the docstring of %s: its imports don't lead to %s. "
                    "Re-export it with `from <module> import %s as %s`.",
                    fullname,
                    " or ".join(candidates),
                    name,
                    fullname.rpartition(".")[2],
                )
            return None

        module_name, name = source

    return CONSTANT_DOCS_CACHE[qualname]


def populate_reexported_docstrings(app, what, name, obj, options, lines):
    """Populate docstrings for re-exported module-level constants.

    When a constant is re-exported via `from module import X as X`, Sphinx's
    autodoc cannot find its docstring. This function follows the imports from
    the public name to the module that assigns the constant, and uses the
    string literal that follows the assignment.
    """
    if what != "data":
        return

    if lines:
        # Already has a docstring, nothing to do
        return

    if (docstring := find_constant_doc(name)) is not None:
        lines.extend(docstring.splitlines())


def rewrite_internal_module_paths(app, doctree, docname):
    """Replace internal module paths with public paths in the rendered output.

    Replaces all occurrences of '._src.lang.' with '.' in text nodes.
    """
    for node in doctree.traverse(docutils.nodes.Text):
        if "._src.lang." in node.astext():
            new_text = node.astext().replace("._src.lang.", ".")
            node.parent.replace(node, docutils.nodes.Text(new_text))


def _is_pass_through(func):
    """Return whether ``func`` takes only ``*args, **kwargs`` after its first parameter."""
    try:
        params = tuple(inspect.signature(func).parameters.values())[1:]
    except (TypeError, ValueError):
        return False

    return tuple(param.kind for param in params) == (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _get_hidden_init(cls):
    """Return the Warp ``__init__`` hidden behind a Warp ``__new__`` that passes its arguments through."""
    new, init = cls.__new__, cls.__init__
    if not inspect.isfunction(new) or not _is_warp_module(new.__module__) or not _is_pass_through(new):
        return None

    if not inspect.isfunction(init) or not _is_warp_module(init.__module__) or _is_pass_through(init):
        return None

    return init


def format_class_signature(app, what, name, obj, options, signature, return_annotation):
    """Replace class signatures that come from Python or hide the constructor arguments.

    Enums get ``(*values)`` from :class:`enum.EnumType`, and protocols get
    ``(*args, **kwargs)`` from :class:`typing.Protocol`. Neither is how Warp
    users create them, so these classes show no signature.

    Some Warp classes define ``__new__(cls, *args, **kwargs)`` only to preset
    attributes that ``__del__`` reads. Sphinx and :func:`inspect.signature`
    prefer ``__new__`` over ``__init__``, so these classes show the signature
    and the parameter types of their ``__init__`` instead.
    """
    if what != "class" or not inspect.isclass(obj):
        return None

    if issubclass(obj, enum.Enum) or getattr(obj, "_is_protocol", False):
        return "", return_annotation

    init = _get_hidden_init(obj)
    # Sphinx's own store for the parameter types it lists in the description.
    # It's undocumented: if it moves, keep Sphinx's signature rather than fail.
    annotations = getattr(getattr(app.env, "current_document", None), "autodoc_annotations", None)
    if init is None or annotations is None:
        return None

    update_defvalue(app, init, bound_method=True)
    sig = sphinx_signature(init, bound_method=True, type_aliases=app.config.autodoc_type_aliases)

    # Sphinx has already recorded the parameter types of `__new__`, before this
    # event; record those of `__init__` the same way.
    short = app.config.autodoc_typehints_format == "short"
    short_literals = app.config.python_display_short_literal_types
    recorded = annotations.setdefault(name, {})
    for param in sig.parameters.values():
        if param.annotation is not param.empty:
            recorded[param.name] = stringify_annotation(
                param.annotation, "smart" if short else "fully-qualified", short_literals=short_literals
            )

    args = stringify_signature(
        sig,
        show_annotation=app.config.autodoc_typehints not in ("none", "description"),
        show_return_annotation=False,
        unqualified_typehints=short,
        short_literals=short_literals,
    )
    return _RE_WP_DOT.sub("warp.", args), return_annotation


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


_RE_REPR_ADDRESS = re.compile(r"<([\w.]+) object at 0x[0-9a-f]+>")


def strip_repr_addresses(app, doctree, docname):
    """Drop memory addresses from default-repr leaks in the resolved doctree.

    A handful of Warp attributes are documented without explicit type
    annotations, so autodoc falls back to ``repr(obj)`` which for objects
    without ``__repr__`` produces strings like ``<warp.types.array object
    at 0x70faf939a250>`` (for class-level array sentinels) or ``<property
    object at 0x...>`` (for descriptors).  The address differs every Python
    process, which makes the rendered HTML byte-unstable across builds —
    every push to ``main`` then writes a different ``gh-pages`` tree even
    when the docs haven't changed.

    ``sphinx.ext.autodoc.typehints`` injects these directly into the doctree
    via the ``object-description-transform`` event, bypassing the
    ``autodoc-process-signature`` and ``autodoc-process-docstring`` hooks,
    so the cleanup has to happen at the doctree level.
    """
    for text_node in list(doctree.findall(docutils.nodes.Text)):
        original = text_node.astext()
        if "object at 0x" not in original:
            continue
        cleaned = _RE_REPR_ADDRESS.sub(r"<\1 object>", original)
        if cleaned != original:
            text_node.parent.replace(text_node, docutils.nodes.Text(cleaned))


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


def resolve_public_builtin_aliases(app, env, node, contnode):
    """Resolve public ``warp.*`` built-in refs to their generated documents."""
    reftarget = node.get("reftarget", "")
    if node.get("reftype") != "func" or not reftarget.startswith("warp."):
        return None

    builtin_name = reftarget[5:]
    if "." in builtin_name:
        return None

    from warp._src.context import builtin_functions  # noqa: PLC0415

    if builtin_name not in builtin_functions:
        return None

    new_target = f"warp._src.lang.{builtin_name}"
    node["reftarget"] = new_target

    domain = env.get_domain("py")
    result = domain.resolve_xref(env, node.get("refdoc", ""), app.builder, "func", new_target, node, contnode)

    node["reftarget"] = reftarget
    return result


def generate_reference_docs(app):
    """Generate API and language reference .rst files before Sphinx reads sources."""
    # The coordinator prepares the shared source tree before starting workers.
    if app.builder.name == "doctest-shard":
        return
    docs.generate_reference.run()


def hide_generated_api_edit_link(_app, pagename, _templatename, context, _doctree):
    """Hide edit links for generated API stubs that are not tracked in Git."""
    if pagename.startswith("api_reference/_generated/"):
        context["theme_use_edit_page_button"] = False


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


# Redirects for documentation pages that moved or were removed during the
# navigation reorganization, so old URLs (bookmarks, external links, search
# results) keep resolving. Maps an old page path without the ``.html`` suffix
# to its new target, both relative to the docs root and without a leading
# slash. Targets include ``.html`` and may include an optional ``#anchor``,
# used when the old URL has no fragment of its own. GitHub Pages has no
# server-side redirects and the site is served with ``.nojekyll`` (so
# ``jekyll-redirect-from`` cannot run), hence
# we emit redirect stubs at build time. Targets are rewritten relative to each
# old page so the stubs work unchanged under any version prefix (``/stable/``,
# ``/latest/``, ``/vX.Y/``).
DOC_REDIRECTS = {
    "user_guide/tiles": "user_guide/programming_model/tiles.html",
    "user_guide/generics": "user_guide/programming_model/generics.html",
    "user_guide/cpp_cuda_workflows": "user_guide/programming_model/cpp_cuda_workflows.html",
    "deep_dive/codegen": "user_guide/programming_model/code_generation.html",
    "deep_dive/concurrency": "user_guide/execution_and_performance/concurrency.html",
    "deep_dive/allocators": "user_guide/execution_and_performance/memory_management.html",
    "deep_dive/profiling": "user_guide/execution_and_performance/profiling.html",
    "user_guide/deterministic_execution": "user_guide/execution_and_performance/deterministic_execution.html",
    "user_guide/interoperability_jax": "user_guide/interoperability/jax.html",
    "user_guide/interoperability_pytorch": "user_guide/interoperability/pytorch.html",
    "user_guide/changelog": "project/changelog.html",
    "user_guide/contribution_guide": "project/contribution_guide.html",
    "user_guide/publications": "project/publications.html",
    "deep_dive/memory_access": "user_guide/execution_and_performance/memory_management.html",
    "user_guide/devices": "user_guide/runtime.html#devices",
    # `warp.build_experimental` moved to `warp.experimental.build` in Warp 1.20.
    "api_reference/warp_build_experimental": "api_reference/warp_experimental_build.html",
    "api_reference/_generated/warp.build_experimental.add_builtin": (
        "api_reference/_generated/warp.experimental.build.add_builtin.html"
    ),
    "api_reference/_generated/warp.build_experimental.add_native_type": (
        "api_reference/_generated/warp.experimental.build.add_native_type.html"
    ),
}

# Legacy fragments whose content moved to a different page or whose generated
# heading ID changed. All other incoming fragments are preserved as-is.
DOC_FRAGMENT_REDIRECTS = {
    "deep_dive/allocators": {
        "allocators": "user_guide/execution_and_performance/memory_management.html#memory-allocation-and-access",
        "introduction": "user_guide/execution_and_performance/memory_management.html#stream-ordered-memory-pool-allocators",
    },
    "deep_dive/memory_access": {
        "cpu-gpu-cross-device-memory-access": (
            "user_guide/execution_and_performance/memory_management.html#cross-device-memory-access"
        ),
        "launching-with-arrays-on-the-same-device": (
            "user_guide/execution_and_performance/memory_management.html#launch-device-versus-allocation-device"
        ),
        "launching-gpu-kernels-with-cpu-arrays": (
            "user_guide/execution_and_performance/memory_management.html#gpu-kernels-using-cpu-arrays"
        ),
        "accessing-gpu-data-from-cpu-code": (
            "user_guide/execution_and_performance/memory_management.html#cpu-code-using-gpu-resident-data"
        ),
        "checking-access-for-a-specific-array-with-wp-can-access": (
            "user_guide/execution_and_performance/memory_management.html#checking-a-concrete-array-with-wp-can-access"
        ),
        "checking-array-access-before-launch": (
            "user_guide/execution_and_performance/memory_management.html#checked-launch-validation"
        ),
    },
    "user_guide/devices": {
        "cuda-peer-access": "user_guide/execution_and_performance/memory_management.html#cuda-peer-access",
        "peer-access": "user_guide/execution_and_performance/memory_management.html#peer-access",
    },
}


def write_redirect_stubs(app, exception):
    """Emit meta-refresh stubs at moved/removed pages' old URLs (see ``DOC_REDIRECTS``)."""
    if exception is not None or app.builder.name != "html":
        return
    logger = sphinx.util.logging.getLogger(__name__)
    written = 0
    for old, new in DOC_REDIRECTS.items():
        dest = os.path.join(app.outdir, old + ".html")
        if os.path.exists(dest):
            logger.warning("skipping redirect for %s: a real page exists there", old)
            continue
        target, sep, frag = new.partition("#")
        if not os.path.exists(os.path.join(app.outdir, target)):
            logger.warning("redirect for %s targets missing page %s", old, target)
        url = os.path.relpath(target, os.path.dirname(old)).replace(os.sep, "/") + sep + frag
        script_url = json.dumps(url).replace("<", r"\u003c")
        fragment_urls = {}
        for old_fragment, fragment_target in DOC_FRAGMENT_REDIRECTS.get(old, {}).items():
            fragment_page, fragment_sep, fragment_anchor = fragment_target.partition("#")
            if not os.path.exists(os.path.join(app.outdir, fragment_page)):
                logger.warning("fragment redirect for %s#%s targets missing page %s", old, old_fragment, fragment_page)
            fragment_urls[f"#{old_fragment}"] = (
                os.path.relpath(fragment_page, os.path.dirname(old)).replace(os.sep, "/")
                + fragment_sep
                + fragment_anchor
            )
        script_fragment_urls = json.dumps(fragment_urls, sort_keys=True).replace("<", r"\u003c")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(
                '<!DOCTYPE html>\n<html><head><meta charset="utf-8">'
                f'<link rel="canonical" href="{url}">'
                f"<script>const target = new URL({script_url}, window.location.href);"
                f"const fragmentTargets = {script_fragment_urls};"
                "const fragmentTarget = fragmentTargets[window.location.hash];"
                "if (fragmentTarget) target.href = new URL(fragmentTarget, window.location.href).href;"
                "else if (window.location.hash) target.hash = window.location.hash;"
                "window.location.replace(target.href);</script>"
                f'<noscript><meta http-equiv="refresh" content="0; url={url}"></noscript></head>'
                f'<body>Redirecting to <a href="{url}">{url}</a>&hellip;</body></html>\n'
            )
        written += 1
    logger.info("wrote %d documentation redirect stub(s)", written)


def setup(app):
    """Sphinx extension setup."""
    # Priority must be lower than autosummary's default (500) so that the
    # reference .rst files exist before autosummary scans for stub directives.
    app.connect("builder-inited", generate_reference_docs, priority=400)
    app.connect("html-page-context", hide_generated_api_edit_link)
    # Before napoleon (priority 500) rewrites the docstring of the value's type,
    # and before `populate_reexported_docstrings` fills the cleared lines.
    app.connect("autodoc-process-docstring", drop_value_type_docstrings, priority=400)
    app.connect("autodoc-process-docstring", rewrite_wp_in_docstrings)
    app.connect("autodoc-process-docstring", populate_reexported_docstrings)
    # Sphinx keeps the first handler result, and `rewrite_wp_aliases` always
    # returns one, so this handler must run before it.
    app.connect("autodoc-process-signature", format_class_signature, priority=400)
    app.connect("autodoc-process-signature", rewrite_wp_aliases)
    app.connect("missing-reference", resolve_wp_aliases)
    app.connect("missing-reference", resolve_public_builtin_aliases)
    # Lower priority runs first; this must precede TocTreeCollector
    # (default 500) so the autosummary wrappers are gone before it
    # populates `env.tocs`.
    app.connect("doctree-read", drop_autosummary_toctrees, priority=400)
    app.connect("doctree-resolved", rewrite_internal_module_paths)
    app.connect("doctree-resolved", strip_repr_addresses)
    app.connect("build-finished", write_redirect_stubs)
