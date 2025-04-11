# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import inspect
import operator
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warp as wp

# -- Project information -----------------------------------------------------

project = "Warp"
version = wp.__version__
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",  # Parse markdown files
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Convert docstrings to reStructuredText
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",  # Markup to shorten external links
    "sphinx.ext.githubpages",
    "sphinx.ext.linkcode",  # Add GitHub source code links
    "sphinx.ext.doctest",  # Test code snippets in docs
    # Third-party extensions:
    "sphinx_copybutton",
    # 'sphinx_tabs.tabs',
    #    'autodocsumm'
]

# put type hints inside the description instead of the signature (easier to read)
autodoc_typehints = "description"
# default argument values of functions will be not evaluated on generating document
autodoc_preserve_defaults = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "exclude-members": "__weakref__",
}

# Mock imports for modules that are not installed by default
autodoc_mock_imports = ["jax", "torch", "paddle", "pxr"]

# autodoc_typehints_format
# add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "pytorch": ("https://pytorch.org/docs/stable", None),
}

extlinks = {
    "github": ("https://github.com/NVIDIA/warp/blob/main/%s", "%s"),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

doctest_global_setup = """
from typing import Any
import numpy as np
import warp as wp
wp.config.quiet = True
wp.init()
"""


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

    return f"https://github.com/NVIDIA/warp/blob/v{version}/warp/{filename}{lines}"


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# sphinx_copybutton settings
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "nvidia_sphinx_theme"
html_title = f"Warp {version}"
html_show_sphinx = False
html_theme_options = {
    "copyright_override": {"start": 2022},
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "footer_links": {},
    "github_url": "https://github.com/NVIDIA/warp",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/warp-lang",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
        {
            "name": "Discord",
            "url": "https://discord.com/channels/827959428476174346/1285719658325999686",
            "icon": "fa-brands fa-discord",
            "type": "fontawesome",
        },
    ],
}
