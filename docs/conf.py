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
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warp as wp  # noqa: E402

# -- Project information -----------------------------------------------------

project = "Warp"
copyright = f"2022-{date.today().year}, NVIDIA Corporation"
author = "NVIDIA Corporation"

version = wp.__version__
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Convert docstrings to reStructuredText
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",  # Markup to shorten external links
    "sphinx.ext.githubpages",
    # Third-party extensions:
    "sphinx_copybutton",
    # 'sphinx_tabs.tabs',
    #    'autodocsumm'
]

# put type hints inside the description instead of the signature (easier to read)
autodoc_typehints = "description"
# document class *and* __init__ methods
autoclass_content = "both"

autodoc_member_order = "bysource"

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
#
html_theme = "furo"

html_title = f"Warp {version}"
html_show_sphinx = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "top_of_page_button": None,
    "light_css_variables": {
        "admonition-title-font-size": "100%",
        "admonition-font-size": "100%",
        "color-api-pre-name": "#4e9a06",  # "#76b900",
        "color-api-name": "#4e9a06",  # "#76b900",
        "color-admonition-title--seealso": "#ffffff",
        "color-admonition-title-background--seealso": "#448aff",
        "color-admonition-title-background--note": "#76b900",
        "color-admonition-title--note": "#ffffff",
    },
    "dark_css_variables": {
        "color-admonition-title-background--note": "#76b900",
        "color-admonition-title--note": "#ffffff",
    },
    "light_logo": "logo-light-mode.png",
    "dark_logo": "logo-dark-mode.png",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA/warp",
            "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
        """,
            "class": "",
        },
    ],
}
