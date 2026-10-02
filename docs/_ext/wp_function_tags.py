# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sphinx extension rendering the property tags and source links of Warp functions.

The node class must live in an importable module rather than in ``conf.py``:
Sphinx executes ``conf.py`` without a module name, so classes defined there are
reported as ``builtins`` members and the doctree pickling step fails.
"""

from typing import ClassVar

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx import addnodes
from sphinx.locale import _ as translate

# Ordered (option name, display label) pairs for the function property tags.
WP_FUNCTION_TAGS = (
    ("kernel", "Kernel"),
    ("python", "Python"),
    ("differentiable", "Differentiable"),
)


class wp_function_tags(nodes.Element):
    """Docutils node holding the true/false/unspecified property tags of one function overload."""


class WpFunctionTagsDirective(Directive):
    """Emit the property tags of a function overload as machine-readable markup.

    The optional ``:source:`` URL is attached to the overload's signature, where
    ``sphinx.ext.linkcode`` places the source links of other objects.
    """

    has_content = False
    option_spec: ClassVar[dict] = {
        **dict.fromkeys((name for name, _ in WP_FUNCTION_TAGS), directives.unchanged_required),
        "source": directives.uri,
    }

    def run(self):
        tags = []
        for name, label in WP_FUNCTION_TAGS:
            if name not in self.options:
                raise self.error(f'"{self.name}" directive is missing the ":{name}:" option.')
            value = self.options[name].strip().lower()
            if value not in ("true", "false", "unknown"):
                raise self.error(
                    f'"{self.name}" option ":{name}:" must be "true", "false", or "unknown", got "{value}".'
                )
            tags.append([name, label, None if value == "unknown" else value == "true"])

        node = wp_function_tags()
        node["wp_tags"] = tags
        node["wp_source"] = self.options.get("source")
        # Keep the labels of the true tags in the doctree so Sphinx's search
        # index still finds them; the HTML writer emits its own markup instead.
        node += nodes.Text(" ".join(label for _, label, value in tags if value))
        return [node]


def attach_source_links(app, doctree):
    """Append each overload's source link to its signature, as ``sphinx.ext.linkcode`` does."""
    for node in doctree.findall(wp_function_tags):
        uri = node.get("wp_source")
        desc = node.parent
        while desc is not None and not isinstance(desc, addnodes.desc):
            desc = desc.parent
        if not uri or desc is None:
            continue
        for signode in desc.children:
            if isinstance(signode, addnodes.desc_signature):
                inline = nodes.inline("", translate("[source]"), classes=["viewcode-link"])
                onlynode = addnodes.only(expr="html")
                onlynode += nodes.reference("", "", inline, internal=False, refuri=uri)
                signode.append(onlynode)


def visit_html(self, node):
    self.body.append('<ul class="wp-function-tags">')
    for name, label, value in node["wp_tags"]:
        if value is None:
            continue
        value_str = "true" if value else "false"
        # `visually-hidden` clips rather than removes, so a property that does not
        # hold is still announced by screen readers and still shows up in the text
        # that documentation tools extract from the rendered page.
        item_class = "wp-function-tag" if value else "wp-function-tag visually-hidden"
        self.body.append(
            f'<li class="{item_class}" data-wp-tag="{name}" data-wp-value="{value_str}">'
            f'<span class="wp-function-tag-name">{label}</span>'
            f'<span class="wp-function-tag-value visually-hidden">: {value_str}</span></li>'
        )
    self.body.append("</ul>")
    raise nodes.SkipNode


def visit_text(self, node):
    self.add_text(
        ", ".join(
            f"{label}: {'true' if value else 'false'}" for _, label, value in node["wp_tags"] if value is not None
        )
    )
    raise nodes.SkipNode


def visit_skip(self, node):
    raise nodes.SkipNode


def setup(app):
    app.add_node(
        wp_function_tags,
        html=(visit_html, None),
        text=(visit_text, None),
        latex=(visit_skip, None),
        man=(visit_skip, None),
        texinfo=(visit_skip, None),
    )
    app.add_directive("wp-function-tags", WpFunctionTagsDirective)
    app.connect("doctree-read", attach_source_links)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
