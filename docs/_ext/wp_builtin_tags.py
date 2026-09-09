# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sphinx extension rendering the property tags of Warp built-ins.

The node class must live in an importable module rather than in ``conf.py``:
Sphinx executes ``conf.py`` without a module name, so classes defined there are
reported as ``builtins`` members and the doctree pickling step fails.
"""

from docutils import nodes
from docutils.parsers.rst import Directive, directives

# Ordered (option name, display label) pairs for the built-in property tags.
WP_BUILTIN_TAGS = (
    ("kernel", "Kernel"),
    ("python", "Python"),
    ("differentiable", "Differentiable"),
)


class wp_builtin_tags(nodes.Element):
    """Docutils node holding the true/false property tags of one built-in overload."""


class WpBuiltinTagsDirective(Directive):
    """Emit the property tags of a built-in overload as machine-readable markup."""

    has_content = False
    option_spec = dict.fromkeys((name for name, _ in WP_BUILTIN_TAGS), directives.unchanged_required)

    def run(self):
        tags = []
        for name, label in WP_BUILTIN_TAGS:
            if name not in self.options:
                raise self.error(f'"{self.name}" directive is missing the ":{name}:" option.')
            value = self.options[name].strip().lower()
            if value not in ("true", "false"):
                raise self.error(f'"{self.name}" option ":{name}:" must be "true" or "false", got "{value}".')
            tags.append([name, label, value == "true"])

        node = wp_builtin_tags()
        node["wp_tags"] = tags
        # Keep the labels of the true tags in the doctree so Sphinx's search
        # index still finds them; the HTML writer emits its own markup instead.
        node += nodes.Text(" ".join(label for _, label, value in tags if value))
        return [node]


def visit_html(self, node):
    self.body.append('<ul class="wp-builtin-tags">')
    for name, label, value in node["wp_tags"]:
        value_str = "true" if value else "false"
        # `visually-hidden` clips rather than removes, so a property that does not
        # hold is still announced by screen readers and still shows up in the text
        # that documentation tools extract from the rendered page.
        item_class = "wp-builtin-tag" if value else "wp-builtin-tag visually-hidden"
        self.body.append(
            f'<li class="{item_class}" data-wp-tag="{name}" data-wp-value="{value_str}">'
            f'<span class="wp-builtin-tag-name">{label}</span>'
            f'<span class="wp-builtin-tag-value visually-hidden">: {value_str}</span></li>'
        )
    self.body.append("</ul>")
    raise nodes.SkipNode


def visit_text(self, node):
    self.add_text(", ".join(f"{label}: {'true' if value else 'false'}" for _, label, value in node["wp_tags"]))
    raise nodes.SkipNode


def visit_skip(self, node):
    raise nodes.SkipNode


def setup(app):
    app.add_node(
        wp_builtin_tags,
        html=(visit_html, None),
        text=(visit_text, None),
        latex=(visit_skip, None),
        man=(visit_skip, None),
        texinfo=(visit_skip, None),
    )
    app.add_directive("wp-builtin-tags", WpBuiltinTagsDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
