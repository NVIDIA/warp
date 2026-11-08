{#-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Towncrier template for CHANGELOG.md. Issue links go inside an entry's
final period: "Fix X (GH-1)." instead of "Fix X. (GH-1)". A one-paragraph
entry is joined into a single line so that towncrier's `wrap = 120` refills
it. Keep the ".md" extension: towncrier picks Markdown or reStructuredText
output from the template's file extension.
-#}
{%- set nl = "\n" %}
{{- nl }}
{%- for section in sections %}
    {%- if section %}
        {{- nl ~ "#" * (header_prefix | length + 1) ~ " " ~ section ~ nl ~ nl }}
    {%- endif %}
    {%- for category, definition in definitions.items() if category in sections[section] %}
        {{- "#" * (header_prefix | length + (2 if section else 1)) ~ " " ~ definition["name"] ~ nl ~ nl }}
        {%- for text, values in sections[section][category].items() %}
            {%- set issues = values | join(", ") %}
            {#- Towncrier indents continuation lines by two spaces. A line that
                starts with a list, quote, table, heading, or deeper indent
                after them marks a structured entry. #}
            {%- set ns = namespace(structured="```" in text) %}
            {%- for line in text.split(nl)[1:] if line %}
                {%- if line[2:3] in " -*+>|#" or line[2:].split(".")[0].isdigit() %}
                    {%- set ns.structured = true %}
                {%- endif %}
            {%- endfor %}
            {%- set entry = text %}
            {%- if not ns.structured and nl ~ nl not in text %}
                {%- set entry = text.split(nl) | map("trim") | join(" ") %}
            {%- endif %}
            {%- if not issues %}
                {{- "- " ~ entry ~ nl }}
            {%- elif ns.structured %}
                {#- Keep the links out of a trailing list or code block. #}
                {{- "- " ~ entry ~ nl ~ nl ~ "  (" ~ issues ~ ")" ~ nl }}
            {%- elif entry.endswith(".") %}
                {{- "- " ~ entry[:-1] ~ " (" ~ issues ~ ")." ~ nl }}
            {%- else %}
                {{- "- " ~ entry ~ " (" ~ issues ~ ")" ~ nl }}
            {%- endif %}
        {%- endfor %}
        {{- nl }}
    {%- endfor %}
{%- endfor %}
{{- nl -}}
