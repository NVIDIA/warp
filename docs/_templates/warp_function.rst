{{ wp_display_name | escape | underline}}

.. currentmodule:: {{ wp_module }}
{%- for overload in wp_overloads %}

.. function:: {{ objname }}({{ overload.args }}){% if overload.return_type is not none %} -> {{ overload.return_type }}{% endif %}
{%- if not loop.first %}
   :noindex:
{%- endif %}

   .. wp-function-tags::
      :kernel: true
      :python: {{ "unknown" if overload.python_callable is none else ("true" if overload.python_callable else "false") }}
      :differentiable: {{ "unknown" if overload.differentiable is none else ("true" if overload.differentiable else "false") }}
{%- if overload.source_url %}
      :source: {{ overload.source_url }}
{%- endif %}

   {{ overload.doc | indent(width=3, first=false, blank=true) }}
{%- endfor %}
