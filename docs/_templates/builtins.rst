{{ fullname | escape | underline}}
{%- for overload in wp_overloads %}

.. function:: {{ fullname }}({{ overload.args }}) -> {{ overload.return_type }}
{%- if not loop.first %}
   :noindex:
{%- endif %}

   .. hlist::
      :columns: 8

      * Kernel
{%- if overload.is_exported %}
      * Python
{%- endif %}
{%- if overload.is_differentiable %}
      * Differentiable
{%- endif %}

   {{ overload.doc | indent(width=3, first=false, blank=true) }}
{%- endfor %}
