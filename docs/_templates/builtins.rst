{{ fullname | escape | underline}}

.. function:: {{ fullname }}({{ wp_args }}) -> {{ wp_return_type }}

   .. hlist::
      :columns: 8

      * Kernel
{% if wp_is_exported       %}      * Python{% endif %}
{% if wp_is_differentiable %}      * Differentiable{% endif %}

   {{ wp_doc }}
