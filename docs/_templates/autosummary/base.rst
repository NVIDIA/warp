{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}{% if wp_annotation_override %}
   :annotation: {{ wp_annotation_override }}
{% endif %}
