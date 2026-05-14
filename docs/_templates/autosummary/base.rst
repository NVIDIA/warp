{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

{# Keep this inline while there is one display-only override. If more config
   defaults need symbolic annotations, move this to a small annotation map. #}
.. auto{{ objtype }}:: {{ objname }}{% if fullname == "warp.config.log_level" %}
   :annotation: : int = warp.LOG_INFO
{% endif %}
