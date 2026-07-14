{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {% block methods %}
   {%- set documented_methods = [] %}
   {%- for item in methods if item != "__init__" %}
   {%- set _ = documented_methods.append(item) %}
   {%- endfor %}
   {% if documented_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in documented_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
