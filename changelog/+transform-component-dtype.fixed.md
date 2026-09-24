Fix reading the `p` and `q` components of a transformation from Python, which always returned `vec3f` and `quatf`
regardless of the transform's own `dtype`. Reading a component of a `wp.transformd` rounded it to `float32`, so
`t.p = t.p` silently discarded precision the transform still held. Components now carry the transform's scalar type,
matching what `wp.transform_get_translation()` and `wp.transform_get_rotation()` already return in kernels.
`wp.transformf` components are unchanged; for `wp.transformd` and `wp.transformh`, `isinstance(t.p, wp.vec3f)` is no
longer true and `t.p[0]` is now a Warp scalar rather than a Python `float`, matching what `t[0]` already returned.
Wrap it in `float()` where a Python `float` is required.
