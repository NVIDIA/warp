Fix the type stubs for built-ins that take a `dtype` argument, such as `wp.vector()`, `wp.quaternion()`,
`wp.quat_identity()`, `wp.identity()`, `wp.tile_astype()`, and `wp.tile_arange()`. Passing `dtype` previously failed
type checking because the parameter was typed as a value rather than a type, and the result type ignored it, so
`wp.quat_identity(dtype=wp.float64)` was reported as `quatf`. Omitting `dtype` now reports the documented result type,
and passing one parameterizes the result.
