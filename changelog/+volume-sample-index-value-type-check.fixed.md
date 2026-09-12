Fix `wp.volume_sample_index()` and `wp.volume_sample_grad_index()` rejecting voxel data whose type is equivalent
to a supported one but not the same object, such as `wp.types.vector(length=4, dtype=wp.float64)` rather than
`wp.vec4d`. The error message now names the type instead of the generic `vec_t`.
