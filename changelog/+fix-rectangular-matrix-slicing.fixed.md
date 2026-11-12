Fix slicing rectangular matrix types from Python, where the row count and the column count were swapped when
resolving the slice bounds. Slicing a `wp.types.matrix(shape=(2, 3), ...)` value returned a truncated `vec2` for
`m[0, :]` and raised `IndexError` for `m[:, 0]`. Square matrix types and slicing inside kernels were unaffected.
