Deprecate `warp.MarchingCubes`; use `warp.geometry.IsoSurfaceMarchingCubes` instead. The top-level name still works
during the deprecation period but warns when accessed. It now resolves to `warp.geometry.IsoSurfaceMarchingCubes`
itself rather than to a subclass, so `isinstance()` checks agree in both directions. The legacy
`domain_bounds_lower_corner` and `domain_bounds_upper_corner` constructor arguments and attributes remain supported as
deprecated aliases of `lower` and `upper`, respectively.
