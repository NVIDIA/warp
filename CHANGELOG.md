# Changelog

## [Unreleased] - 2025-??

### Added

- Add support for dynamic control flow in CUDA graphs, see `wp.capture_if()` and `wp.capture_while()`
  ([GH-597](https://github.com/NVIDIA/warp/issues/597)).
- Add the `Device.sm_count` property to get the number of streaming multiprocessors on a CUDA device
  ([GH-584](https://github.com/NVIDIA/warp/issues/584)).
- Add support for profiling GPU runtime module compilation using the global `wp.config.compile_time_trace`
  setting or the module-level `"compile_time_trace"` option. When used, JSON files in the Trace Event
  format will be written in the kernel cache, which can be opened in a viewer like `chrome://tracing/`
  ([GH-609](https://github.com/NVIDIA/warp/issues/609)).
- Add `wp.tile_squeeze()` ([GH-662](https://github.com/NVIDIA/warp/issues/662)).
- Add `wp.tile_reshape()` ([GH-663](https://github.com/NVIDIA/warp/issues/663)).
- Support in-place tile add and subtract operations ([GH-518](https://github.com/NVIDIA/warp/issues/518)).
- Support in-place tile-component addition and subtraction ([GH-659](https://github.com/NVIDIA/warp/issues/659)).
- Add adjoint method for tile `assign` operations ([GH-680](https://github.com/NVIDIA/warp/issues/680)).
- Add support for returning multiple values from native functions like `wp.svd3()` and `wp.quat_to_axis_angle()`
  ([GH-503](https://github.com/NVIDIA/warp/issues/503)).
- Support attribute indexing for quaternions on the right-hand side of expressions
  ([GH-625](https://github.com/NVIDIA/warp/issues/625))
- Add `transform_compose` and `transform_decompose` math functions for converting between transforms and 4x4 matrices
  with 3D scale information ([GH-576](https://github.com/NVIDIA/warp/issues/576)).
- Add a parameter `as_spheres` to `UsdRenderer.render_points()` in order to choose whether to render
  the points as USD spheres using a point instancer or as simple USD points.
- Add support for animating visibility of objects in the USD renderer
  ([GH-598](https://github.com/NVIDIA/warp/issues/598)).
- Add `wp.sim.VBDIntegrator.rebuild_bvh()`, which rebuilds the BVH used for detecting self contacts.
- Improved consistency of `warp.fem.lookup()` operator across geometries ([GH-618](https://github.com/NVIDIA/warp/pull/618)), added filtering parameters.

### Changed

- Deprecate the `wp.matrix(pos, quat, scale)` built-in function. Use `wp.transform_compose()` instead
  ([GH-576](https://github.com/NVIDIA/warp/issues/576)).
- Change rigid-body-contact handling in `wp.sim.VBDIntegrator` to use only the shape's friction coefficient instead of
  averaging the shape's and the cloth's coefficients.
- Add damping terms for collisions in `wp.sim.VBDIntegrator`, whose strength is controlled by `Model.soft_contact_kd`.
- Exposed new `warp.fem` operators: `node_count`, `node_index`, `element_coordinates`, `element_closest_point`.

### Fixed

- Fix preserving base class of nested struct attributes ([GH-574](https://github.com/NVIDIA/warp/issues/574)).
- Allow recovering from out-of-memory errors during Volume allocation ([GH-611](https://github.com/NVIDIA/warp/issues/611)).
- Address `wp.tile_atomic_add()` compiler errors ([GH-681](https://github.com/NVIDIA/warp/issues/681)).
- Fix the `Formal parameter space overflowed` error when compiling the `wp.sim.VBDIntegrator` kernels for the backward
  pass in CUDA 11 Warp builds. This is done by decoupling the collision evaluation and elasticity evaluations to
  separate kernels, which also increases the parallelism of the collision handling and speeds up the solver
  ([GH-442](https://github.com/NVIDIA/warp/issues/442)).
- Fix `UsdRenderer.render_points()` not supporting multiple colors
  ([GH-634](https://github.com/NVIDIA/warp/issues/634)).
- Fix assembly of rigid body inertia in `ModelBuilder.collapse_fixed_joints()`
  ([GH-631](https://github.com/NVIDIA/warp/issues/631)).
- Fix `OpenGLRenderer.update_shape_instance()` not having color buffers created for the shape instances.
- Fix 2D tile load when source array and tile have incompatible strides
  ([GH-688](https://github.com/NVIDIA/warp/issues/688)).
- Fixed inconsistency in orientation of 2D geometry side normals ([GH-629](https://github.com/NVIDIA/warp/issues/629)).

## [1.7.1] - 2025-04-30

### Added

- Add example of a distributed Jacobi solver using `mpi4py` in `warp/examples/distributed/example_jacobi_mpi.py`
  ([GH-475](https://github.com/NVIDIA/warp/issues/475)).

### Changed

- Improve `repr()` for Warp types, including adding `repr()` for `wp.array`.
- Change the USD renderer to use `framesPerSecond` for time sampling instead of `timeCodesPerSecond`
  to avoid playback speed issues in some viewers ([GH-617](https://github.com/NVIDIA/warp/issues/617)).
- `Model.rigid_contact_tids` are now -1 at non-active contact indices which allows to retrieve the vertex index of a
  mesh collision, see `test_collision.py` ([GH-623](https://github.com/NVIDIA/warp/issues/623)).
- Improve handling of deprecated JAX features ([GH-613](https://github.com/NVIDIA/warp/pull/613)).

### Fixed

- Fix a code generation bug involving return statements in Warp kernels, which could result in some threads in Warp
  being skipped when processed on the GPU ([GH-594](https://github.com/NVIDIA/warp/issues/594)).
- Fix constructing `DeformedGeometry` from `wp.fem.Trimesh3D` geometries
  ([GH-614](https://github.com/NVIDIA/warp/issues/614)).
- Fix `lookup` operator for `wp.fem.Trimesh3D` ([GH-618](https://github.com/NVIDIA/warp/issues/618)).
- Include the block dimension in the LTO file hash for the Cholesky solver
  ([GH-639](https://github.com/NVIDIA/warp/issues/639)).
- Fix tile loads for small tiles with aligned source memory ([GH-622](https://github.com/NVIDIA/warp/issues/622)).
- Fix length/shape matching for vectors and matrices from the Python scope.
- Fix the `dtype` parameter missing for `wp.quaternion()`.
- Fix invalid `dtype` comparison when using the `wp.matrix()`/`wp.vector()`/`wp.quaternion()` constructors
  with literal values and an explicit `dtype` argument ([GH-651](https://github.com/NVIDIA/warp/issues/651)).
- Fix incorrect thread index lookup for the backward pass of `wp.sim.collide()`
  ([GH-459](https://github.com/NVIDIA/warp/issues/459)).
- Fix a bug where `wp.sim.ModelBuilder` adds springs with -1 as vertex indices
  ([GH-621](https://github.com/NVIDIA/warp/issues/621)).
- Fix center of mass, inertia computation for mesh shapes ([GH-251](https://github.com/NVIDIA/warp/issues/251)).
- Fix computation of body center of mass to account for shape orientation
  ([GH-648](https://github.com/NVIDIA/warp/issues/648)).
- Fix `show_joints` not working with `wp.sim.render.SimRenderer` set to render to USD
  ([GH-510](https://github.com/NVIDIA/warp/issues/510)).
- Fix the jitter for the `OgnParticlesFromMesh` node not being computed correctly.
- Fix documentation of `atol` and `rtol` arguments to `wp.autograd.gradcheck()` and `wp.autograd.gradcheck_tape()`
  ([GH-508](https://github.com/NVIDIA/warp/issues/508)).
- Fix an issue where the position of a fixed particle is not copied to the output state ([GH-627](https://github.com/NVIDIA/warp/issues/627)).

## [1.7.0] - 2025-03-30

### Added

- Support JAX foreign function interface (FFI)
  ([docs](https://nvidia.github.io/warp/modules/interoperability.html#jax-foreign-function-interface-ffi),
  [GH-511](https://github.com/NVIDIA/warp/issues/511)).
- Support Python/SASS correlation in Nsight Compute reports by emitting `#line` directives in CUDA-C code.
  This setting is controlled by `wp.config.line_directives` and is `True` by default.
  ([docs](https://nvidia.github.io/warp/profiling.html#nsight-compute-profiling),
   [GH-437](https://github.com/NVIDIA/warp/issues/437))
- Support `vec4f` grid construction in `wp.Volume.allocate_by_tiles()`.
- Add 2D SVD `wp.svd2()` ([GH-436](https://github.com/NVIDIA/warp/issues/436)).
- Add `wp.randu()` for random `uint32` generation.
- Add matrix construction functions `wp.matrix_from_cols()` and `wp.matrix_from_rows()`
  ([GH-278](https://github.com/NVIDIA/warp/issues/278)).
- Add `wp.transform_from_matrix()` to obtain a transform from a 4x4 matrix
  ([GH-211](https://github.com/NVIDIA/warp/issues/211)).
- Add `wp.where()` to select between two arguments conditionally using a
  more intuitive argument order (`cond`, `value_if_true`, `value_if_false`)
  ([GH-469](https://github.com/NVIDIA/warp/issues/469)).
- Add `wp.get_mempool_used_mem_current()` and `wp.get_mempool_used_mem_high()` to
  query the respective current and high-water mark memory pool allocator usage.
  ([GH-446](https://github.com/NVIDIA/warp/issues/446)).
- Add `Stream.is_complete` and `Event.is_complete` properties to query completion status
  ([GH-435](https://github.com/NVIDIA/warp/issues/435)).
- Support timing events inside of CUDA graphs ([GH-556](https://github.com/NVIDIA/warp/issues/556)).
- Add LTO cache to speed up compilation times for kernels using MathDx-based tile functions.
  Use `wp.clear_lto_cache()` to clear the LTO cache ([GH-507](https://github.com/NVIDIA/warp/issues/507)).
- Add example demonstrating gradient checkpointing for fluid optimization in
  `warp/examples/optim/example_fluid_checkpoint.py`.
- Add a hinge-angle-based bending force to `wp.sim.VBDIntegrator`.
- Add an example to show mesh sampling using a CDF
  ([GH-476](https://github.com/NVIDIA/warp/issues/476)).

### Changed

- **Breaking:** Remove CUTLASS dependency and `wp.matmul()` functionality (including batched version).
  Users should use tile primitives for matrix multiplication operations instead.
- Deprecate constructing a matrix from vectors using `wp.matrix()`.
- Deprecate `wp.select()` in favor of `wp.where()`. Users should update their code to use
  `wp.where(cond, value_if_true, value_if_false)` instead of `wp.select(cond, value_if_false, value_if_true)`.
- `wp.sim.Control` no longer has a `model` attribute ([GH-487](https://github.com/NVIDIA/warp/issues/487)).
- `wp.sim.Control.reset()` is deprecated and now only zeros-out the controls (previously restored controls
  to initial `model` state). Use `wp.sim.Control.clear()` instead.
- Vector/matrix/quaternion component assignment operations (e.g., `v[0] = x`) now compile and run faster in the
  backward pass. Note: For correct gradient computation, each component should only be assigned once.
- `@wp.kernel` has now an optional `module` argument that allows passing a `wp.context.Module` to the kernel,
  or, if set to `"unique"` let Warp create a new unique module just for this kernel.
  The default behavior to use the current module is unchanged.
- Default PTX architecture is now automatically determined by the devices present in the system,
  ensuring optimal compatibility and performance ([GH-537](https://github.com/NVIDIA/warp/issues/537)).
- Structs now have a trivial default constructor, allowing for `wp.tile_reduce()` on tiles with struct data types.
- Extend `wp.tile_broadcast()` to support broadcasting to 1D, 3D, and 4D shapes (in addition to existing 2D support).
- `wp.fem.integrate()` and `wp.fem.interpolate()` may now perform parallel evaluation of quadrature points within elements.
- `wp.fem.interpolate()` can now build Jacobian sparse matrices of interpolated functions with respect to a trial field.
- Multiple `wp.sparse` routines (`bsr_set_from_triplets`, `bsr_assign`, `bsr_axpy`, `bsr_mm`) now accept a `masked`
  flag to discard any non-zero not already present in the destination matrix.
- `wp.sparse.bsr_assign()` no longer requires source and destination block shapes to evenly divide each other.
- Extend `wp.expect_near()` to support all vectors and quaternions.
- Extend `wp.quat_from_matrix()` to support 4x4 matrices.
- Update the `OgnClothSimulate` node to use the VBD integrator ([GH-512](https://github.com/NVIDIA/warp/issues/512)).
- Remove the `globalScale` parameter from the `OgnClothSimulate` node.

### Fixed

- Fix an out-of-bounds access bug caused by an unbalanced BVH tree ([GH-536](https://github.com/NVIDIA/warp/issues/536)).
- Fix an error of incorrectly adding the offset to -1 elements in `edge_indices` when adding a ModelBuilder to another
  ([GH-557](https://github.com/NVIDIA/warp/issues/557)).

## [1.6.2] - 2025-03-07

### Changed

- Update project license from *NVIDIA Software License* to *Apache License, Version 2.0* (see `LICENSE.md`).

## [1.6.1] - 2025-03-03

### Added

- Document `wp.Launch` objects ([docs](https://nvidia.github.io/warp/modules/runtime.html#launch-objects),
  [GH-428](https://github.com/NVIDIA/warp/issues/428)).
- Document how overwriting previously computed results can lead to incorrect gradients
  ([docs](https://nvidia.github.io/warp/modules/differentiability.html#array-overwrites),
  [GH-525](https://github.com/NVIDIA/warp/issues/525)).

### Fixed

- Fix unaligned loads with offset 2D tiles in `wp.tile_load()`.
- Fix FP64 accuracy of thread-level matrix-matrix multiplications ([GH-489](https://github.com/NVIDIA/warp/issues/489)).
- Fix `wp.array()` not initializing from arrays defining a CUDA array interface when the target device is CPU
  ([GH-523](https://github.com/NVIDIA/warp/issues/523)).
- Fix `wp.Launch` objects not storing and replaying adjoint kernel launches
  ([GH-449](https://github.com/NVIDIA/warp/issues/449)).
- Fix `wp.config.verify_autograd_array_access` failing to detect overwrites in generic Warp functions
  ([GH-493](https://github.com/NVIDIA/warp/issues/493)).
- Fix an error on Windows when closing an `OpenGLRenderer` app ([GH-488](https://github.com/NVIDIA/warp/issues/488)).
- Fix per-vertex colors not being correctly written out to USD meshes when a constant color is being passed
  ([GH-480](https://github.com/NVIDIA/warp/issues/480)).
- Fix an error in capturing the `wp.sim.VBDIntegrator` with CUDA graphs when `handle_self_contact` is enabled
  ([GH-441](https://github.com/NVIDIA/warp/issues/441)).
- Fix an error of AABB computation in `wp.collide.TriMeshCollisionDetector`.
- Fix URDF-imported planar joints not being set with the intended `target_ke`, `target_kd`, and `mode` parameters
  ([GH-454](https://github.com/NVIDIA/warp/issues/454)).
- Fix `ModelBuilder.add_builder()` to use correct offsets for `ModelBuilder.joint_parent` and `ModelBuilder.joint_child`
  ([GH-432](https://github.com/NVIDIA/warp/issues/432))
- Fix underallocation of contact points for box–sphere and box–capsule collisions.
- Fix `wp.randi()` documentation to show correct output range of `[-2^31, 2^31)`.

## [1.6.0] - 2025-02-03

### Added

- Add preview of Tile Cholesky factorization and solve APIs through `wp.tile_cholesky()`, `tile_cholesky_solve()`
  and `tile_diag_add()` (preview APIs are subject to change).
- Support for loading tiles from arrays whose shapes are not multiples of the tile dimensions.
  Out-of-bounds reads will be zero-filled and out-of-bounds writes will be skipped.
- Support for higher-dimensional (up to 4D) tile shapes and memory operations.
- Add intersection-free self-contact support in `wp.sim.VBDIntegrator` by passing `handle_self_contact=True`.
  See `warp/examples/sim/example_cloth_self_contact.py` for a usage example.
- Add functions `wp.norm_l1()`, `wp.norm_l2()`, `wp.norm_huber()`, `wp.norm_pseudo_huber()`, and `wp.smooth_normalize()`
  for vector types to a new `wp.math` module.
- `wp.sim.SemiImplicitIntegrator` and `wp.sim.FeatherstoneIntegrator` now have an optional `friction_smoothing`
  constructor argument (defaults to 1.0) that controls softness of the friction norm computation.
- Support `assert` statements in kernels ([docs](https://nvidia.github.io/warp/debugging.html#assertions)).
  Assertions can only be triggered in `"debug"` mode ([GH-366](https://github.com/NVIDIA/warp/issues/336)).
- Support CUDA IPC on Linux. Call the `ipc_handle()` method to get an IPC handle for a `wp.Event` or a `wp.array`,
  and call `wp.from_ipc_handle()` or `wp.event_from_ipc_handle()` in another process to open the handle
  ([docs](https://nvidia.github.io/warp/modules/runtime.html#interprocess-communication-ipc)).
- Add per-module option to disable fused floating point operations, use `wp.set_module_options({"fuse_fp": False})`
  ([GH-379](https://github.com/NVIDIA/warp/issues/379)).
- Add per-module option to add CUDA-C line information for profiling, use `wp.set_module_options({"lineinfo": True})`.
- Support operator overloading for `wp.struct` objects by defining `wp.func` functions
  ([GH-392](https://github.com/NVIDIA/warp/issues/392)).
- Add built-in function `wp.len()` to retrieve the number of elements for vectors, quaternions, matrices, and arrays
  ([GH-389](https://github.com/NVIDIA/warp/issues/389)).
- Add `warp/examples/optim/example_softbody_properties.py` as an optimization example for soft-body properties
  ([GH-419](https://github.com/NVIDIA/warp/pull/419)).
- Add `warp/examples/tile/example_tile_walker.py`, which reworks the existing `example_walker.py`
  to use Warp's tile API for matrix multiplication.
- Add `warp/examples/tile/example_tile_nbody.py` as an example of an N-body simulation using Warp tile primitives.

### Changed

- **Breaking:** Change `wp.tile_load()` and `wp.tile_store()` indexing behavior so that indices are now specified in
  terms of *array elements* instead of *tile multiples*.
- **Breaking:** Tile operations now take `shape` and `offset` parameters as tuples,
  e.g.: `wp.tile_load(array, shape=(m,n), offset=(i,j))`.
- **Breaking:** Change exception types and error messages thrown by tile functions for improved consistency.
- Add an implicit tile synchronization whenever a shared memory tile's data is reinitialized (e.g. in dynamic loops).
  This could result in lower performance.
- `wp.Bvh` constructor now supports various construction algorithms via the `constructor` argument, including
  `"sah"` (Surface Area Heuristics), `"median"`, and `"lbvh"` ([docs](https://nvidia.github.io/warp/modules/runtime.html#warp.Bvh.__init__))
- Improve the query efficiency of `wp.Bvh` and `wp.Mesh`.
- Improve memory consumption, compilation and runtime performance when using in-place vector/matrix assignments in
  kernels that have `enable_backward` set to `False` ([GH-332](https://github.com/NVIDIA/warp/issues/332)).
- Vector/matrix/quaternion component `+=` and `-=` operations compile and run faster in the backward pass
  ([GH-332](https://github.com/NVIDIA/warp/issues/332)).
- Name files in the kernel cache according to their directory. Previously, all files began with
  `module_codegen` ([GH-431](https://github.com/NVIDIA/warp/issues/431)).
- Avoid recompilation of modules when changing `block_dim`.
- `wp.autograd.gradcheck_tape()` now has additional optional arguments `reverse_launches` and `skip_to_launch_index`.
- `wp.autograd.gradcheck()`, `wp.autograd.jacobian()`, and `wp.autograd.jacobian_fd()` now also accept
  arbitrary Python functions that have Warp arrays as inputs and outputs.
- `update_vbo_transforms` kernel launches in the OpenGL renderer are no longer recorded onto the tape.
- Skip emitting backward functions/kernels in the generated C++/CUDA code when `enable_backward` is set to `False`.
- Emit deprecation warnings for the use of the `owner` and `length` keywords in the `wp.array` initializer.
- Emit deprecation warnings for the use of `wp.mlp()`, `wp.matmul()`, and `wp.batched_matmul()`.
  Use tile primitives instead.

### Fixed

- Fix unintended modification of non-Warp arrays during the backward pass ([GH-394](https://github.com/NVIDIA/warp/issues/394)).
- Fix so that `wp.Tape.zero()` zeroes gradients passed via the `grads` parameter in `wp.Tape.backward()`
  ([GH-407](https://github.com/NVIDIA/warp/issues/407)).
- Fix errors during graph capture caused by module unloading ([GH-401](https://github.com/NVIDIA/warp/issues/401)).
- Fix potential memory corruption errors when allocating arrays with strides ([GH-404](https://github.com/NVIDIA/warp/issues/404)).
- Fix `wp.array()` not respecting the target `dtype` and `shape` when the given data is an another array with a CUDA interface
  ([GH-363](https://github.com/NVIDIA/warp/issues/363)).
- Negative constants evaluate to compile-time constants ([GH-403](https://github.com/NVIDIA/warp/issues/403))
- Fix `ImportError` exception being thrown during interpreter shutdown on Windows when using the OpenGL renderer
  ([GH-412](https://github.com/NVIDIA/warp/issues/412)).
- Fix the OpenGL renderer not working when multiple instances exist at the same time ([GH-385](https://github.com/NVIDIA/warp/issues/385)).
- Fix `AttributeError` crash in the OpenGL renderer when moving the camera ([GH-426](https://github.com/NVIDIA/warp/issues/426)).
- Fix the OpenGL renderer not correctly displaying duplicate capsule, cone, and cylinder shapes
  ([GH-388](https://github.com/NVIDIA/warp/issues/388)).
- Fix the overriding of `wp.sim.ModelBuilder` default parameters ([GH-429](https://github.com/NVIDIA/warp/pull/429)).
- Fix indexing of `wp.tile_extract()` when the block dimension is smaller than the tile size.
- Fix scale and rotation issues with the rock geometry used in the granular collision SDF example
  ([GH-409](https://github.com/NVIDIA/warp/issues/409)).
- Fix autodiff Jacobian computation in `wp.autograd.jacobian()` where in some cases gradients were not zeroed-out properly.
- Fix plotting issues in `wp.autograd.jacobian_plot()`.
- Fix the `len()` operator returning the total size of a matrix instead of its first dimension.
- Fix gradient instability in rigid-body contact handling for `wp.sim.SemiImplicitIntegrator` and
  `wp.sim.FeatherstoneIntegrator` ([GH-349](https://github.com/NVIDIA/warp/issues/349)).
- Fix overload resolution of generic Warp functions with default arguments.
- Fix rendering of arrows with different `up_axis`, `color` in `OpenGLRenderer` ([GH-448](https://github.com/NVIDIA/warp/issues/448)).

## [1.5.1] - 2025-01-02

### Added

- Add PyTorch basics and custom operators notebooks to the `notebooks` directory.
- Update PyTorch interop docs to include section on custom operators
  ([docs](https://nvidia.github.io/warp/modules/interoperability.html#pytorch-custom-ops-example)).

### Fixed

- warp.sim: Fix a bug in which the color-balancing algorithm was not updating the colorings.
- Fix custom colors being not being updated when rendering meshes with static topology in OpenGL
  ([GH-343](https://github.com/NVIDIA/warp/issues/343)).
- Fix `wp.launch_tiled()` not returning a `Launch` object when passed `record_cmd=True`.
- Fix default arguments not being resolved for `wp.func` when called from Python's runtime
  ([GH-386](https://github.com/NVIDIA/warp/issues/386)).
- Array overwrite tracking: Fix issue with not marking arrays passed to `wp.atomic_add()`, `wp.atomic_sub()`,
  `wp.atomic_max()`, or `wp.atomic_min()` as being written to ([GH-378](https://github.com/NVIDIA/warp/issues/378)).
- Fix for occasional failure to update `.meta` files into Warp kernel cache on Windows.
- Fix the OpenGL renderer not being able to run without a CUDA device available
  ([GH-344](https://github.com/NVIDIA/warp/issues/344)).
- Fix incorrect CUDA driver function versions ([GH-402](https://github.com/NVIDIA/warp/issues/402)).

## [1.5.0] - 2024-12-02

### Added

- Support for cooperative tile-based primitives using cuBLASDx and cuFFTDx, please see the tile
  [documentation](https://nvidia.github.io/warp/modules/tiles.html) for details.
- Expose a `reversed()` built-in for iterators ([GH-311](https://github.com/NVIDIA/warp/issues/311)).
- Support for saving Volumes into `.nvdb` files with the `save_to_nvdb` method.
- warp.fem: Add `wp.fem.Trimesh3D` and `wp.fem.Quadmesh3D` geometry types for 3D surfaces with new `example_distortion_energy` example.
- warp.fem: Add `"add"` option to `wp.fem.integrate()` for accumulating integration result to existing output.
- warp.fem: Add `"assembly"` option to `wp.fem.integrate()` for selecting between more memory-efficient or more
  computationally efficient integration algorithms.
- warp.fem: Add Nédélec (first kind) and Raviart-Thomas vector-valued function spaces
  providing conforming discretization of `curl` and `div` operators, respectively.
- warp.sim: Add a graph coloring module that supports converting trimesh into a vertex graph and applying coloring.
  The `wp.sim.ModelBuilder` now includes methods to color particles for use with `wp.sim.VBDIntegrator()`,
  users should call `builder.color()` before finalizing assets.
- warp.sim: Add support for a per-particle radius for soft-body triangle contact using the `wp.sim.Model.particle_radius`
  array ([docs](https://nvidia.github.io/warp/modules/sim.html#warp.sim.Model.particle_radius)), replacing the previous
  hard-coded value of 0.01 ([GH-329](https://github.com/NVIDIA/warp/issues/329)).
- Add a `particle_radius` parameter to `wp.sim.ModelBuilder.add_cloth_mesh()` and `wp.sim.ModelBuilder.add_cloth_grid()`
  to set a uniform radius for the added particles.
- Document `wp.array` attributes ([GH-364](https://github.com/NVIDIA/warp/issues/364)).
- Document time-to-compile tradeoffs when using vector component assignment statements in kernels.
- Add introductory Jupyter notebooks to the `notebooks` directory.

### Changed

- Drop support for Python 3.7; Python 3.8 is now the minimum-supported version.
- Promote the `wp.Int`, `wp.Float`, and `wp.Scalar` generic annotation types to the public API.
- warp.fem: Simplify querying neighboring cell quantities when integrating on sides using new
  `wp.fem.cells()`, `wp.fem.to_inner_cell()`, `wp.fem.to_outer_cell()` operators.
- Show an error message when the type returned by a function differs from its annotation, which would have led to the compilation stage failing.
- Clarify that `wp.randn()` samples a normal distribution of mean 0 and variance 1.
- Raise error when passing more than 32 variadic argument to the `wp.printf()` built-in.

### Fixed

- Fix `place` setting of paddle backend.
- warp.fem: Fix tri-cubic shape functions on quadrilateral meshes.
- warp.fem: Fix caching of integrand kernels when changing code-generation options.
- Fix `wp.expect_neq()` overloads missing for scalar types.
- Fix an error when a `wp.kernel` or a `wp.func` object is annotated to return a `None` value.
- Fix error when reading multi-volume, BLOSC-compressed `.nvdb` files.
- Fix `wp.printf()` erroring out when no variadic arguments are passed ([GH-333](https://github.com/NVIDIA/warp/issues/333)).
- Fix memory access issues in soft-rigid contact collisions ([GH-362](https://github.com/NVIDIA/warp/issues/362)).
- Fix gradient propagation for in-place addition/subtraction operations on custom vector-type arrays.
- Fix the OpenGL renderer's window not closing when clicking the X button.
- Fix the OpenGL renderer's camera snapping to a different direction from the initial camera's orientation when first looking around.
- Fix custom colors being ignored when rendering meshes in OpenGL ([GH-343](https://github.com/NVIDIA/warp/issues/343)).
- Fix topology updates not being supported by the the OpenGL renderer.

## [1.4.2] - 2024-11-13

### Changed

- Make the output of `wp.print()` in backward kernels consistent for all supported data types.

### Fixed

- Fix to relax the integer types expected when indexing arrays (regression in `1.3.0`).
- Fix printing vector and matrix adjoints in backward kernels.
- Fix kernel compile error when printing structs.
- Fix an incorrect user function being sometimes resolved when multiple overloads are available with array parameters with different `dtype` values.
- Fix error being raised when static and dynamic for-loops are written in sequence with the same iteration variable names ([GH-331](https://github.com/NVIDIA/warp/issues/331)).
- Fix an issue with the `Texture Write` node, used in the Mandelbrot Omniverse sample, sometimes erroring out in multi-GPU environments.
- Code generation of in-place multiplication and division operations (regression introduced in a69d061)([GH-342](https://github.com/NVIDIA/warp/issues/342)).

## [1.4.1] - 2024-10-15

### Fixed

- Fix `iter_reverse()` not working as expected for ranges with steps other than 1 ([GH-311](https://github.com/NVIDIA/warp/issues/311)).
- Fix potential out-of-bounds memory access when a `wp.sparse.BsrMatrix` object is reused for storing matrices of different shapes.
- Fix robustness to very low desired tolerance in `wp.fem.utils.symmetric_eigenvalues_qr`.
- Fix invalid code generation error messages when nesting dynamic and static for-loops.
- Fix caching of kernels with static expressions.
- Fix `ModelBuilder.add_builder(builder)` to correctly update `articulation_start` and thereby `articulation_count` when `builder` contains more than one articulation.
- Re-introduced the `wp.rand*()`, `wp.sample*()`, and `wp.poisson()` onto the Python scope to revert a breaking change.

## [1.4.0] - 2024-10-01

### Added

- Support for a new `wp.static(expr)` function that allows arbitrary Python expressions to be evaluated at the time of
  function/kernel definition ([docs](https://nvidia.github.io/warp/codegen.html#static-expressions)).
- Support for stream priorities to hint to the device that it should process pending work
  in high-priority streams over pending work in low-priority streams when possible
  ([docs](https://nvidia.github.io/warp/modules/concurrency.html#stream-priorities)).
- Adaptive sparse grid geometry to `warp.fem` ([docs](https://nvidia.github.io/warp/modules/fem.html#adaptivity)).
- Support for defining `wp.kernel` and `wp.func` objects from within closures.
- Support for defining multiple versions of kernels, functions, and structs without manually assigning unique keys.
- Support for default argument values for user functions decorated with `wp.func`.
- Allow passing custom launch dimensions to `jax_kernel()` ([GH-310](https://github.com/NVIDIA/warp/pull/310)).
- JAX interoperability examples for sharding and matrix multiplication ([docs](https://nvidia.github.io/warp/modules/interoperability.html#using-shardmap-for-distributed-computation)).
- Interoperability support for the PaddlePaddle ML framework ([GH-318](https://github.com/NVIDIA/warp/pull/318)).
- Support `wp.mod()` for vector types ([GH-282](https://github.com/NVIDIA/warp/issues/282)).
- Expose the modulo operator `%` to Python's runtime scalar and vector types.
- Support for fp64 `atomic_add`, `atomic_max`, and `atomic_min` ([GH-284](https://github.com/NVIDIA/warp/issues/284)).
- Support for quaternion indexing (e.g. `q.w`).
- Support shadowing builtin functions ([GH-308](https://github.com/NVIDIA/warp/issues/308)).
- Support for redefining function overloads.
- Add an ocean sample to the `omni.warp` extension.
- `warp.sim.VBDIntegrator` now supports body-particle collision.
- Add a [contributing guide](https://nvidia.github.io/warp/modules/contribution_guide.html) to the Sphinx docs .
- Add documentation for dynamic code generation ([docs](https://nvidia.github.io/warp/codegen.html#dynamic-kernel-creation)).

### Changed

- `wp.sim.Model.edge_indices` now includes boundary edges.
- Unexposed `wp.rand*()`, `wp.sample*()`, and `wp.poisson()` from the Python scope.
- Skip unused functions in module code generation, improving performance.
- Avoid reloading modules if their content does not change, improving performance.
- `wp.Mesh.points` is now a property instead of a raw data member, its reference can be changed after the mesh is initialized.
- Improve error message when invalid objects are referenced in a Warp kernel.
- `if`/`else`/`elif` statements with constant conditions are resolved at compile time with no branches being inserted in the generated code.
- Include all non-hidden builtins in the stub file.
- Improve accuracy of symmetric eigenvalues routine in `warp.fem`.

### Fixed

- Fix for `wp.func` erroring out when defining a `Tuple` as a return type hint ([GH-302](https://github.com/NVIDIA/warp/issues/302)).
- Fix array in-place op (`+=`, `-=`) adjoints to compute gradients correctly in the backwards pass
- Fix vector, matrix in-place assignment adjoints to compute gradients correctly in the backwards pass, e.g.: `v[1] = x`
- Fix a bug in which Python docstrings would be created as local function variables in generated code.
- Fix a bug with autograd array access validation in functions from different modules.
- Fix a rare crash during error reporting on some systems due to glibc mismatches.
- Handle `--num_tiles 1` in `example_render_opengl.py` ([GH-306](https://github.com/NVIDIA/warp/issues/306)).
- Fix the computation of body contact forces in `FeatherstoneIntegrator` when bodies and particles collide.
- Fix bug in `FeatherstoneIntegrator` where `eval_rigid_jacobian` could give incorrect results or reach an infinite
  loop when the body and joint indices were not in the same order. Added `Model.joint_ancestor` to fix the indexing
  from a joint to its parent joint in the articulation.
- Fix wrong vertex index passed to `add_edges()` called from `ModelBuilder.add_cloth_mesh()` ([GH-319](https://github.com/NVIDIA/warp/issues/319)).
- Add a workaround for uninitialized memory read warning in the `compute-sanitizer` initcheck tool when using `wp.Mesh`.
- Fix name clashes when Warp functions and structs are returned from Python functions multiple times.
- Fix name clashes between Warp functions and structs defined in different modules.
- Fix code generation errors when overloading generic kernels defined in a Python function.
- Fix issues with unrelated functions being treated as overloads (e.g., closures).
- Fix handling of `stream` argument in `array.__dlpack__()`.
- Fix a bug related to reloading CPU modules.
- Fix a crash when kernel functions are not found in CPU modules.
- Fix conditions not being evaluated as expected in `while` statements.
- Fix printing Boolean and 8-bit integer values.
- Fix array interface type strings used for Boolean and 8-bit integer values.
- Fix initialization error when setting struct members.
- Fix Warp not being initialized upon entering a `wp.Tape` context.
- Use `kDLBool` instead of `kDLUInt` for DLPack interop of Booleans.

## [1.3.3] - 2024-09-04

- Bug fixes
  - Fix an aliasing issue with zero-copy array initialization from NumPy introduced in Warp 1.3.0.
  - Fix `wp.Volume.load_from_numpy()` behavior when `bg_value` is a sequence of values ([GH-312](https://github.com/NVIDIA/warp/pull/312)).

## [1.3.2] - 2024-08-30

- Bug fixes
  - Fix accuracy of 3x3 SVD ``wp.svd3`` with fp64 numbers ([GH-281](https://github.com/NVIDIA/warp/issues/281)).
  - Fix module hashing when a kernel argument contained a struct array ([GH-287](https://github.com/NVIDIA/warp/issues/287)).
  - Fix a bug in `wp.bvh_query_ray()` where the direction instead of the reciprocal direction was used ([GH-288](https://github.com/NVIDIA/warp/issues/288)).
  - Fix errors when launching a CUDA graph after a module is reloaded. Modules that were used during graph capture
    will no longer be unloaded before the graph is released.
  - Fix a bug in `wp.sim.collide.triangle_closest_point_barycentric()` where the returned barycentric coordinates may be
    incorrect when the closest point lies on an edge.
  - Fix 32-bit overflow when array shape is specified using `np.int32`.
  - Fix handling of integer indices in the `input_output_mask` argument to `autograd.jacobian` and
    `autograd.jacobian_fd` ([GH-289](https://github.com/NVIDIA/warp/issues/289)).
  - Fix `ModelBuilder.collapse_fixed_joints()` to correctly update the body centers of mass and the
    `ModelBuilder.articulation_start` array.
  - Fix precedence of closure constants over global constants.
  - Fix quadrature point indexing in `wp.fem.ExplicitQuadrature` (regression from 1.3.0).
- Documentation improvements
  - Add missing return types for built-in functions.
  - Clarify that atomic operations also return the previous value.
  - Clarify that `wp.bvh_query_aabb()` returns parts that overlap the bounding volume.

## [1.3.1] - 2024-07-27

- Remove `wp.synchronize()` from PyTorch autograd function example
- `Tape.check_kernel_array_access()` and `Tape.reset_array_read_flags()` are now private methods.
- Fix reporting unmatched argument types

## [1.3.0] - 2024-07-25

- Warp Core improvements
  - Update to CUDA 12.x by default (requires NVIDIA driver 525 or newer), please see [README.md](https://github.com/nvidia/warp?tab=readme-ov-file#installing) for commands to install CUDA 11.x binaries for older drivers
  - Add information to the module load print outs to indicate whether a module was
  compiled `(compiled)`, loaded from the cache `(cached)`, or was unable to be
  loaded `(error)`.
  - `wp.config.verbose = True` now also prints out a message upon the entry to a `wp.ScopedTimer`.
  - Add `wp.clear_kernel_cache()` to the public API. This is equivalent to `wp.build.clear_kernel_cache()`.
  - Add code-completion support for `wp.config` variables.
  - Remove usage of a static task (thread) index for CPU kernels to address multithreading concerns ([GH-224](https://github.com/NVIDIA/warp/issues/224))
  - Improve error messages for unsupported Python operations such as sequence construction in kernels
  - Update `wp.matmul()` CPU fallback to use dtype explicitly in `np.matmul()` call
  - Add support for PEP 563's `from __future__ import annotations` ([GH-256](https://github.com/NVIDIA/warp/issues/256)).
  - Allow passing external arrays/tensors to `wp.launch()` directly via `__cuda_array_interface__` and `__array_interface__`, up to 2.5x faster conversion from PyTorch
  - Add faster Torch interop path using `return_ctype` argument to `wp.from_torch()`
  - Handle incompatible CUDA driver versions gracefully
  - Add `wp.abs()` and `wp.sign()` for vector types
  - Expose scalar arithmetic operators to Python's runtime (e.g.: `wp.float16(1.23) * wp.float16(2.34)`)
  - Add support for creating volumes with anisotropic transforms
  - Allow users to pass function arguments by keyword in a kernel using standard Python calling semantics
  - Add additional documentation and examples demonstrating `wp.copy()`, `wp.clone()`, and `array.assign()` differentiability
  - Add `__new__()` methods for all class `__del__()` methods to handle when a class instance is created but not instantiated before garbage collection
  - Implement the assignment operator for `wp.quat`
  - Make the geometry-related built-ins available only from within kernels
  - Rename the API-facing query types to remove their `_t` suffix: `wp.BVHQuery`, `wp.HashGridQuery`, `wp.MeshQueryAABB`, `wp.MeshQueryPoint`, and `wp.MeshQueryRay`
  - Add `wp.array(ptr=...)` to allow initializing arrays from pointer addresses inside of kernels ([GH-206](https://github.com/NVIDIA/warp/issues/206))

- `warp.autograd` improvements:
  - New `warp.autograd` module with utility functions `gradcheck()`, `jacobian()`, and `jacobian_fd()` for debugging kernel Jacobians ([docs](https://nvidia.github.io/warp/modules/differentiability.html#measuring-gradient-accuracy))
  - Add array overwrite detection, if `wp.config.verify_autograd_array_access` is true in-place operations on arrays on the Tape that could break gradient computation will be detected ([docs](https://nvidia.github.io/warp/modules/differentiability.html#array-overwrite-tracking))
  - Fix bug where modification of `@wp.func_replay` functions and native snippets would not trigger module recompilation
  - Add documentation for dynamic loop autograd limitations

- `warp.sim` improvements:
  - Improve memory usage and performance for rigid body contact handling when `self.rigid_mesh_contact_max` is zero (default behavior).
  - The `mask` argument to `wp.sim.eval_fk()` now accepts both integer and boolean arrays to mask articulations.
  - Fix handling of `ModelBuilder.joint_act` in `ModelBuilder.collapse_fixed_joints()` (affected floating-base systems)
  - Fix and improve implementation of `ModelBuilder.plot_articulation()` to visualize the articulation tree of a rigid-body mechanism
  - Fix ShapeInstancer `__new__()` method (missing instance return and `*args` parameter)
  - Fix handling of `upaxis` variable in `ModelBuilder` and the rendering thereof in `OpenGLRenderer`

- `warp.sparse` improvements:
  - Sparse matrix allocations (from `bsr_from_triplets()`, `bsr_axpy()`, etc.) can now be captured in CUDA graphs; exact number of non-zeros can be optionally requested asynchronously.
  - `bsr_assign()` now supports changing block shape (including CSR/BSR conversions)
  - Add Python operator overloads for common sparse matrix operations, e.g `A += 0.5 * B`, `y = x @ C`

- `warp.fem` new features and fixes:
  - Support for variable number of nodes per element
  - Global `wp.fem.lookup()` operator now supports `wp.fem.Tetmesh` and `wp.fem.Trimesh2D` geometries
  - Simplified defining custom subdomains (`wp.fem.Subdomain`), free-slip boundary conditions
  - New field types: `wp.fem.UniformField`, `wp.fem.ImplicitField` and `wp.fem.NonconformingField`
  - New `streamlines`, `magnetostatics` and `nonconforming_contact` examples, updated `mixed_elasticity` to use a nonlinear model
  - Function spaces can now export VTK-compatible cells for visualization
  - Fixed edge cases with NanoVDB function spaces
  - Fixed differentiability of `wp.fem.PicQuadrature` w.r.t. positions and measures

## [1.2.2] - 2024-07-04

- Fix hashing of replay functions and snippets
- Add additional documentation and examples demonstrating `wp.copy()`, `wp.clone()`, and `array.assign()` differentiability
- Add `__new__()` methods for all class `__del__()` methods to
  handle when a class instance is created but not instantiated before garbage collection.
- Add documentation for dynamic loop autograd limitations
- Allow users to pass function arguments by keyword in a kernel using standard Python calling semantics
- Implement the assignment operator for `wp.quat`

## [1.2.2] - 2024-07-04

- Support for NumPy >= 2.0

## [1.2.1] - 2024-06-14

- Fix generic function caching
- Fix Warp not being initialized when constructing arrays with `wp.array()`
- Fix `wp.is_mempool_access_supported()` not resolving the provided device arguments to `wp.context.Device`

## [1.2.0] - 2024-06-06

- Add a not-a-number floating-point constant that can be used as `wp.NAN` or `wp.nan`.
- Add `wp.isnan()`, `wp.isinf()`, and `wp.isfinite()` for scalars, vectors, matrices, etc.
- Improve kernel cache reuse by hashing just the local module constants. Previously, a
  module's hash was affected by all `wp.constant()` variables declared in a Warp program.
- Revised module compilation process to allow multiple processes to use the same kernel cache directory.
  Cached kernels will now be stored in hash-specific subdirectory.
- Add runtime checks for `wp.MarchingCubes` on field dimensions and size
- Fix memory leak in `wp.Mesh` BVH ([GH-225](https://github.com/NVIDIA/warp/issues/225))
- Use C++17 when building the Warp library and user kernels
- Increase PTX target architecture up to `sm_75` (from `sm_70`), enabling Turing ISA features
- Extended NanoVDB support (see `warp.Volume`):
  - Add support for data-agnostic index grids, allocation at voxel granularity
  - New `wp.volume_lookup_index()`, `wp.volume_sample_index()` and generic `wp.volume_sample()`/`wp.volume_lookup()`/`wp.volume_store()` kernel-level functions
  - Zero-copy aliasing of in-memory grids, support for multi-grid buffers
  - Grid introspection and blind data access capabilities
  - `warp.fem` can now work directly on NanoVDB grids using `warp.fem.Nanogrid`
  - Fixed `wp.volume_sample_v()` and `wp.volume_store_*()` adjoints
  - Prevent `wp.volume_store()` from overwriting grid background values
- Improve validation of user-provided fields and values in `warp.fem`
- Support headless rendering of `wp.render.OpenGLRenderer` via `pyglet.options["headless"] = True`
- `wp.render.RegisteredGLBuffer` can fall back to CPU-bound copying if CUDA/OpenGL interop is not available
- Clarify terms for external contributions, please see CONTRIBUTING.md for details
- Improve performance of `wp.sparse.bsr_mm()` by ~5x on benchmark problems
- Fix for XPBD incorrectly indexing into of joint actuations `joint_act` arrays
- Fix for mass matrix gradients computation in `wp.sim.FeatherstoneIntegrator()`
- Fix for handling of `--msvc_path` in build scripts
- Fix for `wp.copy()` params to record dest and src offset parameters on `wp.Tape()`
- Fix for `wp.randn()` to ensure return values are finite
- Fix for slicing of arrays with gradients in kernels
- Fix for function overload caching, ensure module is rebuilt if any function overloads are modified
- Fix for handling of `bool` types in generic kernels
- Publish CUDA 12.5 binaries for Hopper support, see https://github.com/nvidia/warp?tab=readme-ov-file#installing for details

## 1.1.1 - 2024-05-24

- `wp.init()` is no longer required to be called explicitly and will be performed on first call to the API
- Speed up `omni.warp.core`'s startup time

## [1.1.0] - 2024-05-09

- Support returning a value from `@wp.func_native` CUDA functions using type hints
- Improved differentiability of the `wp.sim.FeatherstoneIntegrator`
- Fix gradient propagation for rigid body contacts in `wp.sim.collide()`
- Added support for event-based timing, see `wp.ScopedTimer()`
- Added Tape visualization and debugging functions, see `wp.Tape.visualize()`
- Support constructing Warp arrays from objects that define the `__cuda_array_interface__` attribute
- Support copying a struct to another device, use `struct.to(device)` to migrate struct arrays
- Allow rigid shapes to not have any collisions with other shapes in `wp.sim.Model`
- Change default test behavior to test redundant GPUs (up to 2x)
- Test each example in an individual subprocess
- Polish and optimize various examples and tests
- Allow non-contiguous point arrays to be passed to `wp.HashGrid.build()`
- Upgrade LLVM to 18.1.3 for from-source builds and Linux x86-64 builds
- Build DLL source code as C++17 and require GCC 9.4 as a minimum
- Array clone, assign, and copy are now differentiable
- Use `Ruff` for formatting and linting
- Various documentation improvements (infinity, math constants, etc.)
- Improve URDF importer, handle joint armature
- Allow builtins.bool to be used in Warp data structures
- Use external gradient arrays in backward passes when passed to `wp.launch()`
- Add Conjugate Residual linear solver, see `wp.optim.linear.cr()`
- Fix propagation of gradients on aliased copy of variables in kernels
- Facilitate debugging and speed up `import warp` by eliminating raising any exceptions
- Improve support for nested vec/mat assignments in structs
- Recommend Python 3.9 or higher, which is required for JAX and soon PyTorch.
- Support gradient propagation for indexing sliced multi-dimensional arrays, i.e. `a[i][j]` vs. `a[i, j]`
- Provide an informative message if setting DLL C-types failed, instructing to try rebuilding the library

## 1.0.3 - 2024-04-17

- Add a `support_level` entry to the configuration file of the extensions

## [1.0.2] - 2024-03-22

- Make examples runnable from any location
- Fix the examples not running directly from their Python file
- Add the example gallery to the documentation
- Update `README.md` examples USD location
- Update `example_graph_capture.py` description

## [1.0.1] - 2024-03-15

- Document Device `total_memory` and `free_memory`
- Documentation for allocators, streams, peer access, and generics
- Changed example output directory to current working directory
- Added `python -m warp.examples.browse` for browsing the examples folder
- Print where the USD stage file is being saved
- Added `examples/optim/example_walker.py` sample
- Make the drone example not specific to USD
- Reduce the time taken to run some examples
- Optimise rendering points with a single colour
- Clarify an error message around needing USD
- Raise exception when module is unloaded during graph capture
- Added `wp.synchronize_event()` for blocking the host thread until a recorded event completes
- Flush C print buffers when ending `stdout` capture
- Remove more unneeded CUTLASS files
- Allow setting mempool release threshold as a fractional value

## [1.0.0] - 2024-03-07

- Add `FeatherstoneIntegrator` which provides more stable simulation of articulated rigid body dynamics in generalized coordinates (`State.joint_q` and `State.joint_qd`)
- Introduce `warp.sim.Control` struct to store control inputs for simulations (optional, by default the `Model` control inputs are used as before); integrators now have a different simulation signature: `integrator.simulate(model: Model, state_in: State, state_out: State, dt: float, control: Control)`
- `joint_act` can now behave in 3 modes: with `joint_axis_mode` set to `JOINT_MODE_FORCE` it behaves as a force/torque, with `JOINT_MODE_VELOCITY` it behaves as a velocity target, and with `JOINT_MODE_POSITION` it behaves as a position target; `joint_target` has been removed
- Add adhesive contact to Euler integrators via `Model.shape_materials.ka` which controls the contact distance at which the adhesive force is applied
- Improve handling of visual/collision shapes in URDF importer so visual shapes are not involved in contact dynamics
- Experimental JAX kernel callback support
- Improve module load exception message
- Add `wp.ScopedCapture`
- Removing `enable_backward` warning for callables
- Copy docstrings and annotations from wrapped kernels, functions, structs

## [0.15.1] - 2024-03-05

- Add examples assets to the wheel packages
- Fix broken image link in documentation
- Fix codegen for custom grad functions calling their respective forward functions
- Fix custom grad function handling for functions that have no outputs
- Fix issues when `wp.config.quiet = True`

## [0.15.0] - 2024-03-04

- Add thumbnails to examples gallery
- Apply colored lighting to examples
- Moved `examples` directory under `warp/`
- Add example usage to `python -m warp.tests --help`
- Adding `torch.autograd.function` example + docs
- Add error-checking to array shapes during creation
- Adding `example_graph_capture`
- Add a Diffsim Example of a Drone
- Fix `verify_fp` causing compiler errors and support CPU kernels
- Fix to enable `matmul` to be called in CUDA graph capture
- Enable mempools by default
- Update `wp.launch` to support tuple args
- Fix BiCGSTAB and GMRES producing NaNs when converging early
- Fix warning about backward codegen being disabled in `test_fem`
- Fix `assert_np_equal` when NaN's and tolerance are involved
- Improve error message to discern between CUDA being disabled or not supported
- Support cross-module functions with user-defined gradients
- Suppress superfluous CUDA error when ending capture after errors
- Make output during initialization atomic
- Add `warp.config.max_unroll`, fix custom gradient unrolling
- Support native replay snippets using `@wp.func_native(snippet, replay_snippet=replay_snippet)`
- Look for the CUDA Toolkit in default locations if the `CUDA_PATH` environment variable or `--cuda_path` build option are not used
- Added `wp.ones()` to efficiently create one-initialized arrays
- Rename `wp.config.graph_capture_module_load_default` to `wp.config.enable_graph_capture_module_load_by_default`

## 0.14.0 - 2024-02-19

- Add support for CUDA pooled (stream-ordered) allocators
  - Support memory allocation during graph capture
  - Support copying non-contiguous CUDA arrays during graph capture
  - Improved memory allocation/deallocation performance with pooled allocators
  - Use `wp.config.enable_mempools_at_init` to enable pooled allocators during Warp initialization (if supported)
  - `wp.is_mempool_supported()` - check if a device supports pooled allocators
  - `wp.is_mempool_enabled()`, `wp.set_mempool_enabled()` - enable or disable pooled allocators per device
  - `wp.set_mempool_release_threshold()`, `wp.get_mempool_release_threshold()` - configure memory pool release threshold
- Add support for direct memory access between devices
  - Improved peer-to-peer memory transfer performance if access is enabled
  - Caveat: enabling peer access may impact memory allocation/deallocation performance and increase memory consumption
  - `wp.is_peer_access_supported()` - check if the memory of a device can be accessed by a peer device
  - `wp.is_peer_access_enabled()`, `wp.set_peer_access_enabled()` - manage peer access for memory allocated using default CUDA allocators
  - `wp.is_mempool_access_supported()` - check if the memory pool of a device can be accessed by a peer device
  - `wp.is_mempool_access_enabled()`, `wp.set_mempool_access_enabled()` - manage access for memory allocated using pooled CUDA allocators
- Refined stream synchronization semantics
  - `wp.ScopedStream` can synchronize with the previous stream on entry and/or exit (only sync on entry by default)
  - Functions taking an optional stream argument do no implicit synchronization for max performance (e.g., `wp.copy()`, `wp.launch()`, `wp.capture_launch()`)
- Support for passing a custom `deleter` argument when constructing arrays
  - Deprecation of `owner` argument - use `deleter` to transfer ownership
- Optimizations for various core API functions (e.g., `wp.zeros()`, `wp.full()`, and more)
- Fix `wp.matmul()` to always use the correct CUDA context
- Fix memory leak in BSR transpose
- Fix stream synchronization issues when copying non-contiguous arrays
- API change: `wp.matmul()` no longer accepts a device as a parameter; instead, it infers the correct device from the arrays being multiplied
- Updated DLPack utilities to the latest published standard
  - External arrays can be imported into Warp directly, e.g., `wp.from_dlpack(external_array)`
  - Warp arrays can be exported to consumer frameworks directly, e.g., `jax.dlpack.from_dlpack(warp_array)`
  - Added CUDA stream synchronization for CUDA arrays
  - The original DLPack protocol can still be used for better performance when stream synchronization is not required, see interoperability docs for details
  - `warp.to_dlpack()` is about 3-4x faster in common cases
  - `warp.from_dlpack()` is about 2x faster when called with a DLPack capsule
  - Fixed a small CPU memory leak related to DLPack interop
- Improved performance of creating arrays

## 0.13.1 - 2024-02-22

- Ensure that the results from the `Noise Deform` are deterministic across different Kit sessions

## [0.13.0] - 2024-02-16

- Update the license to *NVIDIA Software License*, allowing commercial use (see `LICENSE.md`)
- Add `CONTRIBUTING.md` guidelines (for NVIDIA employees)
- Hash CUDA `snippet` and `adj_snippet` strings to fix caching
- Fix `build_docs.py` on Windows
- Add missing `.py` extension to `warp/tests/walkthrough_debug`
- Allow `wp.bool` usage in vector and matrix types

## 0.12.0 - 2024-02-05

- Add a warning when the `enable_backward` setting is set to `False` upon calling `wp.Tape.backward()`
- Fix kernels not being recompiled as expected when defined using a closure
- Change the kernel cache appauthor subdirectory to just "NVIDIA"
- Ensure that gradients attached to PyTorch tensors have compatible strides when calling `wp.from_torch()`
- Add a `Noise Deform` node for OmniGraph that deforms points using a perlin/curl noise

## [0.11.0] - 2024-01-23

- Re-release 1.0.0-beta.7 as a non-pre-release 0.11.0 version so it gets selected by `pip install warp-lang`.
- Introducing a new versioning and release process, detailed in `PACKAGING.md` and resembling that of [Python itself](https://devguide.python.org/developer-workflow/development-cycle/#devcycle):
  - The 0.11 release(s) can be found on the `release-0.11` branch.
  - Point releases (if any) go on the same minor release branch and only contain bug fixes, not new features.
  - The `public` branch, previously used to merge releases into and corresponding with the GitHub `main` branch, is retired.

## 1.0.0-beta.7 - 2024-01-23

- Ensure captures are always enclosed in `try`/`finally`
- Only include .py files from the warp subdirectory into wheel packages
- Fix an extension's sample node failing at parsing some version numbers
- Allow examples to run without USD when possible
- Add a setting to disable the main Warp menu in Kit
- Add iterative linear solvers, see `wp.optim.linear.cg`, `wp.optim.linear.bicgstab`, `wp.optim.linear.gmres`, and `wp.optim.linear.LinearOperator`
- Improve error messages around global variables
- Improve error messages around mat/vec assignments
- Support conversion of scalars to native/ctypes, e.g.: `float(wp.float32(1.23))` or `ctypes.c_float(wp.float32(1.23))`
- Add a constant for infinity, see `wp.inf`
- Add a FAQ entry about array assignments
- Add a mass spring cage diff simulation example, see `examples/example_diffsim_mass_spring_cage.py`
- Add `-s`, `--suite` option for only running tests belonging to the given suites
- Fix common spelling mistakes
- Fix indentation of generated code
- Show deprecation warnings only once
- Improve `wp.render.OpenGLRenderer`
- Create the extension's symlink to the *core library* at runtime
- Fix some built-ins failing to compile the backward pass when nested inside if/else blocks
- Update examples with the new variants of the mesh query built-ins
- Fix type members that weren't zero-initialized
- Fix missing adjoint function for `wp.mesh_query_ray()`

## [1.0.0-beta.6] - 2024-01-10

- Do not create CPU copy of grad array when calling `array.numpy()`
- Fix `assert_np_equal()` bug
- Support Linux AArch64 platforms, including Jetson/Tegra devices
- Add parallel testing runner (invoke with `python -m warp.tests`, use `warp/tests/unittest_serial.py` for serial testing)
- Fix support for function calls in `range()`
- `wp.matmul()` adjoints now accumulate
- Expand available operators (e.g. vector @ matrix, scalar as dividend) and improve support for calling native built-ins
- Fix multi-gpu synchronization issue in `sparse.py`
- Add depth rendering to `wp.render.OpenGLRenderer`, document `wp.render`
- Make `wp.atomic_min()`, `wp.atomic_max()` differentiable
- Fix error reporting using the exact source segment
- Add user-friendly mesh query overloads, returning a struct instead of overwriting parameters
- Address multiple differentiability issues
- Fix backpropagation for returning array element references
- Support passing the return value to adjoints
- Add point basis space and explicit point-based quadrature for `wp.fem`
- Support overriding the LLVM project source directory path using `build_lib.py --build_llvm --llvm_source_path=`
- Fix the error message for accessing non-existing attributes
- Flatten faces array for Mesh constructor in URDF parser

## [1.0.0-beta.5] - 2023-11-22

- Fix for kernel caching when function argument types change
- Fix code-gen ordering of dependent structs
- Fix for `wp.Mesh` build on MGPU systems
- Fix for name clash bug with adjoint code: https://github.com/NVIDIA/warp/issues/154
- Add `wp.frac()` for returning the fractional part of a floating point value
- Add support for custom native CUDA snippets using `@wp.func_native` decorator
- Add support for batched matmul with batch size > 2^16-1
- Add support for transposed CUTLASS `wp.matmul()` and additional error checking
- Add support for quad and hex meshes in `wp.fem`
- Detect and warn when C++ runtime doesn't match compiler during build, e.g.: ``libstdc++.so.6: version `GLIBCXX_3.4.30' not found``
- Documentation update for `wp.BVH`
- Documentation and simplified API for runtime kernel specialization `wp.Kernel`

## 1.0.0-beta.4 - 2023-11-01

- Add `wp.cbrt()` for cube root calculation
- Add `wp.mesh_furthest_point_no_sign()` to compute furthest point on a surface from a query point
- Add support for GPU BVH builds, 10-100x faster than CPU builds for large meshes
- Add support for chained comparisons, i.e.: `0 < x < 2`
- Add support for running `wp.fem` examples headless
- Fix for unit test determinism
- Fix for possible GC collection of array during graph capture
- Fix for `wp.utils.array_sum()` output initialization when used with vector types
- Coverage and documentation updates

## 1.0.0-beta.3 - 2023-10-19

- Add support for code coverage scans (test_coverage.py), coverage at 85% in `omni.warp.core`
- Add support for named component access for vector types, e.g.: `a = v.x`
- Add support for lvalue expressions, e.g.: `array[i] += b`
- Add casting constructors for matrix and vector types
- Add support for `type()` operator that can be used to return type inside kernels
- Add support for grid-stride kernels to support kernels with > 2^31-1 thread blocks
- Fix for multi-process initialization warnings
- Fix alignment issues with empty `wp.struct`
- Fix for return statement warning with tuple-returning functions
- Fix for `wp.batched_matmul()` registering the wrong function in the Tape
- Fix and document for `wp.sim` forward + inverse kinematics
- Fix for `wp.func` to return a default value if function does not return on all control paths
- Refactor `wp.fem` support for new basis functions, decoupled function spaces
- Optimizations for `wp.noise` functions, up to 10x faster in most cases
- Optimizations for `type_size_in_bytes()` used in array construction'

### Breaking Changes

- To support grid-stride kernels, `wp.tid()` can no longer be called inside `wp.func` functions.

## 1.0.0-beta.2 - 2023-09-01

- Fix for passing bool into `wp.func` functions
- Fix for deprecation warnings appearing on `stderr`, now redirected to `stdout`
- Fix for using `for i in wp.hash_grid_query(..)` syntax

## 1.0.0-beta.1 - 2023-08-29

- Fix for `wp.float16` being passed as kernel arguments
- Fix for compile errors with kernels using structs in backward pass
- Fix for `wp.Mesh.refit()` not being CUDA graph capturable due to synchronous temp. allocs
- Fix for dynamic texture example flickering / MGPU crashes demo in Kit by reusing `ui.DynamicImageProvider` instances
- Fix for a regression that disabled bundle change tracking in samples
- Fix for incorrect surface velocities when meshes are deforming in `OgnClothSimulate`
- Fix for incorrect lower-case when setting USD stage "up_axis" in examples
- Fix for incompatible gradient types when wrapping PyTorch tensor as a vector or matrix type
- Fix for adding open edges when building cloth constraints from meshes in `wp.sim.ModelBuilder.add_cloth_mesh()`
- Add support for `wp.fabricarray` to directly access Fabric data from Warp kernels, see https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usdrt_prim_selection.html for examples
- Add support for user defined gradient functions, see `@wp.func_replay`, and `@wp.func_grad` decorators
- Add support for more OG attribute types in `omni.warp.from_omni_graph()`
- Add support for creating NanoVDB `wp.Volume` objects from dense NumPy arrays
- Add support for `wp.volume_sample_grad_f()` which returns the value + gradient efficiently from an NVDB volume
- Add support for LLVM fp16 intrinsics for half-precision arithmetic
- Add implementation of stochastic gradient descent, see `wp.optim.SGD`
- Add `wp.fem` framework for solving weak-form PDE problems (see https://nvidia.github.io/warp/modules/fem.html)
- Optimizations for `omni.warp` extension load time (2.2s to 625ms cold start)
- Make all `omni.ui` dependencies optional so that Warp unit tests can run headless
- Deprecation of `wp.tid()` outside of kernel functions, users should pass `tid()` values to `wp.func` functions explicitly
- Deprecation of `wp.sim.Model.flatten()` for returning all contained tensors from the model
- Add support for clamping particle max velocity in `wp.sim.Model.particle_max_velocity`
- Remove dependency on `urdfpy` package, improve MJCF parser handling of default values

## [0.10.1] - 2023-07-25

- Fix for large multidimensional kernel launches (> 2^32 threads)
- Fix for module hashing with generics
- Fix for unrolling loops with break or continue statements (will skip unrolling)
- Fix for passing boolean arguments to build_lib.py (previously ignored)
- Fix build warnings on Linux
- Fix for creating array of structs from NumPy structured array
- Fix for regression on kernel load times in Kit when using `wp.sim`
- Update `wp.array.reshape()` to handle `-1` dimensions
- Update margin used by for mesh queries when using `wp.sim.create_soft_body_contacts()`
- Improvements to gradient handling with `wp.from_torch()`, `wp.to_torch()` plus documentation

## 0.10.0 - 2023-07-05

- Add support for macOS universal binaries (x86 + aarch64) for M1+ support
- Add additional methods for SDF generation please see the following new methods:
  - `wp.mesh_query_point_nosign()` - closest point query with no sign determination
  - `wp.mesh_query_point_sign_normal()` - closest point query with sign from angle-weighted normal
  - `wp.mesh_query_point_sign_winding_number()` - closest point query with fast winding number sign determination
- Add CSR/BSR sparse matrix support, see `wp.sparse` module:
  - `wp.sparse.BsrMatrix`
  - `wp.sparse.bsr_zeros()`, `wp.sparse.bsr_set_from_triplets()` for construction
  - `wp.sparse.bsr_mm()`, `wp.sparse_bsr_mv()` for matrix-matrix and matrix-vector products respectively
- Add array-wide utilities:
  - `wp.utils.array_scan()` - prefix sum (inclusive or exclusive)
  - `wp.utils.array_sum()` - sum across array
  - `wp.utils.radix_sort_pairs()` - in-place radix sort (key,value) pairs
- Add support for calling `@wp.func` functions from Python (outside of kernel scope)
- Add support for recording kernel launches using a `wp.Launch` object that can be replayed with low overhead, use `wp.launch(..., record_cmd=True)` to generate a command object
- Optimizations for `wp.struct` kernel arguments, up to 20x faster launches for kernels with large structs or number of params
- Refresh USD samples to use bundle based workflow + change tracking
- Add Python API for manipulating mesh and point bundle data in OmniGraph, see `omni.warp.nodes` module, see `omni.warp.nodes.mesh_create_bundle()`, `omni.warp.nodes.mesh_get_points()`, etc
- Improvements to `wp.array`:
  - Fix a number of array methods misbehaving with empty arrays
  - Fix a number of bugs and memory leaks related to gradient arrays
  - Fix array construction when creating arrays in pinned memory from a data source in pageable memory
  - `wp.empty()` no longer zeroes-out memory and returns an uninitialized array, as intended
  - `array.zero_()` and `array.fill_()` work with non-contiguous arrays
  - Support wrapping non-contiguous NumPy arrays without a copy
  - Support preserving the outer dimensions of NumPy arrays when wrapping them as Warp arrays of vector or matrix types
  - Improve PyTorch and DLPack interop with Warp arrays of arbitrary vectors and matrices
  - `array.fill_()` can now take lists or other sequences when filling arrays of vectors or matrices, e.g. `arr.fill_([[1, 2], [3, 4]])`
  - `array.fill_()` now works with arrays of structs (pass a struct instance)
  - `wp.copy()` gracefully handles copying between non-contiguous arrays on different devices
  - Add `wp.full()` and `wp.full_like()`, e.g., `a = wp.full(shape, value)`
  - Add optional `device` argument to `wp.empty_like()`, `wp.zeros_like()`, `wp.full_like()`, and `wp.clone()`
  - Add `indexedarray` methods `.zero_()`, `.fill_()`, and `.assign()`
  - Fix `indexedarray` methods `.numpy()` and `.list()`
  - Fix `array.list()` to work with arrays of any Warp data type
  - Fix `array.list()` synchronization issue with CUDA arrays
  - `array.numpy()` called on an array of structs returns a structured NumPy array with named fields
  - Improve the performance of creating arrays
- Fix for `Error: No module named 'omni.warp.core'` when running some Kit configurations (e.g.: stubgen)
- Fix for `wp.struct` instance address being included in module content hash
- Fix codegen with overridden function names
- Fix for kernel hashing so it occurs after code generation and before loading to fix a bug with stale kernel cache
- Fix for `wp.BVH.refit()` when executed on the CPU
- Fix adjoint of `wp.struct` constructor
- Fix element accessors for `wp.float16` vectors and matrices in Python
- Fix `wp.float16` members in structs
- Remove deprecated `wp.ScopedCudaGuard()`, please use `wp.ScopedDevice()` instead

## [0.9.0] - 2023-06-01

- Add support for in-place modifications to vector, matrix, and struct types inside kernels (will warn during backward pass with `wp.verbose` if using gradients)
- Add support for step-through VSCode debugging of kernel code with standalone LLVM compiler, see `wp.breakpoint()`, and `walkthrough_debug.py`
- Add support for default values on built-in functions
- Add support for multi-valued `@wp.func` functions
- Add support for `pass`, `continue`, and `break` statements
- Add missing `__sincos_stret` symbol for macOS
- Add support for gradient propagation through `wp.Mesh.points`, and other cases where arrays are passed to native functions
- Add support for Python `@` operator as an alias for `wp.matmul()`
- Add XPBD support for particle-particle collision
- Add support for individual particle radii: `ModelBuilder.add_particle` has a new `radius` argument, `Model.particle_radius` is now a Warp array
- Add per-particle flags as a `Model.particle_flags` Warp array, introduce `PARTICLE_FLAG_ACTIVE` to define whether a particle is being simulated and participates in contact dynamics
- Add support for Python bitwise operators `&`, `|`, `~`, `<<`, `>>`
- Switch to using standalone LLVM compiler by default for `cpu` devices
- Split `omni.warp` into `omni.warp.core` for Omniverse applications that want to use the Warp Python module with minimal additional dependencies
- Disable kernel gradient generation by default inside Omniverse for improved compile times
- Fix for bounds checking on element access of vector/matrix types
- Fix for stream initialization when a custom (non-primary) external CUDA context has been set on the calling thread
- Fix for duplicate `@wp.struct` registration during hot reload
- Fix for array `unot()` operator so kernel writers can use `if not array:` syntax
- Fix for case where dynamic loops are nested within unrolled loops
- Change `wp.hash_grid_point_id()` now returns -1 if the `wp.HashGrid` has not been reserved before
- Deprecate `wp.Model.soft_contact_distance` which is now replaced by `wp.Model.particle_radius`
- Deprecate single scalar particle radius (should be a per-particle array)

## 0.8.2 - 2023-04-21

- Add `ModelBuilder.soft_contact_max` to control the maximum number of soft contacts that can be registered. Use `Model.allocate_soft_contacts(new_count)` to change count on existing `Model` objects.
- Add support for `bool` parameters
- Add support for logical boolean operators with `int` types
- Fix for `wp.quat()` default constructor
- Fix conditional reassignments
- Add sign determination using angle weighted normal version of `wp.mesh_query_point()` as `wp.mesh_query_sign_normal()`
- Add sign determination using winding number of `wp.mesh_query_point()` as `wp.mesh_query_sign_winding_number()`
- Add query point without sign determination `wp.mesh_query_no_sign()`

## 0.8.1 - 2023-04-13

- Fix for regression when passing flattened numeric lists as matrix arguments to kernels
- Fix for regressions when passing `wp.struct` types with uninitialized (`None`) member attributes

## 0.8.0 - 2023-04-05

- Add `Texture Write` node for updating dynamic RTX textures from Warp kernels / nodes
- Add multi-dimensional kernel support to Warp Kernel Node
- Add `wp.load_module()` to pre-load specific modules (pass `recursive=True` to load recursively)
- Add `wp.poisson()` for sampling Poisson distributions
- Add support for UsdPhysics schema see `wp.sim.parse_usd()`
- Add XPBD rigid body implementation plus diff. simulation examples
- Add support for standalone CPU compilation (no host-compiler) with LLVM backed, enable with `--standalone` build option
- Add support for per-timer color in `wp.ScopedTimer()`
- Add support for row-based construction of matrix types outside of kernels
- Add support for setting and getting row vectors for Python matrices, see `matrix.get_row()`, `matrix.set_row()`
- Add support for instantiating `wp.struct` types within kernels
- Add support for indexed arrays, `slice = array[indices]` will now generate a sparse slice of array data
- Add support for generic kernel params, use `def compute(param: Any):`
- Add support for `with wp.ScopedDevice("cuda") as device:` syntax (same for `wp.ScopedStream()`, `wp.Tape()`)
- Add support for creating custom length vector/matrices inside kernels, see `wp.vector()`, and `wp.matrix()`
- Add support for creating identity matrices in kernels with, e.g.: `I = wp.identity(n=3, dtype=float)`
- Add support for unary plus operator (`wp.pos()`)
- Add support for `wp.constant` variables to be used directly in Python without having to use `.val` member
- Add support for nested `wp.struct` types
- Add support for returning `wp.struct` from functions
- Add `--quick` build for faster local dev. iteration (uses a reduced set of SASS arches)
- Add optional `requires_grad` parameter to `wp.from_torch()` to override gradient allocation
- Add type hints for generic vector / matrix types in Python stubs
- Add support for custom user function recording in `wp.Tape()`
- Add support for registering CUTLASS `wp.matmul()` with tape backward pass
- Add support for grids with > 2^31 threads (each dimension may be up to INT_MAX in length)
- Add CPU fallback for `wp.matmul()`
- Optimizations for `wp.launch()`, up to 3x faster launches in common cases
- Fix `wp.randf()` conversion to float to reduce bias for uniform sampling
- Fix capture of `wp.func` and `wp.constant` types from inside Python closures
- Fix for CUDA on WSL
- Fix for matrices in structs
- Fix for transpose indexing for some non-square matrices
- Enable Python faulthandler by default
- Update to VS2019

### Breaking Changes

- `wp.constant` variables can now be treated as their true type, accessing the underlying value through `constant.val` is no longer supported
- `wp.sim.model.ground_plane` is now a `wp.array` to support gradient, users should call `builder.set_ground_plane()` to create the ground 
- `wp.sim` capsule, cones, and cylinders are now aligned with the default USD up-axis

## 0.7.2 - 2023-02-15

- Reduce test time for vec/math types
- Clean-up CUDA disabled build pipeline
- Remove extension.gen.toml to make Kit packages Python version independent
- Handle additional cases for array indexing inside Python

## 0.7.1 - 2023-02-14

- Disabling some slow tests for Kit
- Make unit tests run on first GPU only by default

## [0.7.0] - 2023-02-13

- Add support for arbitrary length / type vector and matrices e.g.: `wp.vec(length=7, dtype=wp.float16)`, see `wp.vec()`, and `wp.mat()`
- Add support for `array.flatten()`, `array.reshape()`, and `array.view()` with NumPy semantics
- Add support for slicing `wp.array` types in Python
- Add `wp.from_ptr()` helper to construct arrays from an existing allocation
- Add support for `break` statements in ranged-for and while loops (backward pass support currently not implemented)
- Add built-in mathematic constants, see `wp.pi`, `wp.e`, `wp.log2e`, etc.
- Add built-in conversion between degrees and radians, see `wp.degrees()`, `wp.radians()`
- Add security pop-up for Kernel Node
- Improve error handling for kernel return values

## 0.6.3 - 2023-01-31

- Add DLPack utilities, see `wp.from_dlpack()`, `wp.to_dlpack()`
- Add Jax utilities, see `wp.from_jax()`, `wp.to_jax()`, `wp.device_from_jax()`, `wp.device_to_jax()`
- Fix for Linux Kit extensions OM-80132, OM-80133

## 0.6.2 - 2023-01-19

- Updated `wp.from_torch()` to support more data types
- Updated `wp.from_torch()` to automatically determine the target Warp data type if not specified
- Updated `wp.from_torch()` to support non-contiguous tensors with arbitrary strides
- Add CUTLASS integration for dense GEMMs, see `wp.matmul()` and `wp.matmul_batched()`
- Add QR and Eigen decompositions for `mat33` types, see `wp.qr3()`, and `wp.eig3()`
- Add default (zero) constructors for matrix types
- Add a flag to suppress all output except errors and warnings (set `wp.config.quiet = True`)
- Skip recompilation when Kernel Node attributes are edited
- Allow optional attributes for Kernel Node
- Allow disabling backward pass code-gen on a per-kernel basis, use `@wp.kernel(enable_backward=False)`
- Replace Python `imp` package with `importlib`
- Fix for quaternion slerp gradients (`wp.quat_slerp()`)

## 0.6.1 - 2022-12-05

- Fix for non-CUDA builds
- Fix strides computation in array_t constructor, fixes a bug with accessing mesh indices through mesh.indices[]
- Disable backward pass code generation for kernel node (4-6x faster compilation)
- Switch to linbuild for universal Linux binaries (affects TeamCity builds only)

## 0.6.0 - 2022-11-28

- Add support for CUDA streams, see `wp.Stream`, `wp.get_stream()`, `wp.set_stream()`, `wp.synchronize_stream()`, `wp.ScopedStream`
- Add support for CUDA events, see `wp.Event`, `wp.record_event()`, `wp.wait_event()`, `wp.wait_stream()`, `wp.Stream.record_event()`, `wp.Stream.wait_event()`, `wp.Stream.wait_stream()`
- Add support for PyTorch stream interop, see `wp.stream_from_torch()`, `wp.stream_to_torch()`
- Add support for allocating host arrays in pinned memory for asynchronous data transfers, use `wp.array(..., pinned=True)` (default is non-pinned)
- Add support for direct conversions between all scalar types, e.g.: `x = wp.uint8(wp.float64(3.0))`
- Add per-module option to enable fast math, use `wp.set_module_options({"fast_math": True})`, fast math is now *disabled* by default
- Add support for generating CUBIN kernels instead of PTX on systems with older drivers
- Add user preference options for CUDA kernel output ("ptx" or "cubin", e.g.: `wp.config.cuda_output = "ptx"` or per-module `wp.set_module_options({"cuda_output": "ptx"})`)
- Add kernel node for OmniGraph
- Add `wp.quat_slerp()`, `wp.quat_to_axis_angle()`, `wp.rotate_rodriquez()` and adjoints for all remaining quaternion operations
- Add support for unrolling for-loops when range is a `wp.constant`
- Add support for arithmetic operators on built-in vector / matrix types outside of `wp.kernel`
- Add support for multiple solution variables in `wp.optim` Adam optimization
- Add nested attribute support for `wp.struct` attributes
- Add missing adjoint implementations for spatial math types, and document all functions with missing adjoints
- Add support for retrieving NanoVDB tiles and voxel size, see `wp.Volume.get_tiles()`, and `wp.Volume.get_voxel_size()`
- Add support for store operations on integer NanoVDB volumes, see `wp.volume_store_i()`
- Expose `wp.Mesh` points, indices, as arrays inside kernels, see `wp.mesh_get()`
- Optimizations for `wp.array` construction, 2-3x faster on average
- Optimizations for URDF import
- Fix various deployment issues by statically linking with all CUDA libs
- Update warp.so/warp.dll to CUDA Toolkit 11.5

## 0.5.1 - 2022-11-01

- Fix for unit tests in Kit

## [0.5.0] - 2022-10-31

- Add smoothed particle hydrodynamics (SPH) example, see `example_sph.py`
- Add support for accessing `array.shape` inside kernels, e.g.: `width = arr.shape[0]`
- Add dependency tracking to hot-reload modules if dependencies were modified
- Add lazy acquisition of CUDA kernel contexts (save ~300Mb of GPU memory in MGPU environments)
- Add BVH object, see `wp.Bvh` and `bvh_query_ray()`, `bvh_query_aabb()` functions
- Add component index operations for `spatial_vector`, `spatial_matrix` types
- Add `wp.lerp()` and `wp.smoothstep()` builtins
- Add `wp.optim` module with implementation of the Adam optimizer for float and vector types
- Add support for transient Python modules (fix for Houdini integration)
- Add `wp.length_sq()`, `wp.trace()` for vector / matrix types respectively
- Add missing adjoints for `wp.quat_rpy()`, `wp.determinant()`
- Add `wp.atomic_min()`, `wp.atomic_max()` operators
- Add vectorized version of `wp.sim.model.add_cloth_mesh()`
- Add NVDB volume allocation API, see `wp.Volume.allocate()`, and `wp.Volume.allocate_by_tiles()`
- Add NVDB volume write methods, see `wp.volume_store_i()`, `wp.volume_store_f()`, `wp.volume_store_v()`
- Add MGPU documentation
- Add example showing how to compute Jacobian of multiple environments in parallel, see `example_jacobian_ik.py`
- Add `wp.Tape.zero()` support for `wp.struct` types
- Make SampleBrowser an optional dependency for Kit extension
- Make `wp.Mesh` object accept both 1d and 2d arrays of face vertex indices
- Fix for reloading of class member kernel / function definitions using `importlib.reload()`
- Fix for hashing of `wp.constants()` not invalidating kernels
- Fix for reload when multiple `.ptx` versions are present
- Improved error reporting during code-gen

## [0.4.3] - 2022-09-20

- Update all samples to use GPU interop path by default
- Fix for arrays > 2GB in length
- Add support for per-vertex USD mesh colors with `wp.render` class

## 0.4.2 - 2022-09-07

- Register Warp samples to the sample browser in Kit
- Add NDEBUG flag to release mode kernel builds
- Fix for particle solver node when using a large number of particles
- Fix for broken cameras in Warp sample scenes

## 0.4.1 - 2022-08-30

- Add geometry sampling methods, see `wp.sample_unit_cube()`, `wp.sample_unit_disk()`, etc
- Add `wp.lower_bound()` for searching sorted arrays
- Add an option for disabling code-gen of backward pass to improve compilation times, see `wp.set_module_options({"enable_backward": False})`, True by default
- Fix for using Warp from Script Editor or when module does not have a `__file__` attribute
- Fix for hot reload of modules containing `wp.func()` definitions
- Fix for debug flags not being set correctly on CUDA when `wp.config.mode == "debug"`, this enables bounds checking on CUDA kernels in debug mode
- Fix for code gen of functions that do not return a value

## 0.4.0 - 2022-08-09

- Fix for FP16 conversions on GPUs without hardware support
- Fix for `runtime = None` errors when reloading the Warp module
- Fix for PTX architecture version when running with older drivers, see `wp.config.ptx_target_arch`
- Fix for USD imports from `__init__.py`, defer them to individual functions that need them
- Fix for robustness issues with sign determination for `wp.mesh_query_point()`
- Fix for `wp.HashGrid` memory leak when creating/destroying grids
- Add CUDA version checks for toolkit and driver
- Add support for cross-module `@wp.struct` references
- Support running even if CUDA initialization failed, use `wp.is_cuda_available()` to check availability
- Statically linking with the CUDA runtime library to avoid deployment issues

### Breaking Changes

- Removed `wp.runtime` reference from the top-level module, as it should be considered private

## 0.3.2 - 2022-07-19

- Remove Torch import from `__init__.py`, defer import to `wp.from_torch()`, `wp.to_torch()`

## [0.3.1] - 2022-07-12

- Fix for marching cubes reallocation after initialization
- Add support for closest point between line segment tests, see `wp.closest_point_edge_edge()` builtin
- Add support for per-triangle elasticity coefficients in simulation, see `wp.sim.ModelBuilder.add_cloth_mesh()`
- Add support for specifying default device, see `wp.set_device()`, `wp.get_device()`, `wp.ScopedDevice`
- Add support for multiple GPUs (e.g., `"cuda:0"`, `"cuda:1"`), see `wp.get_cuda_devices()`, `wp.get_cuda_device_count()`, `wp.get_cuda_device()`
- Add support for explicitly targeting the current CUDA context using device alias `"cuda"`
- Add support for using arbitrary external CUDA contexts, see `wp.map_cuda_device()`, `wp.unmap_cuda_device()`
- Add PyTorch device aliasing functions, see `wp.device_from_torch()`, `wp.device_to_torch()`

### Breaking Changes

- A CUDA device is used by default, if available (aligned with `wp.get_preferred_device()`)
- `wp.ScopedCudaGuard` is deprecated, use `wp.ScopedDevice` instead
- `wp.synchronize()` now synchronizes all devices; for finer-grained control, use `wp.synchronize_device()`
- Device alias `"cuda"` now refers to the current CUDA context, rather than a specific device like `"cuda:0"` or `"cuda:1"`

## 0.3.0 - 2022-07-08

- Add support for FP16 storage type, see `wp.float16`
- Add support for per-dimension byte strides, see `wp.array.strides`
- Add support for passing Python classes as kernel arguments, see `@wp.struct` decorator
- Add additional bounds checks for builtin matrix types
- Add additional floating point checks, see `wp.config.verify_fp`
- Add interleaved user source with generated code to aid debugging
- Add generalized GPU marching cubes implementation, see `wp.MarchingCubes` class
- Add additional scalar*matrix vector operators
- Add support for retrieving a single row from builtin types, e.g.: `r = m33[i]`
- Add  `wp.log2()` and `wp.log10()` builtins
- Add support for quickly instancing `wp.sim.ModelBuilder` objects to improve env. creation performance for RL
- Remove custom CUB version and improve compatibility with CUDA 11.7
- Fix to preserve external user-gradients when calling `wp.Tape.zero()`
- Fix to only allocate gradient of a Torch tensor if `requires_grad=True`
- Fix for missing `wp.mat22` constructor adjoint
- Fix for ray-cast precision in edge case on GPU (watertightness issue)
- Fix for kernel hot-reload when definition changes
- Fix for NVCC warnings on Linux
- Fix for generated function names when kernels are defined as class functions
- Fix for reload of generated CPU kernel code on Linux
- Fix for example scripts to output USD at 60 timecodes per-second (better Kit compatibility)

## [0.2.3] - 2022-06-13

- Fix for incorrect 4d array bounds checking
- Fix for `wp.constant` changes not updating module hash
- Fix for stale CUDA kernel cache when CPU kernels launched first
- Array gradients are now allocated along with the arrays and accessible as `wp.array.grad`, users should take care to always call `wp.Tape.zero()` to clear gradients between different invocations of `wp.Tape.backward()`
- Added `wp.array.fill_()` to set all entries to a scalar value (4-byte values only currently)

### Breaking Changes

- Tape `capture` option has been removed, users can now capture tapes inside existing CUDA graphs (e.g.: inside Torch)
- Scalar loss arrays should now explicitly set `requires_grad=True` at creation time

## 0.2.2 - 2022-05-30

- Fix for `from import *` inside Warp initialization
- Fix for body space velocity when using deforming Mesh objects with scale
- Fix for noise gradient discontinuities affecting `wp.curlnoise()`
- Fix for `wp.from_torch()` to correctly preserve shape
- Fix for URDF parser incorrectly passing density to scale parameter
- Optimizations for startup time from 3s -> 0.3s
- Add support for custom kernel cache location, Warp will now store generated binaries in the user's application directory
- Add support for cross-module function references, e.g.: call another modules @wp.func functions
- Add support for overloading `@wp.func` functions based on argument type
- Add support for calling built-in functions directly from Python interpreter outside kernels (experimental)
- Add support for auto-complete and docstring lookup for builtins in IDEs like VSCode, PyCharm, etc
- Add support for doing partial array copies, see `wp.copy()` for details
- Add support for accessing mesh data directly in kernels, see `wp.mesh_get_point()`, `wp.mesh_get_index()`, `wp.mesh_eval_face_normal()`
- Change to only compile for targets where kernel is launched (e.g.: will not compile CPU unless explicitly requested)

### Breaking Changes

- Builtin methods such as `wp.quat_identity()` now call the Warp native implementation directly and will return a `wp.quat` object instead of NumPy array
- NumPy implementations of many builtin methods have been moved to `wp.utils` and will be deprecated
- Local `@wp.func` functions should not be namespaced when called, e.g.: previously `wp.myfunc()` would work even if `myfunc()` was not a builtin
- Removed `wp.rpy2quat()`, please use `wp.quat_rpy()` instead

## 0.2.1 - 2022-05-11

- Fix for unit tests in Kit

## [0.2.0] - 2022-05-02

### Warp Core

- Fix for unrolling loops with negative bounds
- Fix for unresolved symbol `hash_grid_build_device()` not found when lib is compiled without CUDA support
- Fix for failure to load nvrtc-builtins64_113.dll when user has a newer CUDA toolkit installed on their machine
- Fix for conversion of Torch tensors to `wp.array` with a vector dtype (incorrect row count)
- Fix for `warp.dll` not found on some Windows installations
- Fix for macOS builds on Clang 13.x
- Fix for step-through debugging of kernels on Linux
- Add argument type checking for user defined `@wp.func` functions
- Add support for custom iterable types, supports ranges, hash grid, and mesh query objects
- Add support for multi-dimensional arrays, for example use `x = array[i,j,k]` syntax to address a 3-dimensional array
- Add support for multi-dimensional kernel launches, use `launch(kernel, dim=(i,j,k), ...` and `i,j,k = wp.tid()` to obtain thread indices
- Add support for bounds-checking array memory accesses in debug mode, use `wp.config.mode = "debug"` to enable
- Add support for differentiating through dynamic and nested for-loops
- Add support for evaluating MLP neural network layers inside kernels with custom activation functions, see `wp.mlp()`
- Add additional NVDB sampling methods and adjoints, see `wp.volume_sample_i()`, `wp.volume_sample_f()`, and `wp.volume_sample_vec()`
- Add support for loading zlib compressed NVDB volumes, see `wp.Volume.load_from_nvdb()`
- Add support for triangle intersection testing, see `wp.intersect_tri_tri()`
- Add support for NVTX profile zones in `wp.ScopedTimer()`
- Add support for additional transform and quaternion math operations, see `wp.inverse()`, `wp.quat_to_matrix()`, `wp.quat_from_matrix()`
- Add fast math (`--fast-math`) to kernel compilation by default
- Add `wp.torch` import by default (if PyTorch is installed)

### Warp Kit

- Add Kit menu for browsing Warp documentation and example scenes under 'Window->Warp'
- Fix for OgnParticleSolver.py example when collider is coming from Read Prim into Bundle node

### Warp Sim

- Fix for joint attachment forces
- Fix for URDF importer and floating base support
- Add examples showing how to use differentiable forward kinematics to solve inverse kinematics
- Add examples for URDF cartpole and quadruped simulation

### Breaking Changes

- `wp.volume_sample_world()` is now replaced by `wp.volume_sample_f/i/vec()` which operate in index (local) space. Users should use `wp.volume_world_to_index()` to transform points from world space to index space before sampling.
- `wp.mlp()` expects multi-dimensional arrays instead of one-dimensional arrays for inference, all other semantics remain the same as earlier versions of this API.
- `wp.array.length` member has been removed, please use `wp.array.shape` to access array dimensions, or use `wp.array.size` to get total element count
- Marking `dense_gemm()`, `dense_chol()`, etc methods as experimental until we revisit them

## 0.1.25 - 2022-03-20

- Add support for class methods to be Warp kernels
- Add HashGrid reserve() so it can be used with CUDA graphs
- Add support for CUDA graph capture of tape forward/backward passes
- Add support for Python 3.8.x and 3.9.x
- Add hyperbolic trigonometric functions, see `wp.tanh()`, `wp.sinh()`, `wp.cosh()`
- Add support for floored division on integer types
- Move tests into core library so they can be run in Kit environment

## 0.1.24 - 2022-03-03

### Warp Core

- Add NanoVDB support, see `wp.volume_sample*()` methods
- Add support for reading compile-time constants in kernels, see `wp.constant()`
- Add support for __cuda_array_interface__ protocol for zero-copy interop with PyTorch, see `wp.torch.to_torch()`
- Add support for additional numeric types, i8, u8, i16, u16, etc
- Add better checks for device strings during allocation / launch
- Add support for sampling random numbers with a normal distribution, see `wp.randn()`
- Upgrade to CUDA 11.3
- Update example scenes to Kit 103.1
- Deduce array dtype from np.array when one is not provided
- Fix for ranged for loops with negative step sizes
- Fix for 3d and 4d spherical gradient distributions

## 0.1.23 - 2022-02-17

### Warp Core

- Fix for generated code folder being removed during Showroom installation
- Fix for macOS support
- Fix for dynamic for-loop code gen edge case
- Add procedural noise primitives, see `wp.noise()`, `wp.pnoise()`, `wp.curlnoise()`
- Move simulation helpers our of test into `wp.sim` module

## 0.1.22 - 2022-02-14

### Warp Core

- Fix for .so reloading on Linux
- Fix for while loop code-gen in some edge cases
- Add rounding functions `wp.round()`, `wp.rint()`, `wp.trunc()`, `wp.floor()`, `wp.ceil()`
- Add support for printing strings and formatted strings from kernels
- Add MSVC compiler version detection and require minimum

### Warp Sim

- Add support for universal and compound joint types

## 0.1.21 - 2022-01-19

### Warp Core

- Fix for exception on shutdown in empty `wp.array` objects
- Fix for hot reload of CPU kernels in Kit
- Add hash grid primitive for point-based spatial queries, see `wp.hash_grid_query()`, `wp.hash_grid_query_next()`
- Add new PRNG methods using PCG-based generators, see `wp.rand_init()`, `wp.randf()`, `wp.randi()`
- Add support for AABB mesh queries, see `wp.mesh_query_aabb()`, `wp.mesh_query_aabb_next()`
- Add support for all Python `range()` loop variants
- Add builtin vec2 type and additional math operators, `wp.pow()`, `wp.tan()`, `wp.atan()`, `wp.atan2()`
- Remove dependency on CUDA driver library at build time
- Remove unused NVRTC binary dependencies (50mb smaller Linux distribution)

### Warp Sim

- Bundle import of multiple shapes for simulation nodes
- New OgnParticleVolume node for sampling shapes -> particles
- New OgnParticleSolver node for DEM style granular materials

## 0.1.20 - 2021-11-02

- Updates to the ripple solver for GTC (support for multiple colliders, buoyancy, etc)

## 0.1.19 - 2021-10-15

- Publish from 2021.3 to avoid omni.graph database incompatibilities

## 0.1.18 - 2021-10-08

- Enable Linux support (tested on 20.04)

## 0.1.17 - 2021-09-30

- Fix for 3x3 SVD adjoint
- Fix for A6000 GPU (bump compute model to sm_52 minimum)
- Fix for .dll unload on rebuild
- Fix for possible array destruction warnings on shutdown
- Rename spatial_transform -> transform
- Documentation update

## 0.1.16 - 2021-09-06

- Fix for case where simple assignments (a = b) incorrectly generated reference rather than value copy
- Handle passing zero-length (empty) arrays to kernels

## 0.1.15 - 2021-09-03

- Add additional math library functions (asin, etc)
- Add builtin 3x3 SVD support
- Add support for named constants (True, False, None)
- Add support for if/else statements (differentiable)
- Add custom memset kernel to avoid CPU overhead of cudaMemset()
- Add rigid body joint model to `wp.sim` (based on Brax)
- Add Linux, MacOS support in core library
- Fix for incorrectly treating pure assignment as reference instead of value copy
- Removes the need to transfer array to CPU before numpy conversion (will be done implicitly)
- Update the example OgnRipple wave equation solver to use bundles

## 0.1.14 - 2021-08-09

- Fix for out-of-bounds memory access in CUDA BVH
- Better error checking after kernel launches (use `wp.config.verify_cuda=True`)
- Fix for vec3 normalize adjoint code

## 0.1.13 - 2021-07-29

- Remove OgnShrinkWrap.py test node

## 0.1.12 - 2021-07-29

- Switch to Woop et al.'s watertight ray-tri intersection test
- Disable --fast-math in CUDA compilation step for improved precision

## 0.1.11 - 2021-07-28

- Fix for `wp.mesh_query_ray()` returning incorrect t-value

## 0.1.10 - 2021-07-28

- Fix for OV extension fwatcher filters to avoid hot-reload loop due to OGN regeneration

## 0.1.9 - 2021-07-21

- Fix for loading sibling DLL paths
- Better type checking for built-in function arguments
- Added runtime docs, can now list all builtins using `wp.print_builtins()`

## 0.1.8 - 2021-07-14

- Fix for hot-reload of CUDA kernels
- Add Tape object for replaying differentiable kernels
- Add helpers for Torch interop (convert `torch.Tensor` to `wp.Array`)

## 0.1.7 - 2021-07-05

- Switch to NVRTC for CUDA runtime
- Allow running without host compiler
- Disable asserts in kernel release mode (small perf. improvement)

## 0.1.6 - 2021-06-14

- Look for CUDA toolchain in target-deps

## 0.1.5 - 2021-06-14

- Rename OgLang -> Warp
- Improve CUDA environment error checking
- Clean-up some logging, add verbose mode (`wp.config.verbose`)

## 0.1.4 - 2021-06-10

- Add support for mesh raycast

## 0.1.3 - 2021-06-09

- Add support for unary negation operator
- Add support for mutating variables during dynamic loops (non-differentiable)
- Add support for in-place operators
- Improve kernel cache start up times (avoids adjointing before cache check)
- Update README.md with requirements / examples

## 0.1.2 - 2021-06-03

- Add support for querying mesh velocities
- Add CUDA graph support, see `wp.capture_begin()`, `wp.capture_end()`, `wp.capture_launch()`
- Add explicit initialization phase, `wp.init()`
- Add variational Euler solver (sim)
- Add contact caching, switch to nonlinear friction model (sim)

- Fix for Linux/macOS support

## 0.1.1 - 2021-05-18

- Fix bug with conflicting CUDA contexts

## 0.1.0 - 2021-05-17

- Initial publish for alpha testing

[Unreleased]: https://github.com/NVIDIA/warp/compare/v1.7.1...HEAD
[1.7.1]: https://github.com/NVIDIA/warp/releases/tag/v1.7.1
[1.7.0]: https://github.com/NVIDIA/warp/releases/tag/v1.7.0
[1.6.2]: https://github.com/NVIDIA/warp/releases/tag/v1.6.2
[1.6.1]: https://github.com/NVIDIA/warp/releases/tag/v1.6.1
[1.6.0]: https://github.com/NVIDIA/warp/releases/tag/v1.6.0
[1.5.1]: https://github.com/NVIDIA/warp/releases/tag/v1.5.1
[1.5.0]: https://github.com/NVIDIA/warp/releases/tag/v1.5.0
[1.4.2]: https://github.com/NVIDIA/warp/releases/tag/v1.4.2
[1.4.1]: https://github.com/NVIDIA/warp/releases/tag/v1.4.1
[1.4.0]: https://github.com/NVIDIA/warp/releases/tag/v1.4.0
[1.3.3]: https://github.com/NVIDIA/warp/releases/tag/v1.3.3
[1.3.2]: https://github.com/NVIDIA/warp/releases/tag/v1.3.2
[1.3.1]: https://github.com/NVIDIA/warp/releases/tag/v1.3.1
[1.3.0]: https://github.com/NVIDIA/warp/releases/tag/v1.3.0
[1.2.2]: https://github.com/NVIDIA/warp/releases/tag/v1.2.2
[1.2.1]: https://github.com/NVIDIA/warp/releases/tag/v1.2.1
[1.2.0]: https://github.com/NVIDIA/warp/releases/tag/v1.2.0
[1.1.0]: https://github.com/NVIDIA/warp/releases/tag/v1.1.0
[1.0.2]: https://github.com/NVIDIA/warp/releases/tag/v1.0.2
[1.0.1]: https://github.com/NVIDIA/warp/releases/tag/v1.0.1
[1.0.0]: https://github.com/NVIDIA/warp/releases/tag/v1.0.0
[0.15.1]: https://github.com/NVIDIA/warp/releases/tag/v0.15.1
[0.15.0]: https://github.com/NVIDIA/warp/releases/tag/v0.15.0
[0.13.0]: https://github.com/NVIDIA/warp/releases/tag/v0.13.0
[0.11.0]: https://github.com/NVIDIA/warp/releases/tag/v0.11.0
[1.0.0-beta.6]: https://github.com/NVIDIA/warp/releases/tag/v1.0.0-beta.6
[1.0.0-beta.5]: https://github.com/NVIDIA/warp/releases/tag/v1.0.0-beta.5
[0.10.1]: https://github.com/NVIDIA/warp/releases/tag/v0.10.1
[0.9.0]: https://github.com/NVIDIA/warp/releases/tag/v0.9.0
[0.7.0]: https://github.com/NVIDIA/warp/releases/tag/v0.7.0
[0.5.0]: https://github.com/NVIDIA/warp/releases/tag/v0.5.0
[0.4.3]: https://github.com/NVIDIA/warp/releases/tag/v0.4.3
[0.3.1]: https://github.com/NVIDIA/warp/releases/tag/v0.3.1
[0.2.3]: https://github.com/NVIDIA/warp/releases/tag/v0.2.3
[0.2.0]: https://github.com/NVIDIA/warp/releases/tag/v0.2.0
