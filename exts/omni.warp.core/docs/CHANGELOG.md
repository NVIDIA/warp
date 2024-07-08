# CHANGELOG

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

## [1.1.1] - 2024-05-24

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

## [1.0.3] - 2024-04-17

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

## [0.14.0] - 2024-02-19

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

## [0.13.1] - 2024-02-22

- Ensure that the results from the `Noise Deform` are deterministic across different Kit sessions

## [0.13.0] - 2024-02-16

- Update the license to *NVIDIA Software License*, allowing commercial use (see `LICENSE.md`)
- Add `CONTRIBUTING.md` guidelines (for NVIDIA employees)
- Hash CUDA `snippet` and `adj_snippet` strings to fix caching
- Fix `build_docs.py` on Windows
- Add missing `.py` extension to `warp/tests/walkthrough_debug`
- Allow `wp.bool` usage in vector and matrix types

## [0.12.0] - 2024-02-05

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

## [1.0.0-beta.7] - 2024-01-23

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

## [1.0.0-beta.4] - 2023-11-01

- Add `wp.cbrt()` for cube root calculation
- Add `wp.mesh_furthest_point_no_sign()` to compute furthest point on a surface from a query point
- Add support for GPU BVH builds, 10-100x faster than CPU builds for large meshes
- Add support for chained comparisons, i.e.: `0 < x < 2`
- Add support for running `wp.fem` examples headless
- Fix for unit test determinism
- Fix for possible GC collection of array during graph capture
- Fix for `wp.utils.array_sum()` output initialization when used with vector types
- Coverage and documentation updates

## [1.0.0-beta.3] - 2023-10-19

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

## [1.0.0-beta.2] - 2023-09-01

- Fix for passing bool into `wp.func` functions
- Fix for deprecation warnings appearing on `stderr`, now redirected to `stdout`
- Fix for using `for i in wp.hash_grid_query(..)` syntax

## [1.0.0-beta.1] - 2023-08-29

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

## [0.10.0] - 2023-07-05

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

## [0.8.2] - 2023-04-21

- Add `ModelBuilder.soft_contact_max` to control the maximum number of soft contacts that can be registered. Use `Model.allocate_soft_contacts(new_count)` to change count on existing `Model` objects.
- Add support for `bool` parameters
- Add support for logical boolean operators with `int` types
- Fix for `wp.quat()` default constructor
- Fix conditional reassignments
- Add sign determination using angle weighted normal version of `wp.mesh_query_point()` as `wp.mesh_query_sign_normal()`
- Add sign determination using winding number of `wp.mesh_query_point()` as `wp.mesh_query_sign_winding_number()`
- Add query point without sign determination `wp.mesh_query_no_sign()`

## [0.8.1] - 2023-04-13

- Fix for regression when passing flattened numeric lists as matrix arguments to kernels
- Fix for regressions when passing `wp.struct` types with uninitialized (`None`) member attributes

## [0.8.0] - 2023-04-05

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

## [0.7.2] - 2023-02-15

- Reduce test time for vec/math types
- Clean-up CUDA disabled build pipeline
- Remove extension.gen.toml to make Kit packages Python version independent
- Handle additional cases for array indexing inside Python

## [0.7.1] - 2023-02-14

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

## [0.6.3] - 2023-01-31

- Add DLPack utilities, see `wp.from_dlpack()`, `wp.to_dlpack()`
- Add Jax utilities, see `wp.from_jax()`, `wp.to_jax()`, `wp.device_from_jax()`, `wp.device_to_jax()`
- Fix for Linux Kit extensions OM-80132, OM-80133

## [0.6.2] - 2023-01-19

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

## [0.6.1] - 2022-12-05

- Fix for non-CUDA builds
- Fix strides computation in array_t constructor, fixes a bug with accessing mesh indices through mesh.indices[]
- Disable backward pass code generation for kernel node (4-6x faster compilation)
- Switch to linbuild for universal Linux binaries (affects TeamCity builds only)

## [0.6.0] - 2022-11-28

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

## [0.5.1] - 2022-11-01

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

## [0.4.2] - 2022-09-07

- Register Warp samples to the sample browser in Kit
- Add NDEBUG flag to release mode kernel builds
- Fix for particle solver node when using a large number of particles
- Fix for broken cameras in Warp sample scenes

## [0.4.1] - 2022-08-30

- Add geometry sampling methods, see `wp.sample_unit_cube()`, `wp.sample_unit_disk()`, etc
- Add `wp.lower_bound()` for searching sorted arrays
- Add an option for disabling code-gen of backward pass to improve compilation times, see `wp.set_module_options({"enable_backward": False})`, True by default
- Fix for using Warp from Script Editor or when module does not have a `__file__` attribute
- Fix for hot reload of modules containing `wp.func()` definitions
- Fix for debug flags not being set correctly on CUDA when `wp.config.mode == "debug"`, this enables bounds checking on CUDA kernels in debug mode
- Fix for code gen of functions that do not return a value

## [0.4.0] - 2022-08-09

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

## [0.3.2] - 2022-07-19

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

## [0.3.0] - 2022-07-08

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

## [0.2.2] - 2022-05-30

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

## [0.2.1] - 2022-05-11

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

## [0.1.25] - 2022-03-20

- Add support for class methods to be Warp kernels
- Add HashGrid reserve() so it can be used with CUDA graphs
- Add support for CUDA graph capture of tape forward/backward passes
- Add support for Python 3.8.x and 3.9.x
- Add hyperbolic trigonometric functions, see `wp.tanh()`, `wp.sinh()`, `wp.cosh()`
- Add support for floored division on integer types
- Move tests into core library so they can be run in Kit environment

## [0.1.24] - 2022-03-03

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

## [0.1.23] - 2022-02-17

### Warp Core

- Fix for generated code folder being removed during Showroom installation
- Fix for macOS support
- Fix for dynamic for-loop code gen edge case
- Add procedural noise primitives, see `wp.noise()`, `wp.pnoise()`, `wp.curlnoise()`
- Move simulation helpers our of test into `wp.sim` module

## [0.1.22] - 2022-02-14

### Warp Core

- Fix for .so reloading on Linux
- Fix for while loop code-gen in some edge cases
- Add rounding functions `wp.round()`, `wp.rint()`, `wp.trunc()`, `wp.floor()`, `wp.ceil()`
- Add support for printing strings and formatted strings from kernels
- Add MSVC compiler version detection and require minimum

### Warp Sim

- Add support for universal and compound joint types

## [0.1.21] - 2022-01-19

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

## [0.1.20] - 2021-11-02

- Updates to the ripple solver for GTC (support for multiple colliders, buoyancy, etc)

## [0.1.19] - 2021-10-15

- Publish from 2021.3 to avoid omni.graph database incompatibilities

## [0.1.18] - 2021-10-08

- Enable Linux support (tested on 20.04)

## [0.1.17] - 2021-09-30

- Fix for 3x3 SVD adjoint
- Fix for A6000 GPU (bump compute model to sm_52 minimum)
- Fix for .dll unload on rebuild
- Fix for possible array destruction warnings on shutdown
- Rename spatial_transform -> transform
- Documentation update

## [0.1.16] - 2021-09-06

- Fix for case where simple assignments (a = b) incorrectly generated reference rather than value copy
- Handle passing zero-length (empty) arrays to kernels

## [0.1.15] - 2021-09-03

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

## [0.1.14] - 2021-08-09

- Fix for out-of-bounds memory access in CUDA BVH
- Better error checking after kernel launches (use `wp.config.verify_cuda=True`)
- Fix for vec3 normalize adjoint code

## [0.1.13] - 2021-07-29

- Remove OgnShrinkWrap.py test node

## [0.1.12] - 2021-07-29

- Switch to Woop et al.'s watertight ray-tri intersection test
- Disable --fast-math in CUDA compilation step for improved precision

## [0.1.11] - 2021-07-28

- Fix for `wp.mesh_query_ray()` returning incorrect t-value

## [0.1.10] - 2021-07-28

- Fix for OV extension fwatcher filters to avoid hot-reload loop due to OGN regeneration

## [0.1.9] - 2021-07-21

- Fix for loading sibling DLL paths
- Better type checking for built-in function arguments
- Added runtime docs, can now list all builtins using `wp.print_builtins()`

## [0.1.8] - 2021-07-14

- Fix for hot-reload of CUDA kernels
- Add Tape object for replaying differentiable kernels
- Add helpers for Torch interop (convert `torch.Tensor` to `wp.Array`)

## [0.1.7] - 2021-07-05

- Switch to NVRTC for CUDA runtime
- Allow running without host compiler
- Disable asserts in kernel release mode (small perf. improvement)

## [0.1.6] - 2021-06-14

- Look for CUDA toolchain in target-deps

## [0.1.5] - 2021-06-14

- Rename OgLang -> Warp
- Improve CUDA environment error checking
- Clean-up some logging, add verbose mode (`wp.config.verbose`)

## [0.1.4] - 2021-06-10

- Add support for mesh raycast

## [0.1.3] - 2021-06-09

- Add support for unary negation operator
- Add support for mutating variables during dynamic loops (non-differentiable)
- Add support for in-place operators
- Improve kernel cache start up times (avoids adjointing before cache check)
- Update README.md with requirements / examples

## [0.1.2] - 2021-06-03

- Add support for querying mesh velocities
- Add CUDA graph support, see `wp.capture_begin()`, `wp.capture_end()`, `wp.capture_launch()`
- Add explicit initialization phase, `wp.init()`
- Add variational Euler solver (sim)
- Add contact caching, switch to nonlinear friction model (sim)

- Fix for Linux/macOS support

## [0.1.1] - 2021-05-18

- Fix bug with conflicting CUDA contexts

## [0.1.0] - 2021-05-17

- Initial publish for alpha testing
