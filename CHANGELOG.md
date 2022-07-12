# CHANGELOG


## [0.3.1] - 2022-07-12

- Fix for marching cubes reallocation after initialization
- Add support for closest point between line segment tests, see `wp.closest_point_edge_edge()` builtin
- Add support for per-triangle elasticity coefficients in simulation, see `wp.sim.ModelBuilder.add_cloth_mesh()`

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
- Remove custom CUB version and improve compatability with CUDA 11.7
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
- Add support for doing partial array copys, see `wp.copy()` for details
- Add support for accessing mesh data directly in kernels, see `wp.mesh_get_point()`, `wp.mesh_get_index()`, `wp.mesh_eval_face_normal()`
- Change to only compile for targets where kernel is launched (e.g.: will not compile CPU unless explicitly requested)

### Breaking Changes

- Builtin methods such as `wp.quat_identity()` now call the Warp native implementation directly and will return a `wp.quat` object instead of NumPy array
- NumPy implementations of many builtin methods have been moved to `warp.utils` and will be deprecated
- Local `@wp.func` functions should not be namespaced when called, e.g.: previously `wp.myfunc()` would work even if `myfunc()` was not a builtin
- Removed `wp.rpy2quat()`, please use `wp.quat_rpy()` instead

## [0.2.1] - 2022-05-11

- Fix for unit tests in Kit
- 
## [0.2.0] - 2022-05-02

### Warp Core

- Fix for unrolling loops with negative bounds
- Fix for unresolved symbol `hash_grid_build_device()` not found when lib is compiled without CUDA support
- Fix for failure to load nvrtc-builtins64_113.dll when user has a newer CUDA toolkit installed on their machine
- Fix for conversion of Torch tensors to wp.arrays() with a vector dtype (incorrect row count)
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
- Add `warp.torch` import by default (if PyTorch is installed)
  
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
- Add hyperbolic trigonometric functions, see wp.tanh(), wp.sinh(), wp.cosh()
- Add support for floored division on integer types
- Move tests into core library so they can be run in Kit environment


## [0.1.24] - 2022-03-03

### Warp Core

- Add NanoVDB support, see wp.volume_sample*() methods
- Add support for reading compile-time constants in kernels, see wp.constant()
- Add support for __cuda_array_interface__ protocol for zero-copy interop with PyTorch, see wp.torch.to_torch()
- Add support for additional numeric types, i8, u8, i16, u16, etc
- Add better checks for device strings during allocation / launch
- Add support for sampling random numbers with a normal distribution, see wp.randn()
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
- Add procedural noise primitives, see noise(), pnoise(), curlnoise()
- Move simulation helpers our of test into warp.sim module

## [0.1.22] - 2022-02-14

### Warp Core

- Fix for .so reloading on Linux
- Fix for while loop code-gen in some edge cases
- Add rounding functions round(), rint(), trunc(), floor(), ceil() 
- Add support for printing strings and formatted strings from kernels
- Add MSVC compiler version detection and require minimum

### Warp Sim

- Add support for universal and compound joint types

## [0.1.21] - 2022-01-19

### Warp Core

- Fix for exception on shutdown in empty wp.array objects
- Fix for hot reload of CPU kernels in Kit
- Add hash grid primitive for point-based spatial queries, see hash_grid_query(), hash_grid_query_next()
- Add new PRNG methods using PCG-based generators, see rand_init(), randf(), randi()
- Add support for AABB mesh queries, see mesh_query_aabb(), mesh_query_aabb_next()
- Add support for all Python range() loop variants
- Add builtin vec2 type and additional math operators, pow(), tan(), atan(), atan2()
- Remove dependency on CUDA driver library at build time
- Remove unused NVRTC binary dependencies (50mb smaller Linux distribution)

### Warp Sim

- Bundle import of multiple shapes for simulation nodes
- New OgnParticleVolume node for sampling shapes -> particles
- New OgnParticleSolver node for DEM style granular materials

## [0.1.20] - 2021-11-02

- Updates to the ripple solver for GTC (support for multiple colliders, buoyancy, etc)

## [0.1.19] - 2021-10-15

- Publish from 2021.3 to avoid omni.graph database incompatabilities

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
- Add rigid body joint model to warp.sim (based on Brax)
- Add Linux, MacOS support in core library
- Fix for incorrectly treating pure assignment as reference instead of value copy 
- Removes the need to transfer array to CPU before numpy conversion (will be done implicitly)
- Update the example OgnRipple wave equation solver to use bundles

## [0.1.14] - 2021-08-09

- Fix for out-of-bounds memory access in CUDA BVH
- Better error checking after kernel launches (use warp.config.verify_cuda=True)
- Fix for vec3 normalize adjoint code

## [0.1.13] - 2021-07-29

- Remove OgnShrinkWrap.py test node

## [0.1.12] - 2021-07-29

- Switch to Woop et al.'s watertight ray-tri intersection test
- Disable --fast-math in CUDA compilation step for improved precision

## [0.1.11] - 2021-07-28

- Fix for mesh_query_ray() returning incorrect t-value

## [0.1.10] - 2021-07-28

- Fix for OV extension fwatcher filters to avoid hot-reload loop due to OGN regeneration

## [0.1.9] - 2021-07-21

- Fix for loading sibling DLL paths
- Better type checking for built-in function arguments
- Added runtime docs, can now list all builtins using wp.print_builtins()

## [0.1.8] - 2021-07-14

- Fix for hot-reload of CUDA kernels
- Add Tape object for replaying differentiable kernels
- Add helpers for Torch interop (convert torch.Tensor to wp.Array)

## [0.1.7] - 2021-07-05

- Switch to NVRTC for CUDA runtime
- Allow running without host compiler
- Disable asserts in kernel release mode (small perf. improvement)

## [0.1.6] - 2021-06-14

- Look for CUDA toolchain in target-deps

## [0.1.5] - 2021-06-14

- Rename OgLang -> Warp
- Improve CUDA environment error checking
- Clean-up some logging, add verbose mode (warp.config.verbose)

## [0.1.4] - 2021-06-10

- Add support for mesh raycast

## [0.1.3] - 2021-06-09

- Add support for unary negation operator
- Add support for mutating variables during dynamic loops (non-differentiable)
- Add support for inplace operators
- Improve kernel cache start up times (avoids adjointing before cache check)
- Update README.md with requirements / examples

## [0.1.2] - 2021-06-03

- Add support for querying mesh velocities
- Add CUDA graph support, see warp.capture_begin(), warp.capture_end(), warp.capture_launch()
- Add explicit initialization phase, warp.init()
- Add variational Euler solver (sim)
- Add contact caching, switch to nonlinear friction model (sim)

- Fix for Linux/macOS support

## [0.1.1] - 2021-05-18

- Fix bug with conflicting CUDA contexts

## [0.1.0] - 2021-05-17

- Initial publish for alpha testing


