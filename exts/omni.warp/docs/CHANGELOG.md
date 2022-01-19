# CHANGELOG

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


