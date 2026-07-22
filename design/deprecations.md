<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deprecations and Removals

**This is a living tracker, not a design doc.** It records the history of
deprecated and removed Warp API surfaces and tracks the work remaining to
complete pending removals.

This level of detail is aimed at Warp developers, not end users, so it lives
here rather than in the Sphinx documentation under `docs/`.

_Last reconciled against the codebase at 1.15.0.dev0 (2026-06-15)._

## How to use this document

- When you deprecate an API surface, add a row to **Pending Removal**.
- The `Warn` / `Changelog` / `Docstring` columns track whether the deprecation
  has been surfaced through each channel. A `No` is an outstanding gap to close
  before removal; `N/A` means that the channel cannot apply to that deprecation.
- When a deprecation is removed, move its row to **Removed**, fill in the
  removal version and removal commit, and drop the surfacing columns. The full
  historical detail remains in this file's git history.

### Column conventions

- **Deprecated in / Removed in** — the Warp version, without a `v` prefix.
- **Planned removal** — a target version, or one of two sentinels:
  - **Unplanned** — no removal scheduled yet; the decision hasn't been made.
  - **Indefinite** — a deliberate decision (after internal discussion) to keep
    the surface deprecated but _not_ remove it.
- **Warn / Changelog / Docstring** — `Yes`, `No`, `N/A` (the channel cannot
  apply), or `?` (unknown / unverified).
- **Commit / Deprecation commit / Removal commit** — short-SHA link to the
  relevant commit. Rows added or updated in the same branch usually cannot know
  the final main-branch hash because Warp squashes every branch on merge to
  `main`, even single-commit branches. Use the exact marker `Pending main
  merge` for these commit fields, then replace it with the final main-branch
  hash in a follow-up.

## Pending Removal

Sorted by planned removal version (sentinels last). Targets earlier than the
current version are overdue.

| Feature | Deprecated in | Warn | Changelog | Docstring | Planned removal | Commit | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Implicit promotion of scalars to composite types | 1.12.0 | Yes | Yes | N/A | 1.16 | [a23b996](https://github.com/NVIDIA/warp/commit/a23b996271908284136dc86203cf0f882110ef04) | Warns at two behavioral conversion sites: kernel parameters and struct fields. No single API docstring applies. |
| `Texture.copy_from_array()` | 1.13.0 | Yes | Yes | Yes | 1.17 | [5748117](https://github.com/NVIDIA/warp/commit/57481170450954e1229758254c5f77db5955ac86) | Use `Texture.copy_from()`. |
| `Texture.copy_to_array()` | 1.13.0 | Yes | Yes | Yes | 1.17 | [5748117](https://github.com/NVIDIA/warp/commit/57481170450954e1229758254c5f77db5955ac86) | Use `Texture.copy_to()`. |
| `warp.jax_experimental` namespace | 1.14.0 | Yes | Yes | Yes | 1.18 | [604a896](https://github.com/NVIDIA/warp/commit/604a8961df6d40ea64ff1e740b23581e4c72c96f) | Use the top-level `warp` JAX APIs (`wp.jax_kernel`, `wp.jax_callable`, etc.). |
| `get_jax_callable_default_graph_cache_max()` / `set_..()` | 1.14.0 | Yes | Yes | No | 1.18 | [604a896](https://github.com/NVIDIA/warp/commit/604a8961df6d40ea64ff1e740b23581e4c72c96f) | Pass `graph_cache_max` to `wp.jax_callable()` instead. Removed with `warp.jax_experimental`. |
| Legacy `jax_kernel()` (custom-call implementation) | 1.10.0 | Yes | Yes | Yes | 1.18 | [e0eeea2](https://github.com/NVIDIA/warp/commit/e0eeea2f53d460307bf34a714762544937cf9249) | Public only as `warp.jax_experimental.custom_call.jax_kernel()`; removed with `warp.jax_experimental`. Unsupported with JAX ≥ 0.8.0; the FFI implementation is the default. |
| `warp.config.verbose` | 1.14.0 | Yes | Yes | Yes | 1.18 | [110917b](https://github.com/NVIDIA/warp/commit/110917bcfef0aead6cecc7af91345366b365c8f1) | Use `warp.config.log_level = warp.LOG_DEBUG`. |
| `warp.config.quiet` | 1.14.0 | Yes | Yes | Yes | 1.18 | [110917b](https://github.com/NVIDIA/warp/commit/110917bcfef0aead6cecc7af91345366b365c8f1) | Use `warp.config.log_level = warp.LOG_WARNING`. |
| `wp.HashGridQueryH` / `wp.HashGridQueryD` | 1.14.0 | Yes | Yes | N/A | 1.18 | [2717b45](https://github.com/NVIDIA/warp/commit/2717b45bad7919186e36b307c3f4ee0eeeb8c0ad) | Use `wp.HashGridQuery`; the runtime-only aliases are intentionally absent from public docs and stubs, so no API docstring applies. |
| `wp.MarchingCubes` legacy `max_verts`, `max_tris`, and `device` arguments and compatibility attributes | 1.9.0 | Yes | Yes | Yes | 1.19 | [ced4300](https://github.com/NVIDIA/warp/commit/ced43005a6971fcef6738ba785eba056decfd38a) | The `max_verts`/`max_tris` arguments apply to `__init__` and `resize`; `device` applies only to `__init__`. The matching attributes are also deprecated. Runtime warnings were added in 1.15. |
| `wp.MarchingCubes.id` / `wp.MarchingCubes.runtime` compatibility attributes | 1.15.0 | Yes | Yes | Yes | 1.19 | [ced4300](https://github.com/NVIDIA/warp/commit/ced43005a6971fcef6738ba785eba056decfd38a) | The attributes and their runtime warnings were added to the deprecation schedule in 1.15. `id` no longer identifies a native resource; use public Warp APIs instead of accessing `runtime`. |
| `masked=True` in `warp.sparse` topology-changing ops | 1.15.0 | Yes | Yes | Yes | 1.19 | [8d8569e](https://github.com/NVIDIA/warp/commit/8d8569ec2860be1bde278eefbe0e4470470f32d6) | Use `topology="masked"` (`bsr_set_from_triplets`, `bsr_assign`, `bsr_set_transpose`, `bsr_axpy`, `bsr_mm`). |
| Per-environment sequence form of `warp.fem.Nanogrid.from_environment_voxels()` and `warp.fem.AdaptiveNanogrid.from_environment_voxels()` | 1.15.0 | Yes | Yes | Yes | 1.19 | [ed6cd7e](https://github.com/NVIDIA/warp/commit/ed6cd7e2b4dd76ed59d75258f2e38ea86c7123e4) | Pass flat `points`, `cell_levels` where applicable, `point_envs`, and `env_count` instead. |
| `warp.sparse.BsrMatrix.copy_nnz_async()` | 1.10.0 | Yes | Yes | Yes | 1.20 | [4d9b978](https://github.com/NVIDIA/warp/commit/4d9b978c9b84b0f0cd3a29c02e997bd7e590005d) | Prefer `warp.sparse.BsrMatrix.notify_nnz_changed()`. First changelog announcement ships in 1.16. |
| `wp.from_ptr()` requires `length` parameter | 1.1.0 | Yes | No | Yes | Indefinite | [e5ac2d9](https://github.com/NVIDIA/warp/commit/e5ac2d9ad0d4c9d3f695c7206846c9e489c3b83e) | Intentionally retained. The legacy double-pointer form is deprecated: OmniGraph code should use `from_omni_graph_ptr()`; otherwise construct via the `wp.array` `ptr` argument. May be repurposed for regular pointers in the future. |

## Removed

Sorted by removed version.

| Feature | Deprecated in | Removed in | Deprecation commit | Removal commit | Notes |
| --- | --- | --- | --- | --- | --- |
| `wp.matmul()` | 1.6.0 | 1.7.0 | [ba7a865](https://github.com/NVIDIA/warp/commit/ba7a8658a0016a8d966ee8f17c8eed443de20169) | [62141f3](https://github.com/NVIDIA/warp/commit/62141f36070c0bce9d16e6c7496b02dc8980541d) | Replaced by the tile API and other frameworks. |
| Array construction with `length` keyword | 0.2.0 | 1.8.0 | [61fa832](https://github.com/NVIDIA/warp/commit/61fa8327c6e4a557e38383815cd8e891e009f0cb) | [5ac97eb](https://github.com/NVIDIA/warp/commit/5ac97eb7c813e46925a28f86b876a5b347f7d917) | Deprecation surfaces completed in 1.6.0. |
| Array construction with `owner` keyword | 0.14.0 | 1.8.0 | [6ede9b7](https://github.com/NVIDIA/warp/commit/6ede9b7ce6efbe46bfa9f42a6aff1c6eb7efdb0f) | [5ac97eb](https://github.com/NVIDIA/warp/commit/5ac97eb7c813e46925a28f86b876a5b347f7d917) | Deprecation surfaces completed in 1.6.0. |
| `plot_kernel_jacobians()` | 1.4.0 | 1.8.0 | [2ab9695](https://github.com/NVIDIA/warp/commit/2ab969595dbb06c5b3607d8b28be47efb14559ac) | [5ac97eb](https://github.com/NVIDIA/warp/commit/5ac97eb7c813e46925a28f86b876a5b347f7d917) | Use `jacobian_plot()`. |
| `wp.mlp()` | 1.6.0 | 1.8.0 | [ba7a865](https://github.com/NVIDIA/warp/commit/ba7a8658a0016a8d966ee8f17c8eed443de20169) | [5ac97eb](https://github.com/NVIDIA/warp/commit/5ac97eb7c813e46925a28f86b876a5b347f7d917) | |
| `kernel` argument to `wp.autograd.jacobian()` | 1.6.0 | 1.8.0 | [2e8407e](https://github.com/NVIDIA/warp/commit/2e8407ecd658f280dddf079584d8f34f1532470b) | [5ac97eb](https://github.com/NVIDIA/warp/commit/5ac97eb7c813e46925a28f86b876a5b347f7d917) | |
| Calling builtins with non-Warp types | 0.11.0 | 1.10.0 | [d9d9670](https://github.com/NVIDIA/warp/commit/d9d9670e6692b94c0790492178cc58378deac969) | [12cc631](https://github.com/NVIDIA/warp/commit/12cc63175e0d977aa926c7bd95929b044f089868) | |
| `integrate()` with `nodal` keyword | 1.5.0 | 1.10.0 | [ed6445f](https://github.com/NVIDIA/warp/commit/ed6445f072c984c1396f368acf9198af178a50bc) | [3c70ba4](https://github.com/NVIDIA/warp/commit/3c70ba4df4a820b3997b4cad5c360a72e3636702) | Use `assembly="nodal"`. |
| `wp.select()` | 1.7.0 | 1.10.0 | [0cc87b4](https://github.com/NVIDIA/warp/commit/0cc87b4d401cd9047d21030f7e910b1ab061feb5) | [76f5af2](https://github.com/NVIDIA/warp/commit/76f5af2648c216fe0e7bc8d9f8b1250535189f88) | Replaced by `wp.where()`; call now raises. |
| `wp.sim.Control.reset()` | 1.7.0 | 1.10.0 | [b6b35c9](https://github.com/NVIDIA/warp/commit/b6b35c9e3abd0177849b1bd800591ce7fc9c0d18) | [94653a4](https://github.com/NVIDIA/warp/commit/94653a4f1ce3165ce6dad251f4425a4b62395cf1) | `warp.sim` removed entirely. |
| Constructing `wp.sim.Control` with arguments | 1.7.0 | 1.10.0 | [b6b35c9](https://github.com/NVIDIA/warp/commit/b6b35c9e3abd0177849b1bd800591ce7fc9c0d18) | [94653a4](https://github.com/NVIDIA/warp/commit/94653a4f1ce3165ce6dad251f4425a4b62395cf1) | `warp.sim` removed entirely. |
| `warp.sim` | 1.8.0 | 1.10.0 | [b9c4eac](https://github.com/NVIDIA/warp/commit/b9c4eace7f0bc8a00428a13af4269bfbd029bf28) | [94653a4](https://github.com/NVIDIA/warp/commit/94653a4f1ce3165ce6dad251f4425a4b62395cf1) | Superseded by the Newton library. |
| `wp.matrix(pos, quat, scale)` built-in function | 1.8.0 | 1.10.0 | [9eb8b17](https://github.com/NVIDIA/warp/commit/9eb8b1790464df2f73c284b56156e775a902acff) | [53b06e6](https://github.com/NVIDIA/warp/commit/53b06e6db9175aaaa140ca5ed008c3be9d1dba6c) | Use `wp.transform_compose()`; call now raises. |
| Support for Intel-based macOS (x86-64) | 1.9.0 | 1.10.0 | [f4b2445](https://github.com/NVIDIA/warp/commit/f4b2445cbfb733e7d831a31fe2de28be1a830af6) | [bc57ab5](https://github.com/NVIDIA/warp/commit/bc57ab53074c64b587af0004d4ba9d02c075e358) | Now raises a `RuntimeError`. |
| `graph_compatible` for `jax_callable` | 1.8.1 | 1.11.0 | [1bd4fac](https://github.com/NVIDIA/warp/commit/1bd4fac0637cd1eed3abc9af14699b5c9e05b35c) | [93a81e0](https://github.com/NVIDIA/warp/commit/93a81e0f836edb255e91e3961a71ac8f42a509a8) | |
| Construct a matrix from vectors using `wp.matrix()` at kernel scope | 1.7.0 | 1.12.0 | [0b3e4c7](https://github.com/NVIDIA/warp/commit/0b3e4c7bb683829416ac1424329818291ccfede9) | [de4dffe](https://github.com/NVIDIA/warp/commit/de4dffe76136ebea59361774dc047871144fa9f4) | Call now raises. |
| Construct a matrix from vectors using `wp.matrix()` at Python scope | 1.10.0 | 1.12.0 | [e8a3969](https://github.com/NVIDIA/warp/commit/e8a3969d4b908ec24268cc592da701c7692fc7aa) | [de4dffe](https://github.com/NVIDIA/warp/commit/de4dffe76136ebea59361774dc047871144fa9f4) | Call now raises. |
| `warp.fem` `Temporary.array` attribute | 1.10.0 | 1.12.0 | [3c70ba4](https://github.com/NVIDIA/warp/commit/3c70ba4df4a820b3997b4cad5c360a72e3636702) | [08080c4](https://github.com/NVIDIA/warp/commit/08080c4684fa8e2c5e0634777fddaf8157e38c29) | `Temporary` is now a direct alias for `wp.array`. |
| Internal namespaces and symbols not intended for public use | 1.11.0 | 1.13.0 | [6fb821b](https://github.com/NVIDIA/warp/commit/6fb821b4fafa67fc03a325dd9821d3fb54c6a361) | [5cde63d](https://github.com/NVIDIA/warp/commit/5cde63dc820e99521ac98090e31bf1b5edb2efc7) | Removed the private-API forwarding layer, including the deprecated `warp.marching_cubes` namespace shim. |
| `isfinite()`, `isnan()`, and `isinf()` should only accept float types | 1.11.0 | 1.13.0 | [6a68b5d](https://github.com/NVIDIA/warp/commit/6a68b5d13b6668e97d01c41a1843f164596dac75) | [ae53d66](https://github.com/NVIDIA/warp/commit/ae53d66de10ca2922305c0260b49e070cea0c311) | |
| `warp.render.UsdRenderer.update_body_transforms()` | 1.11.0 | 1.15.0 | [acdb500](https://github.com/NVIDIA/warp/commit/acdb500a118b84413c91d07307e03e52191c0e92) | [d76f3fa](https://github.com/NVIDIA/warp/commit/d76f3fa10201c71def518841b947f3fd9aaba4f7) | Removed the non-functional method, which referenced `self.model` and `self.body_names` attributes that `UsdRenderer` does not define. |
| `warp.fem` `quadrature` and `domain` arguments of `interpolate()` | 1.12.0 | 1.15.0 | [08080c4](https://github.com/NVIDIA/warp/commit/08080c4684fa8e2c5e0634777fddaf8157e38c29) | [ad68adc](https://github.com/NVIDIA/warp/commit/ad68adc38b5a45d1b2fcfc52810639ac073914a6) | |
| `warp.fem` `space` argument of `make_space_restriction` and `make_space_partition` | 1.12.0 | 1.15.0 | [08080c4](https://github.com/NVIDIA/warp/commit/08080c4684fa8e2c5e0634777fddaf8157e38c29) | [ad68adc](https://github.com/NVIDIA/warp/commit/ad68adc38b5a45d1b2fcfc52810639ac073914a6) | |
