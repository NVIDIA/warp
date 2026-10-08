<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deprecations and removals

This file tracks deprecated Warp features that still need removal and records
past removals. It covers APIs, supported platforms and toolchains, and
distributed extensions, including features removed without a deprecation
period.

Warp developers use it to plan removal work and check warnings and migration
instructions before removing a feature. It is a maintenance record under
`design/`; user documentation belongs in the Sphinx sources under `docs/`.

A row for a removed namespace covers its contained APIs. Keep separate rows for
features removed before the rest of their namespace. Record changes to accepted
inputs when the old inputs stop working. Keep other behavior changes and
internal implementation changes in the changelog.

_Last reconciled against the codebase at 1.19.0.dev0 (2026-10-08), including
published removals in the changelog and the `release-1.18` tracker._

## Maintaining this file

Update the tracker in the same pull request or merge request as the change it
describes. Add user notices to the code, documentation, and a
[changelog fragment](../changelog/README.md) as appropriate.

### Adding a deprecation

1. Add a row to Pending removal. Name the affected import path, alias,
   argument, attribute, or accepted input precisely enough to check it against
   the code. Split the row if different parts have different deprecation dates
   or removal plans.
2. Record the first release containing the deprecation and the commit that
   introduced it. For unreleased changes, use the intended release version and
   recheck it before release.
3. Check the warning, changelog, and documentation separately. Record missing
   notices as `No` and anything you have not verified as `?`. If notices were
   added in different releases, record those dates in Notes.
4. Use the agreed removal target, or `Unplanned` if no target has been chosen.
   Follow the [compatibility policy](../docs/user_guide/compatibility.rst) when
   choosing a target. Schedule new deprecations and removals for feature
   releases. The usual deprecation period is at least four months, roughly
   four monthly feature releases; document any exception in Notes.
5. Include the replacement or migration instructions in Notes. Explain warning
   conditions, compatibility dependencies, or deferred removal dates that a
   maintainer needs to know.

A later warning, docstring update, or changelog announcement does not change
the original deprecation version. For example, `copy_nnz_async()` has warned
since 1.10.0, while its first changelog announcement shipped in 1.16.0. Keep
both dates, and consider when users received the notices when planning removal.

### Recording a removal

Confirm that the old API or behavior is unavailable before moving the row to
Removed. A name that remains only as a stub that raises an error counts as
removed. A deprecated alias that still works belongs in Pending removal.
Check the affected arguments and attributes as well as the main symbol; use
targeted tests when the code alone does not establish what happens.

Resolve missing notices before removal. If a removal proceeds with a missing
notice, explain the exception in Notes rather than marking that notice `Yes`
or `N/A`.

Move the row to Removed, enter the actual removal version and commit, and
drop the Warn, Changelog, and Docstring columns. Preserve the deprecation
version and commit. Earlier versions of the row remain in Git history.

If only part of a feature is removed, split the row and keep the remaining
deprecated behavior in Pending removal. For a removal with no deprecation
period, add a row directly to Removed, use `N/A` for Deprecated in and
Deprecation commit, and explain the circumstances in Notes. Use `?` when the
historical version or commit is unknown. A missing changelog entry alone does
not establish that there was no deprecation period.

### Checking releases and history

Before a release, review the rows targeting that release or an earlier one.
Confirm which removals are implemented, which still need work, and which have
been deferred. Update deferred targets and explain the change in Notes. Check
overdue targets against the code before recording a removal.

During a reconciliation:

- Compare the tracker with the current code, warning text, docstrings, and
  other API documentation. Check compatibility aliases, Python type stubs,
  and old argument spellings as well as the canonical API.
- Read the relevant released changelog sections and pending fragments. Older
  removals may appear under Changed or Breaking Changes rather than Removed.
- Verify release versions against tags and release branches. A commit on
  `main` may have shipped earlier through a backport, so its current branch
  version is not enough to establish which release contained the change.
- Compare `main` with the latest release branch and tagged release. Bring back
  corrections to this file and any matching docstrings or documentation.
- Check for duplicate rows within each table and features listed in both
  tables. Each entry should describe one removal plan or historical removal;
  keep separate entries only when their scope or dates differ.
- Replace resolved commit placeholders, check the table order, and verify
  that the commit links identify the changes described by their rows.

Update the reconciliation version and date after completing these checks.
An individual row edit does not mean the whole tracker has been reconciled.

### Reading the columns

Feature identifies what is deprecated or removed. Notes provides migration
instructions and any details that the other columns cannot capture.

Deprecated in and Removed in use full release versions without a `v` prefix,
such as `1.17.0`. Keep prerelease identifiers for changes that shipped before a
final release. Removed in records when the API stopped working, even if a
stub or error message remained in the code.

Planned removal records the target release. Use `Unplanned` when no target has
been chosen. Use `Indefinite` only after a deliberate decision to retain the
deprecated feature, and explain that decision in Notes.

Warn, Changelog, and Docstring each accept these values:

- `Yes`: the notice exists and has been checked.
- `No`: the notice is missing and needs to be added before removal.
- `N/A`: that kind of notice does not apply; explain why in Notes.
- `?`: the notice has not been verified.

Warn covers warnings emitted when using or compiling the deprecated feature.
Record conditions such as a version check or a warning that appears only when
compiling an uncached kernel in Notes. Changelog covers a released entry or a
pending fragment that announces the deprecation. Docstring covers the API
docstring notice, or the corresponding documentation for a feature without its
own docstring.

Commit, Deprecation commit, and Removal commit link to the changes that
introduced the deprecation or removal. Use a short SHA as the link text and
the full SHA in the URL. A later documentation correction or tracker update
does not replace the original deprecation commit.

Warp squashes branches when merging to `main`, including branches with one
commit. If the relevant change has not merged, use the exact marker
`Pending main merge` in its commit field. Replace it with the final commit on
`main` after the merge, including when a release branch has a different hash
for the same change.

## Pending removal

Rows are sorted by planned removal version, followed by `Unplanned` and then
`Indefinite`. Targets older than the current feature version are overdue.

| Feature | Deprecated in | Warn | Changelog | Docstring | Planned removal | Commit | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `wp.geometry.IsoSurfaceMarchingCubes` legacy `max_verts`, `max_tris`, and `device` arguments and compatibility attributes | 1.9.0 | Yes | Yes | Yes | 1.19 | [ced4300](https://github.com/NVIDIA/warp/commit/ced43005a6971fcef6738ba785eba056decfd38a) | The `max_verts`/`max_tris` arguments apply to `__init__` and `resize`; `device` applies only to `__init__`. The matching attributes are also deprecated. Runtime warnings were added in 1.15. |
| `wp.geometry.IsoSurfaceMarchingCubes.id` / `.runtime` compatibility attributes | 1.15.0 | Yes | Yes | Yes | 1.19 | [ced4300](https://github.com/NVIDIA/warp/commit/ced43005a6971fcef6738ba785eba056decfd38a) | The attributes and their runtime warnings were added to the deprecation schedule in 1.15. `id` no longer identifies a native resource; use public Warp APIs instead of accessing `runtime`. |
| `masked=True` in `warp.sparse` topology-changing ops | 1.15.0 | Yes | Yes | Yes | 1.19 | [8d8569e](https://github.com/NVIDIA/warp/commit/8d8569ec2860be1bde278eefbe0e4470470f32d6) | Use `topology="masked"` (`bsr_set_from_triplets`, `bsr_assign`, `bsr_set_transpose`, `bsr_axpy`, `bsr_mm`). |
| Per-environment sequence form of `warp.fem.Nanogrid.from_environment_voxels()` and `warp.fem.AdaptiveNanogrid.from_environment_voxels()` | 1.15.0 | Yes | Yes | Yes | 1.19 | [ed6cd7e](https://github.com/NVIDIA/warp/commit/ed6cd7e2b4dd76ed59d75258f2e38ea86c7123e4) | Pass flat `points`, `cell_levels` where applicable, `point_envs`, and `env_count` instead. |
| `warp.sparse.BsrMatrix.copy_nnz_async()` | 1.10.0 | Yes | Yes | Yes | 1.20 | [4d9b978](https://github.com/NVIDIA/warp/commit/4d9b978c9b84b0f0cd3a29c02e997bd7e590005d) | Use `warp.sparse.BsrMatrix.nnz_sync()` to read the exact host-side count; use `warp.sparse.BsrMatrix.notify_nnz_changed()` after modifying sparse storage metadata directly. First changelog announcement shipped in 1.16. |
| Implicit promotion of NumPy integer and floating-point scalars and Python, NumPy, and Warp Boolean values to composite types | 1.17.0 | Yes | Yes | N/A | 1.21 | [085ab77](https://github.com/NVIDIA/warp/commit/085ab77a379a01ab8244048903f7985744ab008f) | Applies to kernel parameters and struct fields. Construct the intended composite value explicitly. These inputs remain temporarily supported because they did not emit the scalar-promotion warning introduced in 1.12, so their deprecation window starts at 1.17 and runs the standard four feature releases. |
| `wp.bvh_query_aabb_tiled()`, `wp.bvh_query_ray_tiled()`, `wp.bvh_query_next_tiled()`, `wp.mesh_query_aabb_tiled()`, `wp.mesh_query_aabb_next_tiled()` | 1.18.0 | Yes | Yes | Yes | 1.22 | [6098a5e](https://github.com/NVIDIA/warp/commit/6098a5e5ad574096ead6d9c13527dc3f259e95fe) | Use the `tile_*` spellings instead: `wp.tile_bvh_query_aabb()`, `wp.tile_bvh_query_ray()`, `wp.tile_bvh_query_next()`, `wp.tile_mesh_query_aabb()`, and `wp.tile_mesh_query_aabb_next()` respectively. Both spellings shipped together in 1.11.0 and lower to the same native functions, so behavior is unchanged. The warning is emitted from each alias's value function during kernel code generation, so it is not seen when the kernel cache is warm. |
| `wp.geometry.IsoSurfaceMarchingCubes.extract_surface_marching_cubes()` | 1.18.0 | Yes | Yes | Yes | Unplanned | [3d10f1d](https://github.com/NVIDIA/warp/commit/3d10f1d12106542bdd124e7fdf5486ad8dece80b) | Alias of `wp.geometry.IsoSurfaceMarchingCubes.extract()`, which is shared across `wp.geometry.IsoSurfaceBase` backends. |
| `wp.geometry.IsoSurfaceMarchingCubes` legacy `domain_bounds_lower_corner` / `domain_bounds_upper_corner` constructor arguments and attributes | 1.18.0 | Yes | Yes | Yes | Unplanned | [9cd151e](https://github.com/NVIDIA/warp/commit/9cd151ec034b475dab7437931970ea6efb73d443) | Use `lower` / `upper`. Retained for compatibility with the deprecated `wp.MarchingCubes` alias and should not be removed before it. |
| `wp.MarchingCubes` top-level alias | 1.18.0 | Yes | Yes | Yes | Unplanned | [3d10f1d](https://github.com/NVIDIA/warp/commit/3d10f1d12106542bdd124e7fdf5486ad8dece80b) | Use `wp.geometry.IsoSurfaceMarchingCubes`. The isosurface API moved to `warp.geometry`; the alias resolves through `warp.__getattr__` to the class itself, so `isinstance` checks agree in both directions, and emits a warning on attribute access. |
| `wp.from_ptr()` legacy double-pointer helper | 1.1.0 | Yes | No | Yes | Indefinite | [e5ac2d9](https://github.com/NVIDIA/warp/commit/e5ac2d9ad0d4c9d3f695c7206846c9e489c3b83e) | Intentionally retained. The legacy double-pointer form is deprecated: OmniGraph code should use `omni.warp.nodes.from_omni_graph_ptr()`; otherwise construct via the `wp.array` `ptr` argument. That helper ships with the Omniverse Kit extension, whose source was removed from this repo, so it cannot be found by grepping here. May be repurposed for regular pointers in the future. |

## Removed

Sorted by removed version.

| Feature | Deprecated in | Removed in | Deprecation commit | Removal commit | Notes |
| --- | --- | --- | --- | --- | --- |
| `wp.spatial_transform` type name | N/A | 0.1.17 | N/A | [9692a80](https://github.com/NVIDIA/warp/commit/9692a803b095c42139221885439231c3f6f7f8cd) | Renamed to `wp.transform`; no separate deprecation period was recorded. |
| `wp.array.length` attribute | N/A | 0.2.0 | N/A | [61fa832](https://github.com/NVIDIA/warp/commit/61fa8327c6e4a557e38383815cd8e891e009f0cb) | Use `shape` for array dimensions and `size` for the total element count. The constructor's legacy `length` keyword remained until 1.8.0. |
| `wp.volume_sample_world()` | N/A | 0.2.0 | N/A | [eaa1ede](https://github.com/NVIDIA/warp/commit/eaa1ede41e283c288f89771e6e1f2c6b3f71da2b) | Replaced by typed index-space volume sampling APIs. Use `wp.volume_world_to_index()` before `wp.volume_sample_f()`, `wp.volume_sample_i()`, or `wp.volume_sample_v()`. No separate deprecation period was recorded. |
| `wp.rpy2quat()` | N/A | 0.2.2 | N/A | [6cb14da](https://github.com/NVIDIA/warp/commit/6cb14da202789c230ca7d8596ecac034be5553db) | Use `wp.quat_rpy()`; no separate deprecation period was recorded. |
| `capture` argument and built-in graph-capture state of `wp.Tape` | N/A | 0.2.3 | N/A | [2478d7f](https://github.com/NVIDIA/warp/commit/2478d7fde4d2f8435bfb2b89150e9109a1cc9245) | Capture tape-recorded launches using externally managed CUDA graphs. Removed the constructor argument and the `capture`, `capture_graph_forward`, and `capture_graph_backward` attributes without a recorded deprecation period. |
| `wp.runtime` top-level reference | N/A | 0.4.0 | N/A | [9cf8964](https://github.com/NVIDIA/warp/commit/9cf8964a4cdef3cfab10c077cb525859db2193e0) | Removed because the runtime is private implementation state; no separate deprecation period was recorded. |
| `.val` attribute of `wp.constant()` results | N/A | 0.8.0 | N/A | [7a302ea](https://github.com/NVIDIA/warp/commit/7a302ea1e676502aa7c95a3bf757ad9b0ef75f7d) | `wp.constant()` now returns the underlying value directly. No separate deprecation period was recorded. |
| `wp.ScopedCudaGuard` | 0.3.1 | 0.10.0 | [5622553](https://github.com/NVIDIA/warp/commit/56225532cd81c13a6c61b64095789626a1b52d24) | [418ccf3](https://github.com/NVIDIA/warp/commit/418ccf34a0179094a4ed97f21eeee785e4eef210) | Use `wp.ScopedDevice()` instead. |
| `wp.config.graph_capture_module_load_default` | N/A | 0.15.0 | N/A | [0fba836](https://github.com/NVIDIA/warp/commit/0fba8364410e2f0193c80fe0f64ce303f60b8d4f) | Renamed to `wp.config.enable_graph_capture_module_load_by_default`; no separate deprecation period was recorded. |
| Calling `wp.tid()` inside `@wp.func` functions | 1.0.0-beta.1 | 1.0.0-beta.3 | [33db40d](https://github.com/NVIDIA/warp/commit/33db40d4f033a7cb7ec1245273bc837e3f432700) | [a1a71bf](https://github.com/NVIDIA/warp/commit/a1a71bf14403c18b2a54a6c5f74f01a4c35497a8) | Obtain thread indices inside `@wp.kernel` and pass them explicitly to Warp functions. Removal accompanied grid-stride kernel support; calls now raise. |
| `wp.sim.Model.flatten()` / `wp.sim.State.flatten()` | 1.0.0-beta.1 | 1.0.0 | [729a4b6](https://github.com/NVIDIA/warp/commit/729a4b6799da44d00972f890b6571ea73cd21953) | [302b79e](https://github.com/NVIDIA/warp/commit/302b79e37725ad46eed4b49b0ba6d4d5160e0dda) | Removed both methods. The beta.1 changelog named `Model.flatten()`, but the runtime deprecation warning was attached to `State.flatten()`. Access the required arrays through their named attributes. |
| `wp.sim.Model.soft_contact_distance` compatibility property | 0.9.0 | 1.0.0 | [c30fc3b](https://github.com/NVIDIA/warp/commit/c30fc3b64d55c50adb91ab4823c0c5199eaa8fc0) | [302b79e](https://github.com/NVIDIA/warp/commit/302b79e37725ad46eed4b49b0ba6d4d5160e0dda) | Replaced by per-particle `Model.particle_radius` values and `Model.particle_max_radius`. The compatibility getter and setter were removed before the later `warp.sim` namespace removal. |
| `wp.sim.Model.joint_target` / `wp.sim.ModelBuilder.joint_target` | N/A | 1.0.0 | N/A | [302b79e](https://github.com/NVIDIA/warp/commit/302b79e37725ad46eed4b49b0ba6d4d5160e0dda) | Use `joint_act` with `joint_axis_mode` to select force/torque, target velocity, or target position control. No separate deprecation period was recorded. |
| Python-scope calls to geometry built-ins | N/A | 1.3.0 | N/A | [ce182d3](https://github.com/NVIDIA/warp/commit/ce182d38310f75e86bd7e2f9ffc3c558190063a0) | Call geometry built-ins inside Warp kernels; their Python-scope exports were removed without a recorded deprecation period. |
| `wp.Tape.check_kernel_array_access()` / `wp.Tape.reset_array_read_flags()` | N/A | 1.3.1 | N/A | [b59e92a](https://github.com/NVIDIA/warp/commit/b59e92a1857e175741dbf6fd114c9f397d3973fc) | Renamed to private methods. Set `wp.config.verify_autograd_array_access` to enable tape-managed checks; no separate deprecation period was recorded. |
| Support for Python 3.7 | N/A | 1.5.0 | N/A | [998ba51](https://github.com/NVIDIA/warp/commit/998ba51d69289ae14c76d79f7c289895fdbe2c6f) | Python 3.8 became the minimum supported version. No separate deprecation announcement was recorded in the changelog. |
| `wp.matmul()` | 1.6.0 | 1.7.0 | [ba7a865](https://github.com/NVIDIA/warp/commit/ba7a8658a0016a8d966ee8f17c8eed443de20169) | [62141f3](https://github.com/NVIDIA/warp/commit/62141f36070c0bce9d16e6c7496b02dc8980541d) | Replaced by the tile API and other frameworks. |
| `wp.batched_matmul()` | 1.6.0 | 1.7.0 | [ba7a865](https://github.com/NVIDIA/warp/commit/ba7a8658a0016a8d966ee8f17c8eed443de20169) | [62141f3](https://github.com/NVIDIA/warp/commit/62141f36070c0bce9d16e6c7496b02dc8980541d) | Removed with the CUTLASS-backed matrix multiplication API. Use tile primitives or other frameworks. |
| `wp.sim.Control.model` attribute | N/A | 1.7.0 | N/A | [b6b35c9](https://github.com/NVIDIA/warp/commit/b6b35c9e3abd0177849b1bd800591ce7fc9c0d18) | Controls no longer retain a model reference; keep the model separately. Constructor arguments and `reset()` remained deprecated until `warp.sim` was removed in 1.10.0. |
| `globalScale` input of the `OgnClothSimulate` Kit node | N/A | 1.7.0 | N/A | [affc3a8](https://github.com/NVIDIA/warp/commit/affc3a8a9ad8d829f60095147b70820e9fbda95b) | Specify cloth and contact coefficients directly. The node no longer multiplies them by a global scale; no separate deprecation period was recorded. |
| Array construction with `length` keyword | 0.2.0 | 1.8.0 | [61fa832](https://github.com/NVIDIA/warp/commit/61fa8327c6e4a557e38383815cd8e891e009f0cb) | [5ac97eb](https://github.com/NVIDIA/warp/commit/5ac97eb7c813e46925a28f86b876a5b347f7d917) | Deprecation surfaces completed in 1.6.0. |
| Array construction with `owner` keyword | 0.14.0 | 1.8.0 | [6ede9b7](https://github.com/NVIDIA/warp/commit/6ede9b7ce6efbe46bfa9f42a6aff1c6eb7efdb0f) | [5ac97eb](https://github.com/NVIDIA/warp/commit/5ac97eb7c813e46925a28f86b876a5b347f7d917) | Deprecation surfaces completed in 1.6.0. |
| `plot_kernel_jacobians()` | 1.4.0 | 1.8.0 | [2ab9695](https://github.com/NVIDIA/warp/commit/2ab969595dbb06c5b3607d8b28be47efb14559ac) | [5ac97eb](https://github.com/NVIDIA/warp/commit/5ac97eb7c813e46925a28f86b876a5b347f7d917) | Use `jacobian_plot()`. |
| `wp.mlp()` | 1.6.0 | 1.8.0 | [ba7a865](https://github.com/NVIDIA/warp/commit/ba7a8658a0016a8d966ee8f17c8eed443de20169) | [5ac97eb](https://github.com/NVIDIA/warp/commit/5ac97eb7c813e46925a28f86b876a5b347f7d917) | Use tile primitives instead. |
| `kernel` argument to `wp.autograd.jacobian()` and `wp.autograd.jacobian_fd()` | 1.6.0 | 1.8.0 | [2e8407e](https://github.com/NVIDIA/warp/commit/2e8407ecd658f280dddf079584d8f34f1532470b) | [5ac97eb](https://github.com/NVIDIA/warp/commit/5ac97eb7c813e46925a28f86b876a5b347f7d917) | Use the `function` argument instead. |
| `outputs` argument to `wp.autograd.jacobian_plot()` | 1.6.0 | 1.8.0 | [2e8407e](https://github.com/NVIDIA/warp/commit/2e8407ecd658f280dddf079584d8f34f1532470b) | [5ac97eb](https://github.com/NVIDIA/warp/commit/5ac97eb7c813e46925a28f86b876a5b347f7d917) | Remove the argument; plotting metadata supplies the output information. |
| Support for building Warp with CUDA Toolkit 11 | N/A | 1.9.0 | N/A | [4bcf6d5](https://github.com/NVIDIA/warp/commit/4bcf6d55348fddea521e1909bf403b27e4f09cab) | Build with CUDA Toolkit 12 or newer. This entry concerns the source-build toolchain, not CPU-only execution; no separate deprecation period was recorded. |
| Passing lists, tuples, and other non-Warp array arguments to built-ins at Python scope | 0.11.0 | 1.10.0 | [d9d9670](https://github.com/NVIDIA/warp/commit/d9d9670e6692b94c0790492178cc58378deac969) | [12cc631](https://github.com/NVIDIA/warp/commit/12cc63175e0d977aa926c7bd95929b044f089868) | Construct the expected Warp composite type explicitly, e.g. `wp.normalize(wp.vec3(1.0, 2.0, 3.0))`. Python numeric scalar arguments remain supported. |
| `integrate()` with `nodal` keyword | 1.5.0 | 1.10.0 | [ed6445f](https://github.com/NVIDIA/warp/commit/ed6445f072c984c1396f368acf9198af178a50bc) | [3c70ba4](https://github.com/NVIDIA/warp/commit/3c70ba4df4a820b3997b4cad5c360a72e3636702) | Use `assembly="nodal"`. |
| `wp.select()` | 1.7.0 | 1.10.0 | [0cc87b4](https://github.com/NVIDIA/warp/commit/0cc87b4d401cd9047d21030f7e910b1ab061feb5) | [76f5af2](https://github.com/NVIDIA/warp/commit/76f5af2648c216fe0e7bc8d9f8b1250535189f88) | Replaced by `wp.where()`; call now raises. |
| `wp.sim.Control.reset()` | 1.7.0 | 1.10.0 | [b6b35c9](https://github.com/NVIDIA/warp/commit/b6b35c9e3abd0177849b1bd800591ce7fc9c0d18) | [94653a4](https://github.com/NVIDIA/warp/commit/94653a4f1ce3165ce6dad251f4425a4b62395cf1) | `warp.sim` removed entirely. |
| Constructing `wp.sim.Control` with arguments | 1.7.0 | 1.10.0 | [b6b35c9](https://github.com/NVIDIA/warp/commit/b6b35c9e3abd0177849b1bd800591ce7fc9c0d18) | [94653a4](https://github.com/NVIDIA/warp/commit/94653a4f1ce3165ce6dad251f4425a4b62395cf1) | `warp.sim` removed entirely. |
| `warp.sim` | 1.8.0 | 1.10.0 | [b9c4eac](https://github.com/NVIDIA/warp/commit/b9c4eace7f0bc8a00428a13af4269bfbd029bf28) | [94653a4](https://github.com/NVIDIA/warp/commit/94653a4f1ce3165ce6dad251f4425a4b62395cf1) | Superseded by the Newton library. |
| `wp.matrix(pos, quat, scale)` built-in function | 1.8.0 | 1.10.0 | [9eb8b17](https://github.com/NVIDIA/warp/commit/9eb8b1790464df2f73c284b56156e775a902acff) | [53b06e6](https://github.com/NVIDIA/warp/commit/53b06e6db9175aaaa140ca5ed008c3be9d1dba6c) | Use `wp.transform_compose()`; call now raises. |
| Support for Intel-based macOS (x86-64) | 1.9.0 | 1.10.0 | [f4b2445](https://github.com/NVIDIA/warp/commit/f4b2445cbfb733e7d831a31fe2de28be1a830af6) | [bc57ab5](https://github.com/NVIDIA/warp/commit/bc57ab53074c64b587af0004d4ba9d02c075e358) | Now raises a `RuntimeError`. |
| `graph_compatible` for `jax_callable` | 1.8.1 | 1.11.0 | [1bd4fac](https://github.com/NVIDIA/warp/commit/1bd4fac0637cd1eed3abc9af14699b5c9e05b35c) | [93a81e0](https://github.com/NVIDIA/warp/commit/93a81e0f836edb255e91e3961a71ac8f42a509a8) | Use `graph_mode` instead. |
| Support for Python 3.8 | N/A | 1.11.0 | N/A | [034475b](https://github.com/NVIDIA/warp/commit/034475bc5fadc0b17b1fe679507dd0473d69a358) | Python 3.9 became the minimum supported version. No separate deprecation announcement was recorded in the changelog. |
| Snake-case `build_lib.py` CLI flags and `--libmathdx` / `--no_libmathdx` | N/A | 1.11.1 | N/A | [b582ad1](https://github.com/NVIDIA/warp/commit/b582ad19233ee73211f979d9523fe674d4423beb) | Use kebab-case flags, e.g. `--cuda-path` and `--llvm-source-path`, and `--use-libmathdx` / `--no-use-libmathdx`. No separate deprecation period was recorded. |
| Construct a matrix from vectors using `wp.matrix()` at kernel scope | 1.7.0 | 1.12.0 | [0b3e4c7](https://github.com/NVIDIA/warp/commit/0b3e4c7bb683829416ac1424329818291ccfede9) | [de4dffe](https://github.com/NVIDIA/warp/commit/de4dffe76136ebea59361774dc047871144fa9f4) | Call now raises. |
| Construct a matrix from vectors using `wp.matrix()` at Python scope | 1.10.0 | 1.12.0 | [e8a3969](https://github.com/NVIDIA/warp/commit/e8a3969d4b908ec24268cc592da701c7692fc7aa) | [de4dffe](https://github.com/NVIDIA/warp/commit/de4dffe76136ebea59361774dc047871144fa9f4) | Call now raises. |
| `warp.fem` `Temporary.array` attribute | 1.10.0 | 1.12.0 | [3c70ba4](https://github.com/NVIDIA/warp/commit/3c70ba4df4a820b3997b4cad5c360a72e3636702) | [08080c4](https://github.com/NVIDIA/warp/commit/08080c4684fa8e2c5e0634777fddaf8157e38c29) | `Temporary` is now a direct alias for `wp.array`. |
| Kit extension source distribution in this repository | N/A | 1.12.1 | N/A | [2d65907](https://github.com/NVIDIA/warp/commit/2d6590791d7061df1ec8f07bbb7fb7d7dc3edaff) | Removed `exts/omni.warp` and `exts/omni.warp.core` source and related packaging from this repository. This is a repository distribution removal, not removal of installed Kit extensions or their APIs; no separate deprecation period was recorded. |
| Internal namespaces and symbols not intended for public use | 1.11.0 | 1.13.0 | [6fb821b](https://github.com/NVIDIA/warp/commit/6fb821b4fafa67fc03a325dd9821d3fb54c6a361) | [5cde63d](https://github.com/NVIDIA/warp/commit/5cde63dc820e99521ac98090e31bf1b5edb2efc7) | Removed the private-API forwarding layer, including the deprecated `warp.marching_cubes` namespace shim. |
| Integer inputs to `wp.isfinite()`, `wp.isnan()`, and `wp.isinf()` | 1.11.0 | 1.13.0 | [6a68b5d](https://github.com/NVIDIA/warp/commit/6a68b5d13b6668e97d01c41a1843f164596dac75) | [ae53d66](https://github.com/NVIDIA/warp/commit/ae53d66de10ca2922305c0260b49e070cea0c311) | Use floating-point inputs. |
| Support for Python 3.9 | 1.12.0 | 1.13.0 | [464a5fc](https://github.com/NVIDIA/warp/commit/464a5fc8c86fede4102cf1ff6e09b98c40f4c858) | [7fa6b22](https://github.com/NVIDIA/warp/commit/7fa6b22940d7b63500098145f88c26fe2df58397) | Python 3.10 became the minimum supported version. Deprecation warnings were emitted at runtime and build time. |
| `warp.render.UsdRenderer.update_body_transforms()` | 1.11.0 | 1.15.0 | [acdb500](https://github.com/NVIDIA/warp/commit/acdb500a118b84413c91d07307e03e52191c0e92) | [d76f3fa](https://github.com/NVIDIA/warp/commit/d76f3fa10201c71def518841b947f3fd9aaba4f7) | Removed the non-functional method, which referenced `self.model` and `self.body_names` attributes that `UsdRenderer` does not define. |
| `warp.fem` `quadrature` and `domain` arguments of `interpolate()` | 1.12.0 | 1.15.0 | [08080c4](https://github.com/NVIDIA/warp/commit/08080c4684fa8e2c5e0634777fddaf8157e38c29) | [ad68adc](https://github.com/NVIDIA/warp/commit/ad68adc38b5a45d1b2fcfc52810639ac073914a6) | Pass a `warp.fem.Quadrature` or `warp.fem.GeometryDomain` to `at` instead. |
| `warp.fem` `space` argument of `make_space_restriction` and `make_space_partition` | 1.12.0 | 1.15.0 | [08080c4](https://github.com/NVIDIA/warp/commit/08080c4684fa8e2c5e0634777fddaf8157e38c29) | [ad68adc](https://github.com/NVIDIA/warp/commit/ad68adc38b5a45d1b2fcfc52810639ac073914a6) | Use `space_topology` for partitions and either `space_topology` or `space_partition` for restrictions. |
| `Texture.copy_from_array()` | 1.13.0 | 1.17.0 | [5748117](https://github.com/NVIDIA/warp/commit/57481170450954e1229758254c5f77db5955ac86) | [89616dd](https://github.com/NVIDIA/warp/commit/89616dd90753d2fc016bea5d945d0df9c9ee6424) | Use `Texture.copy_from()`. |
| `Texture.copy_to_array()` | 1.13.0 | 1.17.0 | [5748117](https://github.com/NVIDIA/warp/commit/57481170450954e1229758254c5f77db5955ac86) | [89616dd](https://github.com/NVIDIA/warp/commit/89616dd90753d2fc016bea5d945d0df9c9ee6424) | Use `Texture.copy_to()`. |
| Implicit promotion of Python and Warp numeric scalars to composite types | 1.12.0 | 1.17.0 | [a23b996](https://github.com/NVIDIA/warp/commit/a23b996271908284136dc86203cf0f882110ef04) | [085ab77](https://github.com/NVIDIA/warp/commit/085ab77a379a01ab8244048903f7985744ab008f) | Removed at the two behavioral conversion sites: kernel parameters now raise a `RuntimeError` and struct fields a `TypeError`. Originally targeted at 1.16, but that release shipped with the warning still in place and no changelog note, so the target moved to 1.17. NumPy integer and floating-point scalars and Python, NumPy, and Warp Boolean values remain temporarily supported with a warning and are tracked separately above. |
| `wp.spatial_jacobian()` / `wp.spatial_mass()` | N/A | 1.17.0 | N/A | [73513a1](https://github.com/NVIDIA/warp/commit/73513a126ac1b077700d850f4fc4deffe52811a9) | Removed non-functional built-in registrations, documentation, and stubs. These functions were never callable from kernels or Python, so no deprecation period applied. |
| `wp.HashGridQueryH` / `wp.HashGridQueryD` | 1.14.0 | 1.18.0 | [2717b45](https://github.com/NVIDIA/warp/commit/2717b45bad7919186e36b307c3f4ee0eeeb8c0ad) | [1d0a154](https://github.com/NVIDIA/warp/commit/1d0a1541b64cee4e5d8e90d9734a755d0281f148) | Use `wp.HashGridQuery`. The removed aliases were runtime-only and intentionally absent from public docs and stubs. |
| `warp.jax_experimental` namespace | 1.14.0 | 1.18.0 | [604a896](https://github.com/NVIDIA/warp/commit/604a8961df6d40ea64ff1e740b23581e4c72c96f) | [23b2449](https://github.com/NVIDIA/warp/commit/23b2449cbb450e3baf383e61a0146c10298317d2) | Use the top-level `warp` JAX APIs (`wp.jax_kernel`, `wp.jax_callable`, etc.). Removal was deferred from 1.16 to 1.18 and announced in the 1.16 changelog. |
| `get_jax_callable_default_graph_cache_max()` / `set_..()` | 1.14.0 | 1.18.0 | [604a896](https://github.com/NVIDIA/warp/commit/604a8961df6d40ea64ff1e740b23581e4c72c96f) | [23b2449](https://github.com/NVIDIA/warp/commit/23b2449cbb450e3baf383e61a0146c10298317d2) | Removed with `warp.jax_experimental`; pass `graph_cache_max` to `wp.jax_callable()` instead. Removal proceeded without standalone docstring notices because all four exposed call sites warned and the migration was announced in the changelog. |
| Legacy `jax_kernel()` (custom-call implementation) | 1.10.0 | 1.18.0 | [e0eeea2](https://github.com/NVIDIA/warp/commit/e0eeea2f53d460307bf34a714762544937cf9249) | [23b2449](https://github.com/NVIDIA/warp/commit/23b2449cbb450e3baf383e61a0146c10298317d2) | Removed with `warp.jax_experimental`; use the FFI implementation exposed as `wp.jax_kernel()`. |
| `warp.config.verbose` | 1.14.0 | 1.18.0 | [110917b](https://github.com/NVIDIA/warp/commit/110917bcfef0aead6cecc7af91345366b365c8f1) | [6465652](https://github.com/NVIDIA/warp/commit/6465652a53e76937bd21cc3e1ea05ce6ddf3fa33) | Use `warp.config.log_level = warp.LOG_DEBUG`. |
| `warp.config.quiet` | 1.14.0 | 1.18.0 | [110917b](https://github.com/NVIDIA/warp/commit/110917bcfef0aead6cecc7af91345366b365c8f1) | [6465652](https://github.com/NVIDIA/warp/commit/6465652a53e76937bd21cc3e1ea05ce6ddf3fa33) | Use `warp.config.log_level = warp.LOG_WARNING`. |
