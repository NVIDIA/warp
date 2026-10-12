<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Public API comments

Read this reference only when the issue or commit changes public API behavior.

Identify:

- Python scope: constructors, functions, arguments, configuration, exceptions, and
  unsupported combinations visible from `warp`.
- Kernel scope: builtins callable only inside `@wp.kernel` or `@wp.func`.
- Behavior type: new API, expanded support, changed semantics, explicit unsupported
  behavior, or deprecation/removal.
- Example accuracy: never show a kernel-only API as a host-side Python call.

Use `### Public API behavior`, not `Public API surface`, when an existing API now
works in more cases. Separate Python and kernel scope when both matter:

```markdown
### Public API behavior

**Python scope**

- `wp.Mesh(..., bvh_constructor="cubql")` now supports the fixed behavior.
- Grouped meshes and winding-number support remain unsupported for cuBQL.

**Kernel scope**

- Existing `wp.mesh_query_point*` and `wp.mesh_query_aabb*` builtins now work with cuBQL-backed meshes.
```
