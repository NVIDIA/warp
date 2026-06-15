# Graph-Capturable NanoVDB Volume Rebuilds

**Status**: Implemented

**Issue**: None

## Motivation

Warp can allocate NanoVDB volumes from point arrays with
``Volume.allocate_by_tiles()`` and ``Volume.allocate_by_voxels()``. The previous
CUDA implementation delegated topology construction to NanoVDB's
``PointsToGrid`` helper. That helper is efficient, but it repeatedly copied
actual topology counts from the device to the host and synchronized the stream
before allocating exact-sized buffers and launching count-dependent work.

Those host round trips make volume construction unusable inside CUDA graph
capture. This is a problem for simulations that rebuild sparse grids every step:
the expensive point-to-grid construction remains outside the captured step even
when the rest of the step can be replayed as a graph.

The goal is to make repeated NanoVDB grid rebuilds from points graph-capturable
for the volume construction paths used by Warp. The first volume allocation is
allowed to synchronize. Subsequent rebuilds into the same volume must be free of
host synchronization and must keep actual topology counts on the device.

## Requirements

| ID | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1 | Rebuild a previously allocated CUDA NanoVDB volume from tile or voxel points during CUDA graph capture. | Must | Covers Warp's ``allocate_by_tiles`` and ``allocate_by_voxels`` use cases. |
| R2 | Keep actual node and voxel counts on the device during rebuild. | Must | Host launches use capacities, not actual counts. |
| R3 | Allow users to provide narrower persistent capacities. | Must | Avoid using ``num_points * 512`` as the persistent voxel bound for tile volumes. |
| R4 | Report insufficient capacity without writing out of bounds. | Must | Rebuild sets a device status flag that the host can inspect after graph launch. |
| R5 | Preserve public non-capture allocation behavior. | Should | Existing APIs and exact-count metadata continue to work through the shared builder. |
| R6 | Use the same broad strategy as the current builder. | Should | Morton-like NanoVDB hierarchy keys plus radix sort and run-length encode. |

**Non-goals**:

- Implement NanoVDB ``Point`` blind-data grids.
- Implement adaptive point-density search.
- Implement grid merging or multi-grid buffers.
- Make first-time allocation graph-capturable.
- Provide exact host counts during captured rebuild without an explicit
  synchronization requested by the caller.

## Design

### Approach

Construction is split into exact allocation and bounded rebuild paths that share
the same device-side counting and grid population implementation.

1. **Exact allocation** counts topology on the device, copies only the final
   counts to the host, allocates an exact-sized NanoVDB buffer, and populates it
   from the already-computed key arrays. This is the default non-capture path and
   may synchronize.
2. **Rebuildable allocation/setup** creates a persistent NanoVDB buffer with
   fixed capacities. This phase may synchronize. If the caller does not provide
   capacities, Warp may count the initial topology and use those exact counts as
   capacities.
3. **Rebuild** overwrites the existing buffer from a new point array. This phase
   is graph-capturable. It never copies actual counts to the host and never
   synchronizes the stream.

Host-side objects store capacities and conservative upper bounds. Device-side
NanoVDB headers and trees store actual counts written by rebuild kernels.
Queries that must launch kernels, such as tile and voxel coordinate extraction,
launch over host capacities and guard on device actual counts.

### Capacity Model

The user-facing capacity knobs are:

- ``max_tiles`` for tile volumes. This is the number of leaf nodes that can be
  present after deduplicating tile points.
- ``max_active_voxels`` for active-voxel index volumes.
- ``max_leaf_nodes``, ``max_lower_nodes``, and ``max_upper_nodes`` for callers
  that need tighter control over internal NanoVDB node capacities. If omitted,
  these default conservatively from the point or tile capacity.

For tile volumes, the persistent voxel upper bound is ``max_tiles * 512``. This
is the number of indexable voxels in fully allocated NanoVDB leaves and is much
tighter than ``num_points * 512`` when many input points map to duplicate tiles.

Scratch arrays can still be sized by the input point count because they are
transient and naturally bounded by the current rebuild input. Persistent storage
is bounded by the user-provided capacities.

### Rebuild Status

Each rebuild clears and then updates a device-side status word. Status bits are:

- tile or leaf capacity exceeded
- lower-node capacity exceeded
- upper-node capacity exceeded
- active-voxel capacity exceeded
- invalid input or unsupported mode

When a capacity is exceeded, rebuild truncates device writes to the relevant
capacity. The resulting volume is memory-safe but incomplete. The caller can
copy the status after graph launch to decide whether to increase capacities and
recreate the volume.

The implementation uses NanoVDB-compatible upper-root-tile keys plus local
upper, lower, and leaf offsets. This preserves the public ``int32`` coordinate
domain while still allowing a compact 64-bit key for sorting within the Warp
builder.

### Rebuild Algorithm

The builder keeps the current strategy: derive hierarchy keys, radix sort,
run-length encode unique topology, and then build NanoVDB nodes from the sorted
keys. Exact allocation and rebuildable setup both reuse the same key/count pass;
only exact allocation copies the final counts back to the host before allocating
the grid buffer.

For tile volumes:

1. Convert each input point to index space when needed.
2. Quantize to NanoVDB leaf origins.
3. Emit leaf keys and point ids.
4. Radix-sort leaf keys.
5. Run-length encode leaf keys to produce unique leaves and actual leaf count.
6. Derive lower and upper keys from leaves, radix-sort the derived parent-key
   arrays, and run-length encode them to produce actual lower and upper counts.
   The extra parent-key sort is required because parent keys are not guaranteed
   to stay contiguous after local child-coordinate bits are stripped.
7. Compare actual counts against capacities and set status bits.
8. Build root, upper, lower, and leaf nodes using kernels launched over
   capacities and guarded by actual device counts.
9. Mark every voxel in each actual leaf active. For regular value grids, set
   background and initial values. For index grids, assign dense leaf-local
   indices.
10. Write actual counts and voxel count into the NanoVDB tree on device.

For active-voxel index volumes:

1. Convert each input point to index space when needed.
2. Emit full voxel keys and point ids.
3. Radix-sort full voxel keys.
4. Run-length encode full voxel keys to produce actual active voxels.
5. Derive leaf, lower, and upper keys from the voxel keys, radix-sort each
   derived parent-key array, and run-length encode them.
6. Compare actual counts against capacities and set status bits.
7. Build root, upper, lower, and leaf nodes.
8. Set leaf value masks for unique active voxels.
9. Compute per-leaf active voxel counts, scan them on device, and write
   NanoVDB ``OnIndex`` offsets and prefix sums.
10. Write actual counts and active voxel count into the NanoVDB tree on device.

### Host Metadata

The existing ``VolumeDesc`` mirrors selected NanoVDB header/tree fields on the
host. Captured rebuilds cannot keep these exact host fields current without a
host round trip, so rebuildable volumes need descriptor fields for capacities.

Exact allocation keeps host metadata exact. Host count APIs may return
capacities for rebuildable volumes. Device kernels that enumerate tiles or
voxels must use the device grid/tree actual counts to avoid exposing stale or
unwritten entries.

### Public API

The initial API should be explicit:

- ``Volume.allocate_by_tiles(..., max_tiles=..., graph_rebuildable=True)``
- ``Volume.allocate_by_voxels(..., max_active_voxels=..., graph_rebuildable=True)``
- ``volume.rebuild_by_tiles(tile_points, status=None)``
- ``volume.rebuild_by_voxels(voxel_points, status=None)``

Exact allocation remains the default. A rebuildable volume is created when the
user opts in with ``graph_rebuildable=True`` or supplies a capacity.

The optional ``status`` argument is a one-element ``uint32`` array on the same
CUDA device. If omitted, Warp allocates and retains an internal status array
that can be queried after synchronization.

### Alternatives Considered

**Patch NanoVDB ``PointsToGrid`` directly.** This would reduce duplicate code,
but the helper is deeply exact-count oriented. It uses host counts for
allocation sizes, host loops over tile runs, host-side density iteration, and a
final stream synchronization. A Warp-owned builder can be narrower, serve both
exact allocation and graph-capturable rebuild, and does not need to preserve the
full upstream helper feature set.

**Use ``num_points`` as every persistent capacity.** This is simple and safe for
node counts, but for tile volumes it implies ``num_points * 512`` indexable
voxels. That can be vastly larger than needed when many points share tiles.
User-provided capacities are a better fit for graph replay, where fixed memory
budgets are expected.

**Make every allocation graph-capturable.** The first allocation still needs to
create host-visible descriptors and can reasonably synchronize. The important
workflow is repeated graph replay, so separating allocation from rebuild keeps
the API and implementation simpler.

## Testing Strategy

Tests should cover:

- Exact non-capture allocation of tile and active-voxel volumes.
- Non-capture creation of rebuildable tile and active-voxel volumes.
- Rebuilding inside ``wp.ScopedCapture`` and replaying the captured graph.
- Duplicate input points, verifying deduplication and status success.
- Capacity overflow, verifying status bits are set and no out-of-bounds writes
  occur.
- Index-space integer inputs and world-space floating-point inputs.
- Device enumeration helpers launched over host capacities but returning only
  actual device entries.

Tests should use ``unittest`` and be registered in the default suite through the
existing volume test module. CUDA graph tests should be restricted to devices
with CUDA memory-pool support.
