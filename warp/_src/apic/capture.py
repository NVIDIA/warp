# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIC capture helper — manages recording state and builds APICLaunchInfo structs."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import warp._src.types
from warp._src.apic.types import (
    APIC_RELOC_DATA_PTR,
    APIC_RELOC_HANDLE,
    APIC_RELOC_NULL,
    APICLaunchInfo,
    APICLaunchParamRecord,
    APICLaunchPtrLocation,
)

if TYPE_CHECKING:
    from warp._src.context import KernelHooks, ModuleExec, Runtime


class APICapture:
    """Manages APIC recording state for a single capture session.

    Created by ``capture_begin()``, attached to the ``Graph`` by ``capture_end()``.
    Handles memory region registration, ``APICLaunchInfo`` construction, and
    metadata collection for serialization.
    """

    def __init__(self, device, runtime: Runtime, apic_savable: bool):
        self.device = device
        self.runtime = runtime
        self.apic_savable = apic_savable  # Whether capture_save() is allowed

        # C++ APICState*
        self.apic_state: ctypes.c_void_p = runtime.core.wp_apic_create_state()

        # Region tracking: id(base_array) -> (region_id, base_ptr, capacity, base_array).
        # The base array is retained so Python cannot recycle its id() (which would
        # otherwise alias unrelated allocations to the same region_id) and so that
        # capture_save still has a valid base_ptr to read from.
        self._regions: dict[int, tuple[int, int, int, object]] = {}

        # Metadata collected during capture. Populated by build_launch_info()
        # on each launch and consumed by capture_save() to register modules,
        # kernels, and meshes with the C++ state before writing the .wrp file.
        self.collected_modules: dict[str, dict] = {}  # module_hash -> info
        self.collected_kernels: dict[tuple[str, str], dict] = {}  # (module_hash, kernel_key) -> info
        self.collected_mesh_ids: set[int] = set()  # mesh IDs seen during capture

    def begin_recording(self):
        self.runtime.core.wp_apic_begin_recording(self.apic_state, 1 if self.device.is_cpu else 0)

    def end_recording(self):
        self.runtime.core.wp_apic_end_recording(self.apic_state)

    def destroy(self):
        if self.apic_state:
            self.runtime.core.wp_apic_destroy_state(self.apic_state)
            self.apic_state = None

    @property
    def operation_count(self) -> int:
        if self.apic_state:
            return self.runtime.core.wp_apic_get_operation_count(self.apic_state)
        return 0

    # ---- Memory region tracking ----

    def _find_base(self, arr) -> tuple:
        """Walk the _ref chain to find the base allocation.

        Returns:
            (base_array, byte_offset) where base_array is the root allocation
            and byte_offset is the offset of arr.ptr from base_array.ptr.
        """
        base = arr
        while hasattr(base, "_ref") and base._ref is not None and hasattr(base._ref, "ptr"):
            base = base._ref
        # Empty arrays (shape=(0,)) have arr.ptr == None; treat as zero offset.
        if arr.ptr is None or base.ptr is None:
            return base, 0
        byte_offset = arr.ptr - base.ptr
        return base, byte_offset

    def _find_handle_offsets(self, dtype, base_offset: int = 0) -> list[int]:
        """Recursively find byte offsets of ``wp.handle`` fields in a type.

        Returns a list of byte offsets where handle pointers are located,
        which APIC uses to remap object handles (Mesh, Volume, BVH) after
        loading a serialized graph.
        """
        import warp._src.codegen  # noqa: PLC0415

        offsets = []
        if dtype is warp._src.types.handle:
            offsets.append(base_offset)
        elif isinstance(dtype, warp._src.codegen.Struct):
            for field_name, var in dtype.vars.items():
                field_offset = getattr(dtype.ctype, field_name).offset
                offsets.extend(self._find_handle_offsets(var.type, base_offset + field_offset))
        return offsets

    def track_array(self, arr: warp.array | None) -> tuple[int, int]:
        """Register an array's base allocation as a memory region.

        ``None`` and zero-``ptr`` arrays are accepted and return ``(-1, 0)``
        without registering anything, so callers do not need to guard the call.

        Returns:
            (region_id, byte_offset) for this array within its base region.
        """
        if arr is None or not arr.ptr:
            return -1, 0

        base, byte_offset = self._find_base(arr)
        base_id = id(base)

        if base_id in self._regions:
            region_id, _, _, _ = self._regions[base_id]
            return region_id, byte_offset

        # Register new region in C++
        region_id = self.runtime.core.wp_apic_register_memory_region_by_ptr(
            self.apic_state,
            ctypes.c_uint64(base.ptr),
            ctypes.c_uint64(base.capacity),
            ctypes.c_uint32(warp._src.types.type_size_in_bytes(base.dtype)),
        )
        if region_id == 0:
            raise RuntimeError(
                f"APIC region registration failed "
                f"(apic_state={self.apic_state}, base_id={base_id}, "
                f"ptr=0x{base.ptr:x}, capacity={base.capacity})"
            )
        self._regions[base_id] = (region_id, base.ptr, base.capacity, base)

        # Register handle pointer locations for APIC fixup
        handle_offsets = self._find_handle_offsets(base.dtype)
        if handle_offsets:
            stride = warp._src.types.type_size_in_bytes(base.dtype)
            for offset in handle_offsets:
                self.runtime.core.wp_apic_register_ptr_location(
                    self.apic_state,
                    region_id,
                    offset,
                    stride,
                )

        return region_id, byte_offset

    def get_region_id(self, arr) -> int:
        """Get the region_id for an array (must have been tracked already)."""
        if arr is None or not arr.ptr:
            return -1
        base, _ = self._find_base(arr)
        base_id = id(base)
        if base_id in self._regions:
            return self._regions[base_id][0]
        # Fall back to tracking
        region_id, _ = self.track_array(arr)
        return region_id

    # ---- APICLaunchInfo construction ----

    @staticmethod
    def _make_reloc(value_byte_offset: int, kind: int, region_id: int, region_offset: int) -> APICLaunchPtrLocation:
        reloc = APICLaunchPtrLocation()
        reloc.value_byte_offset = value_byte_offset
        reloc.region_id = region_id
        reloc.region_offset = region_offset
        reloc.kind = kind
        return reloc

    def _walk_value_pointers(self, arg_type, original_value, packed_value, base_offset: int, out: list) -> None:
        """Append one APICLaunchPtrLocation per pointer field inside a value blob.

        The blob's bytes come from the live ``packed_value`` (the same struct
        the kernel would receive in a non-captured launch). The walker
        recurses through the Warp type tree to discover pointer-typed fields
        and emits relocations that the replay path uses to patch each 8-byte
        slot with a live process-local pointer (or remapped handle id).
        """
        import warp._src.codegen  # noqa: PLC0415

        array_cls = warp._src.types.array
        indexedarray_cls = warp._src.types.indexedarray
        array_t = warp._src.types.array_t
        indexedarray_t = warp._src.types.indexedarray_t

        if warp._src.types.matches_array_class(arg_type, array_cls):
            if isinstance(original_value, array_t):
                raise NotImplementedError(
                    "APIC capture cannot record raw array ctype launch parameters; "
                    "use set_param_at_index() with a Warp array object or None instead."
                )
            data_offset = base_offset + array_t.data.offset
            grad_offset = base_offset + array_t.grad.offset
            if original_value is not None and original_value.ptr:
                region_id, byte_offset = self.track_array(original_value)
                out.append(self._make_reloc(data_offset, APIC_RELOC_DATA_PTR, region_id, byte_offset))
            else:
                out.append(self._make_reloc(data_offset, APIC_RELOC_NULL, -1, 0))
            grad = getattr(original_value, "grad", None) if original_value is not None else None
            if grad is not None and grad.ptr:
                grad_region_id, grad_byte_offset = self.track_array(grad)
                out.append(self._make_reloc(grad_offset, APIC_RELOC_DATA_PTR, grad_region_id, grad_byte_offset))
            else:
                out.append(self._make_reloc(grad_offset, APIC_RELOC_NULL, -1, 0))

        elif warp._src.types.matches_array_class(arg_type, indexedarray_cls):
            if isinstance(original_value, indexedarray_t):
                raise NotImplementedError(
                    "APIC capture cannot record raw indexedarray ctype launch parameters; "
                    "use set_param_at_index() with a Warp indexedarray object or None instead."
                )
            # indexedarray_t lays out a nested array_t at offset 0, then
            # ARRAY_MAX_DIMS indices pointers. Recurse into the nested array_t
            # for data + grad, then emit one DATA_PTR / NULL reloc per indices
            # slot.
            inner_array_type = arg_type if isinstance(arg_type, array_cls) else array_cls(dtype=arg_type.dtype)
            inner_value = original_value.data if original_value is not None else None
            inner_packed = packed_value.data if packed_value is not None else None
            self._walk_value_pointers(
                inner_array_type, inner_value, inner_packed, base_offset + indexedarray_t.data.offset, out
            )
            indices_base = base_offset + indexedarray_t.indices.offset
            ptr_size = ctypes.sizeof(ctypes.c_void_p)
            for d in range(warp._src.types.ARRAY_MAX_DIMS):
                slot_offset = indices_base + d * ptr_size
                idx_arr = None
                if original_value is not None and d < len(original_value.indices):
                    idx_arr = original_value.indices[d]
                if idx_arr is not None and idx_arr.ptr:
                    idx_rid, idx_off = self.track_array(idx_arr)
                    out.append(self._make_reloc(slot_offset, APIC_RELOC_DATA_PTR, idx_rid, idx_off))
                else:
                    out.append(self._make_reloc(slot_offset, APIC_RELOC_NULL, -1, 0))

        elif arg_type is warp._src.types.handle:
            handle_value = int(original_value) if original_value else 0
            out.append(self._make_reloc(base_offset, APIC_RELOC_HANDLE, -1, handle_value))
            if original_value:
                # TODO(apic): APIC_RELOC_HANDLE is generic but the save path only
                # registers meshes. wp.Volume and wp.Bvh handles flow through this
                # branch as raw uint64 ids; there is no runtime registry today to
                # distinguish them, so capture_save would register them as meshes
                # by mistake. Add a kind-aware registry so non-mesh handles
                # round-trip.
                self.collected_mesh_ids.add(handle_value)

        elif isinstance(arg_type, warp._src.codegen.Struct):
            # Recurse into each declared field at its ctypes offset. The
            # codegen contract is that nested array_t / indexedarray_t /
            # @wp.struct fields are embedded by value (not by pointer), so
            # offsets via ``getattr(ctype, name).offset`` match the on-device
            # struct layout.
            for field_name, var in arg_type.vars.items():
                field_offset = base_offset + getattr(arg_type.ctype, field_name).offset
                sub_value = getattr(original_value, field_name, None) if original_value is not None else None
                sub_packed = getattr(packed_value, field_name) if packed_value is not None else None
                self._walk_value_pointers(var.type, sub_value, sub_packed, field_offset, out)

        else:
            # Unsupported array kinds — refuse rather than silently producing
            # a wrong byte stream (the walker would otherwise emit zero
            # relocations and leave stale pointers inside the value blob).
            try:
                fixedarray_cls = warp._src.types.fixedarray
            except AttributeError:
                fixedarray_cls = None
            if fixedarray_cls is not None and warp._src.types.matches_array_class(arg_type, fixedarray_cls):
                raise NotImplementedError("APIC capture does not yet support wp.fixedarray launch params")
            try:
                from warp._src.fabric import fabricarray, indexedfabricarray  # noqa: PLC0415
            except ImportError:
                fabricarray = None
                indexedfabricarray = None
            if fabricarray is not None and warp._src.types.matches_array_class(arg_type, fabricarray):
                raise NotImplementedError("APIC capture does not yet support wp.fabricarray launch params")
            if indexedfabricarray is not None and warp._src.types.matches_array_class(arg_type, indexedfabricarray):
                raise NotImplementedError("APIC capture does not yet support wp.indexedfabricarray launch params")
            # Plain scalars / vec / mat / ctypes Arrays: no relocations.

    @staticmethod
    def _adjoint_blob_type(arg_type):
        """Return the type whose layout matches the *adjoint* value blob.

        For most argument kinds the adjoint pack uses the same ctype as the
        forward pack (scalar, vec/mat, plain ``wp.array``, ``wp.handle``,
        ``@wp.struct``). The one exception is ``wp.indexedarray``: its
        adjoint is the underlying gradient buffer, packed as a plain
        ``array_t`` rather than an ``indexedarray_t``. Walking that blob
        with the forward (indexedarray) type would try to read ``.data`` /
        ``.indices`` slots that don't exist.
        """
        indexedarray_cls = warp._src.types.indexedarray
        array_cls = warp._src.types.array
        if warp._src.types.matches_array_class(arg_type, indexedarray_cls):
            return array_cls(dtype=arg_type.dtype)
        return arg_type

    def _pack_param_record(
        self, rec, arg, original_value, packed_value, value_data_buf, relocs_buf, adjoint: bool = False
    ) -> None:
        """Populate a single APICLaunchParamRecord and append its relocations.

        Every kernel argument is serialized as a value blob — the exact bytes
        the live launch would pass — copied verbatim from ``packed_value``.
        Pointer fields inside the blob (array data / grad, indexedarray
        indices, wp.handle ids, fields inside ``@wp.struct``) are emitted
        as relocation entries that the replay path patches at run time.

        ``adjoint=True`` selects the adjoint-side layout for the walker
        (see ``_adjoint_blob_type``); the forward ``arg.type`` doesn't
        always match the bytes Warp packs for the corresponding adj_arg.
        """
        align = ctypes.alignment(type(packed_value))
        if align <= 0:
            align = 1
        pad = (-len(value_data_buf)) % align
        if pad:
            value_data_buf.extend(b"\x00" * pad)

        value_size = ctypes.sizeof(type(packed_value))
        rec.value_offset = len(value_data_buf)
        rec.value_size = value_size
        rec.value_align = align

        if value_size > 0:
            src_addr = ctypes.addressof(packed_value)
            blob = (ctypes.c_uint8 * value_size).from_address(src_addr)
            value_data_buf.extend(bytes(blob))

        walk_type = self._adjoint_blob_type(arg.type) if adjoint else arg.type
        start = len(relocs_buf)
        self._walk_value_pointers(walk_type, original_value, packed_value, 0, relocs_buf)
        rec.num_relocs = len(relocs_buf) - start

    def build_launch_info(
        self,
        kernel,
        module_exec: ModuleExec,
        hooks: KernelHooks,
        params: list,
        fwd_args: list,
        adjoint: bool,
        adj_args: list | None = None,
    ) -> APICLaunchInfo:
        """Build an APICLaunchInfo ctypes struct for a kernel launch.

        Args:
            kernel: The warp kernel object.
            module_exec: The ModuleExec for this kernel.
            hooks: The KernelHooks (forward/backward function pointers).
            params: The packed parameter list (params[0] is launch_bounds, then
                forward args, then adjoint args when adjoint=True).
            fwd_args: The original forward arguments (warp.array objects, scalars) before packing.
            adjoint: Whether this is a backward pass.
            adj_args: The original adjoint arguments (only used when adjoint=True).

        Returns:
            An APICLaunchInfo ctypes struct ready to pass to C++.
        """
        # Collect module/kernel metadata for later serialization
        self._collect_metadata(kernel, module_exec)

        kernel_args = kernel.adj.args
        num_params = len(kernel_args)
        # APICLaunchRecord.num_params is uint16_t on the wire; refuse anything that
        # would truncate at serialization rather than producing a misaligned record.
        if num_params > 0xFFFF:
            raise ValueError(
                f"APIC capture cannot serialize kernel '{kernel.key}': "
                f"{num_params} parameters exceeds the {0xFFFF}-parameter limit."
            )
        if adjoint:
            actual_adj = 0 if adj_args is None else len(adj_args)
            if actual_adj != num_params:
                raise ValueError(
                    f"APIC capture of backward kernel '{kernel.key}' expects {num_params} adj_args, got {actual_adj}."
                )
        param_array = (APICLaunchParamRecord * num_params)() if num_params > 0 else None

        # Per-launch value-data section + flat reloc table. The same buffers
        # are shared between forward and adjoint param records; each record's
        # ``value_offset`` and ``num_relocs`` slice them.
        value_data_buf = bytearray()
        relocs_buf: list = []

        for i, arg in enumerate(kernel_args):
            rec = param_array[i] if param_array else None
            if rec is None:
                break
            rec.param_index = i + 1  # 1-based (0 is launch_bounds)
            original_value = fwd_args[i]  # Original warp.array or scalar
            packed_value = params[1 + i]  # Packed ctypes value
            self._pack_param_record(rec, arg, original_value, packed_value, value_data_buf, relocs_buf)

        # For backward kernels, also pack the adjoint param records. The
        # validate-at-entry block above guarantees adj_args has exactly
        # num_params entries when we get here.
        adj_param_array = None
        if adjoint and num_params > 0:
            adj_param_array = (APICLaunchParamRecord * num_params)()
            adj_offset = 1 + num_params  # adj args follow fwd in `params`
            for i, arg in enumerate(kernel_args):
                rec = adj_param_array[i]
                rec.param_index = num_params + i + 1
                original_value = adj_args[i]
                packed_value = params[adj_offset + i]
                self._pack_param_record(
                    rec, arg, original_value, packed_value, value_data_buf, relocs_buf, adjoint=True
                )

        # Materialise the value_data buffer for the C++ side.
        if value_data_buf:
            value_buf = (ctypes.c_uint8 * len(value_data_buf)).from_buffer(value_data_buf)
        else:
            value_buf = None

        # Materialise the reloc table for the C++ side. APICLaunchPtrLocation is a
        # packed ctypes struct; copy each entry's bytes into a contiguous
        # array.
        if relocs_buf:
            reloc_array = (APICLaunchPtrLocation * len(relocs_buf))()
            reloc_size = ctypes.sizeof(APICLaunchPtrLocation)
            for i, r in enumerate(relocs_buf):
                ctypes.memmove(ctypes.addressof(reloc_array[i]), ctypes.addressof(r), reloc_size)
        else:
            reloc_array = None

        # Keep references so the buffers outlive the launch info struct
        # (which holds raw pointers into them).
        self._last_value_data = value_buf
        self._last_param_array = param_array
        self._last_adj_param_array = adj_param_array
        self._last_relocs = reloc_array

        # Build APICLaunchInfo
        info = APICLaunchInfo()
        key_str = kernel.key if isinstance(kernel.key, str) else kernel.key.decode("utf-8")
        hash_str = self._hash_to_str(module_exec.module_hash)
        info.kernel_key = key_str.encode("utf-8")
        info.module_hash = hash_str.encode("utf-8")
        info.is_forward = 0 if adjoint else 1
        info.params = param_array
        info.num_params = num_params
        info.kernel_dim = kernel.adj.kernel_dim
        info.value_data_size = len(value_data_buf)
        info.value_data = (
            ctypes.cast(value_buf, ctypes.POINTER(ctypes.c_uint8))
            if value_buf is not None
            else ctypes.cast(None, ctypes.POINTER(ctypes.c_uint8))
        )
        info.adj_params = (
            adj_param_array if adj_param_array is not None else ctypes.cast(None, ctypes.POINTER(APICLaunchParamRecord))
        )
        info.num_relocs = len(relocs_buf)
        info.relocs = (
            reloc_array if reloc_array is not None else ctypes.cast(None, ctypes.POINTER(APICLaunchPtrLocation))
        )

        return info

    @staticmethod
    def _hash_to_str(h) -> str:
        """Convert a module hash to a hex string (safe for filenames and C strings)."""
        if isinstance(h, bytes):
            return h.hex()
        return str(h)

    def _collect_metadata(self, kernel, module_exec: ModuleExec):
        """Collect module and kernel metadata for later serialization."""
        import os  # noqa: PLC0415

        import warp  # noqa: PLC0415

        module_hash = self._hash_to_str(module_exec.module_hash)
        if module_hash not in self.collected_modules:
            # Compute the binary path in the kernel cache (same logic as Module.load)
            module = kernel.module
            output_name = module._get_compile_output_name(self.device, block_dim=module_exec.block_dim)
            module_id = module.get_module_identifier(block_dim=module_exec.block_dim)
            module_dir = os.path.join(warp.config.kernel_cache_dir, module_id)
            binary_path = os.path.join(module_dir, output_name)

            self.collected_modules[module_hash] = {
                "module_hash": module_hash,
                "module_name": kernel.module.name,
                "module_exec": module_exec,
                "binary_path": binary_path,
                "binary_filename": output_name,
            }

        kernel_key = kernel.key
        kernel_id = (module_hash, kernel_key)
        if kernel_id not in self.collected_kernels:
            name = kernel.get_mangled_name()
            options = kernel.module.options | kernel.options

            if self.device.is_cuda:
                forward_name = name + "_cuda_kernel_forward"
                backward_name = (name + "_cuda_kernel_backward") if options.get("enable_backward", True) else ""
            else:
                forward_name = name + "_cpu_forward"
                backward_name = (name + "_cpu_backward") if options.get("enable_backward", True) else ""

            hooks = module_exec.get_kernel_hooks(kernel)
            self.collected_kernels[kernel_id] = {
                "kernel_key": kernel_key,
                "module_hash": module_hash,
                "forward_name": forward_name,
                "backward_name": backward_name,
                "forward_smem_bytes": hooks.forward_smem_bytes,
                "backward_smem_bytes": hooks.backward_smem_bytes,
                "block_dim": module_exec.block_dim,
            }
