# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIC capture helper — manages recording state and builds APICLaunchInfo structs."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import warp._src.types
from warp._src.apic.types import (
    APIC_MAX_DIMS,
    APICLaunchInfo,
    APICLaunchParamRecord,
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

        # C++ APICStateInternal*
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
        self.collected_kernels: dict[str, dict] = {}  # kernel_key -> info
        self.collected_mesh_ids: set[int] = set()  # mesh IDs seen during capture

    def begin_recording(self):
        self.runtime.core.wp_apic_begin_recording(self.apic_state)

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

    def track_array(self, arr) -> tuple[int, int]:
        """Register an array's base allocation as a memory region.

        Returns:
            (region_id, byte_offset) for this array within its base region.
        """
        if arr is None or arr.ptr == 0:
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
        if arr is None or arr.ptr == 0:
            return -1
        base, _ = self._find_base(arr)
        base_id = id(base)
        if base_id in self._regions:
            return self._regions[base_id][0]
        # Fall back to tracking
        region_id, _ = self.track_array(arr)
        return region_id

    # ---- APICLaunchInfo construction ----

    def build_launch_info(
        self,
        kernel,
        module_exec: ModuleExec,
        hooks: KernelHooks,
        params: list,
        fwd_args: list,
        adjoint: bool,
    ) -> APICLaunchInfo:
        """Build an APICLaunchInfo ctypes struct for a kernel launch.

        Args:
            kernel: The warp kernel object.
            module_exec: The ModuleExec for this kernel.
            hooks: The KernelHooks (forward/backward function pointers).
            params: The packed parameter list (params[0] is launch_bounds, params[1:] are ctypes args).
            fwd_args: The original forward arguments (warp.array objects, scalars) before packing.
            adjoint: Whether this is a backward pass.

        Returns:
            An APICLaunchInfo ctypes struct ready to pass to C++.
        """
        # Collect module/kernel metadata for later serialization
        self._collect_metadata(kernel, module_exec)

        # Build parameter records for kernel args
        # Use fwd_args (original warp.array objects) for array tracking,
        # and params (packed ctypes) for scalar byte extraction.
        kernel_args = kernel.adj.args
        num_params = len(kernel_args)
        param_array = (APICLaunchParamRecord * num_params)() if num_params > 0 else None

        for i, arg in enumerate(kernel_args):
            rec = param_array[i] if param_array else None
            if rec is None:
                break

            rec.param_index = i + 1  # 1-based (0 is launch_bounds)
            original_value = fwd_args[i]  # Original warp.array or scalar
            packed_value = params[1 + i]  # Packed ctypes value

            if warp._src.types.is_array(original_value):
                # Array parameter — use original warp.array for tracking
                rec.is_array = 1
                region_id, byte_offset = self.track_array(original_value)
                rec.region_id = region_id
                rec.byte_offset = byte_offset
                rec.ndim = original_value.ndim
                rec.element_size = warp._src.types.type_size_in_bytes(original_value.dtype)
                for d in range(min(original_value.ndim, APIC_MAX_DIMS)):
                    rec.shape[d] = original_value.shape[d]
                    rec.strides[d] = original_value.strides[d]
            elif warp._src.types.is_array(arg.type):
                # Array type but value is None (null array)
                rec.is_array = 1
                rec.region_id = -1
            else:
                # Scalar parameter — pack raw bytes from the ctypes-packed value
                rec.is_array = 0
                rec.region_id = -1

                # Collect mesh IDs for serialization.
                # wp.handle-typed parameters are treated as mesh handles.
                # Volume and BVH serialization is not yet supported.
                if arg.type is warp._src.types.handle and original_value:
                    self.collected_mesh_ids.add(int(original_value))
                scalar_size = ctypes.sizeof(type(packed_value))
                max_scalar_size = APIC_MAX_DIMS * 8 * 2  # 64 bytes (shape + strides)
                if scalar_size > max_scalar_size:
                    raise ValueError(
                        f"Scalar parameter too large for APIC capture: {scalar_size} bytes (max {max_scalar_size})"
                    )
                rec.byte_offset = scalar_size  # Store scalar_size in byte_offset

                shape_space = APIC_MAX_DIMS * 8  # 32 bytes
                src_addr = ctypes.addressof(packed_value)
                first_part = min(scalar_size, shape_space)
                ctypes.memmove(ctypes.addressof(rec.shape), src_addr, first_part)
                if scalar_size > shape_space:
                    ctypes.memmove(
                        ctypes.addressof(rec.strides),
                        src_addr + shape_space,
                        scalar_size - shape_space,
                    )

        # Build APICLaunchInfo
        info = APICLaunchInfo()
        key_str = kernel.key if isinstance(kernel.key, str) else kernel.key.decode("utf-8")
        hash_str = self._hash_to_str(module_exec.module_hash)
        info.kernel_key = key_str.encode("utf-8")
        info.module_hash = hash_str.encode("utf-8")
        info.is_forward = 0 if adjoint else 1
        info.params = param_array
        info.num_params = num_params

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
            output_name = module._get_compile_output_name(self.device)
            module_id = module.get_module_identifier()
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
        if kernel_key not in self.collected_kernels:
            name = kernel.get_mangled_name()
            options = kernel.module.options | kernel.options

            if self.device.is_cuda:
                forward_name = name + "_cuda_kernel_forward"
                backward_name = (name + "_cuda_kernel_backward") if options.get("enable_backward", True) else ""
            else:
                forward_name = name + "_cpu_forward"
                backward_name = (name + "_cpu_backward") if options.get("enable_backward", True) else ""

            hooks = module_exec.get_kernel_hooks(kernel)
            self.collected_kernels[kernel_key] = {
                "kernel_key": kernel_key,
                "module_hash": module_hash,
                "forward_name": forward_name,
                "backward_name": backward_name,
                "forward_smem_bytes": hooks.forward_smem_bytes,
                "backward_smem_bytes": hooks.backward_smem_bytes,
                "block_dim": kernel.options.get("block_dim", 256),
            }
