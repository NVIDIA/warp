# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Texture classes for hardware-accelerated sampling on GPU and software sampling on CPU."""

from __future__ import annotations

import ctypes
import enum
from typing import ClassVar

import numpy as np

from warp._src.types import (
    array,
    float16,
    float32,
    int8,
    int16,
    int32,
    is_array,
    type_is_vector,
    type_length,
    type_scalar_type,
    type_size_in_bytes,
    uint8,
    uint16,
    uint32,
)

# Note: warp._src.context.runtime is accessed lazily via self.runtime = warp._src.context.runtime
# in __init__ methods to avoid circular imports

_SUPPORTED_TEXTURE_DTYPES = (uint8, uint16, uint32, int8, int16, int32, float16, float32)


class TextureFilterMode(enum.IntEnum):
    """Filter modes for texture sampling."""

    CLOSEST = 0
    """Nearest-neighbor (point) filtering."""
    LINEAR = 1
    """Bilinear/trilinear filtering."""


class TextureAddressMode(enum.IntEnum):
    """Address modes for texture coordinates outside [0, 1]."""

    WRAP = 0
    """Wrap coordinates (tile the texture)."""
    CLAMP = 1
    """Clamp coordinates to [0, 1]."""
    MIRROR = 2
    """Mirror coordinates at boundaries."""
    BORDER = 3
    """Return 0 for coordinates outside [0, 1]."""


class MemoryType(enum.IntEnum):
    """Memory types used when copying texture data."""

    HOST = 0x01
    """The memory is a linear array on the host (CPU)."""
    DEVICE = 0x02
    """The memory is a linear array on a CUDA device."""
    ARRAY = 0x03
    """The memory is a CUDA array handle (``cudaArray_t``)."""


class texture1d_t(ctypes.Structure):
    """Structure representing a 1D texture for kernel access.

    This is the struct passed to kernels for tex1d sampling.
    """

    _fields_ = (
        ("tex", ctypes.c_uint64),
        ("width", ctypes.c_int32),
        ("num_channels", ctypes.c_int32),
    )

    def __init__(self, tex=0, width=0, num_channels=0):
        self.tex = tex
        self.width = width
        self.num_channels = num_channels


class texture2d_t(ctypes.Structure):
    """Structure representing a 2D texture for kernel access.

    This is the struct passed to kernels for tex2d sampling.
    """

    _fields_ = (
        ("tex", ctypes.c_uint64),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("num_channels", ctypes.c_int32),
    )

    def __init__(self, tex=0, width=0, height=0, num_channels=0):
        self.tex = tex
        self.width = width
        self.height = height
        self.num_channels = num_channels


class texture3d_t(ctypes.Structure):
    """Structure representing a 3D texture for kernel access.

    This is the struct passed to kernels for tex3d sampling.
    """

    _fields_ = (
        ("tex", ctypes.c_uint64),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("depth", ctypes.c_int32),
        ("num_channels", ctypes.c_int32),
    )

    def __init__(self, tex=0, width=0, height=0, depth=0, num_channels=0):
        self.tex = tex
        self.width = width
        self.height = height
        self.depth = depth
        self.num_channels = num_channels


class cuda_array_desc_t(ctypes.Structure):
    """Structure for querying CUDA array properties (``cudaArray_t``)."""

    _fields_ = (
        ("ndim", ctypes.c_int32),
        ("shape", ctypes.c_int32 * 3),
        ("num_channels", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
    )

    def __init__(self, ndim=0, shape=(), num_channels=0, dtype=0):
        self.ndim = ndim
        for i in range(ndim):
            self.shape[i] = shape[i]
        self.num_channels = num_channels
        self.dtype = dtype


class Texture:
    """Texture base class for hardware-accelerated sampling on GPU and software sampling on CPU.

    .. admonition:: Experimental

        The texture API is experimental and subject to change without a formal deprecation
        cycle.

    Textures provide hardware-accelerated filtering and addressing for regularly gridded data
    on CUDA devices. On CPU, software-based filtering and addressing is used. Supports
    linear/bilinear/trilinear interpolation and various addressing modes (wrap, clamp, mirror, border).

    Supports ``wp.uint8``, ``wp.uint16``, ``wp.uint32``, ``wp.int8``, ``wp.int16``, ``wp.int32``,
    ``wp.float16``, and ``wp.float32`` data types. Unsigned integer textures are read as normalized
    floats in [0, 1]; signed integer textures are normalized to [-1, 1]; float types are returned as-is.

    This class should not be instantiated directly. A specific subclass should be used instead
    (:class:`Texture1D`, :class:`Texture2D`, or :class:`Texture3D`).

    Example::

        import warp as wp
        import numpy as np

        # Create a 1D texture
        data_1d = np.random.rand(256).astype(np.float32)
        tex1d = wp.Texture1D(data_1d, device="cuda:0")
        # Create a 2D texture
        data_2d = np.random.rand(256, 256).astype(np.float32)
        tex2d = wp.Texture2D(data_2d, device="cuda:0")
        # Create a 3D texture
        data_3d = np.random.rand(64, 64, 64).astype(np.float32)
        tex3d = wp.Texture3D(data_3d, device="cuda:0")
    """

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._array_handle = 0
        instance._array_owner = False
        instance._tex_handle = 0
        instance._surface_handle = 0
        return instance

    def __init__(
        self,
        ndim: int,
        data: np.ndarray | array | None = None,
        width: int = 0,
        height: int = 0,
        depth: int = 0,
        num_channels: int = 0,
        dtype=None,
        filter_mode: TextureFilterMode = TextureFilterMode.LINEAR,
        address_mode: TextureAddressMode | tuple[TextureAddressMode, ...] = TextureAddressMode.CLAMP,
        address_mode_u: TextureAddressMode | None = None,
        address_mode_v: TextureAddressMode | None = None,
        address_mode_w: TextureAddressMode | None = None,
        normalized_coords: bool = True,
        device=None,
        surface_access: bool = False,
        cuda_array: int = 0,
    ):
        """Create a texture.

        Args:
            ndim: Number of texture dimensions.
            data: Initial texture data as a NumPy array or Warp array.
                For 1D: shape ``(width,)`` or ``(width, num_channels)``.
                For 2D: shape ``(height, width)`` or ``(height, width, num_channels)``.
                For 3D: shape ``(depth, height, width)`` or ``(depth, height, width, num_channels)``.
                Supported dtypes: ``wp.uint8``, ``wp.uint16``, ``wp.uint32``,
                ``wp.int8``, ``wp.int16``, ``wp.int32``, ``wp.float16``, ``wp.float32``.
            width: Texture width (required if ``data`` is ``None``).
            height: Texture height (required if ``data`` is ``None``).
            depth: Texture depth (required if ``data`` is ``None`` for 3D textures).
            num_channels: Number of channels (1, 2, or 4). Only used when
                ``data`` is ``None``.
            dtype: Data type. Only used when ``data`` is ``None``; otherwise
                inferred from the data.
            filter_mode: Filtering mode, see :class:`TextureFilterMode`.
            address_mode: Address mode for all axes, see :class:`TextureAddressMode`.
                Can be a single int or a tuple of per-axis values.
            address_mode_u: Per-axis address mode for U. Overrides
                ``address_mode`` if specified.
            address_mode_v: Per-axis address mode for V. Overrides
                ``address_mode`` if specified.
            address_mode_w: Per-axis address mode for W (3D only). Overrides
                ``address_mode`` if specified.
            normalized_coords: If ``True``, coordinates are in ``[0, 1]``
                range. If ``False``, coordinates are in texel space.
            device: Device on which to create the texture.
            surface_access: If ``True`` and ``device`` is CUDA, allocates the backing
                CUDA array with surface load/store support so :attr:`cuda_surface`
                can be used.
            cuda_array: CUDA array handle to wrap an external texture (``cudaArray_t``).
        """
        import warp._src.context  # noqa: PLC0415

        if ndim < 1 or ndim > 3:
            raise ValueError(f"Texture dimensionality must be 1, 2, or 3, got {ndim}")

        # Note: get_device() calls wp.init() if needed, so we must capture
        # self.runtime *after* this call to ensure it is not None.
        device = warp._src.context.get_device(device)
        self._runtime = warp._src.context.runtime

        # Resolve address modes
        address_mode_u = self._resolve_address_mode(address_mode, address_mode_u, 0)
        address_mode_v = (
            self._resolve_address_mode(address_mode, address_mode_v, 1) if ndim > 1 else TextureAddressMode.CLAMP
        )
        address_mode_w = (
            self._resolve_address_mode(address_mode, address_mode_w, 2) if ndim > 2 else TextureAddressMode.CLAMP
        )

        # if an external CUDA array was given, infer texture shape and dtype from it
        if cuda_array:
            if not device.is_cuda:
                raise ValueError("Texture cuda_array was given, but the device is not a CUDA device")

            # we don't own this CUDA array, just aliasing
            self._array_owner = False
            self._array_handle = cuda_array

            # get CUDA array descriptor
            texture_desc = cuda_array_desc_t()
            result = warp._src.context.runtime.core.wp_texture_descriptor_from_cuda_array(
                device.context, self._array_handle, ctypes.byref(texture_desc)
            )
            if not result:
                raise RuntimeError(f"Failed to get descriptor for cuda_array {self._array_handle}")

            width = int(texture_desc.shape[0])
            height = int(texture_desc.shape[1])
            depth = int(texture_desc.shape[2])
            ndim = int(texture_desc.ndim)
            num_channels = int(texture_desc.num_channels)
            dtype_code = int(texture_desc.dtype)
            dtype = self._code_to_dtype(dtype_code)

        # if data was given, infer shape and dtype from it and ensure that it's consistent with other arguments
        if data is not None:
            if isinstance(data, np.ndarray):
                # convert to Warp array, but keep it on the CPU for direct texture upload
                data = array(data, device="cpu")
            if not is_array(data):
                raise ValueError("Data must be a Warp array or NumPy array")

            data_width, data_height, data_depth, data_channels, data_dtype = self._shape_from_warp_array(data, ndim)

            # verify argument values match the given data, if specified
            if width == 0:
                width = data_width
            elif data_width != width:
                raise ValueError("Specified texture width does not match data width")
            if height == 0:
                height = data_height
            elif data_height != height:
                raise ValueError("Specified texture height does not match data height")
            if depth == 0:
                depth = data_depth
            elif data_depth != depth:
                raise ValueError("Specified texture depth does not match data depth")
            if num_channels == 0:
                num_channels = data_channels
            elif data_channels != num_channels:
                raise ValueError(
                    f"Specified texture channels do not match data channels (expected {num_channels}, got {data_channels})"
                )
            if dtype is None:
                dtype = data_dtype
            elif data_dtype != dtype:
                raise ValueError("Specified texture dtype does not match data dtype")

        # make sure everything is sane
        if width <= 0:
            raise ValueError("Texture width must be a positive integer")
        if height <= 0 and ndim > 1:
            raise ValueError("Texture height must be a positive integer")
        if depth <= 0 and ndim > 2:
            raise ValueError("Texture depth must be a positive integer")
        if num_channels not in (1, 2, 4):
            raise ValueError("Texture num_channels must be 1, 2, or 4")
        dtype = self._canonicalize_dtype(dtype)
        dtype_code = self._dtype_to_code(dtype)

        self.device = device
        self._width = width
        self._height = height if ndim > 1 else 1
        self._depth = depth if ndim > 2 else 1
        self._ndim = ndim
        self._num_channels = num_channels
        self._dtype = dtype
        self._dtype_code = dtype_code
        self._filter_mode = filter_mode
        self._address_mode_u = address_mode_u
        self._address_mode_v = address_mode_v
        self._address_mode_w = address_mode_w
        self._normalized_coords = normalized_coords
        self._surface_access = bool(surface_access and device.is_cuda)

        # create texture
        if device.is_cuda:
            # create CUDA array if it was not provided
            if not self._array_handle:
                self._array_owner = True
                c_shape = (ctypes.c_int * 3)(width, height, depth)
                self._array_handle = self._runtime.core.wp_texture_create_device(
                    device.context,
                    ndim,
                    c_shape,
                    num_channels,
                    dtype_code,
                    surface_access,
                )
                if not self._array_handle:
                    raise RuntimeError(f"Failed to create CUDA texture: {self._runtime.get_error_string()}")

            # create CUDA texture object
            c_address_modes = (ctypes.c_int * 3)(address_mode_u, address_mode_v, address_mode_w)
            self._tex_handle = self._runtime.core.wp_texture_object_create_device(
                device.context,
                self._array_handle,
                ndim,
                filter_mode,
                c_address_modes,
                normalized_coords,
            )
            if not self._tex_handle:
                raise RuntimeError(f"Failed to create CUDA texture object: {self._runtime.get_error_string()}")
        else:
            # create CPU texture
            c_shape = (ctypes.c_int * 3)(width, height, depth)
            c_address_modes = (ctypes.c_int * 3)(address_mode_u, address_mode_v, address_mode_w)
            host_ptr = ctypes.c_void_p()
            self._tex_handle = self._runtime.core.wp_texture_create_host(
                ndim,
                c_shape,
                num_channels,
                dtype_code,
                surface_access,
                filter_mode,
                c_address_modes,
                normalized_coords,
                ctypes.byref(host_ptr),
            )
            if not self._tex_handle or not host_ptr:
                raise RuntimeError(f"Failed to create CPU texture: {self._runtime.get_error_string()}")
            self._host_ptr = host_ptr.value

        if data is not None:
            self.copy_from(data)

    def __del__(self):
        if self._tex_handle == 0:
            return

        try:
            if self.device.is_cuda:
                with self.device.context_guard:
                    if self._tex_handle:
                        self._runtime.core.wp_texture_object_destroy_device(self.device.context, self._tex_handle)
                    if self._surface_handle:
                        self._runtime.core.wp_surface_object_destroy_device(self.device.context, self._surface_handle)
                    if self._array_handle and self._array_owner:
                        self._runtime.core.wp_texture_destroy_device(self.device.context, self._array_handle)
            else:
                self._runtime.core.wp_texture_destroy_host(self._tex_handle)
        except (TypeError, AttributeError):
            pass

    @staticmethod
    def _validate_texture_dtype(dtype):
        if dtype not in _SUPPORTED_TEXTURE_DTYPES:
            raise ValueError(
                f"Unsupported texture dtype: {dtype}. Supported: uint8, uint16, uint32, int8, int16, int32, float16, float32"
            )

    @staticmethod
    def _canonicalize_dtype(dtype):
        if dtype is float:
            return float32
        if dtype is int:
            return int32
        if dtype in _SUPPORTED_TEXTURE_DTYPES:
            return dtype
        raise ValueError(
            f"Unsupported texture dtype: {dtype}. Supported: uint8, uint16, uint32, int8, int16, int32, float16, float32"
        )

    @classmethod
    def _shape_from_warp_array(cls, wp_data: array, tex_ndim: int):
        dtype = wp_data.dtype
        scalar_dtype = type_scalar_type(dtype)
        cls._validate_texture_dtype(scalar_dtype)
        canonical_dtype = scalar_dtype

        if type_is_vector(dtype):
            num_channels = int(type_length(dtype))
            if tex_ndim == 1:
                if wp_data.ndim != 1:
                    raise ValueError("1D texture data with vector dtype must be a 1D wp.array")
                (width,) = wp_data.shape
                height = 1
                depth = 1
            elif tex_ndim == 2:
                if wp_data.ndim != 2:
                    raise ValueError("2D texture data with vector dtype must be a 2D wp.array")
                height, width = wp_data.shape
                depth = 1
            else:
                if wp_data.ndim != 3:
                    raise ValueError("3D texture data with vector dtype must be a 3D wp.array")
                depth, height, width = wp_data.shape
            return int(width), int(height), int(depth), num_channels, canonical_dtype

        if tex_ndim == 1:
            if wp_data.ndim == 1:
                (width,) = wp_data.shape
                num_channels = 1
                height = 1
                depth = 1
            elif wp_data.ndim == 2:
                width, num_channels = wp_data.shape
                height = 1
                depth = 1
            else:
                raise ValueError("1D texture data must be 1D or 2D array")
        elif tex_ndim == 2:
            if wp_data.ndim == 2:
                height, width = wp_data.shape
                num_channels = 1
                depth = 1
            elif wp_data.ndim == 3:
                height, width, num_channels = wp_data.shape
                depth = 1
            else:
                raise ValueError("2D texture data must be 2D or 3D array")
        else:
            if wp_data.ndim == 3:
                depth, height, width = wp_data.shape
                num_channels = 1
            elif wp_data.ndim == 4:
                depth, height, width, num_channels = wp_data.shape
            else:
                raise ValueError("3D texture data must be 3D or 4D array")

        return int(width), int(height), int(depth), int(num_channels), canonical_dtype

    def _get_shape(self):
        # return the array shape that represents this texture's data
        if self._ndim == 1:
            shape = (self.width,)
        elif self._ndim == 2:
            shape = (self.height, self.width)
        else:
            shape = (self.depth, self.height, self.width)
        if self._num_channels == 1:
            return shape
        else:
            return (*shape, self._num_channels)

    def copy_from(self, src: array | np.ndarray | Texture):
        """Copy texture data from a source.

        Args:
            src: The source can be a Warp array on the same device as the texture, a CPU Warp array,
                a NumPy array, or another texture.
        """
        if isinstance(src, Texture):
            if src.width != self.width or src.height != self.height or src.depth != self.depth:
                raise ValueError("Incompatible texture shapes for copy")
            if src.dtype != self.dtype or src.num_channels != self.num_channels:
                raise ValueError("Incompatible texture data types for copy")
        else:
            if isinstance(src, np.ndarray):
                src = array(src, device="cpu", copy=False)
            elif not isinstance(src, array):
                raise ValueError(f"Expected contiguous array or Texture, got {type(src)}")

            arr_width, arr_height, arr_depth, arr_channels, arr_dtype = self._shape_from_warp_array(src, self._ndim)
            if arr_width != self.width or arr_height != self.height or arr_depth != self.depth:
                raise ValueError("Incompatible array shape for copy")
            if arr_dtype != self.dtype or arr_channels != self.num_channels:
                raise ValueError("Incompatible array data type for copy")
            if not src.contiguous():
                raise ValueError("Source array must be contiguous")

        if self.device.is_cuda:
            if isinstance(src, Texture):
                if src.device == self.device:
                    src_memory_type = MemoryType.ARRAY
                    src_handle = src.cuda_array
                    src_pitch = 0  # ignored
                    src_height = 0  # ignored
                elif src.device.is_cpu:
                    src_memory_type = MemoryType.HOST
                    src_handle = src._host_ptr
                    src_pitch = self.width * self.num_channels * type_size_in_bytes(self.dtype)
                    src_height = self.height
                else:
                    raise ValueError(
                        f"Source texture must be on the same CUDA device (expected {self.device}, got {src.device})"
                    )
            else:
                # src is a contiguous Warp array
                if src.device == self.device:
                    src_memory_type = MemoryType.DEVICE
                    src_handle = src.ptr
                elif src.device.is_cpu:
                    src_memory_type = MemoryType.HOST
                    src_handle = src.ptr
                else:
                    raise ValueError(
                        f"Source array must be on the same CUDA device (expected {self.device}, got {src.device})"
                    )
                src_pitch = self.width * self.num_channels * type_size_in_bytes(self.dtype)
                src_height = arr_height

            width_bytes = self.width * self.num_channels * type_size_in_bytes(self.dtype)
            height = self.height
            depth = self.depth

            result = self._runtime.core.wp_texture_copy_device(
                self.device.context,
                width_bytes,
                height,
                depth,
                MemoryType.ARRAY,
                self.cuda_array,
                0,
                0,
                src_memory_type,
                src_handle,
                src_pitch,
                src_height,
                self.device.stream.cuda_stream,
            )
            if not result:
                raise RuntimeError(f"Failed to copy to texture: {self._runtime.get_error_string()}")
        else:
            import warp._src.context  # noqa: PLC0415

            # CPU textures can be copied as contiguous arrays, but copying to a CUDA texture is a special case.
            if isinstance(src, Texture):
                if src.device.is_cuda:
                    return src.copy_to(self)
                src = array(ptr=src._host_ptr, shape=src._get_shape(), dtype=src.dtype, device=src.device)

            self_array = array(ptr=self._host_ptr, shape=self._get_shape(), dtype=self.dtype, device=self.device)
            warp._src.context.copy(self_array, src)

    def copy_to(self, dst: array | np.ndarray | Texture):
        """Copy texture data to a destination.

        Args:
            dst: The destination can be a Warp array on the same device as the texture, a CPU Warp array,
                a NumPy array, or another texture.
        """
        if isinstance(dst, Texture):
            if dst.width != self.width or dst.height != self.height or dst.depth != self.depth:
                raise ValueError("Incompatible texture shapes for copy")
            if dst.dtype != self.dtype or dst.num_channels != self.num_channels:
                raise ValueError("Incompatible texture data types for copy")
        else:
            if isinstance(dst, np.ndarray):
                dst = array(dst, device="cpu", copy=False)
            elif not isinstance(dst, array):
                raise ValueError(f"Expected contiguous array or Texture, got {type(dst)}")

            arr_width, arr_height, arr_depth, arr_channels, arr_dtype = self._shape_from_warp_array(dst, self._ndim)
            if arr_width != self.width or arr_height != self.height or arr_depth != self.depth:
                raise ValueError("Incompatible array shape for copy")
            if arr_dtype != self.dtype or arr_channels != self.num_channels:
                raise ValueError("Incompatible array data type for copy")
            if not dst.contiguous():
                raise ValueError("Destination array must be contiguous")

        if self.device.is_cuda:
            if isinstance(dst, Texture):
                if dst.device == self.device:
                    dst_memory_type = MemoryType.ARRAY
                    dst_handle = dst.cuda_array
                    dst_pitch = 0  # ignored
                    dst_height = 0  # ignored
                elif dst.device.is_cpu:
                    dst_memory_type = MemoryType.HOST
                    dst_handle = dst._host_ptr
                    dst_pitch = self.width * self.num_channels * type_size_in_bytes(self.dtype)
                    dst_height = self.height
                else:
                    raise ValueError(
                        f"Destination texture must be on the same CUDA device (expected {self.device}, got {dst.device})"
                    )
            else:
                # dst is a contiguous Warp array
                if dst.device == self.device:
                    dst_memory_type = MemoryType.DEVICE
                    dst_handle = dst.ptr
                elif dst.device.is_cpu:
                    dst_memory_type = MemoryType.HOST
                    dst_handle = dst.ptr
                else:
                    raise ValueError(
                        f"Destination array must be on the same CUDA device (expected {self.device}, got {dst.device})"
                    )
                dst_pitch = self.width * self.num_channels * type_size_in_bytes(self.dtype)
                dst_height = arr_height

            width_bytes = self.width * self.num_channels * type_size_in_bytes(self.dtype)
            height = self.height
            depth = self.depth

            result = self._runtime.core.wp_texture_copy_device(
                self.device.context,
                width_bytes,
                height,
                depth,
                dst_memory_type,
                dst_handle,
                dst_pitch,
                dst_height,
                MemoryType.ARRAY,
                self.cuda_array,
                0,
                0,
                self.device.stream.cuda_stream,
            )
            if not result:
                raise RuntimeError(f"Failed to copy to texture: {self._runtime.get_error_string()}")
        else:
            import warp._src.context  # noqa: PLC0415

            # CPU textures can be copied as contiguous arrays, but copying to a CUDA texture is a special case.
            if isinstance(dst, Texture):
                if dst.device.is_cuda:
                    return dst.copy_from(self)
                dst = array(ptr=dst._host_ptr, shape=dst._get_shape(), dtype=dst.dtype, device=dst.device)

            self_array = array(ptr=self._host_ptr, shape=self._get_shape(), dtype=self.dtype, device=self.device)
            warp._src.context.copy(dst, self_array)

    def copy_from_array(self, src: array):
        """Copy from a CUDA Warp array into this texture's CUDA array.
        Deprecated, use ``Texture.copy_from()`` method instead."""
        import warp._src.utils  # noqa: PLC0415

        warp._src.utils.warn(
            "The Texture.copy_from_array() method is deprecated, use Texture.copy_from() instead.",
            DeprecationWarning,
        )

        self.copy_from(src)

    def copy_to_array(self, dst: array):
        """Copy from this texture's CUDA array into a CUDA Warp array.
        Deprecated, use ``Texture.copy_to()`` method instead."""
        import warp._src.utils  # noqa: PLC0415

        warp._src.utils.warn(
            "The Texture.copy_to_array() method is deprecated, use Texture.copy_to() instead.",
            DeprecationWarning,
        )

        self.copy_to(dst)

    @staticmethod
    def _dtype_to_code(dtype):
        """Convert dtype to internal dtype code (must match C++ WP_TEXTURE_DTYPE_* constants)."""
        _code_map = {
            uint8: 0,
            uint16: 1,
            float32: 2,
            int8: 3,
            int16: 4,
            float16: 5,
            uint32: 6,
            int32: 7,
        }
        code = _code_map.get(dtype)
        if code is None:
            raise ValueError(
                f"Unsupported texture dtype: {dtype}. Supported: uint8, uint16, uint32, int8, int16, int32, float16, float32"
            )
        return code

    @staticmethod
    def _code_to_dtype(code):
        """Convert internal type code to Warp dtype (must match C++ WP_TEXTURE_DTYPE_* constants)."""
        _code_map = {
            0: uint8,
            1: uint16,
            2: float32,
            3: int8,
            4: int16,
            5: float16,
            6: uint32,
            7: int32,
        }
        dtype = _code_map.get(code)
        if dtype is None:
            raise ValueError(f"Unsupported texture type code: {code}")
        return dtype

    @staticmethod
    def _resolve_address_mode(address_mode, axis_address_mode, axis_index):
        """Resolve address mode for a single axis."""
        if axis_address_mode is not None:
            return axis_address_mode
        elif address_mode is not None:
            if isinstance(address_mode, tuple):
                return address_mode[axis_index] if axis_index < len(address_mode) else 1
            else:
                return address_mode
        else:
            return TextureAddressMode.CLAMP

    @property
    def ndim(self) -> int:
        """Texture dimensionality (1, 2, or 3)."""
        return self._ndim

    @property
    def width(self) -> int:
        """Texture width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Texture height in pixels."""
        return self._height

    @property
    def depth(self) -> int:
        """Texture depth in pixels (1 for 2D textures)."""
        return self._depth

    @property
    def num_channels(self) -> int:
        """Number of channels."""
        return self._num_channels

    @property
    def dtype(self):
        """Data type of the texture."""
        return self._dtype

    @property
    def address_mode_u(self) -> int:
        """Address mode for U axis."""
        return self._address_mode_u

    @property
    def address_mode_v(self) -> int:
        """Address mode for V axis."""
        return self._address_mode_v

    @property
    def address_mode_w(self) -> int:
        """Address mode for W axis (3D only)."""
        return self._address_mode_w

    @property
    def normalized_coords(self) -> bool:
        """Whether texture uses normalized coordinates."""
        return self._normalized_coords

    @property
    def id(self) -> int:
        """Device-independent texture identifier.

        On CUDA textures, this is the same handle as :attr:`cuda_texture` (``cudaTextureObject_t``).
        On host textures, this is the backing host texture handle.
        """
        return self._tex_handle

    @property
    def cuda_array(self) -> int:
        """CUDA array handle backing this texture.

        Returns:
            CUDA ``cudaArray_t`` handle for CUDA textures.
        """
        if not self.device.is_cuda:
            raise RuntimeError("cuda_array is only supported for CUDA textures.")
        return self._array_handle

    @property
    def cuda_texture(self) -> int:
        """CUDA texture object handle.

        Returns:
            CUDA ``cudaTextureObject_t`` handle for CUDA textures.
        """
        if not self.device.is_cuda:
            raise RuntimeError(
                "cuda_texture is only supported for CUDA textures. Use id for a device-independent handle."
            )
        return self._tex_handle

    @property
    def cuda_surface(self) -> int:
        """CUDA surface object handle backing this texture.

        The surface object is created lazily on first access and cached for the texture lifetime.

        Returns:
            CUDA ``cudaSurfaceObject_t`` handle for CUDA textures with ``surface_access=True``.
        """
        if self._surface_handle:
            return self._surface_handle
        if not self.device.is_cuda:
            raise RuntimeError("cuda_surface is only supported for CUDA textures.")
        if not self._surface_access:
            raise RuntimeError(
                "Texture CUDA array was not created with surface load/store support. "
                "Create the texture with surface_access=True."
            )
        if not self._array_handle:
            raise RuntimeError("Texture has no CUDA array backing storage.")

        self._surface_handle = self._runtime.core.wp_surface_object_create_device(
            self.device.context,
            self._array_handle,
        )
        if not self._surface_handle:
            raise RuntimeError(
                f"Failed to create CUDA surface object from texture array: {self._runtime.get_error_string()}"
            )

        return self._surface_handle


class Texture1D(Texture):
    """1D texture class.

    .. admonition:: Experimental

        The texture API is experimental and subject to change without a formal deprecation
        cycle. See :class:`Texture` for details.

    This is a specialized version of :class:`Texture` with dimensionality fixed to 1.
    Use this for explicit 1D texture creation and as a type hint in kernel parameters.

    Example::

        import warp as wp
        import numpy as np

        data = np.random.rand(256, 4).astype(np.float32)
        tex = wp.Texture1D(data, device="cuda:0")


        @wp.kernel
        def sample_kernel(tex: wp.Texture1D, output: wp.array(dtype=float)):
            tid = wp.tid()
            output[tid] = wp.texture_sample(tex, 0.5, dtype=float)
    """

    _wp_native_name_ = "texture1d_t"
    _wp_ctype_ = texture1d_t  # ctypes struct for arrays of textures

    from warp._src.codegen import Var  # noqa: PLC0415

    vars: ClassVar[dict[str, Var]] = {
        "width": Var("width", int32),
    }

    def __init__(
        self,
        data: np.ndarray | array | None = None,
        width: int = 0,
        num_channels: int = 0,
        dtype=None,
        filter_mode: TextureFilterMode = TextureFilterMode.LINEAR,
        address_mode: TextureAddressMode = TextureAddressMode.CLAMP,
        address_mode_u: TextureAddressMode | None = None,
        normalized_coords: bool = True,
        device=None,
        surface_access: bool = False,
        cuda_array: int = 0,
    ):
        super().__init__(
            ndim=1,
            data=data,
            width=width,
            height=1,
            depth=1,
            num_channels=num_channels,
            dtype=dtype,
            filter_mode=filter_mode,
            address_mode=address_mode,
            address_mode_u=address_mode_u,
            address_mode_v=None,
            address_mode_w=None,
            normalized_coords=normalized_coords,
            device=device,
            surface_access=surface_access,
            cuda_array=cuda_array,
        )

    def __ctype__(self) -> texture1d_t:
        """Return the ctypes structure for passing to kernels."""
        if self._tex_handle == 0:
            raise RuntimeError("Texture was created with data=None but never initialized.")
        return texture1d_t(self._tex_handle, self._width, self._num_channels)


class Texture2D(Texture):
    """2D texture class.

    .. admonition:: Experimental

        The texture API is experimental and subject to change without a formal deprecation
        cycle. See :class:`Texture` for details.

    This is a specialized version of :class:`Texture` with dimensionality fixed to 2.
    Use this for explicit 2D texture creation and as a type hint in kernel parameters.

    Example::

        import warp as wp
        import numpy as np

        data = np.random.rand(256, 256, 4).astype(np.float32)
        tex = wp.Texture2D(data, device="cuda:0")


        @wp.kernel
        def sample_kernel(tex: wp.Texture2D, output: wp.array(dtype=float)):
            tid = wp.tid()
            output[tid] = wp.texture_sample(tex, wp.vec2f(0.5, 0.5), dtype=float)
    """

    _wp_native_name_ = "texture2d_t"
    _wp_ctype_ = texture2d_t  # ctypes struct for arrays of textures

    from warp._src.codegen import Var  # noqa: PLC0415

    vars: ClassVar[dict[str, Var]] = {
        "width": Var("width", int32),
        "height": Var("height", int32),
    }

    def __init__(
        self,
        data: np.ndarray | array | None = None,
        width: int = 0,
        height: int = 0,
        num_channels: int = 0,
        dtype=None,
        filter_mode: TextureFilterMode = TextureFilterMode.LINEAR,
        address_mode: TextureAddressMode | tuple[TextureAddressMode, ...] = TextureAddressMode.CLAMP,
        address_mode_u: TextureAddressMode | None = None,
        address_mode_v: TextureAddressMode | None = None,
        normalized_coords: bool = True,
        device=None,
        surface_access: bool = False,
        cuda_array: int = 0,
    ):
        super().__init__(
            ndim=2,
            data=data,
            width=width,
            height=height,
            depth=1,
            num_channels=num_channels,
            dtype=dtype,
            filter_mode=filter_mode,
            address_mode=address_mode,
            address_mode_u=address_mode_u,
            address_mode_v=address_mode_v,
            normalized_coords=normalized_coords,
            surface_access=surface_access,
            device=device,
            cuda_array=cuda_array,
        )

    def __ctype__(self) -> texture2d_t:
        """Return the ctypes structure for passing to kernels."""
        if self._tex_handle == 0:
            raise RuntimeError("Texture was created with data=None but never initialized.")
        return texture2d_t(self._tex_handle, self._width, self._height, self._num_channels)


class Texture3D(Texture):
    """3D texture class.

    .. admonition:: Experimental

        The texture API is experimental and subject to change without a formal deprecation
        cycle. See :class:`Texture` for details.

    This is a specialized version of :class:`Texture` with dimensionality fixed to 3.
    Use this for explicit 3D texture creation and as a type hint in kernel parameters.

    Example::

        import warp as wp
        import numpy as np

        data = np.random.rand(64, 64, 64).astype(np.float32)
        tex = wp.Texture3D(data, device="cuda:0")


        @wp.kernel
        def sample_kernel(tex: wp.Texture3D, output: wp.array(dtype=float)):
            tid = wp.tid()
            output[tid] = wp.texture_sample(tex, wp.vec3f(0.5, 0.5, 0.5), dtype=float)
    """

    _wp_native_name_ = "texture3d_t"
    _wp_ctype_ = texture3d_t  # ctypes struct for arrays of textures

    from warp._src.codegen import Var  # noqa: PLC0415

    vars: ClassVar[dict[str, Var]] = {
        "width": Var("width", int32),
        "height": Var("height", int32),
        "depth": Var("depth", int32),
    }

    def __init__(
        self,
        data: np.ndarray | array | None = None,
        width: int = 0,
        height: int = 0,
        depth: int = 0,
        num_channels: int = 0,
        dtype=None,
        filter_mode: TextureFilterMode = TextureFilterMode.LINEAR,
        address_mode: TextureAddressMode | tuple[TextureAddressMode, ...] = TextureAddressMode.CLAMP,
        address_mode_u: TextureAddressMode | None = None,
        address_mode_v: TextureAddressMode | None = None,
        address_mode_w: TextureAddressMode | None = None,
        normalized_coords: bool = True,
        device=None,
        surface_access: bool = False,
        cuda_array: int = 0,
    ):
        super().__init__(
            ndim=3,
            data=data,
            width=width,
            height=height,
            depth=depth,
            num_channels=num_channels,
            dtype=dtype,
            filter_mode=filter_mode,
            address_mode=address_mode,
            address_mode_u=address_mode_u,
            address_mode_v=address_mode_v,
            address_mode_w=address_mode_w,
            normalized_coords=normalized_coords,
            surface_access=surface_access,
            device=device,
            cuda_array=cuda_array,
        )

    def __ctype__(self) -> texture3d_t:
        """Return the ctypes structure for passing to kernels."""
        if self._tex_handle == 0:
            raise RuntimeError("Texture was created with data=None but never initialized.")
        return texture3d_t(self._tex_handle, self._width, self._height, self._depth, self._num_channels)


class TextureResourceFlags(enum.IntEnum):
    """Flags specifying how a texture resource will be used by Warp."""

    NONE = 0x00
    """Warp will read and write to this resource (default)."""
    READ_ONLY = 0x01
    """Warp will not write to this resource."""
    WRITE_DISCARD = 0x02
    """Warp will not read from this resource and will write over the entire contents of the resource."""
    SURFACE_LDST = 0x04
    """Warp will bind this resource to a surface reference."""
    TEXTURE_GATHER = 0x08
    """Warp will perform texture gather operations on this resource."""


class GLTextureResource:
    """Register and use an OpenGL texture with Warp.

    .. admonition:: Experimental

        The texture API is experimental and subject to change without a formal deprecation
        cycle. See :class:`Texture` for details.

    The texture must be mapped before it can be accessed by Warp and unmapped before it can
    be used by OpenGL again.

    This class requires ``pyglet`` to be installed (``pip install pyglet``).

    Example::

        import ctypes
        import warp as wp
        from pyglet import gl

        # create OpenGL texture
        tex_id = gl.GLuint()
        gl.glGenTextures(1, ctypes.byref(tex_id))
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            1024,
            768,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # register texture resource
        tex_resource = wp.GLTextureResource(tex_id, gl.GL_TEXTURE_2D)

        a = wp.full((768, 1024), 255, dtype=wp.vec4ub)

        # map and access the texture
        tex = tex_resource.map()
        tex.copy_from(a)
        tex_resource.unmap()
    """

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._resource = None
        instance._mapped_texture = None
        return instance

    def __init__(self, gl_tex_id: int, gl_tex_target: int, device=None, flags: int = TextureResourceFlags.NONE):
        """Register OpenGL texture.

        Args:
            gl_tex_id: OpenGL texture ID, e.g., from ``pyglet.gl.glGenTextures()``.
            gl_tex_target: OpenGL texture target, e.g., ``pyglet.gl.GL_TEXTURE_2D``.
            device: The CUDA device where the texture resides.
            flags: Texture resource flags, see :class:`TextureResourceFlags`.
        """
        from pyglet import gl  # noqa: PLC0415

        import warp._src.context  # noqa: PLC0415

        if gl_tex_target == gl.GL_TEXTURE_1D:
            self._TextureClass = Texture1D
        elif gl_tex_target == gl.GL_TEXTURE_2D:
            self._TextureClass = Texture2D
        elif gl_tex_target == gl.GL_TEXTURE_3D:
            self._TextureClass = Texture3D
        else:
            raise ValueError("Invalid GL texture target")

        # Note: get_device() calls wp.init() if needed, so we must capture
        # self.runtime *after* this call to ensure it is not None.
        self._device = warp._src.context.get_device(device)
        self._runtime = warp._src.context.runtime

        if not self._device.is_cuda:
            raise ValueError("Must be a CUDA device")

        self._resource = self._runtime.core.wp_cuda_graphics_register_gl_image(
            self._device.context, gl_tex_id, gl_tex_target, flags
        )
        if not self._resource:
            raise RuntimeError(f"Failed to register GL texture resource: {self._runtime.get_error_string()}")

        self._mapped_texture = None

    def __del__(self):
        if self._resource is None:
            return

        try:
            # use CUDA context guard to avoid side effects during garbage collection
            with self._device.context_guard:
                self.unmap()
                self._runtime.core.wp_cuda_graphics_unregister_resource(self._device.context, self._resource)
        except (TypeError, AttributeError):
            # Suppress TypeError and AttributeError when callables become None during shutdown
            pass

    def map(self) -> Texture:
        """Map OpenGL texture.

        Returns:
            Mapped :class:`Texture` that can be accessed by Warp.
        """
        if self._mapped_texture is not None:
            return self._mapped_texture

        self._runtime.core.wp_cuda_graphics_map(self._device.context, self._resource)
        cuda_array = self._runtime.core.wp_cuda_graphics_sub_resource_get_mapped_array(
            self._device.context, self._resource, 0, 0
        )
        if not cuda_array:
            raise RuntimeError(f"Failed to map GL texture: {self._runtime.get_error_string()}")

        self._mapped_texture = self._TextureClass(cuda_array=cuda_array)
        return self._mapped_texture

    def unmap(self):
        """Unmap OpenGL texture.

        The texture must be unmapped before it can be used by OpenGL again.
        """
        if self._mapped_texture is not None:
            self._runtime.core.wp_cuda_graphics_unmap(self._device.context, self._resource)
            self._mapped_texture = None
