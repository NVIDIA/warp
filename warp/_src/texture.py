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
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from warp._src.codegen import Var
    from warp._src.context import DeviceLike

from warp._src.types import (
    array,
    dtype_from_numpy,
    dtype_to_numpy,
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


class Texture:
    """Unified texture class for hardware-accelerated sampling on GPU and software sampling on CPU.

    .. admonition:: Experimental

        The texture API is experimental and subject to change without a formal deprecation
        cycle.

    This class handles 1D, 2D, and 3D textures. The dimensionality is provided explicitly via
    ``ndim`` or by using the :class:`Texture1D`, :class:`Texture2D`, or :class:`Texture3D` subclasses.

    Textures provide hardware-accelerated filtering and addressing for regularly gridded data
    on CUDA devices. On CPU, software-based filtering and addressing is used. Supports
    linear/bilinear/trilinear interpolation and various addressing modes (wrap, clamp, mirror, border).

    Supports ``wp.uint8``, ``wp.uint16``, ``wp.uint32``, ``wp.int8``, ``wp.int16``, ``wp.int32``,
    ``wp.float16``, and ``wp.float32`` data types. Unsigned integer textures are read as normalized
    floats in [0, 1]; signed integer textures are normalized to [-1, 1]; float types are returned as-is.

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

    ADDRESS_WRAP = 0
    """Wrap coordinates (tile the texture)."""
    ADDRESS_CLAMP = 1
    """Clamp coordinates to [0, 1]."""
    ADDRESS_MIRROR = 2
    """Mirror coordinates at boundaries."""
    ADDRESS_BORDER = 3
    """Return 0 for coordinates outside [0, 1]."""

    FILTER_POINT = 0
    """Nearest-neighbor (point) filtering."""
    FILTER_LINEAR = 1
    """Bilinear/trilinear filtering."""

    _default_ndim: int | None = None

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._tex_handle = 0
        instance._array_handle = 0
        instance._surface_handle = 0
        instance._surface_access = False
        return instance

    def __init__(
        self,
        data: np.ndarray | array | None = None,
        width: int = 0,
        height: int = 0,
        depth: int = 0,
        num_channels: int = 1,
        dtype=float32,
        filter_mode: int = 1,
        address_mode: int | tuple[int, ...] | None = None,
        address_mode_u: int | None = None,
        address_mode_v: int | None = None,
        address_mode_w: int | None = None,
        normalized_coords: bool = True,
        device: DeviceLike = None,
        ndim: int | None = None,
        surface_access: bool = False,
    ):
        """Create a texture.

        Args:
            data: Initial texture data as a NumPy array or Warp array.
                For 1D: shape ``(width,)`` or ``(width, num_channels)``.
                For 2D: shape ``(height, width)``, ``(height, width, 2)``,
                or ``(height, width, 4)``.
                For 3D: shape ``(depth, height, width)``,
                ``(depth, height, width, 2)``, or ``(depth, height, width, 4)``.
                Supported dtypes: ``wp.uint8``, ``wp.uint16``, ``wp.uint32``,
                ``wp.int8``, ``wp.int16``, ``wp.int32``, ``wp.float16``, ``wp.float32``.
            width: Texture width (required if ``data`` is ``None``).
            height: Texture height (required if ``data`` is ``None``).
            depth: Texture depth (required if ``data`` is ``None`` for 3D textures).
            num_channels: Number of channels (1, 2, or 4). Only used when
                ``data`` is ``None``.
            dtype: Data type. Only used when ``data`` is ``None``; otherwise
                inferred from the data.
            filter_mode: Filtering mode — :attr:`FILTER_POINT` or
                :attr:`FILTER_LINEAR`.
            address_mode: Address mode for all axes —
                :attr:`ADDRESS_WRAP`, :attr:`ADDRESS_CLAMP`,
                :attr:`ADDRESS_MIRROR`, or :attr:`ADDRESS_BORDER`.
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
            ndim: Explicit dimensionality (1, 2, or 3). Required when constructing
                the base :class:`Texture` class directly.
            surface_access: If ``True`` and ``device`` is CUDA, allocates the backing
                CUDA array with surface load/store support so :attr:`cuda_surface`
                can be used.
        """
        import warp._src.context  # noqa: PLC0415

        self.runtime = warp._src.context.runtime

        self._ndim = self._default_ndim if self._default_ndim is not None else ndim
        if self._ndim not in (1, 2, 3):
            raise ValueError(
                "Texture dimensionality must be 1, 2, or 3. Use ndim=1/2/3 or Texture1D/Texture2D/Texture3D."
            )

        # Resolve address modes
        resolved_u = self._resolve_address_mode(address_mode, address_mode_u, 0)
        resolved_v = self._resolve_address_mode(address_mode, address_mode_v, 1) if self._ndim >= 2 else 1
        resolved_w = self._resolve_address_mode(address_mode, address_mode_w, 2) if self._ndim == 3 else 1

        self.device = data.device if is_array(data) and device is None else warp._src.context.get_device(device)

        if data is None:
            self._width = width
            self._height = height if self._ndim >= 2 else 1
            self._depth = depth if self._ndim == 3 else 1
            self._num_channels = num_channels
            self._dtype = self._canonicalize_dtype(dtype)
            self._dtype_code = self._dtype_to_code(self._dtype)
            self._filter_mode = filter_mode
            self._address_mode_u = resolved_u
            self._address_mode_v = resolved_v
            self._address_mode_w = resolved_w
            self._normalized_coords = normalized_coords
            self._surface_access = bool(surface_access and self.device.is_cuda)
            self._tex_handle = 0
            self._array_handle = 0
            self._surface_handle = 0
            return

        # Process input data
        if isinstance(data, np.ndarray):
            np_data = data if data.flags["C_CONTIGUOUS"] else np.ascontiguousarray(data)
            width, height, depth, num_channels = self._shape_from_numpy_data(np_data)
            dtype = dtype_from_numpy(np_data.dtype)
            self._validate_texture_dtype(dtype)
            host_flat = np_data.flatten()
            cuda_copy_src = None
        elif is_array(data):
            wp_data = data.to(self.device).contiguous()
            width, height, depth, num_channels, dtype = self._shape_from_warp_array(wp_data)
            if self.device.is_cuda:
                host_flat = self._make_zero_host_buffer(width, height, depth, num_channels, dtype)
                cuda_copy_src = self._reshape_for_cuda_copy(wp_data)
            else:
                np_data = wp_data.numpy()
                host_flat = np_data.flatten()
                cuda_copy_src = None
        else:
            raise TypeError("data must be a NumPy array or Warp array")

        if num_channels not in (1, 2, 4):
            raise ValueError("num_channels must be 1, 2, or 4")

        self._width = width
        self._height = height if self._ndim >= 2 else 1
        self._depth = depth if self._ndim == 3 else 1
        self._num_channels = num_channels
        self._dtype = dtype
        self._dtype_code = self._dtype_to_code(dtype)
        self._filter_mode = filter_mode
        self._address_mode_u = resolved_u
        self._address_mode_v = resolved_v
        self._address_mode_w = resolved_w
        self._normalized_coords = normalized_coords
        self._surface_access = bool(surface_access and self.device.is_cuda)

        self._create_texture(host_flat)
        if cuda_copy_src is not None:
            if self._ndim == 1:
                self._copy_from_array_1d(cuda_copy_src)
            elif self._ndim == 2:
                self._copy_from_array_2d(cuda_copy_src)
            else:
                self._copy_from_array_3d(cuda_copy_src)

    @staticmethod
    def _validate_texture_dtype(dtype):
        if dtype not in _SUPPORTED_TEXTURE_DTYPES:
            raise ValueError(
                f"Unsupported texture dtype: {dtype}. Supported: uint8, uint16, uint32, int8, int16, int32, float16, float32"
            )

    @classmethod
    def _canonicalize_dtype(cls, dtype):
        if dtype is float:
            return float32
        if dtype in _SUPPORTED_TEXTURE_DTYPES:
            return dtype
        raise ValueError(
            f"Unsupported texture dtype: {dtype}. Supported: uint8, uint16, uint32, int8, int16, int32, float16, float32"
        )

    def _shape_from_numpy_data(self, np_data: np.ndarray):
        if self._ndim == 1:
            if np_data.ndim == 1:
                (width,) = np_data.shape
                num_channels = 1
            elif np_data.ndim == 2:
                width, num_channels = np_data.shape
            else:
                raise ValueError("1D texture data must be 1D or 2D array")
            height = 1
            depth = 1
        elif self._ndim == 2:
            if np_data.ndim == 2:
                height, width = np_data.shape
                num_channels = 1
            elif np_data.ndim == 3:
                height, width, num_channels = np_data.shape
            else:
                raise ValueError("2D texture data must be 2D or 3D array")
            depth = 1
        else:
            if np_data.ndim == 3:
                depth, height, width = np_data.shape
                num_channels = 1
            elif np_data.ndim == 4:
                depth, height, width, num_channels = np_data.shape
            else:
                raise ValueError("3D texture data must be 3D or 4D array")
        return int(width), int(height), int(depth), int(num_channels)

    def _shape_from_warp_array(self, wp_data: array):
        dtype = wp_data.dtype
        scalar_dtype = type_scalar_type(dtype)
        self._validate_texture_dtype(scalar_dtype)
        canonical_dtype = scalar_dtype

        if type_is_vector(dtype):
            num_channels = int(type_length(dtype))
            if self._ndim == 1:
                if wp_data.ndim != 1:
                    raise ValueError("1D texture data with vector dtype must be a 1D wp.array")
                (width,) = wp_data.shape
                height = 1
                depth = 1
            elif self._ndim == 2:
                if wp_data.ndim != 2:
                    raise ValueError("2D texture data with vector dtype must be a 2D wp.array")
                height, width = wp_data.shape
                depth = 1
            else:
                if wp_data.ndim != 3:
                    raise ValueError("3D texture data with vector dtype must be a 3D wp.array")
                depth, height, width = wp_data.shape
            return int(width), int(height), int(depth), num_channels, canonical_dtype

        if self._ndim == 1:
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
        elif self._ndim == 2:
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

    def _reshape_for_cuda_copy(self, wp_data: array) -> array:
        """Reshape a warp array to be compatible with the CUDA array copy helpers.

        For scalar-dtype arrays with a trailing channel dimension, collapse
        it so the array has the right ndim for copy_from_array validation
        (1D for Texture1D, 2D for Texture2D, 3D for Texture3D).
        """
        if type_is_vector(wp_data.dtype):
            return wp_data

        target_ndim = self._ndim
        if wp_data.ndim == target_ndim + 1:
            shape = wp_data.shape
            new_shape = (*shape[:-2], shape[-2] * shape[-1])
            return wp_data.reshape(new_shape)
        return wp_data

    def _make_zero_host_buffer(self, width: int, height: int, depth: int, num_channels: int, dtype):
        np_dtype = dtype_to_numpy(dtype)
        if self._ndim == 1:
            if num_channels == 1:
                shape = (width,)
            else:
                shape = (width, num_channels)
        elif self._ndim == 2:
            if num_channels == 1:
                shape = (height, width)
            else:
                shape = (height, width, num_channels)
        else:
            if num_channels == 1:
                shape = (depth, height, width)
            else:
                shape = (depth, height, width, num_channels)
        return np.zeros(shape, dtype=np_dtype).flatten()

    def _create_texture(self, data_flat):
        """Create the underlying texture resource."""
        tex_handle = ctypes.c_uint64(0)
        array_handle = ctypes.c_uint64(0)
        data_ptr = data_flat.ctypes.data_as(ctypes.c_void_p)

        if self._ndim == 1:
            if self.device.is_cuda:
                success = self.runtime.core.wp_texture1d_create_device(
                    self.device.context,
                    self._width,
                    self._num_channels,
                    self._dtype_code,
                    self._filter_mode,
                    self._address_mode_u,
                    self._normalized_coords,
                    self._surface_access,
                    data_ptr,
                    ctypes.byref(tex_handle),
                    ctypes.byref(array_handle),
                )
            else:
                success = self.runtime.core.wp_texture1d_create_host(
                    self._width,
                    self._num_channels,
                    self._dtype_code,
                    self._filter_mode,
                    self._address_mode_u,
                    self._normalized_coords,
                    data_ptr,
                    ctypes.byref(tex_handle),
                )
        elif self._ndim == 2:
            if self.device.is_cuda:
                success = self.runtime.core.wp_texture2d_create_device(
                    self.device.context,
                    self._width,
                    self._height,
                    self._num_channels,
                    self._dtype_code,
                    self._filter_mode,
                    self._address_mode_u,
                    self._address_mode_v,
                    self._normalized_coords,
                    self._surface_access,
                    data_ptr,
                    ctypes.byref(tex_handle),
                    ctypes.byref(array_handle),
                )
            else:
                success = self.runtime.core.wp_texture2d_create_host(
                    self._width,
                    self._height,
                    self._num_channels,
                    self._dtype_code,
                    self._filter_mode,
                    self._address_mode_u,
                    self._address_mode_v,
                    self._normalized_coords,
                    data_ptr,
                    ctypes.byref(tex_handle),
                )
        else:  # 3D
            if self.device.is_cuda:
                success = self.runtime.core.wp_texture3d_create_device(
                    self.device.context,
                    self._width,
                    self._height,
                    self._depth,
                    self._num_channels,
                    self._dtype_code,
                    self._filter_mode,
                    self._address_mode_u,
                    self._address_mode_v,
                    self._address_mode_w,
                    self._normalized_coords,
                    self._surface_access,
                    data_ptr,
                    ctypes.byref(tex_handle),
                    ctypes.byref(array_handle),
                )
            else:
                success = self.runtime.core.wp_texture3d_create_host(
                    self._width,
                    self._height,
                    self._depth,
                    self._num_channels,
                    self._dtype_code,
                    self._filter_mode,
                    self._address_mode_u,
                    self._address_mode_v,
                    self._address_mode_w,
                    self._normalized_coords,
                    data_ptr,
                    ctypes.byref(tex_handle),
                )

        if not success:
            raise RuntimeError(f"Failed to create Texture{self._ndim}D: {self.runtime.get_error_string()}")

        self._tex_handle = tex_handle.value
        self._array_handle = array_handle.value

    def __del__(self):
        if self._tex_handle == 0 and self._array_handle == 0:
            return

        try:
            if self._ndim == 1:
                if self.device.is_cuda:
                    with self.device.context_guard:
                        if self._surface_handle:
                            self.runtime.core.wp_texture_array_destroy_surface_device(
                                self.device.context, self._surface_handle
                            )
                        self.runtime.core.wp_texture1d_destroy_device(
                            self.device.context, self._tex_handle, self._array_handle
                        )
                else:
                    self.runtime.core.wp_texture1d_destroy_host(self._tex_handle)
            elif self._ndim == 2:
                if self.device.is_cuda:
                    with self.device.context_guard:
                        if self._surface_handle:
                            self.runtime.core.wp_texture_array_destroy_surface_device(
                                self.device.context, self._surface_handle
                            )
                        self.runtime.core.wp_texture2d_destroy_device(
                            self.device.context, self._tex_handle, self._array_handle
                        )
                else:
                    self.runtime.core.wp_texture2d_destroy_host(self._tex_handle)
            else:  # 3D
                if self.device.is_cuda:
                    with self.device.context_guard:
                        if self._surface_handle:
                            self.runtime.core.wp_texture_array_destroy_surface_device(
                                self.device.context, self._surface_handle
                            )
                        self.runtime.core.wp_texture3d_destroy_device(
                            self.device.context, self._tex_handle, self._array_handle
                        )
                else:
                    self.runtime.core.wp_texture3d_destroy_host(self._tex_handle)
        except (TypeError, AttributeError):
            pass

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
    def _resolve_address_mode(address_mode, address_mode_axis, axis_index):
        """Resolve address mode for a single axis."""
        if address_mode_axis is not None:
            return address_mode_axis
        elif address_mode is not None:
            if isinstance(address_mode, tuple):
                return address_mode[axis_index] if axis_index < len(address_mode) else 1
            else:
                return address_mode
        else:
            return 1  # CLAMP

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

        surface_handle = ctypes.c_uint64(0)
        result = self.runtime.core.wp_texture_array_create_surface_device(
            self.device.context,
            self._array_handle,
            ctypes.byref(surface_handle),
        )
        if not result:
            raise RuntimeError(
                f"Failed to create CUDA surface object from texture array: {self.runtime.get_error_string()}"
            )

        self._surface_handle = int(surface_handle.value)
        return self._surface_handle

    def _validate_cuda_copy_array_common(self, arr: array, arg_name: str):
        if not self.device.is_cuda:
            raise RuntimeError("CUDA array copy helpers require a CUDA texture.")
        if not self._array_handle:
            raise RuntimeError("Texture has no CUDA array backing storage.")
        if not isinstance(arr, array):
            raise TypeError(f"{arg_name} must be a wp.array.")
        if not arr.device.is_cuda:
            raise RuntimeError(f"{arg_name} must be a CUDA array.")
        if arr.device != self.device:
            raise RuntimeError(f"{arg_name} must be on the same CUDA device as the texture.")

    def _validate_1d_cuda_copy_array(self, arr: array, arg_name: str) -> int:
        self._validate_cuda_copy_array_common(arr, arg_name)
        if arr.ndim != 1:
            raise ValueError(f"{arg_name} must be a 1D array.")

        elem_size = int(type_size_in_bytes(self._dtype))
        width_bytes = int(arr.shape[0]) * int(arr.strides[0])
        expected_width_bytes = int(self._width) * int(self._num_channels) * elem_size
        if width_bytes != expected_width_bytes:
            raise ValueError(
                f"{arg_name} size ({width_bytes} bytes) does not match texture size ({expected_width_bytes} bytes)."
            )

        return width_bytes

    def _validate_2d_cuda_copy_array(self, arr: array, arg_name: str) -> tuple[int, int, int]:
        self._validate_cuda_copy_array_common(arr, arg_name)
        if arr.ndim != 2:
            raise ValueError(f"{arg_name} must be a 2D array.")

        height = int(arr.shape[0])
        if height != self._height:
            raise ValueError(f"{arg_name} height ({height}) does not match texture height ({self._height}).")

        pitch = int(arr.strides[0])
        width_bytes = int(arr.shape[1]) * int(arr.strides[1])
        elem_size = int(type_size_in_bytes(self._dtype))
        expected_width_bytes = int(self._width) * int(self._num_channels) * elem_size
        if width_bytes != expected_width_bytes:
            raise ValueError(
                f"{arg_name} row size ({width_bytes} bytes) does not match texture row size ({expected_width_bytes} bytes)."
            )

        return pitch, width_bytes, height

    def _validate_3d_cuda_copy_array(self, arr: array, arg_name: str) -> tuple[int, int, int, int]:
        self._validate_cuda_copy_array_common(arr, arg_name)
        if arr.ndim != 3:
            raise ValueError(f"{arg_name} must be a 3D array.")

        depth = int(arr.shape[0])
        if depth != self._depth:
            raise ValueError(f"{arg_name} depth ({depth}) does not match texture depth ({self._depth}).")

        height = int(arr.shape[1])
        if height != self._height:
            raise ValueError(f"{arg_name} height ({height}) does not match texture height ({self._height}).")

        pitch = int(arr.strides[1])
        width_bytes = int(arr.shape[2]) * int(arr.strides[2])
        elem_size = int(type_size_in_bytes(self._dtype))
        expected_width_bytes = int(self._width) * int(self._num_channels) * elem_size
        if width_bytes != expected_width_bytes:
            raise ValueError(
                f"{arg_name} row size ({width_bytes} bytes) does not match texture row size ({expected_width_bytes} bytes)."
            )

        slice_stride = int(arr.strides[0])
        expected_slice_stride = pitch * height
        if slice_stride != expected_slice_stride:
            raise ValueError(
                f"{arg_name} slice stride ({slice_stride} bytes) does not match expected contiguous slice stride ({expected_slice_stride} bytes)."
            )

        return pitch, width_bytes, height, depth

    def _copy_from_array_2d(self, src: array):
        src_pitch, width_bytes, height = self._validate_2d_cuda_copy_array(src, "src")
        result = self.runtime.core.wp_texture2d_copy_from_array_device(
            self.device.context,
            self.device.stream.cuda_stream,
            self._array_handle,
            src.ptr,
            src_pitch,
            width_bytes,
            height,
        )
        if not result:
            raise RuntimeError(f"Texture CUDA array copy (array -> texture) failed: {self.runtime.get_error_string()}")

    def _copy_to_array_2d(self, dst: array):
        dst_pitch, width_bytes, height = self._validate_2d_cuda_copy_array(dst, "dst")
        result = self.runtime.core.wp_texture2d_copy_to_array_device(
            self.device.context,
            self.device.stream.cuda_stream,
            dst.ptr,
            dst_pitch,
            self._array_handle,
            width_bytes,
            height,
        )
        if not result:
            raise RuntimeError(f"Texture CUDA array copy (texture -> array) failed: {self.runtime.get_error_string()}")

    def _copy_from_array_1d(self, src: array):
        width_bytes = self._validate_1d_cuda_copy_array(src, "src")
        result = self.runtime.core.wp_texture1d_copy_from_array_device(
            self.device.context,
            self.device.stream.cuda_stream,
            self._array_handle,
            src.ptr,
            width_bytes,
        )
        if not result:
            raise RuntimeError(f"Texture CUDA array copy (array -> texture) failed: {self.runtime.get_error_string()}")

    def _copy_to_array_1d(self, dst: array):
        width_bytes = self._validate_1d_cuda_copy_array(dst, "dst")
        result = self.runtime.core.wp_texture1d_copy_to_array_device(
            self.device.context,
            self.device.stream.cuda_stream,
            dst.ptr,
            self._array_handle,
            width_bytes,
        )
        if not result:
            raise RuntimeError(f"Texture CUDA array copy (texture -> array) failed: {self.runtime.get_error_string()}")

    def _copy_from_array_3d(self, src: array):
        src_pitch, width_bytes, height, depth = self._validate_3d_cuda_copy_array(src, "src")
        src_height = height
        result = self.runtime.core.wp_texture3d_copy_from_array_device(
            self.device.context,
            self.device.stream.cuda_stream,
            self._array_handle,
            src.ptr,
            src_pitch,
            src_height,
            width_bytes,
            height,
            depth,
        )
        if not result:
            raise RuntimeError(f"Texture CUDA array copy (array -> texture) failed: {self.runtime.get_error_string()}")

    def _copy_to_array_3d(self, dst: array):
        dst_pitch, width_bytes, height, depth = self._validate_3d_cuda_copy_array(dst, "dst")
        dst_height = height
        result = self.runtime.core.wp_texture3d_copy_to_array_device(
            self.device.context,
            self.device.stream.cuda_stream,
            dst.ptr,
            dst_pitch,
            dst_height,
            self._array_handle,
            width_bytes,
            height,
            depth,
        )
        if not result:
            raise RuntimeError(f"Texture CUDA array copy (texture -> array) failed: {self.runtime.get_error_string()}")

    def __ctype__(self):
        """Return the ctypes structure for passing to kernels."""
        if self._tex_handle == 0:
            raise RuntimeError("Texture was created with data=None but never initialized.")
        if self._ndim == 1:
            return texture1d_t(self._tex_handle, self._width, self._num_channels)
        elif self._ndim == 2:
            return texture2d_t(self._tex_handle, self._width, self._height, self._num_channels)
        else:
            return texture3d_t(self._tex_handle, self._width, self._height, self._depth, self._num_channels)


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

    _default_ndim = 1
    _wp_native_name_ = "texture1d_t"
    _wp_ctype_ = texture1d_t  # ctypes struct for arrays of textures

    from warp._src.codegen import Var as _Var  # noqa: PLC0415

    vars: ClassVar[dict[str, Var]] = {
        "width": _Var("width", int32),
    }

    def __init__(
        self,
        data: np.ndarray | array | None = None,
        width: int = 0,
        num_channels: int = 4,
        dtype=float32,
        filter_mode: int = 1,
        address_mode: int | None = None,
        address_mode_u: int | None = None,
        normalized_coords: bool = True,
        device: DeviceLike = None,
        surface_access: bool = False,
    ):
        super().__init__(
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
            ndim=1,
            surface_access=surface_access,
        )

    def copy_from_array(self, src: array):
        """Copy from a CUDA 1D Warp array into this texture's CUDA array."""
        self._copy_from_array_1d(src)

    def copy_to_array(self, dst: array):
        """Copy from this texture's CUDA array into a CUDA 1D Warp array."""
        self._copy_to_array_1d(dst)

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

    _default_ndim = 2
    _wp_native_name_ = "texture2d_t"
    _wp_ctype_ = texture2d_t  # ctypes struct for arrays of textures

    from warp._src.codegen import Var as _Var  # noqa: PLC0415

    vars: ClassVar[dict[str, Var]] = {
        "width": _Var("width", int32),
        "height": _Var("height", int32),
    }

    def __init__(
        self,
        data: np.ndarray | array | None = None,
        width: int = 0,
        height: int = 0,
        num_channels: int = 4,
        dtype=float32,
        filter_mode: int = 1,
        address_mode: int | tuple[int, int] | None = None,
        address_mode_u: int | None = None,
        address_mode_v: int | None = None,
        normalized_coords: bool = True,
        device: DeviceLike = None,
        surface_access: bool = False,
    ):
        super().__init__(
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
            ndim=2,
        )

    def copy_from_array(self, src: array):
        """Copy from a CUDA 2D Warp array into this texture's CUDA array."""
        self._copy_from_array_2d(src)

    def copy_to_array(self, dst: array):
        """Copy from this texture's CUDA array into a CUDA 2D Warp array."""
        self._copy_to_array_2d(dst)

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

    _default_ndim = 3
    _wp_native_name_ = "texture3d_t"
    _wp_ctype_ = texture3d_t  # ctypes struct for arrays of textures

    from warp._src.codegen import Var as _Var  # noqa: PLC0415

    vars: ClassVar[dict[str, Var]] = {
        "width": _Var("width", int32),
        "height": _Var("height", int32),
        "depth": _Var("depth", int32),
    }

    def __init__(
        self,
        data: np.ndarray | array | None = None,
        width: int = 0,
        height: int = 0,
        depth: int = 0,
        num_channels: int = 1,
        dtype=float32,
        filter_mode: int = 1,
        address_mode: int | tuple[int, int, int] | None = None,
        address_mode_u: int | None = None,
        address_mode_v: int | None = None,
        address_mode_w: int | None = None,
        normalized_coords: bool = True,
        device: DeviceLike = None,
        surface_access: bool = False,
    ):
        super().__init__(
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
            ndim=3,
        )

    def copy_from_array(self, src: array):
        """Copy from a CUDA 3D Warp array into this texture's CUDA array."""
        self._copy_from_array_3d(src)

    def copy_to_array(self, dst: array):
        """Copy from this texture's CUDA array into a CUDA 3D Warp array."""
        self._copy_to_array_3d(dst)

    def __ctype__(self) -> texture3d_t:
        """Return the ctypes structure for passing to kernels."""
        if self._tex_handle == 0:
            raise RuntimeError("Texture was created with data=None but never initialized.")
        return texture3d_t(self._tex_handle, self._width, self._height, self._depth, self._num_channels)
