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

from warp._src.types import array, float32, int32, is_array, uint8, uint16

# Note: warp._src.context.runtime is accessed lazily via self.runtime = warp._src.context.runtime
# in __init__ methods to avoid circular imports


class TextureFilterMode(enum.IntEnum):
    """Filter modes for texture sampling."""

    #: Nearest-neighbor (point) filtering
    CLOSEST = 0
    #: Bilinear/trilinear filtering
    LINEAR = 1


class TextureAddressMode(enum.IntEnum):
    """Address modes for texture coordinates outside [0, 1]."""

    #: Wrap coordinates (tile the texture)
    WRAP = 0
    #: Clamp coordinates to [0, 1]
    CLAMP = 1
    #: Mirror coordinates at boundaries
    MIRROR = 2
    #: Return 0 for coordinates outside [0, 1]
    BORDER = 3


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


class _TextureBase:
    """Base class for 2D and 3D textures with shared functionality.

    This class provides common utilities and properties for texture classes.
    It should not be instantiated directly.
    """

    @staticmethod
    def _dtype_to_code(dtype):
        """Convert dtype to internal dtype code.

        Args:
            dtype: One of np.uint8, np.uint16, np.float32, wp.uint8, wp.uint16, wp.float32, or float.

        Returns:
            Internal dtype code (0 for uint8, 1 for uint16, 2 for float32).
        """
        # Handle warp types
        if dtype is uint8 or dtype is np.uint8:
            return 0
        elif dtype is uint16 or dtype is np.uint16:
            return 1
        elif dtype is float32 or dtype is np.float32 or dtype is float:
            return 2
        # Try numpy dtype conversion as fallback
        try:
            dtype = np.dtype(dtype)
            if dtype == np.uint8:
                return 0
            elif dtype == np.uint16:
                return 1
            elif dtype == np.float32:
                return 2
        except (TypeError, AttributeError):
            pass
        raise ValueError(f"Unsupported texture dtype: {dtype}. Supported: uint8, uint16, float32")

    @staticmethod
    def _resolve_address_mode(address_mode, address_mode_axis, axis_index):
        """Resolve address mode for a single axis.

        Args:
            address_mode: The base address_mode (single int or tuple).
            address_mode_axis: Per-axis override (or None).
            axis_index: Index into tuple if address_mode is a tuple.

        Returns:
            Resolved address mode as int (default is CLAMP=1).
        """
        if address_mode_axis is not None:
            return address_mode_axis
        elif address_mode is not None:
            if isinstance(address_mode, tuple):
                return address_mode[axis_index]
            else:
                return address_mode
        else:
            return 1  # CLAMP

    @property
    def width(self) -> int:
        """Texture width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Texture height in pixels."""
        return self._height

    @property
    def num_channels(self) -> int:
        """Number of channels."""
        return self._num_channels

    @property
    def dtype(self):
        """Data type of the texture (uint8, uint16, or float32)."""
        return self._dtype

    @property
    def address_mode_u(self) -> int:
        """Address mode for U axis (0=WRAP, 1=CLAMP, 2=MIRROR, 3=BORDER)."""
        return self._address_mode_u

    @property
    def address_mode_v(self) -> int:
        """Address mode for V axis (0=WRAP, 1=CLAMP, 2=MIRROR, 3=BORDER)."""
        return self._address_mode_v

    @property
    def normalized_coords(self) -> bool:
        """Whether texture uses normalized coordinates [0,1] or texel space."""
        return self._normalized_coords


class Texture2D(_TextureBase):
    """Class representing a 2D texture.

    Textures provide hardware-accelerated filtering and addressing for
    regularly-gridded data on CUDA devices. On CPU, software-based
    filtering and addressing is used. Supports bilinear interpolation and
    various addressing modes (wrap, clamp, mirror, border).

    Supports uint8, uint16, and float32 data types. Integer textures are
    read as normalized floats in the [0, 1] range.

    Example:
        >>> import warp as wp
        >>> import numpy as np
        >>> # Create a 256x256 RGBA float texture on GPU
        >>> data = np.random.rand(256, 256, 4).astype(np.float32)
        >>> tex = wp.Texture2D(data, device="cuda:0")
        >>> # Create a texture on CPU
        >>> tex_cpu = wp.Texture2D(data, device="cpu")
        >>> # Create a compressed 8-bit texture
        >>> data8 = (np.random.rand(256, 256) * 255).astype(np.uint8)
        >>> tex8 = wp.Texture2D(data8, device="cuda:0")
    """

    # Native C++ type name for code generation
    _wp_native_name_ = "texture2d_t"

    from warp._src.codegen import Var as _Var  # noqa: PLC0415

    #: Member attributes available during code-gen (e.g.: w = tex.width)
    vars: ClassVar[dict[str, Var]] = {
        "width": _Var("width", int32),
        "height": _Var("height", int32),
    }

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._tex_handle = 0
        instance._array_handle = 0
        return instance

    def __init__(
        self,
        data: np.ndarray | array | None = None,
        width: int = 0,
        height: int = 0,
        num_channels: int = 4,
        dtype=np.float32,
        filter_mode: int = 1,
        address_mode: int | tuple[int, int] | None = None,
        address_mode_u: int | None = None,
        address_mode_v: int | None = None,
        normalized_coords: bool = True,
        device=None,
    ):
        """Create a 2D texture.

        Args:
            data: Initial texture data as a numpy array or warp array.
                  Shape should be (height, width) for 1-channel,
                  (height, width, 2) for 2-channel, or
                  (height, width, 4) for 4-channel textures.
                  Supported dtypes: uint8, uint16, float32 (or wp.uint8, wp.uint16, wp.float32).
                  If None, width/height/num_channels must be specified.
            width: Texture width (required if data is None).
            height: Texture height (required if data is None).
            num_channels: Number of channels (1, 2, or 4). Default is 4.
            dtype: Data type (uint8, uint16, or float32). Default is float32.
            filter_mode: Filtering mode - CLOSEST (0) or LINEAR (1). Default is LINEAR.
            address_mode: Address mode for all axes - WRAP (0), CLAMP (1), MIRROR (2), or BORDER (3).
                          Can be a single int (applied to all axes) or a tuple (u, v) for per-axis modes.
                          Default is CLAMP. Overridden by address_mode_u/address_mode_v if specified.
            address_mode_u: Per-axis address mode for U (horizontal). Overrides address_mode if specified.
            address_mode_v: Per-axis address mode for V (vertical). Overrides address_mode if specified.
            normalized_coords: If True (default), texture coordinates are in [0, 1] range.
                              If False, coordinates are in texel space [0, width/height].
            device: Device to create the texture on (CPU or CUDA).
        """
        import warp._src.context  # noqa: PLC0415 - lazy import to avoid circular imports

        self.runtime = warp._src.context.runtime

        # Resolve per-axis address modes using base class helper
        resolved_u = self._resolve_address_mode(address_mode, address_mode_u, 0)
        resolved_v = self._resolve_address_mode(address_mode, address_mode_v, 1)

        if data is None:
            # Just store dimensions for lazy creation
            self.device = warp._src.context.get_device(device)
            self._width = width
            self._height = height
            self._num_channels = num_channels
            self._dtype = np.dtype(dtype)
            self._dtype_code = self._dtype_to_code(dtype)
            self._filter_mode = filter_mode
            self._address_mode_u = resolved_u
            self._address_mode_v = resolved_v
            self._normalized_coords = normalized_coords
            self._tex_handle = 0
            self._array_handle = 0
            return

        # Get device and extract data
        if isinstance(data, np.ndarray):
            self.device = warp._src.context.get_device(device)

            # Determine dimensions from numpy array shape
            if data.ndim == 2:
                height, width = data.shape
                num_channels = 1
            elif data.ndim == 3:
                height, width, num_channels = data.shape
            else:
                raise ValueError("Data must be 2D or 3D numpy array")

            # Validate and get dtype code
            dtype_code = self._dtype_to_code(data.dtype)
            dtype = data.dtype

            # Ensure contiguous
            if not data.flags["C_CONTIGUOUS"]:
                data = np.ascontiguousarray(data)

            data_flat = data.flatten()
        elif is_array(data):
            self.device = data.device if device is None else warp._src.context.get_device(device)

            # Copy to numpy for now (could optimize later)
            np_data = data.numpy()
            if np_data.ndim == 2:
                height, width = np_data.shape
                num_channels = 1
            elif np_data.ndim == 3:
                height, width, num_channels = np_data.shape
            else:
                raise ValueError("Data must be 2D or 3D array")

            dtype_code = self._dtype_to_code(np_data.dtype)
            dtype = np_data.dtype

            if not np_data.flags["C_CONTIGUOUS"]:
                np_data = np.ascontiguousarray(np_data)
            data_flat = np_data.flatten()
        else:
            raise TypeError("data must be a numpy array or warp array")

        if num_channels not in (1, 2, 4):
            raise ValueError("num_channels must be 1, 2, or 4")

        self._width = width
        self._height = height
        self._num_channels = num_channels
        self._dtype = np.dtype(dtype)
        self._dtype_code = dtype_code
        self._filter_mode = filter_mode
        self._address_mode_u = resolved_u
        self._address_mode_v = resolved_v
        self._normalized_coords = normalized_coords

        # Create the texture
        tex_handle = ctypes.c_uint64(0)
        array_handle = ctypes.c_uint64(0)

        data_ptr = data_flat.ctypes.data_as(ctypes.c_void_p)

        if self.device.is_cuda:
            # Create CUDA texture
            success = self.runtime.core.wp_texture2d_create_device(
                self.device.context,
                width,
                height,
                num_channels,
                dtype_code,
                filter_mode,
                resolved_u,
                resolved_v,
                normalized_coords,
                data_ptr,
                ctypes.byref(tex_handle),
                ctypes.byref(array_handle),
            )
        else:
            # Create CPU texture
            success = self.runtime.core.wp_texture2d_create_host(
                width,
                height,
                num_channels,
                dtype_code,
                filter_mode,
                resolved_u,
                resolved_v,
                normalized_coords,
                data_ptr,
                ctypes.byref(tex_handle),
            )

        if not success:
            raise RuntimeError("Failed to create Texture2D")

        self._tex_handle = tex_handle.value
        self._array_handle = array_handle.value

    def __del__(self):
        if self._tex_handle == 0 and self._array_handle == 0:
            return

        try:
            if self.device.is_cuda:
                with self.device.context_guard:
                    self.runtime.core.wp_texture2d_destroy_device(
                        self.device.context, self._tex_handle, self._array_handle
                    )
            else:
                self.runtime.core.wp_texture2d_destroy_host(self._tex_handle)
        except (TypeError, AttributeError):
            # Suppress errors during shutdown
            pass

    def __ctype__(self) -> texture2d_t:
        """Return the ctypes structure for passing to kernels."""
        if self._tex_handle == 0:
            raise RuntimeError("Texture was created with data=None but never initialized. Cannot be used in kernels.")
        return texture2d_t(self._tex_handle, self._width, self._height, self._num_channels)


class Texture3D(_TextureBase):
    """Class representing a 3D texture.

    Textures provide hardware-accelerated filtering and addressing for
    regularly-gridded volumetric data on CUDA devices. On CPU, software-based
    filtering and addressing is used. Supports trilinear interpolation
    and various addressing modes (wrap, clamp, mirror, border).

    Supports uint8, uint16, and float32 data types. Integer textures are
    read as normalized floats in the [0, 1] range.

    Example:
        >>> import warp as wp
        >>> import numpy as np
        >>> # Create a 64x64x64 single-channel 3D texture on GPU
        >>> data = np.random.rand(64, 64, 64).astype(np.float32)
        >>> tex = wp.Texture3D(data, device="cuda:0")
        >>> # Create a texture on CPU
        >>> tex_cpu = wp.Texture3D(data, device="cpu")
        >>> # Create a compressed 8-bit 3D texture
        >>> data8 = (np.random.rand(64, 64, 64) * 255).astype(np.uint8)
        >>> tex8 = wp.Texture3D(data8, device="cuda:0")
    """

    # Native C++ type name for code generation
    _wp_native_name_ = "texture3d_t"

    from warp._src.codegen import Var as _Var  # noqa: PLC0415

    #: Member attributes available during code-gen (e.g.: w = tex.width)
    vars: ClassVar[dict[str, Var]] = {
        "width": _Var("width", int32),
        "height": _Var("height", int32),
        "depth": _Var("depth", int32),
    }

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._tex_handle = 0
        instance._array_handle = 0
        return instance

    def __init__(
        self,
        data: np.ndarray | array | None = None,
        width: int = 0,
        height: int = 0,
        depth: int = 0,
        num_channels: int = 1,
        dtype=np.float32,
        filter_mode: int = 1,
        address_mode: int | tuple[int, int, int] | None = None,
        address_mode_u: int | None = None,
        address_mode_v: int | None = None,
        address_mode_w: int | None = None,
        normalized_coords: bool = True,
        device=None,
    ):
        """Create a 3D texture.

        Args:
            data: Initial texture data as a numpy array or warp array.
                  Shape should be (depth, height, width) for 1-channel,
                  (depth, height, width, 2) for 2-channel, or
                  (depth, height, width, 4) for 4-channel textures.
                  Supported dtypes: uint8, uint16, float32 (or wp.uint8, wp.uint16, wp.float32).
                  If None, width/height/depth/num_channels must be specified.
            width: Texture width (required if data is None).
            height: Texture height (required if data is None).
            depth: Texture depth (required if data is None).
            num_channels: Number of channels (1, 2, or 4). Default is 1.
            dtype: Data type (uint8, uint16, or float32). Default is float32.
            filter_mode: Filtering mode - CLOSEST (0) or LINEAR (1). Default is LINEAR.
            address_mode: Address mode for all axes - WRAP (0), CLAMP (1), MIRROR (2), or BORDER (3).
                          Can be a single int (applied to all axes) or a tuple (u, v, w) for per-axis modes.
                          Default is CLAMP. Overridden by address_mode_u/v/w if specified.
            address_mode_u: Per-axis address mode for U. Overrides address_mode if specified.
            address_mode_v: Per-axis address mode for V. Overrides address_mode if specified.
            address_mode_w: Per-axis address mode for W. Overrides address_mode if specified.
            normalized_coords: If True (default), texture coordinates are in [0, 1] range.
                              If False, coordinates are in texel space [0, width/height/depth].
            device: Device to create the texture on (CPU or CUDA).
        """
        import warp._src.context  # noqa: PLC0415 - lazy import to avoid circular imports

        self.runtime = warp._src.context.runtime

        # Resolve per-axis address modes using base class helper
        resolved_u = self._resolve_address_mode(address_mode, address_mode_u, 0)
        resolved_v = self._resolve_address_mode(address_mode, address_mode_v, 1)
        resolved_w = self._resolve_address_mode(address_mode, address_mode_w, 2)

        if data is None:
            # Just store dimensions for lazy creation
            self.device = warp._src.context.get_device(device)
            self._width = width
            self._height = height
            self._depth = depth
            self._num_channels = num_channels
            self._dtype = np.dtype(dtype)
            self._dtype_code = self._dtype_to_code(dtype)
            self._filter_mode = filter_mode
            self._address_mode_u = resolved_u
            self._address_mode_v = resolved_v
            self._address_mode_w = resolved_w
            self._normalized_coords = normalized_coords
            self._tex_handle = 0
            self._array_handle = 0
            return

        # Get device and extract data
        if isinstance(data, np.ndarray):
            self.device = warp._src.context.get_device(device)

            # Determine dimensions from numpy array shape
            if data.ndim == 3:
                depth, height, width = data.shape
                num_channels = 1
            elif data.ndim == 4:
                depth, height, width, num_channels = data.shape
            else:
                raise ValueError("Data must be 3D or 4D numpy array")

            # Validate and get dtype code
            dtype_code = self._dtype_to_code(data.dtype)
            dtype = data.dtype

            # Ensure contiguous
            if not data.flags["C_CONTIGUOUS"]:
                data = np.ascontiguousarray(data)

            data_flat = data.flatten()
        elif is_array(data):
            self.device = data.device if device is None else warp._src.context.get_device(device)

            # Copy to numpy for now (could optimize later)
            np_data = data.numpy()
            if np_data.ndim == 3:
                depth, height, width = np_data.shape
                num_channels = 1
            elif np_data.ndim == 4:
                depth, height, width, num_channels = np_data.shape
            else:
                raise ValueError("Data must be 3D or 4D array")

            dtype_code = self._dtype_to_code(np_data.dtype)
            dtype = np_data.dtype

            if not np_data.flags["C_CONTIGUOUS"]:
                np_data = np.ascontiguousarray(np_data)
            data_flat = np_data.flatten()
        else:
            raise TypeError("data must be a numpy array or warp array")

        if num_channels not in (1, 2, 4):
            raise ValueError("num_channels must be 1, 2, or 4")

        self._width = width
        self._height = height
        self._depth = depth
        self._num_channels = num_channels
        self._dtype = np.dtype(dtype)
        self._dtype_code = dtype_code
        self._filter_mode = filter_mode
        self._address_mode_u = resolved_u
        self._address_mode_v = resolved_v
        self._address_mode_w = resolved_w
        self._normalized_coords = normalized_coords

        # Create the texture
        tex_handle = ctypes.c_uint64(0)
        array_handle = ctypes.c_uint64(0)

        data_ptr = data_flat.ctypes.data_as(ctypes.c_void_p)

        if self.device.is_cuda:
            # Create CUDA texture
            success = self.runtime.core.wp_texture3d_create_device(
                self.device.context,
                width,
                height,
                depth,
                num_channels,
                dtype_code,
                filter_mode,
                resolved_u,
                resolved_v,
                resolved_w,
                normalized_coords,
                data_ptr,
                ctypes.byref(tex_handle),
                ctypes.byref(array_handle),
            )
        else:
            # Create CPU texture
            success = self.runtime.core.wp_texture3d_create_host(
                width,
                height,
                depth,
                num_channels,
                dtype_code,
                filter_mode,
                resolved_u,
                resolved_v,
                resolved_w,
                normalized_coords,
                data_ptr,
                ctypes.byref(tex_handle),
            )

        if not success:
            raise RuntimeError("Failed to create Texture3D")

        self._tex_handle = tex_handle.value
        self._array_handle = array_handle.value

    def __del__(self):
        if self._tex_handle == 0 and self._array_handle == 0:
            return

        try:
            if self.device.is_cuda:
                with self.device.context_guard:
                    self.runtime.core.wp_texture3d_destroy_device(
                        self.device.context, self._tex_handle, self._array_handle
                    )
            else:
                self.runtime.core.wp_texture3d_destroy_host(self._tex_handle)
        except (TypeError, AttributeError):
            # Suppress errors during shutdown
            pass

    @property
    def depth(self) -> int:
        """Texture depth in pixels."""
        return self._depth

    @property
    def address_mode_w(self) -> int:
        """Address mode for W axis (0=WRAP, 1=CLAMP, 2=MIRROR, 3=BORDER)."""
        return self._address_mode_w

    def __ctype__(self) -> texture3d_t:
        """Return the ctypes structure for passing to kernels."""
        if self._tex_handle == 0:
            raise RuntimeError("Texture was created with data=None but never initialized. Cannot be used in kernels.")
        return texture3d_t(self._tex_handle, self._width, self._height, self._depth, self._num_channels)
