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
import warnings
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


class Texture:
    """Unified texture class for hardware-accelerated sampling on GPU and software sampling on CPU.

    This class handles both 2D and 3D textures. The dimensionality is determined automatically
    from the input data shape, or can be set explicitly via the ``dims`` parameter or by using
    the :class:`Texture2D` or :class:`Texture3D` subclasses.

    Textures provide hardware-accelerated filtering and addressing for regularly-gridded data
    on CUDA devices. On CPU, software-based filtering and addressing is used. Supports
    bilinear/trilinear interpolation and various addressing modes (wrap, clamp, mirror, border).

    Supports uint8, uint16, and float32 data types. Integer textures are read as normalized
    floats in the [0, 1] range.

    Class Constants:
        ADDRESS_WRAP (int): Wrap coordinates (tile the texture) = 0
        ADDRESS_CLAMP (int): Clamp coordinates to [0, 1] = 1
        ADDRESS_MIRROR (int): Mirror coordinates at boundaries = 2
        ADDRESS_BORDER (int): Return 0 for coordinates outside [0, 1] = 3
        FILTER_POINT (int): Nearest-neighbor filtering = 0
        FILTER_LINEAR (int): Bilinear/trilinear filtering = 1

    Example::

        import warp as wp
        import numpy as np

        # Create a 2D texture
        data_2d = np.random.rand(256, 256).astype(np.float32)
        tex2d = wp.Texture(data_2d, device="cuda:0")
        # Create a 3D texture
        data_3d = np.random.rand(64, 64, 64).astype(np.float32)
        tex3d = wp.Texture(data_3d, device="cuda:0")
    """

    # Class constants for address modes (matching PR #1153 API)
    ADDRESS_WRAP = 0
    ADDRESS_CLAMP = 1
    ADDRESS_MIRROR = 2
    ADDRESS_BORDER = 3

    # Class constants for filter modes
    FILTER_POINT = 0
    FILTER_LINEAR = 1

    # Default dimensionality (None means auto-detect; subclasses override)
    _default_dims: int | None = None

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
        address_mode: int | tuple[int, ...] | None = None,
        address_mode_u: int | None = None,
        address_mode_v: int | None = None,
        address_mode_w: int | None = None,
        normalized_coords: bool = True,
        device=None,
        dims: int | None = None,
    ):
        """Create a texture.

        Args:
            data: Initial texture data as a numpy array or warp array.
                  For 2D: shape (height, width), (height, width, 2), or (height, width, 4).
                  For 3D: shape (depth, height, width), (depth, height, width, 2), or (depth, height, width, 4).
                  Supported dtypes: uint8, uint16, float32.
            width: Texture width (required if data is None).
            height: Texture height (required if data is None).
            depth: Texture depth (required if data is None for 3D textures).
            num_channels: Number of channels (1, 2, or 4). Only used if data is None. Default is 1.
            dtype: Data type (uint8, uint16, or float32). Only used if data is None;
                   when data is provided, dtype is inferred from the data. Default is float32.
            filter_mode: Filtering mode - FILTER_POINT (0) or FILTER_LINEAR (1). Default is LINEAR.
            address_mode: Address mode for all axes - ADDRESS_WRAP (0), ADDRESS_CLAMP (1),
                          ADDRESS_MIRROR (2), or ADDRESS_BORDER (3). Can be a single int or tuple.
            address_mode_u: Per-axis address mode for U. Overrides address_mode if specified.
            address_mode_v: Per-axis address mode for V. Overrides address_mode if specified.
            address_mode_w: Per-axis address mode for W (3D only). Overrides address_mode if specified.
            normalized_coords: If True (default), coordinates are in [0, 1] range.
                              If False, coordinates are in texel space.
            device: Device to create the texture on (CPU or CUDA).
            dims: Explicit dimensionality (2 or 3). If None, auto-detected from data.
        """
        import warp._src.context  # noqa: PLC0415

        self.runtime = warp._src.context.runtime

        # Determine dimensionality
        if self._default_dims is not None:
            # Subclass has fixed dimensionality
            self._dims = self._default_dims
        elif dims is not None:
            self._dims = dims
        else:
            # Auto-detect from data
            self._dims = self._detect_dims(data, depth)

        # Resolve address modes
        resolved_u = self._resolve_address_mode(address_mode, address_mode_u, 0)
        resolved_v = self._resolve_address_mode(address_mode, address_mode_v, 1)
        resolved_w = self._resolve_address_mode(address_mode, address_mode_w, 2) if self._dims == 3 else 1

        if data is None:
            # Lazy creation - just store dimensions
            self.device = warp._src.context.get_device(device)
            self._width = width
            self._height = height
            self._depth = depth if self._dims == 3 else 1
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

        # Extract data and dimensions
        data_flat, width, height, depth, num_channels, dtype, dtype_code = self._process_data(data, device)
        self.device = (
            warp._src.context.get_device(device)
            if not is_array(data)
            else (data.device if device is None else warp._src.context.get_device(device))
        )

        if num_channels not in (1, 2, 4):
            raise ValueError("num_channels must be 1, 2, or 4")

        self._width = width
        self._height = height
        self._depth = depth if self._dims == 3 else 1
        self._num_channels = num_channels
        self._dtype = np.dtype(dtype)
        self._dtype_code = dtype_code
        self._filter_mode = filter_mode
        self._address_mode_u = resolved_u
        self._address_mode_v = resolved_v
        self._address_mode_w = resolved_w
        self._normalized_coords = normalized_coords

        # Create the texture
        self._create_texture(data_flat)

    def _detect_dims(self, data, depth: int) -> int:
        """Auto-detect texture dimensionality from data or depth parameter."""
        if data is not None:
            if isinstance(data, np.ndarray):
                np_data = data
            elif is_array(data):
                np_data = data.numpy()
            else:
                raise TypeError("data must be a numpy array or warp array")

            ndim = np_data.ndim
            if ndim == 4:
                return 3  # (depth, height, width, channels)
            elif ndim == 3:
                # Could be (height, width, channels) for 2D or (depth, height, width) for 3D
                if np_data.shape[-1] not in (1, 2, 4):
                    return 3  # Last dim is not a valid channel count, so must be 3D
                else:
                    # Ambiguous case: last dim could be channels (for 2D) or width (for 3D)
                    # Default to 3D since the numpy array is 3-dimensional, but warn the user
                    warnings.warn(
                        f"Ambiguous array shape {np_data.shape}: could be interpreted as 2D texture "
                        f"with shape (height={np_data.shape[0]}, width={np_data.shape[1]}, "
                        f"channels={np_data.shape[2]}) or 3D texture with shape "
                        f"(depth={np_data.shape[0]}, height={np_data.shape[1]}, width={np_data.shape[2]}). "
                        f"Defaulting to 3D. Use dims=2 parameter or Texture2D class for 2D textures.",
                        stacklevel=3,
                    )
                    return 3
            else:
                return 2  # ndim == 2 is always 2D
        else:
            return 3 if depth > 1 else 2

    def _process_data(self, data, device):
        """Process input data and extract dimensions."""
        if isinstance(data, np.ndarray):
            np_data = data
        elif is_array(data):
            np_data = data.numpy()
        else:
            raise TypeError("data must be a numpy array or warp array")

        if not np_data.flags["C_CONTIGUOUS"]:
            np_data = np.ascontiguousarray(np_data)

        dtype_code = self._dtype_to_code(np_data.dtype)
        dtype = np_data.dtype

        if self._dims == 2:
            if np_data.ndim == 2:
                height, width = np_data.shape
                num_channels = 1
            elif np_data.ndim == 3:
                height, width, num_channels = np_data.shape
            else:
                raise ValueError("2D texture data must be 2D or 3D array")
            depth = 1
        else:  # 3D
            if np_data.ndim == 3:
                depth, height, width = np_data.shape
                num_channels = 1
            elif np_data.ndim == 4:
                depth, height, width, num_channels = np_data.shape
            else:
                raise ValueError("3D texture data must be 3D or 4D array")

        return np_data.flatten(), width, height, depth, num_channels, dtype, dtype_code

    def _create_texture(self, data_flat):
        """Create the underlying texture resource."""
        tex_handle = ctypes.c_uint64(0)
        array_handle = ctypes.c_uint64(0)
        data_ptr = data_flat.ctypes.data_as(ctypes.c_void_p)

        if self._dims == 2:
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
            raise RuntimeError(f"Failed to create Texture{self._dims}D")

        self._tex_handle = tex_handle.value
        self._array_handle = array_handle.value

    def __del__(self):
        if self._tex_handle == 0 and self._array_handle == 0:
            return

        try:
            if self._dims == 2:
                if self.device.is_cuda:
                    with self.device.context_guard:
                        self.runtime.core.wp_texture2d_destroy_device(
                            self.device.context, self._tex_handle, self._array_handle
                        )
                else:
                    self.runtime.core.wp_texture2d_destroy_host(self._tex_handle)
            else:  # 3D
                if self.device.is_cuda:
                    with self.device.context_guard:
                        self.runtime.core.wp_texture3d_destroy_device(
                            self.device.context, self._tex_handle, self._array_handle
                        )
                else:
                    self.runtime.core.wp_texture3d_destroy_host(self._tex_handle)
        except (TypeError, AttributeError):
            pass

    @staticmethod
    def _dtype_to_code(dtype):
        """Convert dtype to internal dtype code."""
        if dtype is uint8 or dtype is np.uint8:
            return 0
        elif dtype is uint16 or dtype is np.uint16:
            return 1
        elif dtype is float32 or dtype is np.float32 or dtype is float:
            return 2
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
    def dims(self) -> int:
        """Texture dimensionality (2 or 3)."""
        return self._dims

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
        """Texture handle ID (for advanced use)."""
        return self._tex_handle

    def __ctype__(self):
        """Return the ctypes structure for passing to kernels."""
        if self._tex_handle == 0:
            raise RuntimeError("Texture was created with data=None but never initialized.")
        if self._dims == 2:
            return texture2d_t(self._tex_handle, self._width, self._height, self._num_channels)
        else:
            return texture3d_t(self._tex_handle, self._width, self._height, self._depth, self._num_channels)


class Texture2D(Texture):
    """2D texture class.

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

    _default_dims = 2
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
        dtype=np.float32,
        filter_mode: int = 1,
        address_mode: int | tuple[int, int] | None = None,
        address_mode_u: int | None = None,
        address_mode_v: int | None = None,
        normalized_coords: bool = True,
        device=None,
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
            device=device,
        )

    def __ctype__(self) -> texture2d_t:
        """Return the ctypes structure for passing to kernels."""
        if self._tex_handle == 0:
            raise RuntimeError("Texture was created with data=None but never initialized.")
        return texture2d_t(self._tex_handle, self._width, self._height, self._num_channels)


class Texture3D(Texture):
    """3D texture class.

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

    _default_dims = 3
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
        dtype=np.float32,
        filter_mode: int = 1,
        address_mode: int | tuple[int, int, int] | None = None,
        address_mode_u: int | None = None,
        address_mode_v: int | None = None,
        address_mode_w: int | None = None,
        normalized_coords: bool = True,
        device=None,
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
            device=device,
        )

    def __ctype__(self) -> texture3d_t:
        """Return the ctypes structure for passing to kernels."""
        if self._tex_handle == 0:
            raise RuntimeError("Texture was created with data=None but never initialized.")
        return texture3d_t(self._tex_handle, self._width, self._height, self._depth, self._num_channels)
