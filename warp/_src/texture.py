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

from warp._src.types import array, float32, int32, is_array, uint8, uint16

# Note: warp._src.context.runtime is accessed lazily via self.runtime = warp._src.context.runtime
# in __init__ methods to avoid circular imports


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

    # Default dimensionality (None means auto-detect; subclasses override)
    _default_dims: int | None = None

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._tex_handle = 0
        instance._array_handle = 0
        instance._mipmap_handle = 0
        instance._num_mip_levels = 1
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
        device: DeviceLike = None,
        dims: int | None = None,
        num_mip_levels: int = 1,
        mip_filter_mode: int = 1,
    ):
        """Create a texture.

        Args:
            data: Initial texture data as a NumPy array or Warp array.
                For 2D: shape ``(height, width)``, ``(height, width, 2)``,
                or ``(height, width, 4)``.
                For 3D: shape ``(depth, height, width)``,
                ``(depth, height, width, 2)``, or ``(depth, height, width, 4)``.
                Supported dtypes: ``uint8``, ``uint16``, ``float32``.
            width: Texture width (required if ``data`` is ``None``).
            height: Texture height (required if ``data`` is ``None``).
            depth: Texture depth (required if ``data`` is ``None`` for 3D textures).
            num_channels: Number of channels (1, 2, or 4). Only used when
                ``data`` is ``None``.
            dtype: Data type (``uint8``, ``uint16``, or ``float32``). Only used
                when ``data`` is ``None``; otherwise inferred from the data.
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
            dims: Explicit dimensionality (2 or 3). If ``None``,
                auto-detected from ``data``.
            num_mip_levels: Number of mipmap levels. 1 means no mipmaps
                (default). 0 means auto-compute the full mip chain down
                to 1x1.
            mip_filter_mode: Filtering mode between mip levels —
                :attr:`FILTER_POINT` or :attr:`FILTER_LINEAR`.
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
            self._mip_filter_mode = mip_filter_mode
            self._address_mode_u = resolved_u
            self._address_mode_v = resolved_v
            self._address_mode_w = resolved_w
            self._normalized_coords = normalized_coords
            self._tex_handle = 0
            self._array_handle = 0
            return

        # Extract data and dimensions
        np_data, data_flat, width, height, depth, num_channels, dtype, dtype_code = self._process_data(data, device)
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
        self._mip_filter_mode = mip_filter_mode
        self._address_mode_u = resolved_u
        self._address_mode_v = resolved_v
        self._address_mode_w = resolved_w
        self._normalized_coords = normalized_coords

        # Resolve mip level count
        if num_mip_levels == 0:
            self._num_mip_levels = self._compute_mip_count(
                self._width, self._height, self._depth if self._dims == 3 else 1
            )
        else:
            self._num_mip_levels = num_mip_levels

        # Generate mip chain if needed, then create the texture
        if self._num_mip_levels > 1:
            if self._dims == 2:
                mip_flat, level_widths, level_heights = self._generate_mip_chain_2d(
                    np_data, self._num_mip_levels, num_channels
                )
                self._create_texture(mip_flat, level_widths, level_heights, None)
            else:
                mip_flat, level_widths, level_heights, level_depths = self._generate_mip_chain_3d(
                    np_data, self._num_mip_levels, num_channels
                )
                self._create_texture(mip_flat, level_widths, level_heights, level_depths)
        else:
            self._create_texture(data_flat)

    def _detect_dims(self, data, depth: int) -> int:
        """Auto-detect texture dimensionality from data or depth parameter."""
        if data is not None:
            if isinstance(data, np.ndarray):
                np_data = data
            elif is_array(data):
                np_data = data.numpy()
            else:
                raise TypeError("data must be a NumPy array or Warp array")

            ndim = np_data.ndim
            if ndim == 4:
                return 3  # (depth, height, width, channels)
            elif ndim == 3:
                # Could be (height, width, channels) for 2D or (depth, height, width) for 3D
                if np_data.shape[-1] not in (1, 2, 4):
                    return 3  # Last dim is not a valid channel count, so must be 3D
                else:
                    # Ambiguous case: last dim could be channels (for 2D) or width (for 3D)
                    # Default to 3D since the NumPy array is 3-dimensional, but warn the user
                    from warp._src.utils import warn  # noqa: PLC0415

                    warn(
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
            raise TypeError("data must be a NumPy array or Warp array")

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

        return np_data, np_data.flatten(), width, height, depth, num_channels, dtype, dtype_code

    def _create_texture(self, data_flat, mip_widths=None, mip_heights=None, mip_depths=None):
        """Create the underlying texture resource."""
        tex_handle = ctypes.c_uint64(0)
        array_handle = ctypes.c_uint64(0)
        mipmap_handle = ctypes.c_uint64(0)
        data_ptr = data_flat.ctypes.data_as(ctypes.c_void_p)
        widths_ptr = mip_widths.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if mip_widths is not None else None
        heights_ptr = mip_heights.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if mip_heights is not None else None

        if self._dims == 2:
            if self.device.is_cuda:
                success = self.runtime.core.wp_texture2d_create_device(
                    self.device.context,
                    self._width,
                    self._height,
                    self._num_channels,
                    self._dtype_code,
                    self._filter_mode,
                    self._mip_filter_mode,
                    self._address_mode_u,
                    self._address_mode_v,
                    self._normalized_coords,
                    self._num_mip_levels,
                    data_ptr,
                    widths_ptr,
                    heights_ptr,
                    ctypes.byref(tex_handle),
                    ctypes.byref(array_handle),
                    ctypes.byref(mipmap_handle),
                )
            else:
                success = self.runtime.core.wp_texture2d_create_host(
                    self._width,
                    self._height,
                    self._num_channels,
                    self._dtype_code,
                    self._filter_mode,
                    self._mip_filter_mode,
                    self._address_mode_u,
                    self._address_mode_v,
                    self._normalized_coords,
                    self._num_mip_levels,
                    data_ptr,
                    widths_ptr,
                    heights_ptr,
                    ctypes.byref(tex_handle),
                )
        else:  # 3D
            depths_ptr = mip_depths.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if mip_depths is not None else None
            if self.device.is_cuda:
                success = self.runtime.core.wp_texture3d_create_device(
                    self.device.context,
                    self._width,
                    self._height,
                    self._depth,
                    self._num_channels,
                    self._dtype_code,
                    self._filter_mode,
                    self._mip_filter_mode,
                    self._address_mode_u,
                    self._address_mode_v,
                    self._address_mode_w,
                    self._normalized_coords,
                    self._num_mip_levels,
                    data_ptr,
                    widths_ptr,
                    heights_ptr,
                    depths_ptr,
                    ctypes.byref(tex_handle),
                    ctypes.byref(array_handle),
                    ctypes.byref(mipmap_handle),
                )
            else:
                success = self.runtime.core.wp_texture3d_create_host(
                    self._width,
                    self._height,
                    self._depth,
                    self._num_channels,
                    self._dtype_code,
                    self._filter_mode,
                    self._mip_filter_mode,
                    self._address_mode_u,
                    self._address_mode_v,
                    self._address_mode_w,
                    self._normalized_coords,
                    self._num_mip_levels,
                    data_ptr,
                    widths_ptr,
                    heights_ptr,
                    depths_ptr,
                    ctypes.byref(tex_handle),
                )

        if not success:
            raise RuntimeError(f"Failed to create Texture{self._dims}D")

        self._tex_handle = tex_handle.value
        self._array_handle = array_handle.value
        self._mipmap_handle = mipmap_handle.value

    def __del__(self):
        if self._tex_handle == 0 and self._array_handle == 0 and self._mipmap_handle == 0:
            return

        try:
            if self._dims == 2:
                if self.device.is_cuda:
                    with self.device.context_guard:
                        self.runtime.core.wp_texture2d_destroy_device(
                            self.device.context, self._tex_handle, self._array_handle, self._mipmap_handle
                        )
                else:
                    self.runtime.core.wp_texture2d_destroy_host(self._tex_handle)
            else:
                if self.device.is_cuda:
                    with self.device.context_guard:
                        self.runtime.core.wp_texture3d_destroy_device(
                            self.device.context, self._tex_handle, self._array_handle, self._mipmap_handle
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
    def num_mip_levels(self) -> int:
        """Number of mipmap levels (1 means no mipmaps)."""
        return self._num_mip_levels

    @property
    def id(self) -> int:
        """Texture handle ID (for advanced use)."""
        return self._tex_handle

    @staticmethod
    def _compute_mip_count(width: int, height: int, depth: int = 1) -> int:
        """Compute the maximum number of mip levels for given dimensions."""
        max_dim = max(width, height, depth)
        count = 1
        while max_dim > 1:
            max_dim = max(max_dim // 2, 1)
            count += 1
        return count

    @staticmethod
    def _downsample_axis(arr: np.ndarray, axis: int) -> np.ndarray:
        """Downsample array by 2x along the given axis using a box filter."""
        n = arr.shape[axis]
        if n <= 1:
            return arr.copy()
        new_n = max(n // 2, 1)
        slices_even = [slice(None)] * arr.ndim
        slices_odd = [slice(None)] * arr.ndim
        slices_even[axis] = slice(0, new_n * 2, 2)
        slices_odd[axis] = slice(1, new_n * 2, 2)
        return (arr[tuple(slices_even)].astype(np.float64) + arr[tuple(slices_odd)].astype(np.float64)) * 0.5

    @staticmethod
    def _generate_mip_chain_2d(
        np_data: np.ndarray, num_levels: int, num_channels: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a 2D mipmap chain using box filter downsampling.

        Args:
            np_data: Level 0 data, shape ``(H, W)`` or ``(H, W, C)``.
            num_levels: Number of mip levels to generate.
            num_channels: Number of channels.

        Returns:
            Tuple of ``(concatenated_flat, level_widths, level_heights)``.
        """
        original_dtype = np_data.dtype
        if np_data.ndim == 2:
            h, w = np_data.shape
        else:
            h, w, _ = np_data.shape

        levels = [np_data]
        widths = [w]
        heights = [h]

        current = np_data.astype(np.float64)
        for _ in range(1, num_levels):
            # Downsample height (axis 0) then width (axis 1)
            current = Texture._downsample_axis(current, 0)
            current = Texture._downsample_axis(current, 1)

            if original_dtype == np.uint8:
                level_data = np.clip(np.rint(current), 0, 255).astype(np.uint8)
            elif original_dtype == np.uint16:
                level_data = np.clip(np.rint(current), 0, 65535).astype(np.uint16)
            else:
                level_data = current.astype(np.float32)

            levels.append(level_data)
            if np_data.ndim == 2:
                lh, lw = level_data.shape
            else:
                lh, lw, _ = level_data.shape
            widths.append(lw)
            heights.append(lh)

        concatenated = np.concatenate([level.flatten() for level in levels])
        return concatenated, np.array(widths, dtype=np.int32), np.array(heights, dtype=np.int32)

    @staticmethod
    def _generate_mip_chain_3d(
        np_data: np.ndarray, num_levels: int, num_channels: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate a 3D mipmap chain using box filter downsampling.

        Args:
            np_data: Level 0 data, shape ``(D, H, W)`` or ``(D, H, W, C)``.
            num_levels: Number of mip levels to generate.
            num_channels: Number of channels.

        Returns:
            Tuple of ``(concatenated_flat, level_widths, level_heights,
            level_depths)``.
        """
        original_dtype = np_data.dtype
        if np_data.ndim == 3:
            d, h, w = np_data.shape
        else:
            d, h, w, _ = np_data.shape

        levels = [np_data]
        widths = [w]
        heights = [h]
        depths = [d]

        current = np_data.astype(np.float64)
        for _ in range(1, num_levels):
            # Downsample depth (axis 0), height (axis 1), width (axis 2)
            current = Texture._downsample_axis(current, 0)
            current = Texture._downsample_axis(current, 1)
            current = Texture._downsample_axis(current, 2)

            if original_dtype == np.uint8:
                level_data = np.clip(np.rint(current), 0, 255).astype(np.uint8)
            elif original_dtype == np.uint16:
                level_data = np.clip(np.rint(current), 0, 65535).astype(np.uint16)
            else:
                level_data = current.astype(np.float32)

            levels.append(level_data)
            if np_data.ndim == 3:
                ld, lh, lw = level_data.shape
            else:
                ld, lh, lw, _ = level_data.shape
            widths.append(lw)
            heights.append(lh)
            depths.append(ld)

        concatenated = np.concatenate([level.flatten() for level in levels])
        return (
            concatenated,
            np.array(widths, dtype=np.int32),
            np.array(heights, dtype=np.int32),
            np.array(depths, dtype=np.int32),
        )

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
        num_mip_levels: int = 1,
        mip_filter_mode: int = 1,
        device: DeviceLike = None,
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
            num_mip_levels=num_mip_levels,
            mip_filter_mode=mip_filter_mode,
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
        num_mip_levels: int = 1,
        mip_filter_mode: int = 1,
        device: DeviceLike = None,
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
            num_mip_levels=num_mip_levels,
            mip_filter_mode=mip_filter_mode,
            device=device,
        )

    def __ctype__(self) -> texture3d_t:
        """Return the ctypes structure for passing to kernels."""
        if self._tex_handle == 0:
            raise RuntimeError("Texture was created with data=None but never initialized.")
        return texture3d_t(self._tex_handle, self._width, self._height, self._depth, self._num_channels)
