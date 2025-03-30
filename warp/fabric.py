# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes
import math
from typing import Any

import warp
from warp.types import *


class fabricbucket_t(ctypes.Structure):
    _fields_ = [
        ("index_start", ctypes.c_size_t),
        ("index_end", ctypes.c_size_t),
        ("ptr", ctypes.c_void_p),
        ("lengths", ctypes.c_void_p),
    ]

    def __init__(self, index_start=0, index_end=0, ptr=None, lengths=None):
        self.index_start = index_start
        self.index_end = index_end
        self.ptr = ctypes.c_void_p(ptr)
        self.lengths = ctypes.c_void_p(lengths)


class fabricarray_t(ctypes.Structure):
    _fields_ = [
        ("buckets", ctypes.c_void_p),  # array of fabricbucket_t on the correct device
        ("nbuckets", ctypes.c_size_t),
        ("size", ctypes.c_size_t),
    ]

    def __init__(self, buckets=None, nbuckets=0, size=0):
        self.buckets = ctypes.c_void_p(buckets)
        self.nbuckets = nbuckets
        self.size = size


class indexedfabricarray_t(ctypes.Structure):
    _fields_ = [
        ("fa", fabricarray_t),
        ("indices", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
    ]

    def __init__(self, fa=None, indices=None):
        if fa is None:
            self.fa = fabricarray_t()
        else:
            self.fa = fa.__ctype__()

        if indices is None:
            self.indices = ctypes.c_void_p(None)
            self.size = 0
        else:
            self.indices = ctypes.c_void_p(indices.ptr)
            self.size = indices.size


def fabric_to_warp_dtype(type_info, attrib_name):
    if not type_info[0]:
        raise RuntimeError(f"Attribute '{attrib_name}' cannot be used in Warp")

    base_type_dict = {
        "b": warp.bool,  # boolean
        "i1": warp.int8,
        "i2": warp.int16,
        "i4": warp.int32,
        "i8": warp.int64,
        "u1": warp.uint8,
        "u2": warp.uint16,
        "u4": warp.uint32,
        "u8": warp.uint64,
        "f2": warp.float16,
        "f4": warp.float32,
        "f8": warp.float64,
    }

    base_dtype = base_type_dict.get(type_info[1])
    if base_dtype is None:
        raise RuntimeError(f"Attribute '{attrib_name}' base data type '{type_info[1]}' is not supported in Warp")

    elem_count = type_info[2]
    role = type_info[4]

    if role in ("text", "path"):
        raise RuntimeError(f"Attribute '{attrib_name}' role '{role}' is not supported in Warp")

    if elem_count > 1:
        # vector or matrix type
        if role == "quat" and elem_count == 4:
            return quaternion(base_dtype)
        elif role in ("matrix", "transform", "frame"):
            # only square matrices are currently supported
            mat_size = int(math.sqrt(elem_count))
            assert mat_size * mat_size == elem_count
            return matrix((mat_size, mat_size), base_dtype)
        else:
            return vector(elem_count, base_dtype)
    else:
        # scalar type
        return base_dtype


class fabricarray(noncontiguous_array_base[T]):
    # member attributes available during code-gen (e.g.: d = arr.shape[0])
    # (initialized when needed)
    _vars = None

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.deleter = None
        return instance

    def __init__(self, data=None, attrib=None, dtype=Any, ndim=None):
        super().__init__(ARRAY_TYPE_FABRIC)

        if data is not None:
            from .context import runtime

            # ensure the attribute name was also specified
            if not isinstance(attrib, str):
                raise ValueError(f"Invalid attribute name: {attrib}")

            # get the fabric interface dictionary
            if isinstance(data, dict):
                iface = data
            elif hasattr(data, "__fabric_arrays_interface__"):
                iface = data.__fabric_arrays_interface__
            else:
                raise ValueError(
                    "Invalid data argument for fabricarray: expected dict or object with __fabric_arrays_interface__"
                )

            version = iface.get("version")
            if version != 1:
                raise ValueError(f"Unsupported Fabric interface version: {version}")

            device = iface.get("device")
            if not isinstance(device, str):
                raise ValueError(f"Invalid Fabric interface device: {device}")

            self.device = runtime.get_device(device)

            attribs = iface.get("attribs")
            if not isinstance(attribs, dict):
                raise ValueError("Failed to get Fabric interface attributes")

            # look up attribute info by name
            attrib_info = attribs.get(attrib)
            if not isinstance(attrib_info, dict):
                raise ValueError(f"Failed to get attribute '{attrib}'")

            type_info = attrib_info["type"]
            assert len(type_info) == 5

            self.dtype = fabric_to_warp_dtype(type_info, attrib)

            self.access = attrib_info["access"]

            pointers = attrib_info["pointers"]
            counts = attrib_info["counts"]

            if not (hasattr(pointers, "__len__") and hasattr(counts, "__len__") and len(pointers) == len(counts)):
                raise RuntimeError("Attribute pointers and counts must be lists of the same size")

            # check whether it's an array
            array_depth = type_info[3]
            if array_depth == 0:
                self.ndim = 1
                array_lengths = None
            elif array_depth == 1:
                self.ndim = 2
                array_lengths = attrib_info["array_lengths"]
                if not hasattr(array_lengths, "__len__") or len(array_lengths) != len(pointers):
                    raise RuntimeError(
                        "Attribute `array_lengths` must be a list of the same size as `pointers` and `counts`"
                    )
            else:
                raise ValueError(f"Invalid attribute array depth: {array_depth}")

            num_buckets = len(pointers)
            size = 0
            buckets = (fabricbucket_t * num_buckets)()

            if num_buckets > 0:
                for i in range(num_buckets):
                    buckets[i].index_start = size
                    buckets[i].index_end = size + counts[i]
                    buckets[i].ptr = pointers[i]
                    if array_lengths:
                        buckets[i].lengths = array_lengths[i]
                    size += counts[i]

                if self.device.is_cuda:
                    # copy bucket info to device
                    buckets_size = ctypes.sizeof(buckets)
                    allocator = self.device.get_allocator()
                    buckets_ptr = allocator.alloc(buckets_size)
                    cuda_stream = self.device.stream.cuda_stream
                    runtime.core.memcpy_h2d(
                        self.device.context, buckets_ptr, ctypes.addressof(buckets), buckets_size, cuda_stream
                    )
                    self.deleter = allocator.deleter
                else:
                    buckets_ptr = ctypes.addressof(buckets)
            else:
                buckets_ptr = None

            self.buckets = buckets
            self.size = size
            self.shape = (size,)

            self.ctype = fabricarray_t(buckets_ptr, num_buckets, size)

        else:
            # empty array or type annotation
            self.dtype = dtype
            self.ndim = ndim or 1
            self.device = None
            self.access = None
            self.buckets = None
            self.size = 0
            self.shape = (0,)
            self.ctype = fabricarray_t()

    def __del__(self):
        # release the bucket info if needed
        if self.deleter is None:
            return

        buckets_size = ctypes.sizeof(self.buckets)
        with self.device.context_guard:
            self.deleter(self.ctype.buckets, buckets_size)

    def __ctype__(self):
        return self.ctype

    def __len__(self):
        return self.size

    def __str__(self):
        if self.device is None:
            # type annotation
            return f"fabricarray{self.dtype}"
        else:
            return str(self.numpy())

    def __getitem__(self, key):
        if isinstance(key, array):
            return indexedfabricarray(fa=self, indices=key)
        else:
            raise ValueError(f"Fabric arrays only support indexing using index arrays, got key of type {type(key)}")

    @property
    def vars(self):
        # member attributes available during code-gen (e.g.: d = arr.shape[0])
        # Note: we use a shared dict for all fabricarray instances
        if fabricarray._vars is None:
            fabricarray._vars = {"size": warp.codegen.Var("size", uint64)}
        return fabricarray._vars

    def fill_(self, value):
        # TODO?
        # filling Fabric arrays of arrays is not supported, because they are jagged arrays of arbitrary lengths
        if self.ndim > 1:
            raise RuntimeError("Filling Fabric arrays of arrays is not supported")

        super().fill_(value)


# special case for fabric array of arrays
# equivalent to calling fabricarray(..., ndim=2)
def fabricarrayarray(**kwargs):
    kwargs["ndim"] = 2
    return fabricarray(**kwargs)


class indexedfabricarray(noncontiguous_array_base[T]):
    # member attributes available during code-gen (e.g.: d = arr.shape[0])
    # (initialized when needed)
    _vars = None

    def __init__(self, fa=None, indices=None, dtype=None, ndim=None):
        super().__init__(ARRAY_TYPE_FABRIC_INDEXED)

        if fa is not None:
            check_index_array(indices, fa.device)
            self.fa = fa
            self.indices = indices
            self.dtype = fa.dtype
            self.ndim = fa.ndim
            self.device = fa.device
            self.size = indices.size
            self.shape = (indices.size,)
            self.ctype = indexedfabricarray_t(fa, indices)
        else:
            # allow empty indexedarrays in type annotations
            self.fa = None
            self.indices = None
            self.dtype = dtype
            self.ndim = ndim or 1
            self.device = None
            self.size = 0
            self.shape = (0,)
            self.ctype = indexedfabricarray_t()

    def __ctype__(self):
        return self.ctype

    def __len__(self):
        return self.size

    def __str__(self):
        if self.device is None:
            # type annotation
            return f"indexedfabricarray{self.dtype}"
        else:
            return str(self.numpy())

    @property
    def vars(self):
        # member attributes available during code-gen (e.g.: d = arr.shape[0])
        # Note: we use a shared dict for all indexedfabricarray instances
        if indexedfabricarray._vars is None:
            indexedfabricarray._vars = {"size": warp.codegen.Var("size", uint64)}
        return indexedfabricarray._vars

    def fill_(self, value):
        # TODO?
        # filling Fabric arrays of arrays is not supported, because they are jagged arrays of arbitrary lengths
        if self.ndim > 1:
            raise RuntimeError("Filling indexed Fabric arrays of arrays is not supported")

        super().fill_(value)


# special case for indexed fabric array of arrays
# equivalent to calling fabricarray(..., ndim=2)
def indexedfabricarrayarray(**kwargs):
    kwargs["ndim"] = 2
    return indexedfabricarray(**kwargs)
