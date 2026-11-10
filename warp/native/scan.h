// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

template <typename T> void scan_host(const T* values_in, T* values_out, int n, bool inclusive = true);
template <typename T>
void scan_host(
    const T* values_in,
    T* values_out,
    int n,
    int in_byte_stride,
    int out_byte_stride,
    int type_length,
    bool inclusive = true
);
template <typename T> bool scan_device(const T* values_in, T* values_out, int n, bool inclusive = true);
// When temp_storage is null, the scan allocates and frees its own temporary
// storage. Otherwise it uses the caller's buffer, which must hold at least
// scan_device_temp_storage_size() bytes, and neither allocates nor frees memory.
template <typename T>
bool scan_device(
    const T* values_in,
    T* values_out,
    int n,
    int in_byte_stride,
    int out_byte_stride,
    int type_length,
    bool inclusive = true,
    void* temp_storage = nullptr,
    size_t temp_storage_bytes = 0
);
// Bytes of temporary storage that scan_device() needs for n elements on the given
// CUDA context. The size covers both inclusive and exclusive scans and any vector
// length, since vector components are scanned one after another.
template <typename T> bool scan_device_temp_storage_size(void* context, int n, size_t* temp_storage_bytes);
