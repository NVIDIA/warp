// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

template <typename T> void scan_host(const T* values_in, T* values_out, int n, bool inclusive = true);
template <typename T> void scan_device(const T* values_in, T* values_out, int n, bool inclusive = true);
