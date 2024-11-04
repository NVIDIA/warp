/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

template<typename T>
void scan_host(const T* values_in, T* values_out, int n, bool inclusive = true);
template<typename T>
void scan_device(const T* values_in, T* values_out, int n, bool inclusive = true);
