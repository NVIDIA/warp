#pragma once

template<typename T>
void scan_host(const T* values_in, T* values_out, int n, bool inclusive = true);
template<typename T>
void scan_device(const T* values_in, T* values_out, int n, bool inclusive = true);

