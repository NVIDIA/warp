// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// fieldkit - resample_field
//
// ONE source, three targets. build_backends.py translates this file for each
// supported vendor; there is no vendor-specific copy of this kernel anywhere
// in the tree, and the release checklist blocks a change that lands on only
// one of them.
//
//   NVIDIA : nvcc  -x cu       resample_field.cu
//   AMD    : hipify + hipcc    resample_field.cu   (automatic, in-tree)
//   CPU    : c++ -DFIELDKIT_HOST resample_field.cu
//
// Anything added here is inherited by all three. Anything that cannot be
// expressed here becomes a second implementation, and then every future change
// to resampling has to be written, reviewed, tested and released three times.

#ifdef FIELDKIT_HOST
#define FK_DEVICE
#define FK_GLOBAL
#else
#define FK_DEVICE __device__
#define FK_GLOBAL __global__
#endif

FK_GLOBAL void resample_field(
    const float* __restrict__ pts,      // (N, 3)
    const float* __restrict__ val,      // (N,)
    const float* __restrict__ centres,  // (M, 3)
    float* __restrict__ out,            // (M,)
    int n, int m, float radius) {
  // Irregular by construction: the neighbourhood size varies per centre, the
  // loop exits early, and the accumulation order is fixed so the result is
  // reproducible across all three targets.
  const float r2 = radius * radius;
  for (int i = FK_INDEX; i < m; i += FK_STRIDE) {
    const float cx = centres[3 * i + 0];
    const float cy = centres[3 * i + 1];
    const float cz = centres[3 * i + 2];
    float acc = 0.0f;
    int cnt = 0;
    for (int j = 0; j < n; ++j) {
      const float dx = cx - pts[3 * j + 0];
      if (dx > radius || dx < -radius) continue;   // early exit
      const float dy = cy - pts[3 * j + 1];
      const float dz = cz - pts[3 * j + 2];
      const float d2 = dx * dx + dy * dy + dz * dz;
      if (d2 <= r2) {
        acc += val[j] / (d2 + 1e-6f);
        ++cnt;
      }
    }
    out[i] = cnt > 0 ? acc / (float)cnt : 0.0f;
  }
}
