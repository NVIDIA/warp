Mark `byte_offset_helper` as `CUDA_CALLABLE`. It is reachable from device code
through array slicing in kernels, so a strict Clang-based CUDA compile rejected
the call as host-only. Tile warp-scan code now derives its lane mask and lane
indices from `WP_TILE_WARP_SIZE` instead of repeating the width as literals;
values are unchanged on all supported architectures.
