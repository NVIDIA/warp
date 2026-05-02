# APIC Wave Simulation Example (CPU)

This example demonstrates Warp's APIC mode for capturing and executing graphs
with **CPU-only graph replay**. The Python side captures a simulation loop on
the CPU device and saves it as a `.wrp` file; the C++ side loads that file and
replays it on the host.

Note: loading a `.wrp` file currently requires a **CUDA-enabled Warp library**
even when the graph targets CPU — the `.wrp` parser and graph setup code still
live in `apic.cu`. Once loaded, *graph replay* runs entirely on the CPU and
makes no GPU or CUDA runtime calls. Porting the loading path into `apic.cpp`
(so non-CUDA builds can load `.wrp` files too) is planned future work.

## Overview

1. **Python Phase**: Capture a multi-kernel simulation loop on the CPU device
2. **C++ Phase**: Load the graph, replay it on the CPU, and visualize with OpenGL

### Key Differences from the CUDA Version

- **Replay runs CPU-only** — uses `wp_apic_cpu_replay_graph()` instead of
  `cudaGraphLaunch()`, with no GPU/CUDA dependency at replay time
- **No CUDA linking in the example** — `main.cpp` links only against `warp`
  and `warp-clang`; no `cuda.lib`/`cudart` needed
- **Host memory** — all parameter arrays are in regular system memory
- **CUDA-enabled Warp library still required** for `wp_apic_load_graph()` to
  parse the `.wrp` file (see the note above)

## Files

- `capture_wave.py` — Python script that captures the wave simulation graph on CPU
- `main.cpp` — C++ program (pure C++, no CUDA) with OpenGL visualization
- `Makefile` / `CMakeLists.txt` — Build systems (Make for Unix, CMake for cross-platform; both fetch glad v2 and link `warp.so` directly)
- `generated/` — Directory containing generated files:
  - `wave_sim.wrp` — Serialized graph (Warp Recorded Program)
  - `wave_sim_modules/` — Compiled CPU modules (.o files)

## Quick Start

### Prerequisites

- **Python 3.8+** with Warp installed
- **CMake 3.20+**
- **OpenGL 3.3** support
- **Warp native library** (`warp.dll` on Windows, `warp.so` on Linux,
  `libwarp.dylib` on macOS), built with CUDA support so
  `wp_apic_load_graph()` is available
- **Warp LLVM library** (`warp-clang.dll` on Windows, `warp-clang.so` on
  Linux, `libwarp-clang.dylib` on macOS) for CPU JIT

A GPU is not required at runtime — the graph replay is CPU-only — but the
Warp library must have been built with CUDA enabled so that `.wrp` loading
is compiled in. Once that is moved to `apic.cpp` in a future update, this
example will run on CPU-only builds too.

### Build and Run

**Using Make (Unix/Linux)**:

```bash
cd warp/examples/cpp/03_apic_visualization_cpu
make                            # auto-runs capture_wave.py and fetches glad v2
./03_apic_visualization_cpu     # interactive run
```

**Using CMake (cross-platform)**:

```bash
cd warp/examples/cpp/03_apic_visualization_cpu

# Step 1: Capture the graph (requires Python + Warp)
uv run capture_wave.py    # or: python capture_wave.py

# Step 2: Configure and build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Step 3: Run
./build/03_apic_visualization_cpu
```

**Headless smoke mode**:

The example also accepts `--smoke` as a single argv to run a headless
sanity check that loads the graph, queries parameters, and replays it
on the CPU 10 times without opening a GLFW window. CTest registers this
mode as `apic_visualization_cpu_smoke` so the example runs in CI on
hosts without a display server.

```bash
./03_apic_visualization_cpu --smoke    # exits 0 with "smoke OK (10 replay iterations)"
```

## Controls

- **Left-click**: Create waves at mouse position
- **Right-drag**: Rotate camera
- **Scroll**: Zoom in/out
- **ESC**: Exit

## APIC C API (CPU)

```cpp
// Load a graph for CPU execution (device_type=1, context=NULL)
APICGraph graph = wp_apic_load_graph(NULL, "path/to/graph", 1);

// Set/get named parameters (host memory)
wp_apic_set_param(graph, "heights", host_ptr, size);
wp_apic_get_param(graph, "heights_out", host_ptr, size);

// Execute all recorded operations on the CPU
wp_apic_cpu_replay_graph(graph);

// Cleanup
wp_apic_destroy_graph(graph);
```

## Current Limitations

- The `.wrp` loading code is currently in `apic.cu` (compiled only with CUDA).
  On non-CUDA builds of Warp, `wp_apic_load_graph` is stubbed to return NULL.
  A future update will port the CPU loading path to `apic.cpp` so this example
  runs on CPU-only Warp builds.
- Graph replay itself is already CPU-only and has no CUDA runtime dependency
  (see `wp_apic_cpu_replay_graph` in `apic.cpp`).
