# APIC Wave Simulation Example (CPU)

This example demonstrates Warp's CPU graph capture and API Capture (APIC) save/load workflow with **CPU-only graph replay**. The Python side captures a simulation loop on the CPU device and saves it as a `.wrp` file; the C++ side loads that file and replays its operation stream on the host.

CPU `.wrp` graph loading uses the pure-C++ APIC loader in the Warp native
library and does not require a CUDA-enabled build. Replay also runs entirely on
the CPU, but it still needs the `warp-clang` library and the companion
`_modules` directory with the recorded CPU kernel object files.

## Overview

1. **Python Phase**: Capture a multi-kernel simulation loop on the CPU device
2. **C++ Phase**: Load the graph, replay it on the CPU, and visualize with OpenGL

### Key Differences from the CUDA Version

- **Replay runs CPU-only** — uses `wp_apic_cpu_replay_graph()` instead of
  `cudaGraphLaunch()` for host execution
- **No CUDA linking in the example** — `main.cpp` links only against `warp`
  and `warp-clang`; no `cuda.lib`/`cudart` needed
- **Host memory** — all parameter arrays are in regular system memory
- **CPU-only Warp builds supported** — `wp_apic_load_graph()` can load CPU
  `.wrp` graphs when Warp is built without CUDA

## Files

- `capture_wave.py` — Python script that captures the wave simulation graph on CPU
- `main.cpp` — C++ program (pure C++, no CUDA) with OpenGL visualization
- `Makefile` / `CMakeLists.txt` — Build systems (Make for Unix, CMake for
  cross-platform; both link the Warp and `warp-clang` shared libraries)
- `generated/` — Directory containing generated files:
  - `wave_sim.wrp` — Serialized APIC graph representation
  - `wave_sim_modules/` — Compiled CPU modules (.o files)

## Quick Start

### Prerequisites

- **Python 3.10+** with Warp installed
- **CMake 3.20+**
- **OpenGL 3.3** support
- **Warp native library** (`warp.dll` and `warp.lib` on Windows, `warp.so` on
  Linux, or `libwarp.dylib` on macOS)
- **Warp LLVM library** (`warp-clang.dll` and `warp-clang.lib` on Windows,
  `warp-clang.so` on Linux, or `libwarp-clang.dylib` on macOS) for CPU JIT

The generated `wave_sim_modules/` directory must be available next to the
`.wrp` graph for replay.

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

If using the Visual Studio generator on Windows instead of Ninja, run:

```powershell
.\build\Release\03_apic_visualization_cpu.exe
```

**Headless smoke mode**:

The example also accepts `--smoke` as a single argv to run a headless
sanity check that loads the graph, queries parameters, and replays it
on the CPU 10 times without opening a GLFW window. CTest registers this
mode as `apic_visualization_cpu_smoke` so the example runs in CI on
hosts without a display server.

**Using Make (Unix/Linux)**:

```bash
./03_apic_visualization_cpu --smoke    # exits 0 with "smoke OK (10 replay iterations)"
```

**Using CMake (Ninja / Unix Makefiles)**:

```bash
./build/03_apic_visualization_cpu --smoke
```

**Using CMake (Visual Studio on Windows)**:

```powershell
.\build\Release\03_apic_visualization_cpu.exe --smoke
```

## Controls

- **Left-click**: Create waves at mouse position
- **Right-drag**: Rotate camera
- **Scroll**: Zoom in/out
- **ESC**: Exit

## APIC C API (CPU)

```cpp
// Load a graph for CPU execution (device_type=1, context=NULL)
APICGraph* graph = wp_apic_load_graph(NULL, "path/to/graph", 1);

// Set/get named parameters (host memory)
wp_apic_set_param(graph, "heights", host_ptr, size);
wp_apic_get_param(graph, "heights_out", host_ptr, size);

// Execute all recorded operations on the CPU
wp_apic_cpu_replay_graph(graph);

// Cleanup
wp_apic_destroy_graph(graph);
```

CPU replay also needs kernel function pointers from the companion `.o` files.
The example includes `warp_clang.h` and links `warp-clang`, then loads each
object with `wp_load_obj()`, resolves the recorded forward and backward symbols
with `wp_lookup()`, and registers them with
`wp_apic_register_loaded_cpu_kernel()` using the recorded kernel key and module
hash. Before loading the graph, it verifies that both native libraries match
the version declared by the installed headers.

## Current Limitations

- CPU replay requires `warp-clang` and the companion `wave_sim_modules/`
  directory so the recorded CPU kernel object files can be loaded.
