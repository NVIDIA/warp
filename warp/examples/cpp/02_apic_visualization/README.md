# APIC Wave Simulation Example

This example demonstrates Warp's APIC (API Capture) mode for capturing and executing CUDA graphs from C++.
It implements an interactive wave simulation where users can create ripples by clicking, visualized in real-time using GLFW/OpenGL.

## Overview

APIC mode captures CUDA kernel launches into a graph during Python execution, then allows efficient replay from C++ without the Python runtime. This example demonstrates:

1. **Python Phase**: Capture a multi-kernel simulation loop as a single CUDA graph
2. **C++ Phase**: Load the graph, execute it with dynamic inputs, and visualize results

### Key APIC Benefits Demonstrated

- **Graph Complexity**: The captured graph contains 17 kernel launches (1 displacement + 16 wave solve iterations)
- **Single Launch**: C++ executes the entire frame with ONE `cudaGraphLaunch()` call
- **Dynamic Parameters**: Mouse position is passed as a parameter each frame - no graph rebuilding needed
- **Zero Python Runtime**: Production deployment requires only CUDA and the Warp native library

## Files

- `capture_wave.py` - Python script that captures the wave simulation graph
- `main.cu` - C++ program that loads the graph and provides interactive visualization
- `Makefile` / `CMakeLists.txt` - Build systems (Make for Unix, CMake for cross-platform)
- `generated/` - Directory containing generated files (auto-created by capture script):
  - `wave_sim.wrp` - Serialized CUDA graph (Warp Recorded Program)
  - `wave_sim_modules/` - Compiled CUDA modules (cubins)

## Quick Start

### Prerequisites

This example runs from within the Warp repository.

**Requirements:**
- **Python 3.8+**
- **CUDA Toolkit 12.0+** (includes `nvcc` compiler)
- **NVIDIA GPU**
- **GLFW library** (for OpenGL window creation)
- **Warp native library** (built as part of Warp)
- **Build System**: GNU Make (Unix/Linux) or CMake 3.20+ (cross-platform)
- **Note**: macOS is **not supported** (CUDA not available on macOS)

**Setup:**

Clone and build Warp:
```bash
git clone https://github.com/NVIDIA/warp.git
cd warp

# Build and install (choose one):
uv run build_lib.py  # Option A (recommended)
# or: python build_lib.py && pip install -e .  # Option B

cd warp/examples/cpp/02_apic_visualization
```

### Build and Run

**Using Make (Unix/Linux)**:

```bash
make              # Build everything (auto-captures graph if needed)
./02_apic_visualization  # Run
```

**Make Targets**:

- `make` (default) - Build everything, auto-capture graph if needed
- `make cpp` - Build only C++ program (fast iteration, assumes graph exists)
- `make capture` - Capture the APIC graph only (no C++ build)
- `make clean` - Remove executable only
- `make distclean` - Remove executable and `generated/` directory

**Using CMake (cross-platform)**:

```bash
python capture_wave.py                        # Step 1: Capture the graph
cmake -B build -DCMAKE_BUILD_TYPE=Release     # Step 2: Configure
cmake --build build --config Release          # Step 3: Build
./build/02_apic_visualization                 # Step 4: Run
```

**Headless smoke mode**:

The example also accepts `--smoke` as a single argv to run a headless
sanity check that loads the graph, queries parameters, and replays the
graph 10 times without opening a GLFW window. CTest registers this mode
as `apic_visualization_smoke` so the example runs in CI on hosts without
a display server.

```bash
./02_apic_visualization --smoke    # exits 0 with "smoke OK (10 graph launches)"
```

## How It Works

### 1. Python: Capture Multi-Kernel Graph

The Python script captures a **full simulation frame** (multiple substeps) as a single graph:

```python
# Simulation arrays
grid0 = wp.zeros(width * height, dtype=float, device=device)
grid1 = wp.zeros(width * height, dtype=float, device=device)
mouse_pos = wp.zeros(1, dtype=wp.vec2, device=device)

# Capture ALL substeps in one graph
wp.capture_begin(device=device, apic=True)

for s in range(substeps):  # e.g., 16 iterations
    if s == 0:
        # First substep: apply mouse displacement
        wp.launch(wave_displace, dim=width * height,
                  inputs=[grid0, grid1, mouse_pos, ...])

    # Every substep: integrate wave equation
    wp.launch(wave_solve, dim=width * height,
              inputs=[grid0, grid1, ...])

    # Swap buffers
    grid0, grid1 = grid1, grid0

graph = wp.capture_end(device=device)

# Save with named bindings
wp.capture_save(graph, "generated/wave_sim",
                inputs={"heights_in": grid1, "mouse_pos": mouse_pos},
                outputs={"heights_out": grid0})
```

This creates `wave_sim.wrp` containing 17 kernel launches (1 displace + 16 solve).

### 2. C++: Load Graph and Visualize

The C++ program loads the pre-captured graph and executes it each frame:

```cpp
#include "aot.h"   // Warp AOT utilities
#include "warp.h"  // Warp C API including APIC functions

// Load the pre-captured graph
APICGraph graph = wp_apic_load_graph(context, "generated/wave_sim", 0);  // 0 = CUDA

// Query parameters
int n_params = wp_apic_get_num_params(graph);
for (int i = 0; i < n_params; i++) {
    const char* name = wp_apic_get_param_name(graph, i);
    size_t size = wp_apic_get_param_size(graph, name);
    printf("  %s: %zu bytes\n", name, size);
}

// Get CUDA graph executable (built on first call)
cudaGraphExec_t exec = (cudaGraphExec_t)wp_apic_get_cuda_graph_exec(graph);

// Main loop - ONE graph launch per frame!
while (!glfwWindowShouldClose(window)) {
    // Update mouse position from GLFW input
    float mouse_grid[2] = { ... };
    cudaMemcpy(d_mouse_pos, mouse_grid, 2 * sizeof(float), cudaMemcpyHostToDevice);

    // Set input parameters
    wp_apic_set_param(graph, "heights_in", d_heights, heights_size);
    wp_apic_set_param(graph, "mouse_pos", d_mouse_pos, 2 * sizeof(float));

    // Execute the graph - runs ALL 17 kernels in one call!
    cudaGraphLaunch(exec, stream);

    // Get output
    wp_apic_get_param(graph, "heights_out", d_heights, heights_size);

    // Render with OpenGL...
}

wp_apic_destroy_graph(graph);
```

## APIC C API Reference

```cpp
// Load a graph from .wrp file
APICGraph wp_apic_load_graph(void* context, const char* path, int device_type);

// Set/get named parameters (return true on success, false on failure)
bool wp_apic_set_param(APICGraph graph, const char* name, const void* data, size_t size);
bool wp_apic_get_param(APICGraph graph, const char* name, void* data, size_t size);
void* wp_apic_get_param_ptr(APICGraph graph, const char* name);

// Get CUDA graph handles
void* wp_apic_get_cuda_graph(APICGraph graph);
void* wp_apic_get_cuda_graph_exec(APICGraph graph);  // Creates executable on first call

// Query parameters
int wp_apic_get_num_params(APICGraph graph);
const char* wp_apic_get_param_name(APICGraph graph, int index);
size_t wp_apic_get_param_size(APICGraph graph, const char* name);

// Cleanup
void wp_apic_destroy_graph(APICGraph graph);
```

## Controls

- **Left-click**: Create waves at mouse position
- **Right-drag**: Rotate camera
- **Scroll**: Zoom in/out
- **ESC**: Exit

## Performance

The key advantage of APIC is efficiency:

- **Without APIC**: 17 separate kernel launches per frame, each with CPU-GPU synchronization overhead
- **With APIC**: 1 `cudaGraphLaunch()` per frame, CUDA handles all scheduling internally

This is especially beneficial for simulations with many small kernels where launch overhead would otherwise dominate.

## Troubleshooting

**"Failed to load graph"**: Ensure the graph was captured first:
```bash
python capture_wave.py
```

**"Could not find Warp library"**: Set the library path:
```bash
export LD_LIBRARY_PATH=/path/to/warp/bin:$LD_LIBRARY_PATH  # Linux
# or
cmake -DWARP_BIN_DIR=/path/to/warp/bin ...  # CMake
```

**GLFW not found**: Install GLFW development package:
```bash
sudo apt install libglfw3-dev  # Ubuntu/Debian
sudo dnf install glfw-devel    # Fedora
brew install glfw              # macOS (note: CUDA still not supported)
```
