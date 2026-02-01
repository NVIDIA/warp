# Warp C++ Integration: Source Inclusion with Automatic Differentiation

This example demonstrates statically including Warp-generated CUDA source (`.cu`) in a C++ program.
It showcases Warp's automatic differentiation capabilities by implementing gradient descent for linear regression
using auto-generated forward and backward kernels.

**Builds on**: Example `00_cubin_launch` showed runtime CUBIN loading. This example shows compile-time source inclusion,
enabling use of standard `<<<>>>` launch syntax and showcasing Warp's automatic differentiation.

## Overview

Workflow:

1. **Python**: Define a differentiable kernel, compile to `.cu` source
2. **C++**: Include generated source, launch kernels with standard `<<<>>>` syntax
3. **Training**: Forward pass computes loss, backward pass computes gradients

**Problem**: Fit linear model `y = a*x + b` to noisy data using gradient descent.

**Kernels**:

- `compute_loss` (forward) - Computes MSE loss
- `compute_loss` (backward) - Computes ∂loss/∂params (auto-generated)
- `update_params` - Updates parameters on GPU

## Files

- `compile_kernel.py` - Python script that compiles Warp kernels to `.cu` source
- `main.cu` - C++ program with gradient descent training loop
- `Makefile` / `CMakeLists.txt` - Build systems (Make for Unix, CMake for cross-platform)
- `generated/` - Directory containing generated files (auto-created):
  - `*.cu` - Generated CUDA source code (included in C++)
  - `*.ptx` / `*.cubin` - Compiled binaries (not used in this example)

## Quick Start

### Prerequisites

This example runs from within the Warp repository.

**Requirements:**
- **Python 3.9+**
- **CUDA Toolkit 12.8+** (includes `nvcc` compiler; 12.8+ required for `sm_120` support)
- **NVIDIA GPU** (supports multiple architectures via multi-arch compilation)
- **Build System**: GNU Make (Unix/Linux) or CMake 3.20+ (cross-platform)
- **Note**: macOS is **not supported** (CUDA not available on macOS)

**Setup:**

Clone and build Warp:
```bash
git clone https://github.com/NVIDIA/warp.git
cd warp

# Build and install (choose one):
uv run build_lib.py  # Option A (recommended - handles everything automatically)
# or: python build_lib.py && pip install -e .  # Option B (use a virtual environment)

cd warp/examples/cpp/01_source_include
```

### Build and Run

**Using Make (Unix/Linux)**:

```bash
make                 # Build everything (auto-compiles kernel if needed)
./01_source_include  # Run
```

**Make Targets**:

- `make` (default) - Build everything, auto-compile kernel if needed
- `make cpp` - Build only C++ program (fast iteration, assumes kernel exists)
- `make clean` - Remove executable only
- `make distclean` - Remove executable and `generated/` directory
- `make help` - Show all available targets

**Using CMake (cross-platform)**:

```bash
python compile_kernel.py                  # Step 1: Compile kernels to .cu
cmake -B build -DCMAKE_BUILD_TYPE=Release # Step 2: Configure
cmake --build build --config Release      # Step 3: Build
./build/01_source_include                 # Step 4: Run (Unix) or build\Release\01_source_include.exe (Windows)
```

**Output**:

```text
Iter 0: MSE=321.055, params=[2.56704, 0.813697]
Iter 10: MSE=0.0161513, params=[3.53469, 0.971546]
...
Iter 90: MSE=0.0092167, params=[3.52368, 1.04446]

Learned: y = 3.52269*x + 1.05101
True:    y = 3.5*x + 1.2
Error:   Δa=0.022687, Δb=0.148992
✓ Training converged successfully!
```

## Implementation

### Python Kernel Generation

```python
@wp.kernel
def compute_loss(
    params: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
    y_true: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    a, b = params[0], params[1]
    y_pred = a * x[tid] + b
    error = y_pred - y_true[tid]
    wp.atomic_add(loss, 0, error * error)

@wp.kernel
def update_params(learning_rate: wp.float32, grads: wp.array(dtype=wp.float32), params: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    params[tid] -= learning_rate * grads[tid]

# Generate .cu source
wp.compile_aot_module("__main__", module_dir="generated/", strip_hash=True)
```

Generates `generated/wp___main__.cu` with:

- `compute_loss_cuda_kernel_forward` / `_backward`
- `update_params_cuda_kernel_forward`

### C++ Integration

```cpp
#include "aot.h"  // Warp types and utilities
#include "generated/wp___main__.cu"

// Setup Warp structures
wp::launch_bounds_t loss_dim = {{N_SAMPLES, 0, 0, 0}, 1, size_t(N_SAMPLES)};
wp::launch_bounds_t update_dim = {{2, 0, 0, 0}, 1, 2};
wp::array_t<wp::float32> arr_params(d_params, 2);
wp::array_t<wp::float32> arr_x(d_x, N_SAMPLES);
// ... other arrays ...

for (int iter = 0; iter < N_ITERATIONS; iter++) {
    // Zero loss and gradients
    cudaMemset(d_loss, 0, sizeof(float));
    cudaMemset(d_adj_params, 0, 2 * sizeof(float));
    
    // Seed output gradient (∂loss/∂loss = 1)
    float adj_loss_val = 1.0f;
    cudaMemcpy(d_adj_loss, &adj_loss_val, sizeof(float), cudaMemcpyHostToDevice);
    
    // Forward pass
    compute_loss_cuda_kernel_forward<<<grid_dim, block_size>>>(
        loss_dim, arr_params, arr_x, arr_y_true, arr_loss
    );
    
    // Backward pass
    compute_loss_cuda_kernel_backward<<<grid_dim, block_size>>>(
        loss_dim, arr_params, arr_x, arr_y_true, arr_loss,
        adj_params, adj_x, adj_y_true, adj_loss
    );
    
    // Update parameters on GPU
    update_params_cuda_kernel_forward<<<1, 2>>>(
        update_dim, LEARNING_RATE / N_SAMPLES, adj_params, arr_params
    );
}
```

## Key Concepts

**Adjoint Arrays**: Each forward array has a corresponding adjoint (gradient) array in the backward pass:

- `params` → `adj_params` (∂loss/∂params)
- `loss` → `adj_loss` (seed gradient, set to 1.0)

**Gradient Seeding**: Initialize `adj_loss = 1.0f` to compute ∂loss/∂params.

**Kernel Naming**: `strip_hash=True` generates predictable names without hash suffixes:

- `compute_loss` → `compute_loss_cuda_kernel_forward` / `_backward`
- `update_params` → `update_params_cuda_kernel_forward`

**Source Generation**: `wp.compile_aot_module()` generates:

- `.cu` source file (used for C++ inclusion)
- `.ptx` or `.cubin` file (not used in this example)

## See Also

- [Warp Autodiff Guide](https://nvidia.github.io/warp/autodiff.html)
- Example `00_cubin_launch` - Runtime CUBIN loading approach
