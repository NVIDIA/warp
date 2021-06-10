# NVIDIA OgLang

OgLang is a Python framework for writing high-performance simulation and graphics code. Kernels are defined in Python syntax and JIT converted to C++/CUDA and compiled at runtime.

##  Installation

### Local Python

To install in your local Python environment use:

    pip install -e .


## Requirements

For developers writing their own kernels the following are required:

    * Microsoft Visual Studio 2015 upwards (Windows)
    * GCC 4.0 upwards (Linux)
    * CUDA 11.0 upwards

To build CUDA kernels ensure that either `CUDA_HOME` or `CUDA_PATH` should be set as environment variables pointing to the CUDA installation directory.

To run built-in tests you should install the USD Core library to your Python environment using `pip install usd-core`.

## Example Usage

To define a computational kernel use the following syntax with the `@og.kernel` decorator. Note that all input arguments must be typed, and that the function can not access any global state.

```python
@og.kernel
def simple_kernel(a: og.array(dtype=vec3),
                  b: og.array(dtype=vec3),
                  c: og.array(dtype=float)):

    # get thread index
    tid = og.tid()

    # load two vec3s
    x = og.load(a, tid)
    y = og.load(b, tid)

    # compute the dot product between vectors
    r = og.dot(x, y)

    # write result back to memory
    og.store(c, tid, r)
```

Arrays can be allocated similar to PyTorch:

```python
    # allocate an uninitizalized array of vec3s
    v = og.empty(dim=n, dtype=og.vec3, device="cuda")

    # allocate a zero-initialized array of quaternions    
    q = og.zeros(dim=n, dtype=og.quat, device="cuda")

    # allocate and initialize an array from a numpy array
    # will be automatically transferred to the specified device
    v = og.from_numpy(array, dtype=og.vec3, device="cuda")
```

To launch a kernel use the following syntax:

```python
    og.launch(kernel=simple_kernel, # kernel to launch
              dim=1024,             # number of threads
              inputs=[a, b, c],     # parameters
              device="cuda")        # execution device

```

Note that all input and output buffers must exist on the same device as the one specified for execution.

Often we need to read data back to main (CPU) memory which can be done conveniently as follows:

```python
    # automatically bring data from device back to host
    view = device_array.to("cpu").numpy()
```

This pattern will allocate a temporary CPU buffer, perform a copy from device->host memory, and return a numpy view onto it. To avoid allocating temporary buffers this process can be managed explicitly:

```python
    # manually bring data back to host
    og.copy(dest=host_array, src=device_array)
    og.synchronize()

    view = host_array.numpy()
```

All copy operations are performed asynchronously and must be synchronized explicitly to ensure data is visible. For best performance multiple copies should be queued together:

```python
    # launch multiple copy operations asynchronously
    og.copy(dest=host_array_0, src=device_array_0)
    og.copy(dest=host_array_1, src=device_array_1)
    og.copy(dest=host_array_2, src=device_array_2)
    og.synchronize()
```

## Memory Model

Memory allocations are exposed via. the `array` type. Arrays wrap an underlying piece of memory that may live in either host (CPU), or device (GPU) memory. Arrays are strongly typed and store a linear sequence of built-in structures (`vec3`, `matrix33`, etc).

Arrays may be constructed from Python lists or numpy arrays, by default data will be copied to new memory for the device specified. However, it is possible for arrays to alias user memory using the `copy=False` parameter to the array constructor.

## Compilation Model

OgLang uses a Python->C++/CUDA compilation model that generates kernel code from Python function definitions. All kernels belonging to a Python module are runtime compiled into dynamic libraries (.dll/.so) and cached between application restarts.

Note that compilation is triggered on the first kernel launch for that module. Any kernels registered in the module with `@og.kernel` will be included in the shared library.

## Language Details

To support GPU computation and differentiability there are some differences from the CPython runtime.

### Built-in Types

OgLang supports a number of built-in math types similar to high-level shading languages, for example `vec2, vec3, vec4, mat22, mat33, mat44, quat, array`. All built-in types have value semantics so that expressions such as `a = b` generate a copy of the variable b rather than a reference.

### Strong Typing

Unlike Python, in oglang all variables must be typed. Types are inferred from source expressions and function signatures using the Python typing extensions. All kernel parameters must be annotated with the appropriate type, for example:

```python
@og.kernel
def simple_kernel(a: og.array(dtype=vec3),
                  b: og.array(dtype=vec3),
                  c: float):
...
```

Tuple initialization is not supported, instead variables should be explicitly typed:

```python
# invalid
a = (1.0, 2.0, 3.0)        

# valid
a = vec3(1.0, 2.0, 3.0) 
```
### Built-in Functions

#### Core functions

```python
tid, print, load, store, atomic_add, atomic_sub
```

#### Math functions

```python
add, sub, mod, mul, div, neg, min, max, clamp, step, sign, abs, sin, cos, acos, sqrt, select
```

#### Vector functions

```python
dot, cross, skew, length, normalize, rotate, rotate_inv, determinant, transpose, vec3, quat, quat_identity, quat_from_axis_angle, mat22, mat33, mat44, transform_point, transform_vector
```

#### Spatial Functions

Spatial vectors are often used in the implementation of articulated body algorithms. The following methods can be used to operate on `spatial_vector` and `spatial_matrix` types.

```python
spatial_vector, spatial_transform, spatial_transform_identity, inverse, 
spatial_transform_get_translation, spatial_transform_get_rotation, spatial_transform_multiply, spatial_adjoint, spatial_dot, spatial_cross, spatial_cross_dual, spatial_transform_point, spatial_transform_vector, spatial_top, spatial_bottom, spatial_jacobian, spatial_mass
```

#### Dense-Matrix Functions

```python
dense_gemm, dense_gemm_batched, dense_chol, dense_chol_batched, dense_subs, dense_solve, dense_solve_batched
```

#### Mesh Functions

```python
mesh_query_point, mesh_query_ray, mesh_eval_position, mesh_eval_velocity
```

### Unsupported Features

To achieve high performance some dynamic language features are not supported:

* Array slicing notation
* Lambda expressions
* Exceptions
* Class definitions
* Runtime evaluation of expressions, e.g.: eval()
* Recursion
* Dynamic allocation, lists, sets, dictionaries

## Source

https://gitlab-master.nvidia.com/mmacklin/oglang


