# NVIDIA OgLang

OgLang is a Python DSL and framework for writing high-performance code. Kernels are defined in Python syntax and JIT converted to C++/CUDA and compiled at runtime.

A simple example is the following:

```python
og.kernel
def simple_kernel(a: og.array(vec3),
                  b: og.array(vec3),
                  c: og.array(float):

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

##  Installation

### Local Python


To install in your local Python environment use:

    pip install -e .


### Omniverse

To install in the Omniverse Python environment, use:

```python

    import omni.kit.pipapi as pip

    pip.install(package="F:\gitlab\oglang", module="oglang")
```

Where the package path is where the oglang package lives on your local machine. See the following FAQ for more details on custom Python packages in Kit:

http://omnidocs-internal.nvidia.com/py/docs/guide/faq.html#can-i-use-packages-from-pip


## Requirements

For developers writing their own kernels the following are required:

    * Microsoft Visual Studio 2015 upwards (Windows)
    * GCC 4.0 upwards (Linux)
    * CUDA 11.0 upwards

To build CUDA kernels either `CUDA_HOME` or `CUDA_PATH` should be set as environment variables pointing to the CUDA installation directory.

To run built-in tests you should install the USD Core library to your Python environment using `pip install usd-core`.

## Source

https://gitlab-master.nvidia.com/mmacklin/oglang

## Contact

mmacklin@nvidia.com

