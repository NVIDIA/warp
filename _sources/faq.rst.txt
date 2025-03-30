FAQ
===

How does Warp relate to other Python projects for GPU programming, e.g.: Numba, Taichi, cuPy, PyTorch, etc.?
------------------------------------------------------------------------------------------------------------

Warp is inspired by many of these projects, and is closely related to
Numba and Taichi which both expose kernel programming to Python. These
frameworks map to traditional GPU programming models, so many of the
high-level concepts are similar, however there are some functionality
and implementation differences.

Compared to Numba, Warp supports a smaller subset of Python, but
offering auto-differentiation of kernel programs, which is useful for
machine learning. Unlike Numba and Taichi, Warp uses C++/CUDA as an
intermediate representation, which makes it convenient to implement and
expose low-level routines and leverage existing C++ libraries in kernels.
In addition, Warp has built in data structures to support geometry processing (meshes, sparse volumes,
point clouds, USD data) as first-class citizens that are not exposed in
other runtimes.

Warp does not offer a full tensor-based programming model like PyTorch
and JAX, but is designed to work well with these frameworks through data
sharing mechanisms like ``__cuda_array_interface__``. For computations
that map well to tensors (e.g.: neural-network inference) it makes sense
to use these existing tools. For problems with a lot of sparsity,
conditional logic, heterogeneous workloads (like the ones we often find in
simulation and graphics), etc., then kernel-based programming models like
the one in Warp are often more convenient since users have control over
individual threads.

What are some examples of projects that use Warp?
-------------------------------------------------

* `MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`__: A GPU-optimized version of the MuJoCo physics simulator,
  designed for NVIDIA hardware. Maintained by Google DeepMind and NVIDIA.
* `Rewarped <https://github.com/rewarped/rewarped>`__: A platform for reinforcement learning in parallel differentiable multi-physics simulation.
* `XLB (Accelerated Lattice Boltzmann) <https://github.com/Autodesk/XLB>`__: A lattice Boltzmann solver with a backend option using Warp.
  Maintained by Autodesk.
* `warp-mpm <https://github.com/zeshunzong/warp-mpm>`__: An MPM simulator used in projects like
  `Neural Stress Fields for Reduced-order Elastoplasticity and Fracture <https://zeshunzong.github.io/reduced-order-mpm/>`__
  and `PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics <https://xpandora.github.io/PhysGaussian/>`__.

Does Warp support all of the Python language?
---------------------------------------------

No, Warp supports a subset of Python that maps well to the GPU. Our goal
is to not have any performance cliffs so that users can expect
consistently good behavior from kernels that is close to native code.
Examples of unsupported concepts that don't map well to the GPU are
dynamic types, list comprehensions, exceptions, garbage collection, etc.

When should I call ``wp.synchronize()``?
----------------------------------------

One of the common sources of confusion for new users is when calls to
:func:`wp.synchronize() <warp.synchronize>` are necessary. The answer is "almost never"!
Synchronization is quite expensive and should generally be avoided
unless necessary. Warp naturally takes care of synchronization between
operations (e.g.: kernel launches, device memory copies).

For example, the following requires no manual synchronization, as the
conversion to NumPy will automatically synchronize:

.. code:: python

    # run some kernels
    wp.launch(kernel_1, dim, [array_x, array_y], device="cuda")
    wp.launch(kernel_2, dim, [array_y, array_z], device="cuda")

    # bring data back to host (and implicitly synchronize)
    x = array_z.numpy()

The *only* case where manual synchronization is needed is when copies
are being performed back to CPU asynchronously, e.g.:

.. code:: python

    # copy data back to cpu from gpu, all copies will happen asynchronously to Python
    wp.copy(cpu_array_1, gpu_array_1)
    wp.copy(cpu_array_2, gpu_array_2)
    wp.copy(cpu_array_3, gpu_array_3)

    # ensure that the copies have finished
    wp.synchronize()

    # return a numpy wrapper around the cpu arrays, note there is no implicit synchronization here
    a1 = cpu_array_1.numpy()
    a2 = cpu_array_2.numpy()
    a3 = cpu_array_3.numpy()

For more information about asynchronous operations, please refer to the :doc:`concurrency documentation <modules/concurrency>`
and :ref:`synchronization guidance <synchronization_guidance>`.

What happens when you differentiate a function like ``wp.abs(x)``?
------------------------------------------------------------------

Non-smooth functions such as :math:`y=|x|` do not have a single unique
gradient at :math:`x=0`, rather they have what is known as a
*subgradient*, which is formally the convex hull of directional
derivatives at that point. The way that Warp (and most
auto-differentiation frameworks) handles these points is to pick an
arbitrary gradient from this set, e.g.: for ``wp.abs()``, it will
arbitrarily choose the gradient to be 1.0 at the origin. You can find
the implementation for these functions in
`warp/native/builtin.h <https://github.com/NVIDIA/warp/blob/main/warp/native/builtin.h>`_.

Most optimizers (particularly ones that exploit stochasticity), are not
sensitive to the choice of which gradient to use from the subgradient,
although there are exceptions.

Does Warp support multi-GPU programming?
----------------------------------------

Yes! Since version ``0.4.0`` we support allocating, launching, and
copying between multiple GPUs in a single process. We follow the naming
conventions of PyTorch and use aliases such as ``cuda:0``, ``cuda:1``,
``cpu`` to identify individual devices. For more information, see the
:doc:`modules/devices` documentation.

Warp applications can also be parallelized over multiple GPUs using
`mpi4py <https://github.com/mpi4py/mpi4py>`__. Warp arrays on the GPU may be
passed directly to MPI calls if mpi4py is built against a CUDA-aware MPI installation.

Should I switch to Warp over IsaacGym/PhysX?
----------------------------------------------

Warp is not a replacement for IsaacGym, IsaacSim, or PhysX—while Warp
does offer some physical simulation capabilities, this is primarily aimed
at developers who need differentiable physics, rather than a fully
featured physics engine. Warp is also integrated with IsaacGym and is
great for performing auxiliary tasks such as reward and observation
computations for reinforcement learning.

Why aren't assignments to Warp arrays supported outside of kernels?
------------------------------------------------------------------------

For best performance, reading and writing data that is living on the GPU can 
only be performed inside Warp CUDA kernels. Otherwise individual element accesses
such as ``array[i] = 1.0`` in Python scope would require prohibitively slow device
synchronization and copies.

We recommend to either initialize Warp arrays from other native arrays
(Python lists, NumPy arrays, etc.) or by launching a kernel to set its values.

For the common use case of filling an array with a given value, we
also support the following forms:

- ``wp.full(8, 1.23, dtype=float)``: initializes a new array of 8 float values set
  to ``1.23``.
- ``arr.fill_(1.23)``: sets the content of an existing float array to ``1.23``.
- ``arr[:4].fill(1.23)``: sets the four first values of an existing float array to ``1.23``.

How can I contact the Warp team directly?
-----------------------------------------

For bug reports, feature requests, and technical questions, we recommend using `GitHub Issues <https://github.com/NVIDIA/warp/issues>`_.

The Warp team also monitors the **#warp** forum on the public `Omniverse Discord <https://discord.com/invite/nvidiaomniverse>`_ server.

For inquiries not suited for GitHub Issues or Discord, please email warp-python@nvidia.com.
