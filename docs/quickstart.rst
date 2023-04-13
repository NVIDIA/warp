Quickstart
==========

Basic example
-------------

An example first program that computes the lengths of random 3D vectors is given below::

    import warp as wp
    import numpy as np

    wp.init()

    num_points = 1024

    @wp.kernel
    def length(points: wp.array(dtype=wp.vec3),
            lengths: wp.array(dtype=float)):

        # thread index
        tid = wp.tid()
        
        # compute distance of each point from origin
        lengths[tid] = wp.length(points[tid])


    # allocate an array of 3d points
    points = wp.array(np.random.rand(num_points, 3), dtype=wp.vec3)
    lengths = wp.zeros(num_points, dtype=float)

    # launch kernel
    wp.launch(kernel=length,
            dim=len(points),
            inputs=[points, lengths])

    print(lengths)

Additional examples
-------------------
The `examples <https://github.com/NVIDIA/warp/tree/main/examples>`__ directory in
the Github repository contains a number of scripts that show how to
implement different simulation methods using the Warp API. Most examples
will generate USD files containing time-sampled animations in the
``examples/outputs`` directory. Before running examples users should
ensure that the ``usd-core`` package is installed using:

::

   pip install usd-core

USD files can be viewed or rendered inside NVIDIA
`Omniverse <https://developer.nvidia.com/omniverse>`__,
Pixar's UsdView, and Blender. Note that Preview in macOS is not
recommended as it has limited support for time-sampled animations.

Built-in unit tests can be run from the command-line as follows:

::

   python -m warp.tests
