# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental task-parallelism support for Warp via CUDASTF.

This module exposes a small set of helpers that let Warp kernels participate
in a CUDASTF (``cuda.stf``) task graph: tasks declare which logical data they
read and write, CUDASTF chooses streams and inserts the synchronization
needed to honor those dependencies, and the Warp helpers wrap CUDASTF
streams as :class:`warp.Stream` objects and CUDASTF array views as zero-copy
:class:`warp.array` views.

.. caution::
    This module is experimental and less stable than the core Warp API. The
    interface may change as new functionality is added and to accommodate
    changes in upcoming ``cuda.stf`` library versions.

Usage:
    This module must be explicitly imported::

        import warp.stf_experimental

See Also:
    :doc:`../deep_dive/stf_task_graphs` in the Deep Dive section for detailed
    examples and usage patterns.
"""

# isort: skip_file

from warp._src.stf_experimental.context import context as context
from warp._src.stf_experimental.context import is_available as is_available
from warp._src.stf_experimental.context import task as task
from warp._src.stf_experimental.context import task_graph as task_graph
from warp._src.stf_experimental.context import warmup as warmup
from warp._src.stf_experimental.graph_debug import dump_dot as dump_dot
