"""
This example illustrates using domain decomposition to solve a diffusion PDE over multiple devices
"""

from typing import Tuple

import warp as wp
from warp.sparse import bsr_axpy, bsr_mv

from warp.fem.types import Field, Sample
from warp.fem import Grid2D, LinearGeometryPartition
from warp.fem import make_polynomial_space, make_space_partition
from warp.fem import make_test, make_trial
from warp.fem import Cells, BoundarySides
from warp.fem import integrate
from warp.fem import integrand

from warp.utils import array_cast

from example_diffusion import linear_form, diffusion_form
from bsr_utils import bsr_cg
from plot_utils import plot_grid_surface

import matplotlib.pyplot as plt


@integrand
def mass_form(
    s: Sample,
    u: Field,
    v: Field,
):
    return u(s) * v(s)


@wp.kernel
def scal_kernel(a: wp.array(dtype=wp.float64), alpha: wp.float64):
    a[wp.tid()] = a[wp.tid()] * alpha


@wp.kernel
def sum_kernel(a: wp.indexedarray(dtype=wp.float64), b: wp.array(dtype=wp.float64)):
    a[wp.tid()] = a[wp.tid()] + b[wp.tid()]


def sum_vecs(vecs, indices, sum: wp.array, tmp: wp.array):
    for v, idx in zip(vecs, indices):
        wp.copy(dest=tmp, src=v)
        idx_sum = wp.indexedarray(sum, idx)
        wp.launch(kernel=sum_kernel, dim=idx.shape, device=sum.device, inputs=[idx_sum, tmp])

    return sum


class DistributedSystem:
    device = None
    scalar_type: type
    tmp_buf: wp.array

    nrow: int
    shape = Tuple[int, int]
    rank_data = None


def mv_routine(A: DistributedSystem, x: wp.array, y: wp.array, alpha=1.0, beta=0.0):
    """Distributed matrix-vector multiplication routine, for example purposes"""

    tmp = A.tmp_buf

    wp.launch(kernel=scal_kernel, dim=y.shape, device=y.device, inputs=[y, wp.float64(beta)])

    stream = wp.get_stream()

    for mat_i, x_i, y_i, idx in zip(*A.rank_data):
        # WAR copy with indexed array requiring matching shape
        tmp_i = wp.array(
            ptr=tmp.ptr, device=tmp.device, capacity=tmp.capacity, dtype=tmp.dtype, shape=idx.shape, owner=False
        )

        # Compress rhs on rank 0
        x_idx = wp.indexedarray(x, idx)
        wp.copy(dest=tmp_i, src=x_idx, count=idx.size, stream=stream)

        # Send to rank i
        wp.copy(dest=x_i, src=tmp_i, count=idx.size, stream=stream)

        with wp.ScopedDevice(x_i.device):
            wp.wait_stream(stream)
            bsr_mv(A=mat_i, x=x_i, y=y_i, alpha=alpha, beta=0.0)

        wp.wait_stream(wp.get_stream(x_i.device))

        # Back to rank 0 for sum
        wp.copy(dest=tmp_i, src=y_i, count=idx.size, stream=stream)
        y_idx = wp.indexedarray(y, idx)
        wp.launch(kernel=sum_kernel, dim=idx.shape, device=y_idx.device, inputs=[y_idx, tmp_i], stream=stream)


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})

    geo = Grid2D(res=wp.vec2i(25))
    bd_weight = 100.0

    scalar_space = make_polynomial_space(geo, degree=3)

    devices = wp.get_cuda_devices()

    rhs_vecs = []
    res_vecs = []
    matrices = []
    indices = []

    main_device = devices[0]

    # Build local system for each device
    for k, device in enumerate(devices):
        with wp.ScopedDevice(device):
            # Construct the partition corresponding to the k'th device
            geo_partition = LinearGeometryPartition(geo, k, len(devices))
            space_partition = make_space_partition(scalar_space, geo_partition)

            domain = Cells(geometry=geo_partition)

            # Right-hand-side
            test = make_test(space=scalar_space, space_partition=space_partition, domain=domain)
            rhs = integrate(linear_form, fields={"v": test})

            # Weakly-imposed boundary conditions on all sides
            boundary = BoundarySides(geometry=geo_partition)
            bd_test = make_test(space=scalar_space, space_partition=space_partition, domain=boundary)
            bd_trial = make_trial(space=scalar_space, space_partition=space_partition, domain=boundary)
            bd_matrix = integrate(mass_form, fields={"u": bd_trial, "v": bd_test})

            # Diffusion form
            trial = make_trial(space=scalar_space, space_partition=space_partition, domain=domain)
            matrix = integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": 1.0})

            bsr_axpy(y=matrix, x=bd_matrix, alpha=bd_weight)

            rhs_vecs.append(rhs)
            res_vecs.append(wp.empty_like(rhs))
            matrices.append(matrix)
            indices.append(space_partition.space_node_indices().to(main_device))


    # Global rhs as sum of all local rhs
    glob_rhs = wp.zeros(n=scalar_space.node_count(), dtype=wp.float64, device=main_device)
    tmp = wp.empty_like(glob_rhs)
    sum_vecs(rhs_vecs, indices, glob_rhs, tmp)

    # Distributed CG
    glob_res = wp.zeros_like(glob_rhs)
    A = DistributedSystem()
    A.device = device
    A.scalar_type = glob_rhs.dtype
    A.nrow = scalar_space.node_count()
    A.shape = (A.nrow, A.nrow)
    A.tmp_buf = tmp
    A.rank_data = (matrices, rhs_vecs, res_vecs, indices)

    with wp.ScopedDevice(main_device):
        bsr_cg(A, x=glob_res, b=glob_rhs, use_diag_precond=False, mv_routine=mv_routine, device=main_device)

    scalar_field = scalar_space.make_field()

    array_cast(in_array=glob_res, out_array=scalar_field.dof_values)

    plot_grid_surface(scalar_field)

    plt.show()
