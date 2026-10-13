# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An example implementation of a distributed FDTD solver using NCCL.

This example is the multi-GPU counterpart of ``warp/examples/core/example_fdtd_3d.py``.
It solves the same problem: a 3-D Luneburg lens collimates a point source. Refer to the
docstring of that example for the equations, the Yee grid, the time staggering, the
absorbing layer, the lens profile, and the source. The two examples use the same constants,
except that the default ``N_GRID`` is 384 instead of 128. The kernels apply the same updates
as the serial kernels, but they index the fields with local indices that are shifted by the
halo slots. This docstring describes only the changes for multiple GPUs.

The x-y plane is decomposed across a ``num_ranks_x x num_ranks_y`` grid of MPI ranks.
Each rank owns a pencil of ``nx_local x ny_local`` nodes that spans the full z extent.
Each field array has one halo slot on each x and y side, so its shape is
``(nx_local + 2, ny_local + 2, N_GRID)``. The kernels take the ``x_offset`` and
``y_offset`` of the pencil and convert local thread indices to global grid indices. The
loss profiles, the lens, the source, and the wall conditions therefore use the same
global-index logic as the serial kernels.

Each timestep does two depth-1 halo exchanges with NCCL point-to-point calls:

1. Before ``update_h``, each rank receives the first ``E`` plane of its upper x and y
   neighbors, because the ``H`` update takes forward differences of ``E``.
2. Before ``update_e``, each rank receives the last ``H`` plane of its lower x and y
   neighbors, because the ``E`` update takes backward differences of ``H``.

The x faces are contiguous and go directly from the field arrays. The y faces are
strided, so ``pack_y_face`` and ``unpack_y_face`` copy them through contiguous staging
buffers. The halo exchanges and the kernels run on one stream, so one CUDA graph
captures a full timestep.

The energy diagnostic sums over each pencil and then reduces the sums across ranks with
an NCCL all-reduce. In interactive mode, rank 0 renders the ``E_x`` center plane, which
it gathers from the ranks that hold it. In headless mode, the example reports the global
throughput, using the elapsed time of the slowest rank.

This example requires nccl4py, mpi4py, and an MPI implementation. Install the nccl4py extra
that matches your CUDA major version: python -m pip install "nccl4py[cu12]" or
python -m pip install "nccl4py[cu13]".

Requires Matplotlib for interactive visualization (use --headless to run without it).

Usage:
    mpirun -n 4 python example_fdtd_3d_nccl.py --num-ranks-x 2 --num-ranks-y 2

Both ``--num-ranks-x`` and ``--num-ranks-y`` are required, and their product must equal
the number of MPI ranks.
"""

import math
import traceback

import nccl.core as nccl
import numpy as np
from mpi4py import MPI

import warp as wp

wp.config.log_level = wp.LOG_WARNING
wp.config.enable_backward = False

# physical inputs, in normalized units (c = mu_0 = eps_0 = lambda_0 = f_0 = T_0 = 1)
HALF_WIDTH = 4.25  # half-extent of the cubic domain
R_LENS = 3.0  # lens radius in vacuum wavelengths
L_ABSORBER = 2.0 / 3.0  # depth of the attenuation layer lining each wall
E0 = 1.0  # source amplitude, tuned at the reference grid spacing DX_REF
DX_REF = 1.0 / 30.0  # grid spacing at which E0 and PLOT_E_SCALE were tuned
SOURCE_RAMP = 3.0  # periods over which the source envelope rises from rest
SIM_DURATION = 36.0  # default headless run length, in source periods
DIAG_INTERVAL = 100  # timesteps between electromagnetic energy diagnostics
PLOT_E_SCALE = 2.5e-4  # colorbar half-range for the E_x frames; SOURCE_SCALE keeps it valid at every resolution
GAMMA_MAX = -2.0 * math.log(1.0e-3) / L_ABSORBER

# resolution and derived quantities
N_GRID = 384  # stored nodes per axis across the fixed domain
DX = 2.0 * HALF_WIDTH / (N_GRID - 1)
DT = 0.95 * DX / math.sqrt(3.0)  # 3-D CFL limit with a 0.95 safety factor
SOURCE_SCALE = (DX_REF / DX) ** 2  # holds the dipole moment, fixed across resolutions
SOURCE_IJ = (N_GRID - 1) // 2
SOURCE_K = round((HALF_WIDTH - R_LENS) / DX)


def calc_default_device(mpi_comm: "MPI.Comm") -> wp.Device:
    """Return the Warp device this rank should use.

    Splits the communicator into a node-local group. The mapping depends on the devices
    that this rank can see, which ``CUDA_VISIBLE_DEVICES`` controls for each rank:

    - If the rank sees more than one device, it takes the device at its node-local rank.
    - If the rank sees one device, it takes that device. For example, a launcher can set
      a different ``CUDA_VISIBLE_DEVICES`` for each rank, so each rank gets its own GPU.

    Ranks share a GPU only when they all see the same single device. NCCL allows this
    only in version 2.30 or later with ``NCCL_MULTI_RANK_GPU_ENABLE=1``, and the feature
    is experimental. Otherwise, the NCCL communicator initialization fails.

    Args:
        mpi_comm: The global MPI communicator.

    Returns:
        The Warp device assigned to this rank.

    Raises:
        RuntimeError: If no CUDA device is visible, or if more than one device is
            visible but the node has fewer devices than ranks.
    """
    local_mpi_comm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)

    local_size = local_mpi_comm.Get_size()
    local_rank = local_mpi_comm.Get_rank()

    num_cuda_devices = wp.get_cuda_device_count()

    if num_cuda_devices == 0:
        raise RuntimeError("This example requires at least one CUDA device, but Warp sees none.")

    if 1 < num_cuda_devices < local_size:
        raise RuntimeError(
            f"Number of visible devices ({num_cuda_devices}) is less than number of ranks on the node ({local_size})"
        )

    if num_cuda_devices > 1:
        return wp.get_cuda_device(local_rank)
    else:
        return wp.get_cuda_device()


@wp.func
def loss_coeffs(gamma: float) -> tuple[float, float]:
    """Return the matched-loss coefficients for a field component at its position on the Yee grid."""
    gamma_dt_half = 0.5 * DT * gamma
    return (1.0 - gamma_dt_half) / (1.0 + gamma_dt_half), 1.0 / (1.0 + gamma_dt_half)


@wp.func
def inv_eps_r(x: float, y: float, z: float) -> float:
    """Return the inverse relative permittivity of the Luneburg lens at a point given by (x, y, z)."""
    r_sq = x * x + y * y + z * z

    eps_r = float(1.0)
    if r_sq < R_LENS * R_LENS:
        eps_r = 2.0 - r_sq / (R_LENS * R_LENS)

    return 1.0 / eps_r


@wp.func
def source_amplitude(timestep_index: int) -> float:
    """Return the scaled point-source term ``DX * J_x`` sampled at ``t^{n+1/2}``.

    ``SOURCE_SCALE`` keeps the dipole moment ``J_x * DX^3``, and with it the radiated
    field and ``PLOT_E_SCALE``, independent of resolution.
    """
    t = (float(timestep_index) + 0.5) * DT
    return -E0 * SOURCE_SCALE * wp.sin(2.0 * wp.pi * t) * (1.0 - wp.exp(-((t / SOURCE_RAMP) ** 2.0)))


@wp.kernel
def update_h(
    e_x: wp.array3d[float],
    e_y: wp.array3d[float],
    e_z: wp.array3d[float],
    gamma_integer: wp.array[float],
    gamma_half_offset: wp.array[float],
    x_offset: int,
    y_offset: int,
    h_x: wp.array3d[float],
    h_y: wp.array3d[float],
    h_z: wp.array3d[float],
):
    """Advance ``H`` from ``t = (n - 1/2) dt`` to ``t = (n + 1/2) dt`` using ``E^n``.

    The representative component update (the others follow by cyclic permutation) is::

        H_x^{n+1/2}|_{i, j+1/2, k+1/2} = a * H_x^{n-1/2}
            - (b * dt / dx) * [(E_z|_{j+1} - E_z|_j) - (E_y|_{k+1} - E_y|_k)]

    with the matched-loss coefficients of ``loss_coeffs`` sampled from the 1-D profiles
    ``gamma_integer`` and ``gamma_half_offset`` at each component's staggered coordinates
    (the grid is cubic, so all three axes index the same two profiles).
    """
    i_local, j_local, k = wp.tid()

    i = x_offset + i_local
    j = y_offset + j_local
    field_i = i_local + 1
    field_j = j_local + 1

    dt_dx = DT / DX
    # last index whose half-offset indices are still inside the walls
    last_staggered = N_GRID - 2

    # H_x at (i, j+1/2, k+1/2)
    if j <= last_staggered and k <= last_staggered:
        a_x, b_x = loss_coeffs(gamma_integer[i] + gamma_half_offset[j] + gamma_half_offset[k])
        curl_x = (e_z[field_i, field_j + 1, k] - e_z[field_i, field_j, k]) - (
            e_y[field_i, field_j, k + 1] - e_y[field_i, field_j, k]
        )
        h_x[field_i, field_j, k] = a_x * h_x[field_i, field_j, k] - b_x * dt_dx * curl_x

    # H_y at (i+1/2, j, k+1/2)
    if i <= last_staggered and k <= last_staggered:
        a_y, b_y = loss_coeffs(gamma_half_offset[i] + gamma_integer[j] + gamma_half_offset[k])
        curl_y = (e_x[field_i, field_j, k + 1] - e_x[field_i, field_j, k]) - (
            e_z[field_i + 1, field_j, k] - e_z[field_i, field_j, k]
        )
        h_y[field_i, field_j, k] = a_y * h_y[field_i, field_j, k] - b_y * dt_dx * curl_y

    # H_z at (i+1/2, j+1/2, k)
    if i <= last_staggered and j <= last_staggered:
        a_z, b_z = loss_coeffs(gamma_half_offset[i] + gamma_half_offset[j] + gamma_integer[k])
        curl_z = (e_y[field_i + 1, field_j, k] - e_y[field_i, field_j, k]) - (
            e_x[field_i, field_j + 1, k] - e_x[field_i, field_j, k]
        )
        h_z[field_i, field_j, k] = a_z * h_z[field_i, field_j, k] - b_z * dt_dx * curl_z


@wp.kernel
def update_e(
    h_x: wp.array3d[float],
    h_y: wp.array3d[float],
    h_z: wp.array3d[float],
    gamma_integer: wp.array[float],
    gamma_half_offset: wp.array[float],
    x_offset: int,
    y_offset: int,
    timestep_counter: wp.array[wp.int32],
    e_x: wp.array3d[float],
    e_y: wp.array3d[float],
    e_z: wp.array3d[float],
):
    """Advance ``E`` from ``t = n dt`` to ``t = (n + 1) dt`` using ``H^{n+1/2}``.

    The representative component update (the others follow by cyclic permutation) is::

        E_x^{n+1}|_{i+1/2, j, k} = a * E_x^n + (b * dt / (eps_r * dx))
            * [(H_z|_{j+1/2} - H_z|_{j-1/2}) - (H_y|_{k+1/2} - H_y|_{k-1/2}) - dx * J_x]

    where ``eps_r`` is the Luneburg profile of ``inv_eps_r``, evaluated analytically. A
    point source drives ``J_x`` at the single node ``(SOURCE_IJ, SOURCE_IJ, SOURCE_K)``,
    with ``source_amplitude`` supplying ``dx * J_x``.

    """
    i_local, j_local, k = wp.tid()

    i = x_offset + i_local
    j = y_offset + j_local
    field_i = i_local + 1
    field_j = j_local + 1

    timestep_index = timestep_counter[0]

    dt_dx = DT / DX

    last_staggered = N_GRID - 2
    last_interior = N_GRID - 2

    x_node = float(i) * DX - HALF_WIDTH
    y_node = float(j) * DX - HALF_WIDTH
    z_node = float(k) * DX - HALF_WIDTH
    half_dx = 0.5 * DX

    # E_x at (i+1/2, j, k), carrying the point source
    if i <= last_staggered and 1 <= j <= last_interior and 1 <= k <= last_interior:
        a_x, b_x = loss_coeffs(gamma_half_offset[i] + gamma_integer[j] + gamma_integer[k])
        curl_x = (h_z[field_i, field_j, k] - h_z[field_i, field_j - 1, k]) - (
            h_y[field_i, field_j, k] - h_y[field_i, field_j, k - 1]
        )
        if i == SOURCE_IJ and j == SOURCE_IJ and k == SOURCE_K:
            curl_x -= source_amplitude(timestep_index)
        inv_eps_x = inv_eps_r(x_node + half_dx, y_node, z_node)
        new_e_x = a_x * e_x[field_i, field_j, k] + b_x * dt_dx * inv_eps_x * curl_x
        e_x[field_i, field_j, k] = new_e_x

    # E_y at (i, j+1/2, k)
    if 1 <= i <= last_interior and j <= last_staggered and 1 <= k <= last_interior:
        a_y, b_y = loss_coeffs(gamma_integer[i] + gamma_half_offset[j] + gamma_integer[k])
        curl_y = (h_x[field_i, field_j, k] - h_x[field_i, field_j, k - 1]) - (
            h_z[field_i, field_j, k] - h_z[field_i - 1, field_j, k]
        )
        inv_eps_y = inv_eps_r(x_node, y_node + half_dx, z_node)
        new_e_y = a_y * e_y[field_i, field_j, k] + b_y * dt_dx * inv_eps_y * curl_y
        e_y[field_i, field_j, k] = new_e_y

    # E_z at (i, j, k+1/2)
    if 1 <= i <= last_interior and 1 <= j <= last_interior and k <= last_staggered:
        a_z, b_z = loss_coeffs(gamma_integer[i] + gamma_integer[j] + gamma_half_offset[k])
        curl_z = (h_y[field_i, field_j, k] - h_y[field_i - 1, field_j, k]) - (
            h_x[field_i, field_j, k] - h_x[field_i, field_j - 1, k]
        )
        inv_eps_z = inv_eps_r(x_node, y_node, z_node + half_dx)
        new_e_z = a_z * e_z[field_i, field_j, k] + b_z * dt_dx * inv_eps_z * curl_z
        e_z[field_i, field_j, k] = new_e_z


@wp.kernel
def compute_em_energy(
    e_x: wp.array3d[float],
    e_y: wp.array3d[float],
    e_z: wp.array3d[float],
    h_x: wp.array3d[float],
    h_y: wp.array3d[float],
    h_z: wp.array3d[float],
    x_offset: int,
    y_offset: int,
    electric_energy_sum: wp.array[float],
    magnetic_energy_sum: wp.array[float],
):
    """Accumulate ``eps_r * E^2`` and ``H^2`` over the Yee grid."""
    i_local, j_local, k = wp.tid()

    i = x_offset + i_local
    j = y_offset + j_local
    field_i = i_local + 1
    field_j = j_local + 1

    x_node = float(i) * DX - HALF_WIDTH
    y_node = float(j) * DX - HALF_WIDTH
    z_node = float(k) * DX - HALF_WIDTH
    half_dx = 0.5 * DX

    # Note: DO NOT interpret `electric_energy_density` below as the energy density at a point `i, j, k` in space.
    # The ``compute_em_energy`` integrates each component over the entire domain at its own respective Yee cell locations.
    # It is valid here to do so because we are ultimately reporting a single summed electrical and magnetic energy value
    # over the entire domain.
    electric_energy_density = (
        e_x[field_i, field_j, k] * e_x[field_i, field_j, k] / inv_eps_r(x_node + half_dx, y_node, z_node)
        + e_y[field_i, field_j, k] * e_y[field_i, field_j, k] / inv_eps_r(x_node, y_node + half_dx, z_node)
        + e_z[field_i, field_j, k] * e_z[field_i, field_j, k] / inv_eps_r(x_node, y_node, z_node + half_dx)
    )
    magnetic_energy_density = (
        h_x[field_i, field_j, k] * h_x[field_i, field_j, k]
        + h_y[field_i, field_j, k] * h_y[field_i, field_j, k]
        + h_z[field_i, field_j, k] * h_z[field_i, field_j, k]
    )

    # `tile_sum` reduces per-thread values to one sum per block. Each
    # `wp.tile_atomic_add` then issues one global atomic per block instead
    # of one global atomic per thread (which would have been the case with wp.atomic_add()).
    electric_tile_sum = wp.tile_sum(wp.tile(electric_energy_density))
    magnetic_tile_sum = wp.tile_sum(wp.tile(magnetic_energy_density))
    wp.tile_atomic_add(electric_energy_sum, electric_tile_sum)
    wp.tile_atomic_add(magnetic_energy_sum, magnetic_tile_sum)


@wp.kernel
def increment_timestep(timestep_counter: wp.array[wp.int32]):
    timestep_counter[0] = timestep_counter[0] + 1


@wp.kernel
def pack_y_face(field: wp.array3d[float], slot: wp.array2d[float], j_local: int):
    """Pack the y face of a field at local index ``j_local`` into a contiguous staging slot."""
    i, k = wp.tid()
    slot[i, k] = field[i, j_local, k]


@wp.kernel
def unpack_y_face(field: wp.array3d[float], slot: wp.array2d[float], j_local: int):
    """Unpack a contiguous staging slot into the y face of a field at local index ``j_local``."""
    i, k = wp.tid()
    field[i, j_local, k] = slot[i, k]


def build_loss_profiles() -> tuple[wp.array[float], wp.array[float]]:
    """Build the 1-D matched-loss rate profiles at grid nodes and half-cell offsets.

    The loss rate is additive per axis, ``gamma_q(d) = GAMMA_MAX * (d / L_ABSORBER)^3``,
    where ``d`` runs from zero at the inner edge of the attenuation layer to
    ``L_ABSORBER`` at the wall. The grid is cubic and uniform, so the three axes share the
    same two profiles.
    """

    def gamma_profile(coords: np.ndarray) -> np.ndarray:
        depth = np.clip((np.abs(coords) - (HALF_WIDTH - L_ABSORBER)) / L_ABSORBER, 0.0, 1.0)
        return GAMMA_MAX * depth**3

    node_coords = (np.arange(N_GRID) - (N_GRID - 1) / 2.0) * DX
    half_coords = node_coords + DX / 2.0

    gamma_integer = wp.array(gamma_profile(node_coords).astype(np.float32), dtype=float)
    gamma_half_offset = wp.array(gamma_profile(half_coords).astype(np.float32), dtype=float)
    return gamma_integer, gamma_half_offset


class Example:
    """FDTD solver for a point source collimated by a Luneburg lens.

    Each rank owns an ``nx_local x ny_local`` pencil of stored nodes with the full z extent,
    stored as ``(nx_local + 2, ny_local + 2, N_GRID)`` fp32 arrays with one halo slot on each
    x and y side. Simulation parameters (``N_GRID``, ``DX``, ``DT``, etc.) are defined as
    module-level constants and captured directly by Warp kernels.

    Args:
        num_ranks_x: Number of MPI ranks along x.
        num_ranks_y: Number of MPI ranks along y.
    """

    def __init__(self, num_ranks_x: int, num_ranks_y: int):
        self.steps_per_frame = 1  # maximum timesteps per rendered frame
        self.gather_planes = True  # gather the E_x plane each frame; the gather is collective, so ranks must agree

        self.mpi_comm = MPI.COMM_WORLD
        self.rank = self.mpi_comm.Get_rank()
        self.size = self.mpi_comm.Get_size()

        if num_ranks_x <= 0 or num_ranks_y <= 0:
            raise ValueError(f"Process-grid dimensions must be positive, got {num_ranks_x} x {num_ranks_y}.")
        if num_ranks_x * num_ranks_y != self.size:
            raise ValueError(
                f"Process grid {num_ranks_x} x {num_ranks_y} requires {num_ranks_x * num_ranks_y} ranks, but MPI launched {self.size}."
            )
        if N_GRID % num_ranks_x != 0 or N_GRID % num_ranks_y != 0:
            raise ValueError(
                f"Cannot decompose {N_GRID} stored nodes over a {num_ranks_x} x {num_ranks_y} process grid; "
                f"N_GRID must be divisible by both factors. Adjust the process grid."
            )

        self.rank_x = self.rank % num_ranks_x
        self.rank_y = self.rank // num_ranks_x

        self.nx_local = N_GRID // num_ranks_x
        self.ny_local = N_GRID // num_ranks_y
        self.x_offset = self.rank_x * self.nx_local
        self.y_offset = self.rank_y * self.ny_local

        self.x_left_neighbor = self.rank - 1 if self.rank_x > 0 else None
        self.x_right_neighbor = self.rank + 1 if self.rank_x < num_ranks_x - 1 else None
        self.y_bottom_neighbor = self.rank - num_ranks_x if self.rank_y > 0 else None
        self.y_top_neighbor = self.rank + num_ranks_x if self.rank_y < num_ranks_y - 1 else None

        self.device = calc_default_device(self.mpi_comm)
        wp.set_device(self.device)
        self.compute_stream = wp.Stream()

        # use CUDA graph if GPU is available
        self.use_cuda_graph = self.device.is_cuda
        self.graph = None

        field_shape = (self.nx_local + 2, self.ny_local + 2, N_GRID)
        self.face_dim = (self.nx_local + 2, N_GRID)

        self.e_x = wp.zeros(field_shape, dtype=float)
        self.e_y = wp.zeros(field_shape, dtype=float)
        self.e_z = wp.zeros(field_shape, dtype=float)
        self.h_x = wp.zeros(field_shape, dtype=float)
        self.h_y = wp.zeros(field_shape, dtype=float)
        self.h_z = wp.zeros(field_shape, dtype=float)

        self.electric_energy_sum = wp.zeros(1, dtype=float)
        self.magnetic_energy_sum = wp.zeros(1, dtype=float)

        # global timestep index counter on device to make CUDA graph capture possible
        self.timestep_counter = wp.zeros(1, dtype=wp.int32)

        # the three axes index the same integer and half-offset profiles for a cubic uniform grid
        self.gamma_integer, self.gamma_half_offset = build_loss_profiles()

        self._alloc_staging()
        self._init_nccl()

        self.mpi_comm.Barrier()
        print(
            f"Rank {self.rank} on device {self.device.pci_bus_id}: "
            f"pencil ({self.rank_x}, {self.rank_y}) of {num_ranks_x} x {num_ranks_y}, "
            f"x nodes [{self.x_offset}, {self.x_offset + self.nx_local}), "
            f"y nodes [{self.y_offset}, {self.y_offset + self.ny_local})"
        )
        self.mpi_comm.Barrier()

        if self.rank == 0:
            points_per_wavelength = (N_GRID - 1) / (2.0 * HALF_WIDTH)
            print(
                f"FDTD Luneburg lens: {N_GRID}^3 grid, dx = {DX:.6f} "
                f"({points_per_wavelength:.1f} points per wavelength), dt = {DT:.6f}"
            )

    def _alloc_staging(self) -> None:
        """Allocate the y-face staging slots.

        Two fp32 send slots and two receive slots, one per tangential component, each a contiguous
        ``face_dim`` array reused for the opposite-direction ``E`` and ``H`` halo exchanges.
        """
        self.send_slot1 = wp.zeros(self.face_dim, dtype=float)
        self.send_slot2 = wp.zeros(self.face_dim, dtype=float)
        self.recv_slot1 = wp.zeros(self.face_dim, dtype=float)
        self.recv_slot2 = wp.zeros(self.face_dim, dtype=float)

        # separate slot for the animation plane so a frame grab can never race a halo exchange
        self.plane_slot = wp.zeros(self.face_dim, dtype=float)

    def _init_nccl(self) -> None:
        """Create the NCCL communicator used for halo exchanges."""
        unique_id = nccl.get_unique_id() if self.rank == 0 else None
        unique_id = self.mpi_comm.bcast(unique_id, root=0)
        self.mpi_comm.Barrier()
        self.nccl_comm = nccl.Communicator.init(nranks=self.size, rank=self.rank, unique_id=unique_id)

    def _exchange_faces(self, transfers: list[tuple]) -> None:
        """Issue one NCCL group of paired halo sends and receives on the compute stream.

        Each transfer is a ``(send_buf, send_peer, recv_buf, recv_peer)`` tuple of
        contiguous buffers and neighbor ranks; a ``None`` peer at a domain boundary
        skips that half of the pair. NCCL matches point-to-point messages per peer by
        issue order, so every rank must pass component-matched transfers in the same
        list order.
        """
        with nccl.group():
            for send_buf, send_peer, recv_buf, recv_peer in transfers:
                if send_peer is not None:
                    self.nccl_comm.send(send_buf, send_peer, stream=self.compute_stream)
                if recv_peer is not None:
                    self.nccl_comm.recv(recv_buf, recv_peer, stream=self.compute_stream)

    def _exchange_e_halos(self) -> None:
        """Exchange the depth-1 ``E`` halo faces with the four pencil neighbors.

        ``(E_y, E_z)`` travel toward -x and arrive from +x; ``(E_x, E_z)`` travel toward -y and
        arrive from +y. The x faces are contiguous slices and go straight from the field arrays,
        while the y faces are strided and stage through the packing slots. All four transfers
        share one NCCL group and keep a fixed component order that every peer must match.
        """
        wp.launch(
            pack_y_face,
            dim=self.face_dim,
            inputs=[self.e_x, self.send_slot1, 1],
            stream=self.compute_stream,
        )
        wp.launch(
            pack_y_face,
            dim=self.face_dim,
            inputs=[self.e_z, self.send_slot2, 1],
            stream=self.compute_stream,
        )

        self._exchange_faces(
            [
                (self.e_y[1], self.x_left_neighbor, self.e_y[self.nx_local + 1], self.x_right_neighbor),
                (self.e_z[1], self.x_left_neighbor, self.e_z[self.nx_local + 1], self.x_right_neighbor),
                (self.send_slot1, self.y_bottom_neighbor, self.recv_slot1, self.y_top_neighbor),
                (self.send_slot2, self.y_bottom_neighbor, self.recv_slot2, self.y_top_neighbor),
            ]
        )

        wp.launch(
            unpack_y_face,
            dim=self.face_dim,
            inputs=[self.e_x, self.recv_slot1, self.ny_local + 1],
            stream=self.compute_stream,
        )
        wp.launch(
            unpack_y_face,
            dim=self.face_dim,
            inputs=[self.e_z, self.recv_slot2, self.ny_local + 1],
            stream=self.compute_stream,
        )

    def _exchange_h_halos(self) -> None:
        """Exchange the depth-1 ``H`` halo faces with the four pencil neighbors.

        ``(H_y, H_z)`` travel toward +x and arrive from -x; ``(H_x, H_z)`` travel toward +y and
        arrive from -y. The directions reverse relative to the ``E`` round because the ``H``
        update takes forward differences of ``E`` while the ``E`` update takes backward
        differences of ``H``: a rank needs its upper neighbor's first ``E`` plane and its lower
        neighbor's last ``H`` plane. Same ordering, grouping, packing, and stream rules as that
        round.
        """
        wp.launch(
            pack_y_face,
            dim=self.face_dim,
            inputs=[self.h_x, self.send_slot1, self.ny_local],
            stream=self.compute_stream,
        )
        wp.launch(
            pack_y_face,
            dim=self.face_dim,
            inputs=[self.h_z, self.send_slot2, self.ny_local],
            stream=self.compute_stream,
        )

        self._exchange_faces(
            [
                (self.h_y[self.nx_local], self.x_right_neighbor, self.h_y[0], self.x_left_neighbor),
                (self.h_z[self.nx_local], self.x_right_neighbor, self.h_z[0], self.x_left_neighbor),
                (self.send_slot1, self.y_top_neighbor, self.recv_slot1, self.y_bottom_neighbor),
                (self.send_slot2, self.y_top_neighbor, self.recv_slot2, self.y_bottom_neighbor),
            ]
        )

        wp.launch(
            unpack_y_face,
            dim=self.face_dim,
            inputs=[self.h_x, self.recv_slot1, 0],
            stream=self.compute_stream,
        )
        wp.launch(
            unpack_y_face,
            dim=self.face_dim,
            inputs=[self.h_z, self.recv_slot2, 0],
            stream=self.compute_stream,
        )

    def _check_em_energy(self, timestep: int) -> None:
        """Compute and report global electromagnetic energy, raising if it is not finite."""
        with wp.ScopedStream(self.compute_stream, sync_enter=False):
            self.electric_energy_sum.zero_()
            self.magnetic_energy_sum.zero_()
            wp.launch(
                compute_em_energy,
                dim=(self.nx_local, self.ny_local, N_GRID),
                inputs=[
                    self.e_x,
                    self.e_y,
                    self.e_z,
                    self.h_x,
                    self.h_y,
                    self.h_z,
                    self.x_offset,
                    self.y_offset,
                ],
                outputs=[self.electric_energy_sum, self.magnetic_energy_sum],
            )
            self.nccl_comm.allreduce(
                self.electric_energy_sum,
                self.electric_energy_sum,
                nccl.SUM,
                stream=self.compute_stream,
            )
            self.nccl_comm.allreduce(
                self.magnetic_energy_sum,
                self.magnetic_energy_sum,
                nccl.SUM,
                stream=self.compute_stream,
            )

        w_e = 0.5 * DX**3 * float(self.electric_energy_sum.numpy()[0])
        w_h = 0.5 * DX**3 * float(self.magnetic_energy_sum.numpy()[0])
        if not (np.isfinite(w_e) and np.isfinite(w_h)):
            raise RuntimeError(f"Rank {self.rank}: non-finite electromagnetic energy at timestep {timestep}.")
        if self.rank == 0:
            print(f"step {timestep}, t = {timestep * DT:.6f}, W_E = {w_e:.6e}, W_H = {w_h:.6e}", flush=True)

    def step(self) -> None:
        """Advance ``H`` from curl ``E``, then ``E`` from curl ``H`` in leapfrog fashion.

        Each update runs after the halo exchange of the field that it differentiates.
        """
        with wp.ScopedStream(self.compute_stream, sync_enter=False):
            self._exchange_e_halos()

            wp.launch(
                update_h,
                dim=(self.nx_local, self.ny_local, N_GRID),
                inputs=[
                    self.e_x,
                    self.e_y,
                    self.e_z,
                    self.gamma_integer,
                    self.gamma_half_offset,
                    self.x_offset,
                    self.y_offset,
                ],
                outputs=[self.h_x, self.h_y, self.h_z],
            )

            self._exchange_h_halos()

            wp.launch(
                update_e,
                dim=(self.nx_local, self.ny_local, N_GRID),
                inputs=[
                    self.h_x,
                    self.h_y,
                    self.h_z,
                    self.gamma_integer,
                    self.gamma_half_offset,
                    self.x_offset,
                    self.y_offset,
                    self.timestep_counter,
                ],
                outputs=[self.e_x, self.e_y, self.e_z],
            )

            # Advance the device-side timestep counter after an electric-field update.
            wp.launch(increment_timestep, dim=1, inputs=[self.timestep_counter])

    def gather_e_x_plane(self) -> np.ndarray | None:
        """Gather the instantaneous ``E_x`` on the y-center node plane to rank 0.

        The ranks holding the center plane pack their strip with ``pack_y_face`` so only
        one face, not the whole pencil, crosses the PCIe bus each frame. Returns the
        assembled ``(N_GRID, N_GRID)`` x-z plane on rank 0 and ``None`` elsewhere.
        """
        j_center = (N_GRID - 1) // 2  # global y node index of the center plane
        local_strip = None
        if self.y_offset <= j_center < self.y_offset + self.ny_local:
            wp.launch(
                pack_y_face,
                dim=self.face_dim,
                inputs=[self.e_x, self.plane_slot, j_center - self.y_offset + 1],
                stream=self.compute_stream,
            )
            local_strip = self.plane_slot.numpy()[1 : self.nx_local + 1, :]

        strips = self.mpi_comm.gather((self.x_offset, local_strip), root=0)
        if self.rank != 0:
            return None

        strips = [(x0, strip) for x0, strip in strips if strip is not None]
        strips.sort(key=lambda item: item[0])
        return np.concatenate([strip for _, strip in strips], axis=0)

    def step_and_render_frame(self, frame_num: int, img=None, num_steps: int | None = None) -> tuple:
        """Advance up to ``steps_per_frame`` timesteps and update the Matplotlib image."""
        first_timestep = frame_num * self.steps_per_frame
        steps_this_frame = self.steps_per_frame
        if num_steps is not None:
            steps_this_frame = min(steps_this_frame, num_steps - first_timestep)

        with wp.ScopedStream(self.compute_stream, sync_enter=False):
            self.timestep_counter.fill_(first_timestep)

        for substep in range(steps_this_frame):
            timestep_index = first_timestep + substep
            if self.use_cuda_graph:
                wp.capture_launch(self.graph, stream=self.compute_stream)
            else:
                self.step()
            if (timestep_index + 1) % DIAG_INTERVAL == 0:
                self._check_em_energy(timestep_index + 1)

        plane = self.gather_e_x_plane() if self.gather_planes else None

        if img is not None:
            sim_timestep = first_timestep + steps_this_frame
            img.set_array(plane)
            img.axes.set_title(f"Luneburg lens, {N_GRID}$^3$, $E_x$ at timestep {sim_timestep}")

        return (img,)

    def drive_animation_frame(self, frame_num: int, img=None, num_steps: int | None = None) -> tuple:
        """Render one frame on rank 0, holding the other ranks in lockstep.

        The GUI event loop swallows callback exceptions, which would stall the animation
        and leave the followers blocked on the next broadcast, so abort the whole job on
        any failure.
        """
        try:
            self.mpi_comm.bcast(frame_num, root=0)
            return self.step_and_render_frame(frame_num, img, num_steps)
        except Exception:
            traceback.print_exc()
            self.mpi_comm.Abort(1)
            raise

    def follow_animation(self, num_steps: int | None = None) -> None:
        """Run the frames rank 0 asks for, until it signals that the window closed."""
        while (frame_num := self.mpi_comm.bcast(None, root=0)) is not None:
            self.step_and_render_frame(frame_num, num_steps=num_steps)

    def stop_animation(self) -> None:
        """Release the ranks waiting in :meth:`follow_animation`."""
        self.mpi_comm.bcast(None, root=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="3-D FDTD simulation of a point source collimated by a Luneburg lens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-ranks-x",
        type=int,
        required=True,
        default=argparse.SUPPRESS,
        help="Number of MPI ranks along x. The product of --num-ranks-x and --num-ranks-y must equal "
        "the number of MPI ranks.",
    )
    parser.add_argument(
        "--num-ranks-y",
        type=int,
        required=True,
        default=argparse.SUPPRESS,
        help="Number of MPI ranks along y. The product of --num-ranks-x and --num-ranks-y must equal "
        "the number of MPI ranks.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=N_GRID,
        help="Grid resolution: nodes per axis across the fixed 8.5-wavelength domain. "
        "Refining the grid leaves the physical extent, lens, and attenuation-layer depth unchanged. "
        "The resolution must be divisible by both rank counts.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Total number of leapfrog timesteps. Defaults to unlimited in interactive mode and to "
        "SIM_DURATION source periods, rounded to whole steps, in headless mode.",
    )
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=20,
        help="Maximum number of leapfrog timesteps per rendered frame.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    for flag, value in (("--num-ranks-x", args.num_ranks_x), ("--num-ranks-y", args.num_ranks_y)):
        if value < 1:
            parser.error(f"{flag} must be at least 1, got {value}")
    if args.num_steps is not None and args.num_steps < 0:
        parser.error(f"--num-steps must be non-negative, got {args.num_steps}")
    if args.steps_per_frame < 1:
        parser.error(f"--steps-per-frame must be at least 1, got {args.steps_per_frame}")
    if args.resolution < 48:
        parser.error(
            f"--resolution {args.resolution} is too coarse: below 48 nodes per axis the absorbing "
            f"layer spans fewer than four cells and the grid falls under six points per wavelength."
        )

    # Rebind the resolution-derived module constants before the first kernel launch.
    N_GRID = args.resolution
    DX = 2.0 * HALF_WIDTH / (N_GRID - 1)
    DT = 0.95 * DX / math.sqrt(3.0)
    SOURCE_SCALE = (DX_REF / DX) ** 2
    SOURCE_IJ = (N_GRID - 1) // 2
    SOURCE_K = round((HALF_WIDTH - R_LENS) / DX)

    # dt tracks dx, so the headless default timestep count derives from a fixed physical duration
    num_steps = args.num_steps
    if num_steps is None and args.headless:
        num_steps = round(SIM_DURATION / DT)

    if not args.headless and MPI.COMM_WORLD.Get_rank() == 0:
        # fail fast before any GPU or NCCL setup; mpirun tears down the other ranks on exit
        try:
            import matplotlib.pyplot  # noqa: F401
        except ImportError as exc:
            raise ImportError("Interactive mode requires Matplotlib; install it or pass --headless.") from exc

    example = Example(num_ranks_x=args.num_ranks_x, num_ranks_y=args.num_ranks_y)
    example.steps_per_frame = args.steps_per_frame
    example.gather_planes = not args.headless  # skip the collective plane gather in headless mode

    if example.use_cuda_graph:
        # capture one leapfrog timestep, halo exchanges included, in a CUDA graph
        wp.synchronize_device()
        with wp.ScopedCapture(stream=example.compute_stream) as capture:
            example.step()
        example.graph = capture.graph

    if args.headless:
        import time

        num_frames = (num_steps + args.steps_per_frame - 1) // args.steps_per_frame

        # start every rank's clock together, after all setup work has finished
        wp.synchronize_device()
        example.mpi_comm.Barrier()
        start_time = time.perf_counter()

        for i in range(num_frames):
            example.step_and_render_frame(frame_num=i, num_steps=num_steps)

        wp.synchronize_device()
        stop_time = time.perf_counter()

        # the slowest rank sets the run time
        elapsed = example.mpi_comm.allreduce(stop_time - start_time, op=MPI.MAX)
        if example.rank == 0:
            mcells_per_s = num_steps * N_GRID**3 / elapsed / 1e6
            print(
                f"{example.size} rank(s), {num_steps} timesteps on a {N_GRID}^3 grid: "
                f"{elapsed:.3f} s, {mcells_per_s:.1f} Mcell-updates/s"
            )
    elif example.rank == 0:
        import matplotlib.animation as anim
        import matplotlib.pyplot as plt

        num_frames = (num_steps + args.steps_per_frame - 1) // args.steps_per_frame if num_steps is not None else None

        # the finally releases the followers from their broadcast wait even if rank 0 fails
        try:
            fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
            ax.set_title(f"Luneburg lens, {N_GRID}$^3$, $E_x$ at timestep 0")
            ax.set_xlabel("z")
            ax.set_ylabel("x")

            img = ax.imshow(
                example.gather_e_x_plane(),
                origin="lower",
                cmap="seismic",
                interpolation="antialiased",
                vmin=-PLOT_E_SCALE,
                vmax=PLOT_E_SCALE,
                extent=[-HALF_WIDTH, HALF_WIDTH, -HALF_WIDTH, HALF_WIDTH],
            )
            fig.colorbar(img, ax=ax, label="$E_x$")
            ax.add_patch(plt.Circle((0.0, 0.0), R_LENS, fill=False, color="0.3", lw=0.8, ls="--"))

            seq = anim.FuncAnimation(
                fig,
                example.drive_animation_frame,
                fargs=(img, num_steps),
                frames=num_frames,
                init_func=lambda: (),  # without this, Matplotlib delivers frame 0 twice and replays the source window
                blit=False,
                interval=1,
                repeat=False,
                cache_frame_data=False,
            )

            plt.show()
        finally:
            example.stop_animation()
    else:
        example.gather_e_x_plane()  # match rank 0's pre-animation frame grab
        example.follow_animation(num_steps)
