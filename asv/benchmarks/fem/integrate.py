# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

import warp as wp
import warp._src.fem.integrate as fem_integrate_mod
import warp.fem as fem
import warp.sparse as wps
from warp.examples.fem.utils import gen_tetmesh

# ruff: noqa: RUF059


@wp.func
def symmetric_strain(sig: wp.vec3, V: wp.mat33):
    return V * wp.diag(sig) * wp.transpose(V)


@wp.func
def symmetric_strain_delta(U: wp.mat33, sig: wp.vec3, V: wp.mat33, dF: wp.mat33):
    # see supplementary of `WRAPD: Weighted Rotation-aware ADMM`, Brown and Narain 21

    Ut = wp.transpose(U)
    Vt = wp.transpose(V)

    dF_loc = Ut * dF * V
    SigdF_loc = wp.diag(sig) * dF_loc

    sig_op = wp.matrix_from_cols(wp.vec3(sig[0]), wp.vec3(sig[1]), wp.vec3(sig[2]))
    dSig = wp.cw_div(SigdF_loc + wp.transpose(SigdF_loc), sig_op + wp.transpose(sig_op))
    dS = V * dSig * Vt

    return dS


@fem.integrand
def defgrad(u: fem.Field, s: fem.Sample):
    return fem.grad(u, s) + wp.identity(n=3, dtype=float)


@wp.func
def hooke_stress(S: wp.mat33, lame: wp.vec2):
    strain = S - wp.identity(n=3, dtype=float)
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(n=3, dtype=float)


@wp.func
def hooke_energy(S: wp.mat33, lame: wp.vec2):
    strain = S - wp.identity(n=3, dtype=float)
    return 0.5 * wp.ddot(strain, hooke_stress(S, lame))


@wp.func
def hooke_hessian(S: wp.mat33, tau: wp.mat33, sig: wp.mat33, lame: wp.vec2):
    return wp.ddot(hooke_stress(sig + wp.identity(n=3, dtype=float), lame), tau)


@fem.integrand
def cr_elastic_energy(s: fem.Sample, u_cur: fem.Field, lame: fem.Field):
    F = defgrad(u_cur, s)
    U, sig, V = wp.svd3(F)

    S = symmetric_strain(sig, V)
    return hooke_energy(S, lame(s))


@fem.integrand
def cr_elastic_forces(s: fem.Sample, u_cur: fem.Field, v: fem.Field, lame: fem.Field):
    F = defgrad(u_cur, s)
    U, sig, V = wp.svd3(F)

    S = symmetric_strain(sig, V)
    tau = symmetric_strain_delta(U, sig, V, fem.grad(v, s))
    return -wp.ddot(tau, hooke_stress(S, lame(s)))


@fem.integrand
def cr_elasticity_hessian(s: fem.Sample, u_cur: fem.Field, u: fem.Field, v: fem.Field, lame: fem.Field):
    F = defgrad(u_cur, s)
    U, sig, V = wp.svd3(F)

    S_s = symmetric_strain(sig, V)
    tau_s = symmetric_strain_delta(U, sig, V, fem.grad(v, s))
    sig_s = symmetric_strain_delta(U, sig, V, fem.grad(u, s))
    lame_s = lame(s)

    return hooke_hessian(S_s, tau_s, sig_s, lame_s)


@fem.integrand
def diffusion_form_scalar(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def diffusion_form_vector(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.ddot(fem.D(u, s), fem.D(v, s))


def _supports_padded_bsr_assembly():
    return (
        "bsr_options" in inspect.signature(fem.integrate).parameters
        and "topology" in inspect.signature(wps.bsr_set_from_triplets).parameters
        and hasattr(wps, "bsr_compress")
        and hasattr(fem_integrate_mod, "bsr_compress")
        and hasattr(fem_integrate_mod, "_BSR_CONSTRUCTION_ROW_COMPRESS")
    )


def _require_padded_bsr_assembly():
    if not _supports_padded_bsr_assembly():
        raise NotImplementedError("Padded BSR assembly is not available")


def _triplet_bsr_options():
    if "bsr_options" not in inspect.signature(fem.integrate).parameters:
        return None
    if not hasattr(fem_integrate_mod, "_BSR_CONSTRUCTION_TRIPLETS"):
        return None
    return {"construction": "triplets"}


def _make_corotated_hessian_integrator(space: fem.FunctionSpace, assembly: str):
    u = fem.make_trial(space)
    v = fem.make_test(space)
    u_cur = fem.make_discrete_field(space)

    lame = fem.UniformField(v.domain, value=wp.vec2(1.0, 0.3))
    quadrature = fem.RegularQuadrature(v.domain, order=2 * u_cur.degree)

    def integrate_hessian(output=None, bsr_options=None):
        kwargs = {}
        if "bsr_options" in inspect.signature(fem.integrate).parameters and bsr_options is not None:
            kwargs["bsr_options"] = bsr_options

        return fem.integrate(
            cr_elasticity_hessian,
            fields={"u_cur": u_cur, "u": u, "v": v, "lame": lame},
            output_dtype=float,
            output=output,
            assembly=assembly,
            quadrature=quadrature,
            kernel_options={"enable_backward": False},
            **kwargs,
        )

    return integrate_hessian


class FemCorotatedElasticity:
    """Utility base class for building FEM matrices to test BSR matrix-vector multiplication."""

    def __init__(self, use_graph: bool = True):
        self._use_graph = use_graph

    def make_integrate_func(self, space: fem.FunctionSpace, fn: str, assembly: str):
        integrate_hessian = _make_corotated_hessian_integrator(space, assembly)

        v = fem.make_test(space)
        u_cur = fem.make_discrete_field(space)

        lame = fem.UniformField(v.domain, value=wp.vec2(1.0, 0.3))
        E = wp.array(shape=1, dtype=float)

        quadrature = fem.RegularQuadrature(v.domain, order=2 * u_cur.degree)

        compressed_hessian_fn = None
        self._compressed_hessian_output = None
        if fn == "hessian_compressed":
            _require_padded_bsr_assembly()
            hessian_bsr_options = {"topology": "padded", "construction": "row_compress"}
            hessian_bsr_reuse_options = {**hessian_bsr_options, "capacity": "reuse"}
            self._compressed_hessian_output = integrate_hessian(bsr_options=hessian_bsr_options)

            if self._compressed_hessian_output.status_sync() != 0:
                raise RuntimeError("Padded FEM Hessian assembly exceeded row capacity")

            def integrate_compressed_hessian():
                return integrate_hessian(output=self._compressed_hessian_output, bsr_options=hessian_bsr_reuse_options)

            compressed_hessian_fn = integrate_compressed_hessian

        self._integrate_funcs = {
            "energy": lambda: fem.integrate(
                cr_elastic_energy,
                domain=v.domain,
                fields={"u_cur": u_cur, "lame": lame},
                output=E,
                quadrature=quadrature,
                kernel_options={"enable_backward": False},
            ),
            "forces": lambda: fem.integrate(
                cr_elastic_forces,
                fields={"u_cur": u_cur, "v": v, "lame": lame},
                output_dtype=float,
                assembly=assembly,
                quadrature=quadrature,
                kernel_options={"enable_backward": False},
            ),
            "hessian": lambda: integrate_hessian(bsr_options=_triplet_bsr_options()),
        }
        if compressed_hessian_fn is not None:
            self._integrate_funcs["hessian_compressed"] = compressed_hessian_fn

        integrate_fn = self._integrate_funcs[fn]
        integrate_fn()
        if fn == "hessian_compressed" and self._compressed_hessian_output.status_sync() != 0:
            raise RuntimeError("Padded FEM Hessian assembly exceeded row capacity")

        if self._use_graph:
            with wp.ScopedCapture() as capture:
                integrate_fn()
            self._graph = capture.graph

    def run(self, fn: str):
        if self._use_graph:
            wp.capture_launch(self._graph)
        else:
            self._integrate_funcs[fn]()
        wp.synchronize_device(self.device)


class _SparseAssemblyCapture:
    def __init__(self):
        self.kind = None
        self.options = None
        self.dest = None
        self.rows = None
        self.columns = None
        self.values = None
        self.raw_offsets = None
        self.raw_row_counts = None
        self.raw_columns = None
        self.raw_values = None

    def capture_from_integrate(self, integrate_once, bsr_options):
        orig_set_from_triplets = fem_integrate_mod.bsr_set_from_triplets
        orig_compress = getattr(fem_integrate_mod, "bsr_compress", None)

        def capture_set_from_triplets(dest, rows, columns, values=None, **kwargs):
            self.kind = "triplets"
            self.dest = dest
            self.options = dict(kwargs)
            self.rows = wp.clone(rows)
            self.columns = wp.clone(columns)
            self.values = wp.clone(values) if values is not None else None
            return orig_set_from_triplets(dest, rows, columns, values, **kwargs)

        def capture_compress(src, **kwargs):
            self.kind = "compress"
            self.dest = src
            self.options = dict(kwargs)
            self.raw_offsets = wp.clone(src.offsets)
            self.raw_row_counts = wp.clone(src.row_counts)
            self.raw_columns = wp.clone(src.columns[: src.nnz])
            self.raw_values = wp.clone(src.values[: src.nnz])
            return orig_compress(src, **kwargs)

        fem_integrate_mod.bsr_set_from_triplets = capture_set_from_triplets
        if orig_compress is not None:
            fem_integrate_mod.bsr_compress = capture_compress
        try:
            output = integrate_once(None, bsr_options)
            if _bsr_status_failed(output):
                raise RuntimeError("FEM sparse assembly capture exceeded row capacity")
            wp.synchronize_device(output.device)
        finally:
            fem_integrate_mod.bsr_set_from_triplets = orig_set_from_triplets
            if orig_compress is not None:
                fem_integrate_mod.bsr_compress = orig_compress

        if self.kind is None:
            raise RuntimeError("FEM sparse assembly capture did not reach a sparse construction call")

        return self


def _bsr_status_failed(matrix):
    status_sync = getattr(matrix, "status_sync", None)
    return status_sync is not None and status_sync() != 0


def _make_raw_bsr_copy(capture: _SparseAssemblyCapture):
    bsr = wps.bsr_matrix_t(capture.raw_values.dtype)()
    bsr.nrow = capture.dest.nrow
    bsr.ncol = capture.dest.ncol
    bsr.nnz = capture.raw_columns.shape[0]
    bsr.offsets = wp.clone(capture.raw_offsets)
    bsr.row_counts = wp.clone(capture.raw_row_counts)
    bsr.columns = wp.clone(capture.raw_columns)
    bsr.values = wp.clone(capture.raw_values)
    return bsr


class FemCorotatedElasticitySparseAssembly:
    """Benchmark only the sparse construction stage after bilinear dispatch."""

    repeat = 5
    number = 10
    pool_size = 64

    def make_sparse_assembly_func(self, space: fem.FunctionSpace, fn: str, assembly: str):
        integrate_hessian = _make_corotated_hessian_integrator(space, assembly)

        if fn == "hessian":
            bsr_options = _triplet_bsr_options()
        elif fn == "hessian_compressed":
            _require_padded_bsr_assembly()
            bsr_options = {"topology": "padded", "construction": "row_compress"}
        else:
            raise ValueError(f"Unsupported sparse assembly function: {fn}")

        self._capture = _SparseAssemblyCapture().capture_from_integrate(integrate_hessian, bsr_options)
        self._src_index = 0
        self._src_pool = []
        if self._capture.kind == "compress":
            self._src_pool = [_make_raw_bsr_copy(self._capture) for _ in range(self.pool_size)]

    def _run_impl(self):
        capture = self._capture
        if capture.kind == "triplets":
            wps.bsr_set_from_triplets(capture.dest, capture.rows, capture.columns, capture.values, **capture.options)
            if _bsr_status_failed(capture.dest):
                raise RuntimeError("FEM sparse assembly exceeded row capacity")
        else:
            if self._src_index == len(self._src_pool):
                self._src_pool.append(_make_raw_bsr_copy(capture))

            src = self._src_pool[self._src_index]
            self._src_index += 1
            wps.bsr_compress(src, **capture.options)

    def run(self):
        self._run_impl()
        wp.synchronize_device(self.device)


class FemCorotatedElasticityQuadraticTetmesh(FemCorotatedElasticity):
    """Test corotated elasticity with quadratic tetrahedral elements."""

    repeat = 5
    number = 15

    params = (["energy", "forces", "hessian", "hessian_compressed"], ["dispatch"])
    param_names = ["fn", "assembly"]

    def setup(self, fn: str, assembly: str):
        wp.init()
        self.device = wp.get_device("cuda:0")

        res = 16
        with wp.ScopedDevice(self.device):
            pos, cells = gen_tetmesh(res=(res, res, res))
            geo = fem.Tetmesh(cells, pos)
            space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec3)
            self.make_integrate_func(space, fn, assembly)

    def time_cuda(self, fn, assembly):
        self.run(fn)


class FemCorotatedElasticityLinearGrid(FemCorotatedElasticity):
    """Test corotated elasticity with linear grid elements."""

    repeat = 5
    number = 15

    params = (["energy", "forces", "hessian", "hessian_compressed"], ["generic", "dispatch"])
    param_names = ["fn", "assembly"]

    def setup(self, fn: str, assembly: str):
        wp.init()
        self.device = wp.get_device("cuda:0")

        res = 32
        with wp.ScopedDevice(self.device):
            geo = fem.Grid3D(res=(res, res, res))
            space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec3)
            self.make_integrate_func(space, fn, assembly)

    def time_cuda(self, fn, assembly):
        self.run(fn)


class FemCorotatedElasticityVeryHighOrder(FemCorotatedElasticity):
    """Test corotated elasticity with high order elements."""

    repeat = 15
    number = 50

    params = (["energy", "forces", "hessian", "hessian_compressed"], ["dispatch"])
    param_names = ["fn", "assembly"]

    def setup(self, fn: str, assembly: str):
        wp.init()
        self.device = wp.get_device("cuda:0")

        res = 2
        with wp.ScopedDevice(self.device):
            geo = fem.Grid3D(res=(res, res, res))
            space = fem.make_polynomial_space(geo, degree=4, dtype=wp.vec3)
            self.make_integrate_func(space, fn, assembly)

    def time_cuda(self, fn, assembly):
        self.run(fn)


class FemCorotatedElasticitySparseAssemblyQuadraticTetmesh(FemCorotatedElasticitySparseAssembly):
    """Test post-dispatch sparse assembly for quadratic tetrahedral elements."""

    params = (["hessian", "hessian_compressed"], ["dispatch"])
    param_names = ["fn", "assembly"]

    def setup(self, fn: str, assembly: str):
        wp.init()
        self.device = wp.get_device("cuda:0")

        res = 16
        with wp.ScopedDevice(self.device):
            pos, cells = gen_tetmesh(res=(res, res, res))
            geo = fem.Tetmesh(cells, pos)
            space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec3)
            self.make_sparse_assembly_func(space, fn, assembly)

    def time_cuda(self, fn, assembly):
        self.run()


class FemCorotatedElasticitySparseAssemblyLinearGrid(FemCorotatedElasticitySparseAssembly):
    """Test post-dispatch sparse assembly for linear grid elements."""

    params = (["hessian", "hessian_compressed"], ["generic", "dispatch"])
    param_names = ["fn", "assembly"]

    def setup(self, fn: str, assembly: str):
        wp.init()
        self.device = wp.get_device("cuda:0")

        res = 32
        with wp.ScopedDevice(self.device):
            geo = fem.Grid3D(res=(res, res, res))
            space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec3)
            self.make_sparse_assembly_func(space, fn, assembly)

    def time_cuda(self, fn, assembly):
        self.run()


class FemCorotatedElasticitySparseAssemblyVeryHighOrder(FemCorotatedElasticitySparseAssembly):
    """Test post-dispatch sparse assembly for high order elements."""

    params = (["hessian", "hessian_compressed"], ["dispatch"])
    param_names = ["fn", "assembly"]

    def setup(self, fn: str, assembly: str):
        wp.init()
        self.device = wp.get_device("cuda:0")

        res = 2
        with wp.ScopedDevice(self.device):
            geo = fem.Grid3D(res=(res, res, res))
            space = fem.make_polynomial_space(geo, degree=4, dtype=wp.vec3)
            self.make_sparse_assembly_func(space, fn, assembly)

    def time_cuda(self, fn, assembly):
        self.run()


if __name__ == "__main__":
    for cases in (
        (FemCorotatedElasticityQuadraticTetmesh, "quadratic_tetmesh"),
        (FemCorotatedElasticityLinearGrid, "linear_grid"),
        (FemCorotatedElasticityVeryHighOrder, "very_high_order"),
    ):
        case = cases[0](use_graph=False)

        for fn in case.params[0]:
            for asm in case.params[1]:
                case.setup(fn, asm)
                for _k in range(3):
                    with wp.ScopedTimer(f"{cases[1]}_{fn}_{asm}", synchronize=True):
                        case.time_cuda(fn, asm)
