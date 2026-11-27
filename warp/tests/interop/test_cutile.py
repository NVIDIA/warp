# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the experimental cuTile foreign-function bridge (warp.experimental.cutile).

The tests target the public surface and are tiered by environment:

- ``TestCutileHelpers``: the host helpers, toolchain lookup, and packing
  errors; no cuTile needed (the CUDA array operand check runs on every CUDA
  test device);
- ``TestCutileDeclaration``: the errors ``export_func`` reports before any
  toolchain or GPU work; needs ``cuda.tile`` importable;
- ``TestCutileKernels``: end-to-end calls from cuTile kernels, one test per
  documented call form or contract; needs ``cuda.tile_preview``, a supported
  ``tileiras`` (from the ``nvidia-cuda-tileiras`` wheel the ``cutile`` extra installs,
  ``$WARP_TILEIR_TOOLCHAIN``, or ``PATH``), and a CUDA device.

Classes skip cleanly when their environment is missing, so the module is safe
to run (and register in the default suite) everywhere.

Names keep the two kernel kinds apart: ``@ct.kernel`` functions end in
``_cutile_kernel``, the ``@wp.kernel`` functions that compute reference results
end in ``_reference``, and ``export_func()`` handles are
``wp_<function name>``. Every cuTile kernel costs a ``tileiras`` compile, so each
test folds its call sites into as few cuTile kernels as the checked behavior allows.
"""

import contextlib
import ctypes
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock
import warnings

import numpy as np

import warp as wp
import warp._src.cutile as _cutile
import warp._src.logger
import warp.experimental.cutile as wpct
from warp.tests.unittest_utils import *

NUM_THREADS = 128
NUM_WARPS = NUM_THREADS // 32
VEC_PAD = 4  # vec3 tile extents are padded to the next power of two
cuda_test_devices = get_cuda_test_devices()
# cuTile supports Ampere and newer; the end-to-end kernel tests run only there
cutile_test_devices = [d for d in cuda_test_devices if d.arch >= 80]


# Keep fixture functions at module scope: defining them is environment-independent
# (no cuTile import), and codegen only happens when a test builds them.


@wp.struct
class BridgeParams:
    """Mixed-field struct exercised through the int32 byte-blob path."""

    scale: float
    offset: wp.vec3
    exponent: int


@wp.struct
class WideAlignParams:
    """Struct with an 8-byte-aligned field to exercise blob alignment."""

    a: wp.float64
    b: int


@wp.struct
class ArrayHolder:
    """Struct with an array field, which cannot cross the boundary as a return value."""

    data: wp.array[float]


@wp.func
def blend(x: float, y: wp.vec3) -> float:
    """Mix several Warp builtins over scalar + vec3 inputs."""
    n = wp.normalize(y)
    return x * wp.length(y) + wp.sin(x) + wp.dot(n, wp.vec3(1.0, 2.0, 3.0))


@wp.func
def square(x: float) -> float:
    """Square a scalar (the target of an explicit wp.grad() call)."""
    return x * x


@wp.func
def grad_square(x: float) -> float:
    """Differentiate square() explicitly inside the function body."""
    return wp.grad(square)(x)


@wp.func
def custom_square(x: float) -> float:
    """Square a scalar whose derivative is supplied by a @wp.func_grad callback."""
    return x * x


@wp.func_grad(custom_square)
def adj_custom_square(x: float, adj_ret: float):
    """Deliberately wrong derivative (10, not 2x) so the custom rule is visible in results."""
    wp.adjoint[x] += 10.0 * adj_ret


@wp.func
def grad_custom_square(x: float) -> float:
    """Differentiate custom_square() explicitly, exercising the deferred callback build."""
    return wp.grad(custom_square)(x)


@wp.func
def scale_add(x: float, y: float) -> float:
    """Return 2x + sin(y), a two-scalar fixture whose arguments can each be shared or per-invocation."""
    return x * 2.0 + wp.sin(y)


@wp.func
def gather_scale(data: wp.array[float], i: int, scale: float) -> float:
    """Read data[i] and scale it."""
    return data[i] * scale


@wp.func
def accumulate(hist: wp.array[float], i: int, v: float) -> float:
    """Atomically add v into hist[i], returning the previous value."""
    return wp.atomic_add(hist, i, v)


@wp.func
def make_params(x: float) -> BridgeParams:
    """Build a struct from a scalar; composes with apply_params across two tilelibs."""
    p = BridgeParams()
    p.scale = x
    p.offset = wp.vec3(x, 0.0, -x)
    p.exponent = 1
    return p


@wp.func
def apply_params(p: BridgeParams, y: wp.vec3) -> float:
    """Combine struct fields with a per-thread vector."""
    v = y + p.offset
    r = wp.length(v) * p.scale
    for _ in range(p.exponent):
        r = r * 0.5
    return r


@wp.func
def wide_params_sum(q: WideAlignParams, x: float) -> float:
    """Combine an 8-byte-aligned struct's fields with a per-invocation scalar."""
    return wp.float32(q.a) + wp.float32(q.b) + x


@wp.func
def read2d(m: wp.array2d[float], i: int, j: int) -> float:
    """Read one element of a 2d array."""
    return m[i, j]


@wp.func
def block_sum(x: float) -> float:
    """Sum x across the block using Warp tile builtins (needs shared memory)."""
    t = wp.tile(x)
    s = wp.tile_sum(t)
    return s[0]


@wp.func
def no_return(x: float):
    """Compute something without returning it (illegal for the bridge)."""
    _y = x * 2.0


@wp.func
def two_returns(x: float):
    """Return two values (unsupported by the tilelib surface)."""
    return x + 1.0, x * 2.0


@wp.func
def closest_point_u(mesh: wp.handle, x: wp.vec3) -> float:
    """Return the barycentric u of the mesh point closest to x (point query, thread-local traversal stack)."""
    q = wp.mesh_query_point(mesh, x, 1.0e6)
    return wp.where(q.result, q.u, 0.0)


@wp.func
def mesh_aabb_count(mesh: wp.uint64, x: wp.vec3) -> float:
    """Count triangles overlapping a box around x (block-wide traversal stack, served from scratch)."""
    h = wp.vec3(0.05, 0.05, 0.05)
    q = wp.mesh_query_aabb(mesh, x - h, x + h)
    n = float(0.0)
    face = int(0)
    while wp.mesh_query_aabb_next(q, face):
        n += 1.0
    return n


@wp.func
def tiled_mesh_aabb_count(mesh: wp.uint64, x: wp.vec3) -> float:
    """Count triangles overlapping a box around x with the block-cooperative tiled query."""
    h = wp.vec3(0.05, 0.05, 0.05)
    q = wp.tile_mesh_query_aabb(mesh, x - h, x + h)
    n = float(0.0)
    while wp.tile_query_valid(q):
        idx = wp.untile(wp.tile_mesh_query_aabb_next(q))
        if idx >= 0:
            n += 1.0
    return n


@wp.func
def repeated_mesh_aabb_count(mesh: wp.uint64, x: wp.vec3) -> float:
    """Count triangles overlapping eight growing boxes around x, constructing a fresh query each time."""
    n = float(0.0)
    for i in range(8):
        r = 0.01 * float(i + 1)
        h = wp.vec3(r, r, r)
        q = wp.mesh_query_aabb(mesh, x - h, x + h)
        face = int(0)
        while wp.mesh_query_aabb_next(q, face):
            n += 1.0
    return n


@wp.func
def bvh_sphere_count(bvh: wp.uint64, x: wp.vec3) -> float:
    """Count BVH boxes overlapping a sphere around x (BVH traversal stack, served from scratch)."""
    q = wp.bvh_query_sphere(bvh, x, 0.05)
    n = float(0.0)
    idx = int(0)
    while wp.bvh_query_next(q, idx):
        n += 1.0
    return n


@wp.func
def nested_mesh_bvh_count(mesh: wp.uint64, bvh: wp.uint64, x: wp.vec3) -> float:
    """Run a BVH query inside each mesh query hit; the two traversal stacks must not alias."""
    h = wp.vec3(0.05, 0.05, 0.05)
    n = float(0.0)
    face = int(0)
    q = wp.mesh_query_aabb(mesh, x - h, x + h)
    while wp.mesh_query_aabb_next(q, face):
        idx = int(0)
        qb = wp.bvh_query_aabb(bvh, x - h, x + h)
        while wp.bvh_query_next(qb, idx):
            n += 1.0
        n += 1000.0
    return n


@wp.func
def sin_probe(x: float) -> float:
    """Return sin(x); under fast_math NVRTC substitutes the approximate intrinsic."""
    return wp.sin(x)


@wp.func_native(
    snippet="""
#ifdef WP_CUTILE_TEST_SEVEN
    return WP_CUTILE_TEST_SEVEN;
#else
    return -1.0f;
#endif
"""
)
def native_seven() -> float:
    """Return the constant the module's extra CUDA preamble defines, or -1 without that preamble."""
    ...


@wp.func
def preamble_probe(x: float) -> float:
    """Depend on a wp.func_native that needs the module's extra_cuda_preamble."""
    return x + native_seven()


@wp.func
def cpp_keyword_params(auto: float, register: float) -> float:
    """Use C++ keywords as parameter names: legal in Warp, so the wrapper must accept them."""
    return auto * 2.0 + register


@wp.func
def colliding_names(x: wp.array[float], x_0: float) -> float:
    """Name a scalar like an array's first generated element so the call body's names must not collide."""
    return x[0] + x_0


@wp.func
def mat_vec(m: wp.mat33, v: wp.vec3) -> wp.vec3:
    """Multiply a broadcast matrix by a per-thread vector."""
    return m * v


@wp.func
def overloaded_probe(x: wp.float32) -> wp.float32:
    """First overload of a deliberately overloaded fixture."""
    return x


@wp.func
def overloaded_probe(x: wp.int32) -> wp.int32:
    """Second overload of a deliberately overloaded fixture."""
    return x


@wp.kernel
def blend_reference(xs: wp.array[float], ys: wp.array[wp.vec3], out: wp.array[float]):
    """Evaluate blend per thread."""
    tid = wp.tid()
    out[tid] = blend(xs[tid], ys[tid])


@wp.kernel
def sin_probe_reference(xs: wp.array[float], out: wp.array[float]):
    """Evaluate sin_probe per thread under the module's own build options."""
    tid = wp.tid()
    out[tid] = sin_probe(xs[tid])


@wp.kernel
def mesh_queries_reference(
    mesh: wp.uint64, pts: wp.array[wp.vec3], closest_u: wp.array[float], aabb_count: wp.array[float]
):
    """Evaluate closest_point_u and mesh_aabb_count per thread."""
    tid = wp.tid()
    closest_u[tid] = closest_point_u(mesh, pts[tid])
    aabb_count[tid] = mesh_aabb_count(mesh, pts[tid])


# Own module: this kernel holds a static mesh stack and a static BVH stack
# (32 ints per thread each), 64 KiB at the test module's default block_dim of
# 256, over the 48 KiB ptxas limit on most architectures. A unique module
# compiles only when launched, at the launch's block_dim (128 below).
@wp.kernel(module="unique")
def bvh_queries_reference(
    mesh: wp.uint64,
    bvh: wp.uint64,
    pts: wp.array[wp.vec3],
    sphere_count: wp.array[float],
    nested_count: wp.array[float],
    pair_count: wp.array[float],
):
    """Evaluate the BVH sphere count, the nested count, and the repeated mesh count plus sphere count per thread."""
    tid = wp.tid()
    sphere = bvh_sphere_count(bvh, pts[tid])
    sphere_count[tid] = sphere
    nested_count[tid] = nested_mesh_bvh_count(mesh, bvh, pts[tid])
    pair_count[tid] = repeated_mesh_aabb_count(mesh, pts[tid]) + sphere


def _pad_vec3(a: np.ndarray) -> np.ndarray:
    """Pad an (n, 3) float32 array to (n, VEC_PAD) with zeros."""
    p = np.zeros((a.shape[0], VEC_PAD), dtype=np.float32)
    p[:, :3] = a
    return p


def _dense_triangle_mesh(rng: np.random.Generator, device) -> wp.Mesh:
    """Build a dense soup of small triangles so mesh queries traverse deep into the BVH.

    Deep traversals fill the per-thread stacks, so an under-sized scratch slab
    faults instead of passing by luck (a handful of triangles never leaves the
    top levels).
    """
    ntri = 20000
    centers = rng.uniform(0.0, 1.0, (ntri, 3)).astype(np.float32)
    offsets = rng.uniform(-0.01, 0.01, (ntri, 3, 3)).astype(np.float32)
    points = (centers[:, None, :] + offsets).reshape(-1, 3)
    return wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=device),
        indices=wp.array(np.arange(ntri * 3, dtype=np.int32), dtype=int, device=device),
    )


def test_array_pointer_args(test, device):
    """Expand a contiguous CUDA array into (ptr, dims) and reject empty arrays."""
    arr = wp.zeros((4, 6), dtype=wp.float32, device=device)
    test.assertEqual(wpct.array_pointer_args(arr), (arr.ptr, 4, 6))
    empty = wp.zeros(0, dtype=wp.float32, device=device)
    with test.assertRaisesRegex(ValueError, "non-empty"):
        wpct.array_pointer_args(empty)


class TestCutileHelpers(unittest.TestCase):
    """Host-side operand, toolchain-lookup, and packing tests; no cuTile needed."""

    def test_encode_struct_roundtrip(self):
        """Round-trip a struct through encode_struct, verify pow2 padding, and reject non-struct inputs."""
        p = BridgeParams()
        p.scale = 3.0
        p.offset = wp.vec3(0.5, -1.0, 2.0)
        p.exponent = 2

        blob = wpct.encode_struct(p)
        self.assertEqual(blob.dtype, np.int32)
        # blob length is the struct's int32 word count rounded up to a power of
        # two: float + vec3 + int = 20 bytes = 5 words -> 8
        size = ctypes.sizeof(BridgeParams.ctype)
        self.assertEqual(size, 20)
        self.assertEqual(len(blob), 8)
        # float64 + int pads to 16 bytes = 4 words, already a power of two
        self.assertEqual(len(wpct.encode_struct(WideAlignParams())), 4)

        # leading bytes reproduce the ctypes representation, the padding is zero
        c = p.__ctype__()
        raw = ctypes.string_at(ctypes.addressof(c), size)
        blob_bytes = blob.view(np.uint8).tobytes()
        self.assertEqual(blob_bytes[:size], raw)
        self.assertEqual(blob_bytes[size:], b"\x00" * (len(blob_bytes) - size))
        # only Warp struct instances are accepted
        with self.assertRaisesRegex(TypeError, "Warp struct instance, got dict"):
            wpct.encode_struct({"scale": 1.0})

    def test_array_pointer_args_invalid_inputs(self):
        """Reject non-contiguous, CPU, and non-Warp inputs as array operands."""
        strided = wp.zeros(8, dtype=wp.float32, device="cpu")[::2]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            wpct.array_pointer_args(strided)

        host = wp.zeros(4, dtype=wp.float32, device="cpu")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            wpct.array_pointer_args(host)
        with self.assertRaisesRegex(TypeError, "expects a wp.array, got ndarray"):
            wpct.array_pointer_args(np.zeros(4, dtype=np.float32))

    def test_public_module_surface(self):
        """Import the public warp.experimental.cutile module and resolve its exports."""
        public = ["array_pointer_args", "encode_struct", "export_func"]
        self.assertEqual(sorted(wpct.__all__), public)
        # the API reference documents every non-underscore name, so nothing else may leak
        self.assertEqual(sorted(n for n in dir(wpct) if not n.startswith("_")), public)

    def test_bvh_stack_depth_matches_header(self):
        """Keep the scratch slab's per-thread depth equal to BVH_QUERY_STACK_SIZE in bvh.h."""
        header = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(_cutile.__file__))), "native", "bvh.h")
        with open(header) as f:
            match = re.search(r"^#define[ \t]+BVH_QUERY_STACK_SIZE[ \t]+\(?(\d+)\)?[ \t]*$", f.read(), re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(_cutile._BVH_QUERY_STACK_SIZE, int(match.group(1)))

    def test_pack_tilelib_failure_is_reported(self):
        """Surface a failing tileiras with its output and publish nothing."""

        def failing_run(cmd, **kwargs):
            """Fail like tileiras does, after creating its output file."""
            with open(cmd[cmd.index("-o") + 1], "wb") as f:
                f.write(b"partial")
            raise subprocess.CalledProcessError(2, cmd, "", "bad bytecode")

        with tempfile.TemporaryDirectory(prefix="wp_cutile_pack_test_") as tmp:
            ltoir_path = os.path.join(tmp, "impl.ltoir")
            with open(ltoir_path, "wb") as f:
                f.write(b"LTOIR")
            out_path = os.path.join(tmp, "fail.tilelib")
            with unittest.mock.patch("subprocess.run", side_effect=failing_run):
                with self.assertRaisesRegex(RuntimeError, r"(?s)tileiras failed .exit 2.*bad bytecode"):
                    _cutile._pack_tilelib(b"IFACE", b"IMPL", ltoir_path, out_path, "tileiras")
            self.assertFalse(os.path.exists(out_path))

    def test_tileiras_lookup_and_release(self):
        """Resolve tileiras from WARP_TILEIR_TOOLCHAIN before the pip package, with clear errors, and gate on its release."""
        exe = "tileiras.exe" if sys.platform == "win32" else "tileiras"  # Windows resolves by PATHEXT

        def executable(directory):
            """Create an executable tileiras stub in ``directory`` and return its path."""
            path = os.path.join(directory, exe)
            with open(path, "w") as f:
                f.write("#!/bin/sh\n")
            os.chmod(path, 0o700)
            return path

        with tempfile.TemporaryDirectory() as tmp:
            env_dir, pip_dir = os.path.join(tmp, "env"), os.path.join(tmp, "pip")
            os.makedirs(env_dir)
            os.makedirs(pip_dir)
            pip_tileiras = executable(pip_dir)
            # nothing configured, no pip package, nothing on PATH
            with (
                unittest.mock.patch.dict(os.environ, {"WARP_TILEIR_TOOLCHAIN": ""}),
                unittest.mock.patch.object(_cutile, "_pip_tileiras_dir", return_value=None),
                unittest.mock.patch("shutil.which", return_value=None),
                self.assertRaisesRegex(RuntimeError, "not found"),
            ):
                _cutile._find_tileiras()
            # a configured toolchain that holds no tileiras never falls through to another one
            with (
                unittest.mock.patch.dict(os.environ, {"WARP_TILEIR_TOOLCHAIN": env_dir}),
                unittest.mock.patch.object(_cutile, "_pip_tileiras_dir", return_value=pip_dir),
                self.assertRaisesRegex(RuntimeError, "does not contain a tileiras executable"),
            ):
                _cutile._find_tileiras()
            # an explicit toolchain wins over the pip package
            env_tileiras = executable(env_dir)
            with (
                unittest.mock.patch.dict(os.environ, {"WARP_TILEIR_TOOLCHAIN": env_dir}),
                unittest.mock.patch.object(_cutile, "_pip_tileiras_dir", return_value=pip_dir),
            ):
                self.assertTrue(os.path.samefile(_cutile._find_tileiras(), env_tileiras))
            with (
                unittest.mock.patch.dict(os.environ, {"WARP_TILEIR_TOOLCHAIN": ""}),
                unittest.mock.patch.object(_cutile, "_pip_tileiras_dir", return_value=pip_dir),
            ):
                self.assertTrue(os.path.samefile(_cutile._find_tileiras(), pip_tileiras))
            # the nvidia.cu13 namespace can span several site directories (e.g. a
            # `uv run --with` overlay in front of the project environment); the wheel
            # holding bin/tileiras need not be the first entry
            ns_front, ns_back = os.path.join(tmp, "ns_front", "cu13"), os.path.join(tmp, "ns_back", "cu13")
            os.makedirs(os.path.join(ns_front, "lib"))
            os.makedirs(os.path.join(ns_back, "bin"))
            ns_tileiras = executable(os.path.join(ns_back, "bin"))
            ns_module = types.ModuleType("nvidia.cu13")
            ns_module.__path__ = [ns_front, ns_back]
            ns_parent = types.ModuleType("nvidia")  # `import nvidia.cu13` binds through the parent's attribute
            ns_parent.__path__ = []
            ns_parent.cu13 = ns_module
            # a decoy on PATH: the package must win over it
            path_dir = os.path.join(tmp, "path")
            os.makedirs(path_dir)
            executable(path_dir)
            with (
                unittest.mock.patch.dict(sys.modules, {"nvidia": ns_parent, "nvidia.cu13": ns_module}),
                unittest.mock.patch.dict(os.environ, {"WARP_TILEIR_TOOLCHAIN": "", "PATH": path_dir}),
            ):
                self.assertTrue(os.path.samefile(_cutile._find_tileiras(), ns_tileiras))

        # the release gate reads the --version banner: older toolkits are rejected with both remedies
        banners = {
            "/tk/13.4/tileiras": "Cuda compilation tools, release 13.4, V13.4.92\n",
            "/tk/13.5/tileiras": "Cuda compilation tools, release 13.5, V13.5.37\n",
            "/tk/13.6/tileiras": "Cuda compilation tools, release 13.6, V13.6.10\n",
            "/tk/odd/tileiras": "tileiras: no release information\n",
        }
        _cutile._tileiras_version_banner.cache_clear()
        self.addCleanup(_cutile._tileiras_version_banner.cache_clear)
        with unittest.mock.patch(
            "subprocess.run", side_effect=lambda cmd, **kw: unittest.mock.Mock(stdout=banners[cmd[0]])
        ):
            # the preview bytecode is specific to one release: older and newer both fail
            with self.assertRaisesRegex(RuntimeError, "13.4 tileiras; the cuTile bridge requires the 13.5 tileiras"):
                _cutile._check_tileiras_release("/tk/13.4/tileiras")
            with self.assertRaisesRegex(RuntimeError, "13.6 tileiras; the cuTile bridge requires the 13.5 tileiras"):
                _cutile._check_tileiras_release("/tk/13.6/tileiras")
            # a binary whose release cannot be read is refused, not trusted
            with self.assertRaisesRegex(RuntimeError, "Could not determine the CUDA Toolkit release"):
                _cutile._check_tileiras_release("/tk/odd/tileiras")
            _cutile._check_tileiras_release("/tk/13.5/tileiras")


add_function_test(TestCutileHelpers, "test_array_pointer_args", test_array_pointer_args, devices=cuda_test_devices)


@unittest.skipUnless(cutile_available(), "cuda.tile and cuda.tile_preview are not installed")
class TestCutileDeclaration(unittest.TestCase):
    """Operand planning and toolchain-free error paths (needs cuda.tile for dtypes only)."""

    def test_export_func_validation_errors(self):
        """Reject bad declarations before any toolchain or GPU work happens."""
        # not a @wp.func
        with self.assertRaisesRegex(TypeError, "expects a @wp.func"):
            wpct.export_func(lambda x: x)
        # num_worker_warps must be a power of two between 1 and 32 (Tile IR
        # rejects non-power-of-two tile dimensions, and larger pinned tiles
        # would be undeclarable or unlaunchable), so e.g. 3 must fail up front
        for bad in (0, 3, 64):
            with self.assertRaisesRegex(ValueError, "power of two"):
                wpct.export_func(scale_add, num_worker_warps=bad)
        # the declaration targets a CUDA device, by default the current one;
        # this is checked before any toolchain lookup
        with unittest.mock.patch.object(_cutile, "_find_tileiras", side_effect=AssertionError("reached")):
            with wp.ScopedDevice("cpu"):
                with self.assertRaisesRegex(ValueError, "must target a CUDA device"):
                    wpct.export_func(scale_add)
            with self.assertRaisesRegex(ValueError, "must target a CUDA device"):
                wpct.export_func(scale_add, device="cpu")
        # unknown scope
        with self.assertRaisesRegex(ValueError, "Unknown scope"):
            wpct.export_func(scale_add, scope="warp")
        # functions must return at least one value
        with self.assertRaisesRegex(TypeError, "no return value"):
            wpct.export_func(no_return)
        # shared-memory-using functions are rejected at thread scope
        with self.assertRaisesRegex(ValueError, "shared memory"):
            wpct.export_func(block_sum, scope="thread")
        # so are block-cooperative tiled queries, whose static result buffers
        # and barriers assume the function owns the whole block
        with self.assertRaisesRegex(ValueError, "tiled query"):
            wpct.export_func(tiled_mesh_aabb_count, scope="thread")
        # overloaded functions have no single concrete signature to package
        with self.assertRaisesRegex(NotImplementedError, "overloads"):
            wpct.export_func(overloaded_probe)

        # parameter and return types the boundary cannot carry
        @wp.func
        def half_param(h: wp.float16) -> float:
            return float(h)

        @wp.func
        def half_vec_param(v: wp.vec3h) -> float:
            return 1.0

        @wp.func
        def indexed_param(ia: wp.indexedarray[float]) -> float:
            return ia[0]

        @wp.func
        def array_return(a: wp.array[float]) -> wp.array[float]:
            return a

        @wp.func
        def array_holder_return(x: float) -> ArrayHolder:
            return ArrayHolder()

        for bad, pattern in (
            (half_param, "Unsupported Warp type"),
            (half_vec_param, "Unsupported element type"),
            (indexed_param, "Unsupported array kind"),
            (array_return, "cannot return a wp.array"),
            (array_holder_return, "holds an array field"),
        ):
            with self.assertRaisesRegex(TypeError, pattern):
                wpct.export_func(bad)
        # builtins are wp.Function objects too, but have no Python body to package
        with self.assertRaisesRegex(TypeError, "'length' is a Warp builtin"):
            wpct.export_func(wp.length)

        # default values cannot be honored (every argument is passed at the call site)
        @wp.func
        def with_default(x: float, s: float = 2.0) -> float:
            return x * s

        with self.assertRaisesRegex(ValueError, "default values"):
            wpct.export_func(with_default)

        # non-ASCII parameter names would only fail deep inside NVRTC/MLIR
        @wp.func
        def unicode_param(α: float) -> float:  # noqa: PLC2401 (deliberate non-ASCII fixture)
            return α

        with self.assertRaisesRegex(ValueError, "not a plain ASCII identifier"):
            wpct.export_func(unicode_param)
        # multi-return functions are not implemented on the tilelib surface
        with self.assertRaisesRegex(NotImplementedError, "returns 2 values"):
            wpct.export_func(two_returns)


def _import_ct():
    """Import cuTile Python lazily so this module loads where it is absent."""
    import cuda.tile as ct  # noqa: PLC0415

    return ct


def _launch_cutile(device, kernel, args):
    """Launch a single-block cuTile kernel on ``device``'s Warp stream.

    No explicit synchronization: the kernel is stream-ordered with all
    subsequent Warp work, and every test reads results via ``.numpy()``,
    which synchronizes implicitly.
    """
    _import_ct().launch(wp.get_stream(device).cuda_stream, (1, 1, 1), kernel, args)


def test_scalar_and_vec3_call(test, device):
    """Match a call with per-invocation scalar and vec3 tiles, by position and by keyword, against wp.launch."""
    ct = _import_ct()
    wp_blend = wpct.export_func(blend, device=device)

    @ct.kernel(num_worker_warps=NUM_WARPS)
    def blend_cutile_kernel(xs, ys, out, out_by_keyword):
        x = ct.load(xs, (0,), shape=(NUM_THREADS,))
        y = ct.reshape(ct.load(ys, (0, 0), shape=(NUM_THREADS, VEC_PAD)), (NUM_THREADS * VEC_PAD,))
        ct.store(out, (0,), wp_blend(x, y))
        # arguments may also be passed by Warp parameter name, in any order
        ct.store(out_by_keyword, (0,), wp_blend(y=y, x=x))

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal(NUM_THREADS).astype(np.float32)
    y_np = rng.standard_normal((NUM_THREADS, 3)).astype(np.float32)

    xs = wp.array(x_np, dtype=wp.float32, device=device)
    ys_padded = wp.array(_pad_vec3(y_np), dtype=wp.float32, device=device)
    out = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    out_by_keyword = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    _launch_cutile(device, blend_cutile_kernel, (xs, ys_padded, out, out_by_keyword))

    ys_vec3 = wp.array(y_np, dtype=wp.vec3, device=device)
    expected = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    wp.launch(blend_reference, dim=NUM_THREADS, inputs=[xs, ys_vec3, expected], device=device)
    np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(out_by_keyword.numpy(), out.numpy())


def test_explicit_gradient_inside_function(test, device):
    """Keep wp.grad() derivatives, including @wp.func_grad callbacks, correct inside a forward-only foreign function."""
    ct = _import_ct()
    wp_grad_square = wpct.export_func(grad_square, device=device)
    # a @wp.func_grad callback is a deferred build dependency; the
    # standalone build must include it like a module build does
    wp_grad_custom_square = wpct.export_func(grad_custom_square, device=device)
    x_np = np.arange(NUM_THREADS, dtype=np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)
    out = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    out_custom = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)

    @ct.kernel(num_worker_warps=NUM_WARPS)
    def grad_square_cutile_kernel(xs, out, out_custom):
        x = ct.load(xs, (0,), shape=(NUM_THREADS,))
        ct.store(out, (0,), wp_grad_square(x))
        ct.store(out_custom, (0,), wp_grad_custom_square(x))

    _launch_cutile(device, grad_square_cutile_kernel, (xs, out, out_custom))
    # d/dx x^2 = 2x: the derivative code must be emitted even though no
    # backward kernel uses the function
    np.testing.assert_allclose(out.numpy(), 2.0 * x_np)
    # the custom rule (deliberately 10, not 2x) replaces the automatic one
    np.testing.assert_allclose(out_custom.numpy(), 10.0)


def test_call_site_mistakes_reported_at_trace(test, device):
    """Report wrong operand rank, length, element type, plain-vs-tile mix-ups, and bad array tuples at trace time."""
    ct = _import_ct()
    wp_blend = wpct.export_func(blend, device=device)
    wp_gather_scale = wpct.export_func(gather_scale, device=device)
    xs = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    ys = wp.zeros((NUM_THREADS, VEC_PAD), dtype=wp.float32, device=device)
    ids = wp.zeros(NUM_THREADS, dtype=wp.int32, device=device)
    out = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)

    def assert_trace_error(pattern, kernel, args=(xs, ys, out)):
        """Assert that tracing ``kernel`` fails with a message matching ``pattern``."""
        with test.assertRaisesRegex(ct.TileError, pattern):
            _launch_cutile(device, kernel, args)

    @ct.kernel(num_worker_warps=NUM_WARPS)
    def unpadded_vec3_cutile_kernel(xs, ys, out):
        x = ct.load(xs, (0,), shape=(NUM_THREADS,))
        ct.store(out, (0,), wp_blend(x, x))

    assert_trace_error(r"'y'.*\(512,\)", unpadded_vec3_cutile_kernel)

    @ct.kernel(num_worker_warps=NUM_WARPS)
    def number_for_vec3_cutile_kernel(xs, ys, out):
        x = ct.load(xs, (0,), shape=(NUM_THREADS,))
        ct.store(out, (0,), wp_blend(x, 1.0))

    assert_trace_error(r"'y'.*TILE \* 4.*got 1\.0", number_for_vec3_cutile_kernel)

    @ct.kernel(num_worker_warps=NUM_WARPS)
    def undeclared_length_cutile_kernel(xs, ys, out):
        x = ct.load(xs, (0,), shape=(16,))
        y = ct.reshape(ct.load(ys, (0, 0), shape=(16, VEC_PAD)), (16 * VEC_PAD,))
        ct.store(out, (0,), wp_blend(x, y))

    assert_trace_error(r"tile length 16.*'x'.*not a declared length", undeclared_length_cutile_kernel)

    @ct.kernel(num_worker_warps=NUM_WARPS)
    def rank2_vec3_cutile_kernel(xs, ys, out):
        x = ct.load(xs, (0,), shape=(NUM_THREADS,))
        ct.store(out, (0,), wp_blend(x, ct.load(ys, (0, 0), shape=(NUM_THREADS, VEC_PAD))))

    assert_trace_error(r"'y'.*rank-1", rank2_vec3_cutile_kernel)

    @ct.kernel(num_worker_warps=NUM_WARPS)
    def int_for_float_cutile_kernel(ys, ids, out):
        y = ct.reshape(ct.load(ys, (0, 0), shape=(NUM_THREADS, VEC_PAD)), (NUM_THREADS * VEC_PAD,))
        ct.store(out, (0,), wp_blend(ct.load(ids, (0,), shape=(NUM_THREADS,)), y))

    assert_trace_error(r"'x'.*float32.*int32", int_for_float_cutile_kernel, (ys, ids, out))

    @ct.kernel(num_worker_warps=NUM_WARPS)
    def missing_argument_cutile_kernel(xs, ys, out):
        ct.store(out, (0,), wp_blend(ct.load(xs, (0,), shape=(NUM_THREADS,))))

    assert_trace_error(r"missing.*'y'", missing_argument_cutile_kernel)

    # an array tuple with an extra value fails instead of silently dropping it
    @ct.kernel(num_worker_warps=NUM_WARPS)
    def long_array_tuple_cutile_kernel(ptr: ct.Constant[int], length: ct.Constant[int], ids, out):
        i = ct.load(ids, (0,), shape=(NUM_THREADS,))
        ct.store(out, (0,), wp_gather_scale((ptr, length, 999), i, 2.5))

    assert_trace_error(
        r"'data'.*tuple of 2 plain values",
        long_array_tuple_cutile_kernel,
        (*wpct.array_pointer_args(xs), ids, out),
    )

    # a non-integer extent fails instead of being truncated by the int32 materialization
    @ct.kernel(num_worker_warps=NUM_WARPS)
    def float_array_extent_cutile_kernel(ptr: ct.Constant[int], ids, out):
        i = ct.load(ids, (0,), shape=(NUM_THREADS,))
        ct.store(out, (0,), wp_gather_scale((ptr, 2.5), i, 2.5))

    assert_trace_error(
        r"element 1 of parameter 'data'.*integer.*got 2\.5", float_array_extent_cutile_kernel, (xs.ptr, ids, out)
    )


def test_thread_scope_tile_lengths(test, device):
    """Serve call sites of different tile lengths in one kernel from one thread-scope declaration."""
    ct = _import_ct()
    wp_scale_add = wpct.export_func(scale_add, device=device)
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal(1024).astype(np.float32)
    y_np = rng.standard_normal(1024).astype(np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)
    ys = wp.array(y_np, dtype=wp.float32, device=device)
    outs = {length: wp.zeros(length, dtype=wp.float32, device=device) for length in (32, 64, 256, 1024)}

    # each call site dispatches to the declared length its tiles carry, and
    # cuTile schedules the invocations onto however many workers the kernel
    # has: fewer and more invocations than the 128 workers, up to both ends of
    # the declared range
    @ct.kernel(num_worker_warps=NUM_WARPS)
    def mixed_lengths_cutile_kernel(xs, ys, out_32, out_64, out_256, out_1024):
        ct.store(out_32, (0,), wp_scale_add(ct.load(xs, (0,), shape=(32,)), ct.load(ys, (0,), shape=(32,))))
        ct.store(out_64, (0,), wp_scale_add(ct.load(xs, (0,), shape=(64,)), ct.load(ys, (0,), shape=(64,))))
        ct.store(out_256, (0,), wp_scale_add(ct.load(xs, (0,), shape=(256,)), ct.load(ys, (0,), shape=(256,))))
        ct.store(out_1024, (0,), wp_scale_add(ct.load(xs, (0,), shape=(1024,)), ct.load(ys, (0,), shape=(1024,))))

    _launch_cutile(device, mixed_lengths_cutile_kernel, (xs, ys, *outs.values()))
    expected = x_np * 2.0 + np.sin(y_np)
    for length, out in outs.items():
        np.testing.assert_allclose(out.numpy(), expected[:length], rtol=1e-6, atol=1e-6)


def test_mesh_query_scratch_stack(test, device):
    """Run point and iterator mesh queries at any tile length and block size, hint-independent."""
    ct = _import_ct()
    wp_closest_point_u = wpct.export_func(closest_point_u, device=device)
    wp_mesh_aabb_count = wpct.export_func(mesh_aabb_count, num_worker_warps=NUM_WARPS, device=device)
    # a hint above 8 warps grows the slab, so it must key the identity:
    # reusing the smaller slab would trap in the 16-warp block below
    wp_mesh_aabb_count_16_warps = wpct.export_func(mesh_aabb_count, num_worker_warps=16, device=device)
    test.assertNotEqual(wp_mesh_aabb_count_16_warps.version, wp_mesh_aabb_count.version)
    # the scratch slab costs 128 bytes per worker thread per query family
    # (documented): a 32-warp block needs 128 KiB, more than some GPUs
    # allow per block, and that is reported at declaration, not as a fault
    if 128 * 1024 > device.max_shared_memory_per_block:
        with test.assertRaisesRegex(ValueError, "compiler scratch .* shared memory per block"):
            wpct.export_func(mesh_aabb_count, num_worker_warps=32, device=device)

    rng = np.random.default_rng(13)
    mesh = _dense_triangle_mesh(rng, device)
    pts_np = rng.uniform(0.0, 1.0, (512, 3)).astype(np.float32)
    pts_padded = wp.array(_pad_vec3(pts_np), dtype=wp.float32, device=device)
    pts_vec3 = wp.array(pts_np, dtype=wp.vec3, device=device)
    expected_u = wp.zeros(512, dtype=wp.float32, device=device)
    expected_count = wp.zeros(512, dtype=wp.float32, device=device)
    wp.launch(
        mesh_queries_reference,
        dim=512,
        inputs=[mesh.id, pts_vec3, expected_u, expected_count],
        device=device,
    )

    # the point query keeps its traversal stack per thread, so it runs at any
    # tile length, and the mesh id may also arrive per invocation as an int64
    # tile (Tile IR has no unsigned dtypes; the 64 bits pass through unchanged)
    ids = wp.array(np.full(256, mesh.id, dtype=np.uint64).view(np.int64), dtype=wp.int64, device=device)
    out_u = wp.zeros(256, dtype=wp.float32, device=device)
    out_u_by_id = wp.zeros(256, dtype=wp.float32, device=device)
    out_u_32 = wp.zeros(32, dtype=wp.float32, device=device)
    out_count = wp.zeros(256, dtype=wp.float32, device=device)

    @ct.kernel(num_worker_warps=8)
    def mesh_queries_cutile_kernel(mesh_id: ct.Constant[int], ids, pts, out_u, out_u_by_id, out_u_32, out_count):
        p = ct.reshape(ct.load(pts, (0, 0), shape=(256, VEC_PAD)), (256 * VEC_PAD,))
        ct.store(out_u, (0,), wp_closest_point_u(mesh_id, p))
        ct.store(out_u_by_id, (0,), wp_closest_point_u(ct.load(ids, (0,), shape=(256,)), p))
        p_32 = ct.reshape(ct.load(pts, (0, 0), shape=(32, VEC_PAD)), (32 * VEC_PAD,))
        ct.store(out_u_32, (0,), wp_closest_point_u(mesh_id, p_32))
        # every thread of the block traverses at once, so the iterator query's
        # slab must cover the calling block: a 4-warp hint still covers 8
        # warps, the slab's minimum
        ct.store(out_count, (0,), wp_mesh_aabb_count(mesh_id, p))

    _launch_cutile(
        device, mesh_queries_cutile_kernel, (mesh.id, ids, pts_padded, out_u, out_u_by_id, out_u_32, out_count)
    )
    np.testing.assert_allclose(out_u.numpy(), expected_u.numpy()[:256], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out_u_by_id.numpy(), expected_u.numpy()[:256], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out_u_32.numpy(), expected_u.numpy()[:32], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(out_count.numpy(), expected_count.numpy()[:256])

    def check_aabb_count(wp_func, warps):
        """Run the iterator query through ``wp_func`` on a ``warps``-warp block and match the wp.launch reference."""
        threads = warps * 32

        @ct.kernel(num_worker_warps=warps)
        def aabb_count_cutile_kernel(mesh_id: ct.Constant[int], pts, out):
            p = ct.reshape(ct.load(pts, (0, 0), shape=(threads, VEC_PAD)), (threads * VEC_PAD,))
            ct.store(out, (0,), wp_func(mesh_id, p))

        out = wp.zeros(threads, dtype=wp.float32, device=device)
        _launch_cutile(device, aabb_count_cutile_kernel, (mesh.id, pts_padded, out))
        np.testing.assert_array_equal(out.numpy(), expected_count.numpy()[:threads])

    # larger blocks are declared through num_worker_warps ...
    check_aabb_count(wp_mesh_aabb_count_16_warps, 16)
    # ... and at block scope the slab is sized to the pinned block exactly
    check_aabb_count(wpct.export_func(mesh_aabb_count, num_worker_warps=8, scope="block", device=device), 8)


def test_bvh_and_nested_query_scratch_stacks(test, device):
    """Serve BVH, nested mesh+BVH, and back-to-back mesh and BVH queries from separate scratch slabs on an 8-warp block."""
    ct = _import_ct()
    rng = np.random.default_rng(29)
    # a dense mesh and a dense set of small boxes so both traversals go deep
    mesh = _dense_triangle_mesh(rng, device)
    box_centers = rng.uniform(0.0, 1.0, (20000, 3)).astype(np.float32)
    bvh = wp.Bvh(
        lowers=wp.array(box_centers - 0.01, dtype=wp.vec3, device=device),
        uppers=wp.array(box_centers + 0.01, dtype=wp.vec3, device=device),
    )
    pts_np = rng.uniform(0.0, 1.0, (256, 3)).astype(np.float32)
    pts_padded = wp.array(_pad_vec3(pts_np), dtype=wp.float32, device=device)
    pts_vec3 = wp.array(pts_np, dtype=wp.vec3, device=device)

    # every BVH query kind gets the scratch slab, sized for the block
    wp_bvh_sphere_count = wpct.export_func(bvh_sphere_count, num_worker_warps=NUM_WARPS, device=device)
    wp_nested_mesh_bvh_count = wpct.export_func(nested_mesh_bvh_count, num_worker_warps=NUM_WARPS, device=device)
    wp_repeated_mesh_aabb_count = wpct.export_func(repeated_mesh_aabb_count, num_worker_warps=NUM_WARPS, device=device)

    @ct.kernel(num_worker_warps=8)
    def bvh_queries_cutile_kernel(
        mesh_id: ct.Constant[int], bvh_id: ct.Constant[int], pts, out_sphere, out_nested, out_pair
    ):
        p = ct.reshape(ct.load(pts, (0, 0), shape=(256, VEC_PAD)), (256 * VEC_PAD,))
        # a BVH query nested in a mesh query: the two slabs are laid out
        # back to back, so the inner traversal cannot clobber the outer one
        ct.store(out_nested, (0,), wp_nested_mesh_bvh_count(mesh_id, bvh_id, p))
        # two foreign functions of different query families back to back, with
        # no barrier between the calls: each wrapper binds only the slab it
        # owns, so a warp still re-reading the mesh slab pointer inside the
        # repeated mesh query never observes the BVH-only call's binding
        aabb_count = wp_repeated_mesh_aabb_count(mesh_id, p)
        sphere_count = wp_bvh_sphere_count(bvh_id, p)
        ct.store(out_sphere, (0,), sphere_count)
        ct.store(out_pair, (0,), aabb_count + sphere_count)

    expected_sphere = wp.zeros(256, dtype=wp.float32, device=device)
    expected_nested = wp.zeros(256, dtype=wp.float32, device=device)
    expected_pair = wp.zeros(256, dtype=wp.float32, device=device)
    # 128-thread blocks keep the reference's two static stacks at 32 KiB
    # (see the kernel's module="unique" note)
    wp.launch(
        bvh_queries_reference,
        dim=256,
        inputs=[mesh.id, bvh.id, pts_vec3, expected_sphere, expected_nested, expected_pair],
        device=device,
        block_dim=128,
    )
    test.assertGreater(expected_sphere.numpy().max(), 0.0)
    test.assertGreater(expected_nested.numpy().max(), 1000.0)

    # the race depends on warp timing, so repeat the launch and check every result
    out_sphere = wp.zeros(256, dtype=wp.float32, device=device)
    out_nested = wp.zeros(256, dtype=wp.float32, device=device)
    out_pair = wp.zeros(256, dtype=wp.float32, device=device)
    for _ in range(4):
        for out in (out_sphere, out_nested, out_pair):
            out.zero_()
        _launch_cutile(
            device, bvh_queries_cutile_kernel, (mesh.id, bvh.id, pts_padded, out_sphere, out_nested, out_pair)
        )
        np.testing.assert_array_equal(out_sphere.numpy(), expected_sphere.numpy())
        np.testing.assert_array_equal(out_nested.numpy(), expected_nested.numpy())
        np.testing.assert_array_equal(out_pair.numpy(), expected_pair.numpy())


def test_module_build_options_honored(test, device):
    """Compile the foreign function with its module's fast_math setting, key the cache on it, and warn on fuse_fp=False."""
    if wp.config.llvm_cuda:
        # the wp.launch reference is built by Clang, which ignores fast_math; the foreign build is NVRTC
        test.skipTest("fast_math comparison needs the NVRTC reference path")
    ct = _import_ct()
    module = sin_probe.module
    # large arguments expose the approximate __sinf that fast_math selects
    x_np = np.linspace(1.0e3, 1.0e4, NUM_THREADS, dtype=np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)

    def run(fast_math):
        """Declare and call sin_probe under the given fast_math option; return (version, result, wp.launch reference)."""
        wp.set_module_options({"fast_math": fast_math}, module=module)
        wp_sin_probe = wpct.export_func(sin_probe, device=device)

        # one kernel per build: two builds linked into one kernel share one math mode
        @ct.kernel(num_worker_warps=NUM_WARPS)
        def sin_probe_cutile_kernel(xs, out):
            ct.store(out, (0,), wp_sin_probe(ct.load(xs, (0,), shape=(NUM_THREADS,))))

        out = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
        _launch_cutile(device, sin_probe_cutile_kernel, (xs, out))
        expected = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
        wp.launch(sin_probe_reference, dim=NUM_THREADS, inputs=[xs, expected], device=device)
        return wp_sin_probe.version, out.numpy(), expected.numpy()

    original = module.options.get("fast_math", False)
    try:
        version_fast, out_fast, expected_fast = run(True)
        version_precise, out_precise, expected_precise = run(False)
    finally:
        wp.set_module_options({"fast_math": original}, module=module)
    # the two builds must differ on these inputs, or the comparisons below prove nothing
    test.assertFalse(np.array_equal(expected_fast, expected_precise))
    # each setting matches wp.launch of the same function under the same option,
    # and the tilelib identity tracks the option
    np.testing.assert_array_equal(out_fast, expected_fast)
    np.testing.assert_array_equal(out_precise, expected_precise)
    test.assertNotEqual(version_fast, version_precise)

    # fuse_fp=False cannot be guaranteed: contraction is decided when
    # tileiras links the kernel, so declaring under it warns (Warp's logger
    # bypasses warnings.catch_warnings(record=True), so capture stderr)
    had_fuse_fp = "fuse_fp" in module.options
    original_fuse_fp = module.options.get("fuse_fp")
    try:
        wp.set_module_options({"fuse_fp": False}, module=module)
        # the warning is emitted once per process; reset that so every device sees it
        with (
            unittest.mock.patch.object(warp._src.logger, "_warnings_seen", set()),
            warnings.catch_warnings(),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            warnings.simplefilter("always")
            wpct.export_func(sin_probe, device=device)
    finally:
        if had_fuse_fp:
            wp.set_module_options({"fuse_fp": original_fuse_fp}, module=module)
        else:
            module.options.pop("fuse_fp", None)
    test.assertIn("fuse_fp=False is not honored", stderr.getvalue())


def test_module_preamble_and_build_dependencies(test, device):
    """Compile with the module's extra_cuda_preamble and rebuild when a registered build-dependency header changes."""
    ct = _import_ct()
    module = preamble_probe.module
    x_np = np.arange(NUM_THREADS, dtype=np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)

    # a wp.func_native dependency reads a define from a header that only the
    # module's preamble includes, and the header is a registered build dependency
    tmp = tempfile.mkdtemp(prefix="wp_cutile_dep_test_")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    header = os.path.join(tmp, "wp_cutile_dep.h")

    def export_with_header(value):
        """Write the header with ``value``, re-register it, and declare preamble_probe against it."""
        with open(header, "w") as f:
            f.write(f"#define WP_CUTILE_TEST_SEVEN {value}f\n")
        # re-setting the option re-reads the dependency's content hash
        wp.set_module_options(
            {
                "extra_build_options": wp.ModuleBuildOptions(
                    extra_cuda_include_dirs=[tmp],
                    extra_cuda_preamble='#include "wp_cutile_dep.h"\n',
                    extra_build_dependencies=[header],
                )
            },
            module=module,
        )
        return wpct.export_func(preamble_probe, device=device)

    original = module.options.get("extra_build_options")
    try:
        wp_preamble_probe_7 = export_with_header(7.0)
        wp_preamble_probe_11 = export_with_header(11.0)
    finally:
        wp.set_module_options({"extra_build_options": original}, module=module)
    # the same source with a changed header must not reuse the old tilelib
    test.assertNotEqual(wp_preamble_probe_11.version, wp_preamble_probe_7.version)

    @ct.kernel(num_worker_warps=NUM_WARPS)
    def preamble_probe_cutile_kernel(xs, out_7, out_11):
        x = ct.load(xs, (0,), shape=(NUM_THREADS,))
        ct.store(out_7, (0,), wp_preamble_probe_7(x))
        ct.store(out_11, (0,), wp_preamble_probe_11(x))

    out_7 = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    out_11 = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    _launch_cutile(device, preamble_probe_cutile_kernel, (xs, out_7, out_11))
    np.testing.assert_array_equal(out_7.numpy(), x_np + 7.0)
    np.testing.assert_array_equal(out_11.numpy(), x_np + 11.0)


def test_struct_operands(test, device):
    """Pass a struct from one tilelib to another, broadcast struct blobs to every invocation, and keep field alignment."""
    ct = _import_ct()
    wp_make_params = wpct.export_func(make_params, device=device)
    wp_apply_params = wpct.export_func(apply_params, device=device)
    wp_wide_params_sum = wpct.export_func(wide_params_sum, device=device)

    rng = np.random.default_rng(11)
    x_np = rng.uniform(0.5, 2.0, NUM_THREADS).astype(np.float32)
    y_np = rng.standard_normal((NUM_THREADS, 3)).astype(np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)
    ys_padded = wp.array(_pad_vec3(y_np), dtype=wp.float32, device=device)

    params = BridgeParams()
    params.scale = 3.0
    params.offset = wp.vec3(0.5, -1.0, 2.0)
    params.exponent = 2
    params_blob = wp.array(wpct.encode_struct(params), dtype=wp.int32, device=device)
    wide_params = WideAlignParams()
    wide_params.a = 1.5
    wide_params.b = 7
    wide_params_blob = wp.array(wpct.encode_struct(wide_params), dtype=wp.int32, device=device)
    params_words = params_blob.shape[0]
    wide_params_words = wide_params_blob.shape[0]

    @ct.kernel(num_worker_warps=NUM_WARPS)
    def struct_operands_cutile_kernel(xs, ys, params_blob, wide_params_blob, out_returned, out_shared, out_wide):
        x = ct.load(xs, (0,), shape=(NUM_THREADS,))
        y = ct.reshape(ct.load(ys, (0, 0), shape=(NUM_THREADS, VEC_PAD)), (NUM_THREADS * VEC_PAD,))
        # both tilelibs instantiate BridgeParams and its helper functions, which
        # must link into one kernel without multiply-defined symbols
        ct.store(out_returned, (0,), wp_apply_params(wp_make_params(x), y))
        # one int32 blob tile holds a struct shared by every invocation
        ct.store(out_shared, (0,), wp_apply_params(ct.load(params_blob, (0,), shape=(params_words,)), y))
        # a struct with an 8-byte-aligned field: its blob is declared with a
        # two-word alignment and the field offsets survive the trip
        wide = ct.load(wide_params_blob, (0,), shape=(wide_params_words,))
        ct.store(out_wide, (0,), wp_wide_params_sum(wide, x))

    out_returned = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    out_shared = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    out_wide = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    _launch_cutile(
        device,
        struct_operands_cutile_kernel,
        (xs, ys_padded, params_blob, wide_params_blob, out_returned, out_shared, out_wide),
    )
    # apply_params(make_params(x), y) = length(y + (x, 0, -x)) * x * 0.5
    expected_returned = np.linalg.norm(y_np + np.stack([x_np, np.zeros_like(x_np), -x_np], axis=1), axis=1) * x_np * 0.5
    np.testing.assert_allclose(out_returned.numpy(), expected_returned.astype(np.float32), rtol=1e-5, atol=1e-6)
    # apply_params(params, y) = length(y + offset) * scale * 0.5^exponent
    expected_shared = np.linalg.norm(y_np + np.array([0.5, -1.0, 2.0], dtype=np.float32), axis=1) * 3.0 * 0.25
    np.testing.assert_allclose(out_shared.numpy(), expected_shared.astype(np.float32), rtol=1e-5)
    np.testing.assert_allclose(out_wide.numpy(), x_np + 1.5 + 7.0, rtol=1e-6, atol=1e-6)


def test_parameter_name_clashes(test, device):
    """Call functions whose parameter names are C++ keywords or match a generated operand name."""
    ct = _import_ct()
    wp_cpp_keyword_params = wpct.export_func(cpp_keyword_params, device=device)
    # an array parameter and a scalar named after one of its generated elements coexist
    wp_colliding_names = wpct.export_func(colliding_names, device=device)
    rng = np.random.default_rng(5)
    x_np = rng.standard_normal(NUM_THREADS).astype(np.float32)
    y_np = rng.standard_normal(NUM_THREADS).astype(np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)
    ys = wp.array(y_np, dtype=wp.float32, device=device)

    @ct.kernel(num_worker_warps=NUM_WARPS)
    def name_clash_cutile_kernel(
        xs_ptr: ct.Constant[int], xs_len: ct.Constant[int], xs, ys, out_cpp_keywords, out_colliding
    ):
        x = ct.load(xs, (0,), shape=(NUM_THREADS,))
        y = ct.load(ys, (0,), shape=(NUM_THREADS,))
        ct.store(out_cpp_keywords, (0,), wp_cpp_keyword_params(x, y))
        ct.store(out_colliding, (0,), wp_colliding_names((xs_ptr, xs_len), y))

    out_cpp_keywords = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    out_colliding = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    _launch_cutile(
        device, name_clash_cutile_kernel, (*wpct.array_pointer_args(xs), xs, ys, out_cpp_keywords, out_colliding)
    )
    np.testing.assert_allclose(out_cpp_keywords.numpy(), x_np * 2.0 + y_np, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_colliding.numpy(), x_np[0] + y_np, rtol=1e-6, atol=1e-6)


def test_matrix_operand_and_vec3_return(test, device):
    """Broadcast a matrix to every invocation and return padded vec3 values at several tile lengths."""
    ct = _import_ct()
    rng = np.random.default_rng(7)
    wp_mat_vec = wpct.export_func(mat_vec, device=device)

    m_np = rng.standard_normal((3, 3)).astype(np.float32)
    m_padded = np.zeros(16, dtype=np.float32)
    m_padded[:9] = m_np.reshape(-1)  # row-major, padded to the pow2 tile width
    ms = wp.array(m_padded, dtype=wp.float32, device=device)
    v_np = rng.standard_normal((512, 3)).astype(np.float32)
    vs_padded = wp.array(_pad_vec3(v_np), dtype=wp.float32, device=device)
    # poison the destinations so the zero-filled pad lanes are proven
    outs = {length: wp.full(length * VEC_PAD, 777.0, dtype=wp.float32, device=device) for length in (64, 128, 512)}

    # the lengths other than the 128-invocation fallback exercise the width-4
    # dispatch constants (compare against length * VEC_PAD, not length)
    @ct.kernel(num_worker_warps=NUM_WARPS)
    def mat_vec_cutile_kernel(ms, vs, out_64, out_128, out_512):
        m = ct.load(ms, (0,), shape=(16,))
        ct.store(out_64, (0,), wp_mat_vec(m, ct.reshape(ct.load(vs, (0, 0), shape=(64, VEC_PAD)), (64 * VEC_PAD,))))
        ct.store(out_128, (0,), wp_mat_vec(m, ct.reshape(ct.load(vs, (0, 0), shape=(128, VEC_PAD)), (128 * VEC_PAD,))))
        ct.store(out_512, (0,), wp_mat_vec(m, ct.reshape(ct.load(vs, (0, 0), shape=(512, VEC_PAD)), (512 * VEC_PAD,))))

    _launch_cutile(device, mat_vec_cutile_kernel, (ms, vs_padded, *outs.values()))
    for length, out in outs.items():
        got = out.numpy().reshape(length, VEC_PAD)
        np.testing.assert_allclose(got[:, :3], v_np[:length] @ m_np.T, rtol=1e-5, atol=1e-5)
        # padding lanes of the returned vec3 are zero-filled by the wrapper
        np.testing.assert_allclose(got[:, 3], 0.0)


def test_array_operands(test, device):
    """Gather from a 1d array, mutate an array atomically, and index a 2d array through foreign calls."""
    ct = _import_ct()
    rng = np.random.default_rng(11)
    wp_gather_scale = wpct.export_func(gather_scale, device=device)
    wp_accumulate = wpct.export_func(accumulate, device=device)
    wp_read2d = wpct.export_func(read2d, device=device)

    data_np = rng.standard_normal(1000).astype(np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    ids_np = rng.integers(0, 1000, NUM_THREADS).astype(np.int32)
    ids = wp.array(ids_np, dtype=wp.int32, device=device)
    hist = wp.zeros(16, dtype=wp.float32, device=device)
    bins_np = rng.integers(0, 16, NUM_THREADS).astype(np.int32)
    bins = wp.array(bins_np, dtype=wp.int32, device=device)
    m_np = rng.standard_normal((32, 8)).astype(np.float32)
    m = wp.array(m_np, dtype=wp.float32, device=device)
    rows_np = rng.integers(0, 32, NUM_THREADS).astype(np.int32)
    cols_np = rng.integers(0, 8, NUM_THREADS).astype(np.int32)
    rows = wp.array(rows_np, dtype=wp.int32, device=device)
    cols = wp.array(cols_np, dtype=wp.int32, device=device)

    @ct.kernel(num_worker_warps=NUM_WARPS)
    def array_operands_cutile_kernel(
        data_ptr: ct.Constant[int],
        data_len: ct.Constant[int],
        ids,
        out_gather,
        hist_ptr: ct.Constant[int],
        hist_len: ct.Constant[int],
        bins,
        m_ptr: ct.Constant[int],
        m_rows: ct.Constant[int],
        m_cols: ct.Constant[int],
        rows,
        cols,
        out_2d,
    ):
        # an array parameter takes its (address, length) tuple, and plain
        # numbers materialize as broadcast tiles automatically
        i = ct.load(ids, (0,), shape=(NUM_THREADS,))
        ct.store(out_gather, (0,), wp_gather_scale((data_ptr, data_len), i, 2.5))
        # the call's side effect survives when its result is discarded: foreign
        # calls carry a memory effect, so the unused-result call is not
        # eliminated; explicit rank-0 tiles also pass through unchanged inside
        # the tuple (raw pointers travel as int64: Tile IR dtypes are signed only)
        hist_ptr_tile = ct.full((), hist_ptr, dtype=ct.int64)
        hist_len_tile = ct.full((), hist_len, dtype=ct.int32)
        b = ct.load(bins, (0,), shape=(NUM_THREADS,))
        wp_accumulate((hist_ptr_tile, hist_len_tile), b, ct.full((NUM_THREADS,), 1.0, dtype=ct.float32))
        # a 2d array expands to its address and two extents
        r = ct.load(rows, (0,), shape=(NUM_THREADS,))
        c = ct.load(cols, (0,), shape=(NUM_THREADS,))
        ct.store(out_2d, (0,), wp_read2d((m_ptr, m_rows, m_cols), r, c))

    out_gather = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    out_2d = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    _launch_cutile(
        device,
        array_operands_cutile_kernel,
        (
            *wpct.array_pointer_args(data),
            ids,
            out_gather,
            *wpct.array_pointer_args(hist),
            bins,
            *wpct.array_pointer_args(m),
            rows,
            cols,
            out_2d,
        ),
    )
    np.testing.assert_allclose(out_gather.numpy(), data_np[ids_np] * np.float32(2.5), rtol=1e-6)
    np.testing.assert_allclose(hist.numpy(), np.bincount(bins_np, minlength=16).astype(np.float32))
    np.testing.assert_allclose(out_2d.numpy(), m_np[rows_np, cols_np], rtol=1e-6)


def test_block_scope_reduction(test, device):
    """Run a block-cooperative reduction at block scope and reject a mismatched calling block at compile time."""
    ct = _import_ct()
    wp_block_sum = wpct.export_func(block_sum, num_worker_warps=NUM_WARPS, scope="block", device=device)
    # block scope pins the declaration: the block size is baked in
    test.assertEqual(wp_block_sum.lengths, (NUM_THREADS,))

    @ct.kernel(num_worker_warps=NUM_WARPS)
    def block_sum_cutile_kernel(xs, out):
        ct.store(out, (0,), wp_block_sum(ct.load(xs, (0,), shape=(NUM_THREADS,))))

    rng = np.random.default_rng(5)
    x_np = rng.standard_normal(NUM_THREADS).astype(np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)
    out = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    _launch_cutile(device, block_sum_cutile_kernel, (xs, out))

    expected = np.full(NUM_THREADS, x_np.sum(dtype=np.float64), dtype=np.float32)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-4)

    # a kernel whose block differs from the pinned size is rejected when
    # tileiras compiles it, before any launch
    @ct.kernel(num_worker_warps=2 * NUM_WARPS)
    def mismatched_block_cutile_kernel(xs, out):
        ct.store(out, (0,), wp_block_sum(ct.load(xs, (0,), shape=(NUM_THREADS,))))

    with test.assertRaises(ct.TileCompilerExecutionError):
        _launch_cutile(device, mismatched_block_cutile_kernel, (xs, out))


def test_tilelib_caching(test, device):
    """Reuse the cached .tilelib for an identical declaration without re-entering the packer."""
    # a private cache directory so the test never depends on, or pollutes,
    # artifacts another process (or test) may be linking against
    tmp = tempfile.mkdtemp(prefix="wp_cutile_tilelib_cache_test_")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    with unittest.mock.patch.object(_cutile, "_cache_dir", return_value=tmp):
        wp_scale_add = wpct.export_func(scale_add, device=device)
        test.assertTrue(os.path.exists(wp_scale_add.path))
        test.assertEqual(os.path.dirname(wp_scale_add.path), tmp)

        # a cache hit must not re-enter the packing path at all
        with unittest.mock.patch.object(
            _cutile, "_pack_tilelib", side_effect=AssertionError("cache miss: _pack_tilelib was re-invoked")
        ):
            wp_scale_add_cached = wpct.export_func(scale_add, device=device)
        # redeclarations are equal handles (same name, symbol, path, version, and lengths)
        test.assertEqual(wp_scale_add_cached, wp_scale_add)

        # compiling a backward-enabled kernel (or a wp.grad() call) that uses
        # the function leaves adjoint flags set on it process-wide; the
        # foreign build stays forward-only, so the declaration's identity
        # does not depend on what else the process compiled
        with (
            unittest.mock.patch.object(scale_add.adj, "used_by_backward_kernel", True),
            unittest.mock.patch.object(scale_add.adj, "force_adjoint_codegen", True),
        ):
            test.assertEqual(wpct.export_func(scale_add, device=device).version, wp_scale_add.version)
        # the identity follows every declaration choice a user can make
        wp_scale_add_block = wpct.export_func(scale_add, scope="block", device=device)
        test.assertNotEqual(wp_scale_add_block.version, wp_scale_add.version)
        test.assertNotEqual(
            wpct.export_func(scale_add, scope="block", num_worker_warps=2, device=device).version,
            wp_scale_add_block.version,
        )


def test_shared_argument_inference(test, device):
    """Infer shared vs per-invocation arguments from each call site, mixing patterns and runtime parameters in one kernel."""
    ct = _import_ct()
    wp_scale_add = wpct.export_func(scale_add, device=device)
    wp_gather_scale = wpct.export_func(gather_scale, device=device)
    wp_blend = wpct.export_func(blend, device=device)
    rng = np.random.default_rng(11)
    x_np = rng.standard_normal(NUM_THREADS).astype(np.float32)
    y_np = rng.standard_normal(NUM_THREADS).astype(np.float32)
    shared_y_np = np.array([0.3, -1.2, 2.0], dtype=np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)
    ys = wp.array(y_np, dtype=wp.float32, device=device)
    shared_y_padded = wp.array(_pad_vec3(shared_y_np[None, :])[0], dtype=wp.float32, device=device)
    ids = wp.array(np.arange(NUM_THREADS, dtype=np.int32)[::-1].copy(), dtype=wp.int32, device=device)
    out_tiles = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    out_literal = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    out_param = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    out_shared_x = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    out_all_shared = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    out_array = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    out_shared_vec3 = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)

    @ct.kernel(num_worker_warps=NUM_WARPS)
    def shared_arguments_cutile_kernel(
        xs,
        ys,
        s: float,
        ptr: ct.Constant[int],
        n: int,
        ids,
        shared_y,
        out_tiles,
        out_literal,
        out_param,
        out_shared_x,
        out_all_shared,
        out_array,
        out_shared_vec3,
    ):
        x = ct.load(xs, (0,), shape=(NUM_THREADS,))
        y = ct.load(ys, (0,), shape=(NUM_THREADS,))
        i = ct.load(ids, (0,), shape=(NUM_THREADS,))
        ct.store(out_tiles, (0,), wp_scale_add(x, y))  # both per-invocation
        ct.store(out_literal, (0,), wp_scale_add(x, 0.5))  # y shared: a literal
        ct.store(out_param, (0,), wp_scale_add(x, s))  # y shared: a runtime kernel parameter
        ct.store(out_shared_x, (0,), wp_scale_add(2.0, y=y))  # x shared, y by keyword
        # every argument shared: no tile sets the length, so the call runs
        # num_worker_warps * 32 invocations
        ct.store(out_all_shared, (0,), wp_scale_add(1.5, 2.0))
        ct.store(out_array, (0,), wp_gather_scale((ptr, n), i, s))  # array with a runtime extent, shared scale
        # a shared vector takes a single (width,) tile, broadcast to every invocation
        ct.store(out_shared_vec3, (0,), wp_blend(x, ct.load(shared_y, (0,), shape=(VEC_PAD,))))

    _launch_cutile(
        device,
        shared_arguments_cutile_kernel,
        (
            xs,
            ys,
            1.5,
            xs.ptr,
            NUM_THREADS,
            ids,
            shared_y_padded,
            out_tiles,
            out_literal,
            out_param,
            out_shared_x,
            out_all_shared,
            out_array,
            out_shared_vec3,
        ),
    )
    np.testing.assert_allclose(out_tiles.numpy(), x_np * 2.0 + np.sin(y_np), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_literal.numpy(), x_np * 2.0 + np.sin(0.5), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_param.numpy(), x_np * 2.0 + np.sin(1.5), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_shared_x.numpy(), 4.0 + np.sin(y_np), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_all_shared.numpy(), np.full(NUM_THREADS, 3.0 + np.sin(2.0)), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_array.numpy(), x_np[::-1] * 1.5, rtol=1e-6, atol=1e-6)
    ys_vec3 = wp.array(np.tile(shared_y_np, (NUM_THREADS, 1)), dtype=wp.vec3, device=device)
    expected_shared_vec3 = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    wp.launch(blend_reference, dim=NUM_THREADS, inputs=[xs, ys_vec3, expected_shared_vec3], device=device)
    np.testing.assert_allclose(out_shared_vec3.numpy(), expected_shared_vec3.numpy(), rtol=1e-5, atol=1e-6)


class TestCutileKernels(unittest.TestCase):
    """End-to-end tests of the .tilelib surface (export_func) on every CUDA test device."""

    def setUp(self):
        """Skip on Windows and when the cuTile frontend or a supported tileiras is missing."""
        if sys.platform == "win32":
            self.skipTest(CUTILE_WINDOWS_SKIP_REASON)
        if not cutile_toolchain_available():
            self.skipTest(CUTILE_SKIP_REASON)


add_function_test(
    TestCutileKernels,
    "test_scalar_and_vec3_call",
    test_scalar_and_vec3_call,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_explicit_gradient_inside_function",
    test_explicit_gradient_inside_function,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_call_site_mistakes_reported_at_trace",
    test_call_site_mistakes_reported_at_trace,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_thread_scope_tile_lengths",
    test_thread_scope_tile_lengths,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_mesh_query_scratch_stack",
    test_mesh_query_scratch_stack,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_bvh_and_nested_query_scratch_stacks",
    test_bvh_and_nested_query_scratch_stacks,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_module_build_options_honored",
    test_module_build_options_honored,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_module_preamble_and_build_dependencies",
    test_module_preamble_and_build_dependencies,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_struct_operands",
    test_struct_operands,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_parameter_name_clashes",
    test_parameter_name_clashes,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_matrix_operand_and_vec3_return",
    test_matrix_operand_and_vec3_return,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_array_operands",
    test_array_operands,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_block_scope_reduction",
    test_block_scope_reduction,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_tilelib_caching",
    test_tilelib_caching,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_shared_argument_inference",
    test_shared_argument_inference,
    devices=cutile_test_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
