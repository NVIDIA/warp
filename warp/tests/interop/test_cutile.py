# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the experimental cuTile foreign-function bridge (warp.experimental.cutile).

The tests target the public surface and are tiered by environment:

- ``TestCutileHelpers``: the host helpers, toolchain lookup, and packing
  errors; no cuTile needed (the CUDA array operand check runs on every CUDA
  test device);
- ``TestCutileDeclaration``: every error ``export_func`` reports before any
  toolchain or GPU work; needs ``cuda.tile`` importable;
- ``TestCutileKernels``: end-to-end calls from cuTile kernels, one test per
  documented call form or contract; needs ``cuda.tile_preview``, a supported
  ``tileiras`` (from the ``nvidia-cuda-tileiras`` wheel the ``cutile`` extra installs,
  ``$WARP_TILEIR_TOOLCHAIN``, or ``PATH``), and a CUDA device.

Classes skip cleanly when their environment is missing, so the module is safe
to run (and register in the default suite) everywhere.
"""

import ctypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock

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
def dsquare(x: float) -> float:
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
def dcustom_square(x: float) -> float:
    """Differentiate custom_square() explicitly, exercising the deferred callback build."""
    return wp.grad(custom_square)(x)


@wp.func
def scale_add(x: float, y: float) -> float:
    """Trivial per-thread math used for tilelib caching checks."""
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
def probe_mesh_sdf(mesh: wp.handle, x: wp.vec3) -> float:
    """Query a mesh point (thread-local traversal stack)."""
    q = wp.mesh_query_point(mesh, x, 1.0e6)
    return wp.where(q.result, q.u, 0.0)


@wp.func
def aabb_count(mesh: wp.uint64, x: wp.vec3) -> float:
    """Count triangles overlapping a box around x (block-wide traversal stack, served from scratch)."""
    h = wp.vec3(0.05, 0.05, 0.05)
    q = wp.mesh_query_aabb(mesh, x - h, x + h)
    n = float(0.0)
    face = int(0)
    while wp.mesh_query_aabb_next(q, face):
        n += 1.0
    return n


@wp.func
def tiled_aabb_count(mesh: wp.uint64, x: wp.vec3) -> float:
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


@wp.kernel
def preamble_probe_reference(x: wp.array[float], out: wp.array[float]):
    """Evaluate preamble_probe per thread under the module's own build options."""
    tid = wp.tid()
    out[tid] = preamble_probe(x[tid])


@wp.func
def keyword_params(auto: float, register: float) -> float:
    """Use C++ keywords as parameter names: legal in Warp, so the wrapper must accept them."""
    return auto * 2.0 + register


@wp.func
def colliding_names(x: wp.array[float], x_0: float) -> float:
    """Name a scalar like an array's first generated element so the call body's names must not collide."""
    return x[0] + x_0


@wp.kernel
def bvh_sphere_count_reference(bvh: wp.uint64, pts: wp.array[wp.vec3], out: wp.array[float]):
    """Evaluate bvh_sphere_count per thread as the BVH-query reference."""
    tid = wp.tid()
    out[tid] = bvh_sphere_count(bvh, pts[tid])


# Own module: this kernel holds a static mesh stack and a static BVH stack
# (32 ints per thread each), 64 KiB at the test module's default block_dim of
# 256, over the 48 KiB ptxas limit on most architectures. A unique module
# compiles only when launched, at the launch's block_dim (128 below).
@wp.kernel(module="unique")
def nested_mesh_bvh_count_reference(mesh: wp.uint64, bvh: wp.uint64, pts: wp.array[wp.vec3], out: wp.array[float]):
    """Evaluate nested_mesh_bvh_count per thread as the nested-query reference."""
    tid = wp.tid()
    out[tid] = nested_mesh_bvh_count(mesh, bvh, pts[tid])


@wp.kernel
def sin_probe_reference(x: wp.array[float], out: wp.array[float]):
    """Evaluate sin_probe per thread under the module's own build options."""
    tid = wp.tid()
    out[tid] = sin_probe(x[tid])


@wp.kernel
def aabb_count_reference(mesh: wp.uint64, pts: wp.array[wp.vec3], out: wp.array[float]):
    """Evaluate aabb_count per thread as the iterator-query reference."""
    tid = wp.tid()
    out[tid] = aabb_count(mesh, pts[tid])


@wp.func
def aabb_count_repeat(mesh: wp.uint64, x: wp.vec3) -> float:
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


# Own module for the same reason as nested_mesh_bvh_count_reference: one
# static mesh stack plus one static BVH stack exceed 48 KiB at block_dim 256.
@wp.kernel(module="unique")
def aabb_sphere_pair_reference(mesh: wp.uint64, bvh: wp.uint64, pts: wp.array[wp.vec3], out: wp.array[float]):
    """Evaluate aabb_count_repeat plus bvh_sphere_count per thread as the two-function reference."""
    tid = wp.tid()
    out[tid] = aabb_count_repeat(mesh, pts[tid]) + bvh_sphere_count(bvh, pts[tid])


@wp.kernel
def probe_mesh_sdf_reference(mesh: wp.handle, pts: wp.array[wp.vec3], out: wp.array[float]):
    """Evaluate probe_mesh_sdf per thread as the mesh-query reference."""
    tid = wp.tid()
    out[tid] = probe_mesh_sdf(mesh, pts[tid])


@wp.func
def all_uniform_sum(a: float, b: float) -> float:
    """Add two broadcast scalars (all-uniform signature, pinned declaration)."""
    return a + b


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


@wp.func
def ret_collision(ret0: float) -> float:
    """Fixture whose parameter name collides with the generated return operand."""
    return ret0 * 2.0


@wp.func(name="collide__WP_FOREIGN_SYMBOL__")
def placeholder_named(x: float) -> float:
    """Fixture whose custom key contains a reserved placeholder token."""
    return x


@wp.func(name="not an identifier")
def bad_key_named(x: float) -> float:
    """Fixture whose custom key is not a valid identifier."""
    return x


@wp.func(name="fóó")
def unicode_key_named(x: float) -> float:
    """Fixture whose custom key is a Unicode (non-ASCII) identifier."""
    return x


@wp.kernel
def blend_reference(xs: wp.array[float], ys: wp.array[wp.vec3], out: wp.array[float]):
    """Per-thread reference for the blend function."""
    i = wp.tid()
    out[i] = blend(xs[i], ys[i])


def _pad_vec3(a: np.ndarray) -> np.ndarray:
    """Pad an (n, 3) float32 array to (n, VEC_PAD) with zeros."""
    p = np.zeros((a.shape[0], VEC_PAD), dtype=np.float32)
    p[:, :3] = a
    return p


def test_array_pointer_args_device(test, device):
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

    def test_array_pointer_args(self):
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
        for name in wpct.__all__:
            self.assertTrue(callable(getattr(wpct, name)) or isinstance(getattr(wpct, name), type))

    def test_bvh_stack_depth_matches_header(self):
        """Keep the scratch slab's per-thread depth equal to BVH_QUERY_STACK_SIZE in bvh.h."""
        header = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(_cutile.__file__))), "native", "bvh.h")
        with open(header) as f:
            match = re.search(r"#define BVH_QUERY_STACK_SIZE \((\d+)\)", f.read())
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
            with (
                unittest.mock.patch.dict(sys.modules, {"nvidia": ns_parent, "nvidia.cu13": ns_module}),
                unittest.mock.patch.dict(os.environ, {"WARP_TILEIR_TOOLCHAIN": ""}),
                unittest.mock.patch("shutil.which", wraps=shutil.which) as which,
            ):
                self.assertEqual(_cutile._pip_tileiras_dir(), os.path.join(ns_back, "bin"))
                self.assertTrue(os.path.samefile(_cutile._find_tileiras(), ns_tileiras))
                which.assert_called_once()  # resolved from the package, never from PATH

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


add_function_test(
    TestCutileHelpers, "test_array_pointer_args_device", test_array_pointer_args_device, devices=cuda_test_devices
)


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
        for bad in (0, 3, -1, 5, 48, 64):
            with self.assertRaisesRegex(ValueError, "power of two"):
                wpct.export_func(scale_add, num_worker_warps=bad)
        for bad in (True, "4", 4.0):
            with self.assertRaisesRegex(TypeError, "num_worker_warps must be an int"):
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
            wpct.export_func(tiled_aabb_count, scope="thread")
        # overloaded functions have no single concrete signature to package
        with self.assertRaisesRegex(NotImplementedError, "overloads"):
            wpct.export_func(overloaded_probe)
        # a non-int in the num_worker_warps slot is reported by type
        with self.assertRaisesRegex(TypeError, "num_worker_warps must be an int"):
            wpct.export_func(scale_add, {"x"})

        # parameter and return types the boundary cannot carry
        @wp.func
        def half_param(h: wp.float16) -> float:
            return float(h)

        @wp.func
        def bool_param(b: bool) -> float:
            return 1.0

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
            (bool_param, "Unsupported Warp type"),
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

        # the generated call body owns the _wp_ prefix
        @wp.func
        def reserved_param(_wp_ct: float) -> float:
            return _wp_ct

        with self.assertRaisesRegex(ValueError, "reserved prefix '_wp_'"):
            wpct.export_func(reserved_param)

        @wp.func(name="_wp_ct")
        def reserved_key(x: float) -> float:
            return x

        with self.assertRaisesRegex(ValueError, "reserved prefix '_wp_'"):
            wpct.export_func(reserved_key)

        # non-ASCII parameter names would only fail deep inside NVRTC/MLIR
        @wp.func
        def unicode_param(α: float) -> float:  # noqa: PLC2401 (deliberate non-ASCII fixture)
            return α

        with self.assertRaisesRegex(ValueError, "not a plain ASCII identifier"):
            wpct.export_func(unicode_param)
        # multi-return functions are not implemented on the tilelib surface
        with self.assertRaisesRegex(NotImplementedError, "returns 2 values"):
            wpct.export_func(two_returns)
        # parameter names may not collide with generated operand names
        with self.assertRaisesRegex(ValueError, "operand name collision"):
            wpct.export_func(ret_collision)
        # custom function keys may not contain the reserved placeholder tokens
        with self.assertRaisesRegex(ValueError, "reserved token"):
            wpct.export_func(placeholder_named)
        # custom function keys must be plain ASCII identifiers: the key names
        # the generated C++ function itself, so no sanitization can rescue it,
        # and NVRTC/MLIR reject Unicode word characters
        for bad_func in (bad_key_named, unicode_key_named):
            with self.assertRaisesRegex(ValueError, "not a valid identifier"):
                wpct.export_func(bad_func)


def _import_ct():
    """Import cuTile Python lazily so this module loads where it is absent."""
    import cuda.tile as ct  # noqa: PLC0415

    return ct


def _launch(device, kernel, args):
    """Launch a single-block cuTile kernel on ``device``'s Warp stream.

    No explicit synchronization: the kernel is stream-ordered with all
    subsequent Warp work, and every test reads results via ``.numpy()``,
    which synchronizes implicitly.
    """
    _import_ct().launch(wp.get_stream(device).cuda_stream, (1, 1, 1), kernel, args)


def test_per_thread_scalar_vec_call(test, device):
    """Match a per-thread scalar+vec3 tilelib call against a wp.launch reference."""
    ct = _import_ct()
    d_blend = wpct.export_func(blend, num_worker_warps=NUM_WARPS, device=device)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def kernel(xs, ys, out):
        x = ct.load(xs, (0,), shape=(NUM_THREADS,))
        y = ct.reshape(ct.load(ys, (0, 0), shape=(NUM_THREADS, VEC_PAD)), (NUM_THREADS * VEC_PAD,))
        r = d_blend(x, y)
        ct.store(out, (0,), r)

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal(NUM_THREADS).astype(np.float32)
    y_np = rng.standard_normal((NUM_THREADS, 3)).astype(np.float32)

    xs = wp.array(x_np, dtype=wp.float32, device=device)
    ys_pad = wp.array(_pad_vec3(y_np), dtype=wp.float32, device=device)
    out_ct = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    _launch(device, kernel, (xs, ys_pad, out_ct))

    ys_vec = wp.array(y_np, dtype=wp.vec3, device=device)
    out_ref = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    wp.launch(blend_reference, dim=NUM_THREADS, inputs=[xs, ys_vec, out_ref], device=device)
    np.testing.assert_allclose(out_ct.numpy(), out_ref.numpy(), rtol=1e-6, atol=1e-6)

    # arguments may also be passed by Warp parameter name
    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def kernel_kw(xs, ys, out):
        x = ct.load(xs, (0,), shape=(NUM_THREADS,))
        y = ct.reshape(ct.load(ys, (0, 0), shape=(NUM_THREADS, VEC_PAD)), (NUM_THREADS * VEC_PAD,))
        ct.store(out, (0,), d_blend(y=y, x=x))

    out_kw = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    _launch(device, kernel_kw, (xs, ys_pad, out_kw))
    np.testing.assert_array_equal(out_kw.numpy(), out_ct.numpy())


def test_explicit_gradient_inside_function(test, device):
    """Keep wp.grad() derivatives, including @wp.func_grad callbacks, correct inside a forward-only foreign function."""
    ct = _import_ct()
    d = wpct.export_func(dsquare, device=device)
    x_np = np.arange(NUM_THREADS, dtype=np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)
    out = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def kernel(xs_a, out_a):
        ct.store(out_a, (0,), d(ct.load(xs_a, (0,), shape=(NUM_THREADS,))))

    _launch(device, kernel, (xs, out))
    # d/dx x^2 = 2x: the derivative code must be emitted even though no
    # backward kernel uses the function
    np.testing.assert_allclose(out.numpy(), 2.0 * x_np)

    # a @wp.func_grad callback is a deferred build dependency; the
    # standalone build must include it like a module build does
    d_custom = wpct.export_func(dcustom_square, device=device)
    out_custom = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def kernel_custom(xs_a, out_a):
        ct.store(out_a, (0,), d_custom(ct.load(xs_a, (0,), shape=(NUM_THREADS,))))

    _launch(device, kernel_custom, (xs, out_custom))
    np.testing.assert_allclose(out_custom.numpy(), 10.0)


def test_call_site_mistakes_reported_at_trace(test, device):
    """Report wrong operand rank, length, element type, and plain-vs-tile mix-ups when cuTile traces the call."""
    ct = _import_ct()
    d_blend = wpct.export_func(blend, num_worker_warps=NUM_WARPS, device=device)
    xs = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    ys = wp.zeros((NUM_THREADS, VEC_PAD), dtype=wp.float32, device=device)
    ids = wp.zeros(NUM_THREADS, dtype=wp.int32, device=device)
    out = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)

    def expect(pattern, kernel, args=(xs, ys, out)):
        """Assert that tracing ``kernel`` fails with a message matching ``pattern``."""
        with test.assertRaisesRegex(ct.TileError, pattern):
            _launch(device, kernel, args)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def unpadded_vec(xs_a, ys_a, out_a):
        x = ct.load(xs_a, (0,), shape=(NUM_THREADS,))
        ct.store(out_a, (0,), d_blend(x, x))

    expect(
        r"blend\(\): parameter 'y' \(wp.vec3f, width 4\) expects shape \(512,\) to match parameter 'x'",
        unpadded_vec,
    )

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def plain_for_tile(xs_a, ys_a, out_a):
        x = ct.load(xs_a, (0,), shape=(NUM_THREADS,))
        ct.store(out_a, (0,), d_blend(x, 1.0))

    expect(
        r"blend\(\): parameter 'y' \(wp.vec3f\) expects a rank-1 tile of shape \(TILE \* 4,\), "
        r"or one shared \(4,\) tile, got 1.0",
        plain_for_tile,
    )

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def undeclared_length(xs_a, ys_a, out_a):
        x = ct.load(xs_a, (0,), shape=(16,))
        y = ct.reshape(ct.load(ys_a, (0, 0), shape=(16, VEC_PAD)), (16 * VEC_PAD,))
        ct.store(out_a, (0,), d_blend(x, y))

    expect(r"blend\(\): tile length 16 \(from parameter 'x'\) is not a declared length", undeclared_length)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def rank2_vec(xs_a, ys_a, out_a):
        x = ct.load(xs_a, (0,), shape=(NUM_THREADS,))
        ct.store(out_a, (0,), d_blend(x, ct.load(ys_a, (0, 0), shape=(NUM_THREADS, VEC_PAD))))

    expect(r"parameter 'y' \(wp.vec3f\) expects a rank-1 tile of shape \(TILE \* 4,\)", rank2_vec)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def int_for_float(xs_a, ids_a, out_a):
        y = ct.reshape(ct.load(xs_a, (0, 0), shape=(NUM_THREADS, VEC_PAD)), (NUM_THREADS * VEC_PAD,))
        ct.store(out_a, (0,), d_blend(ct.load(ids_a, (0,), shape=(NUM_THREADS,)), y))

    expect(r"parameter 'x' expects float32 elements, got int32", int_for_float, (ys, ids, out))

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def missing_argument(xs_a, ys_a, out_a):
        ct.store(out_a, (0,), d_blend(ct.load(xs_a, (0,), shape=(NUM_THREADS,))))

    expect(r"blend\(\): missing a required argument: 'y'", missing_argument)


def test_thread_scope_universal_lengths(test, device):
    """Serve several call-site tile lengths from one thread-scope declaration."""
    ct = _import_ct()
    rng = np.random.default_rng(3)

    d_sa = wpct.export_func(scale_add, device=device)
    test.assertEqual(d_sa.lengths, _cutile._UNIVERSAL_LENGTHS)
    # the num_worker_warps hint does not participate in the cache identity
    test.assertEqual(wpct.export_func(scale_add, num_worker_warps=16, device=device).version, d_sa.version)

    def run(length):
        """Call the shared declaration at the given tile length."""
        x_np = rng.standard_normal(length).astype(np.float32)
        y_np = rng.standard_normal(length).astype(np.float32)
        xs = wp.array(x_np, dtype=wp.float32, device=device)
        ys = wp.array(y_np, dtype=wp.float32, device=device)
        out = wp.zeros(length, dtype=wp.float32, device=device)

        @ct.kernel(num_worker_warps=NUM_THREADS // 32)
        def kernel(xs_a, ys_a, out_a):
            x = ct.load(xs_a, (0,), shape=(length,))
            y = ct.load(ys_a, (0,), shape=(length,))
            r = d_sa(x, y)
            ct.store(out_a, (0,), r)

        _launch(device, kernel, (xs, ys, out))
        np.testing.assert_allclose(out.numpy(), x_np * 2.0 + np.sin(y_np), rtol=1e-6, atol=1e-6)

    # cuTile schedules the invocations onto however many workers the
    # kernel has: fewer and more invocations than the 128 workers
    for length in (32, 64, 256, 1024):
        run(length)

    # two call sites of different lengths in one kernel dispatch to
    # different declared lengths of the same tilelib
    x_np = rng.standard_normal(256).astype(np.float32)
    y_np = rng.standard_normal(256).astype(np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)
    ys = wp.array(y_np, dtype=wp.float32, device=device)
    out_a = wp.zeros(64, dtype=wp.float32, device=device)
    out_b = wp.zeros(256, dtype=wp.float32, device=device)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def kernel_mixed(xs_a, ys_a, out_a_, out_b_):
        ct.store(out_a_, (0,), d_sa(ct.load(xs_a, (0,), shape=(64,)), ct.load(ys_a, (0,), shape=(64,))))
        ct.store(out_b_, (0,), d_sa(ct.load(xs_a, (0,), shape=(256,)), ct.load(ys_a, (0,), shape=(256,))))

    _launch(device, kernel_mixed, (xs, ys, out_a, out_b))
    expected = x_np * 2.0 + np.sin(y_np)
    np.testing.assert_allclose(out_a.numpy(), expected[:64], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_b.numpy(), expected, rtol=1e-6, atol=1e-6)

    # sharing is inferred per call, so an all-scalar signature is declared
    # for every length like any other; a call whose arguments are all shared
    # values runs num_worker_warps * 32 invocations
    d_uni = wpct.export_func(all_uniform_sum, num_worker_warps=2, device=device)
    test.assertEqual(d_uni.lengths, _cutile._UNIVERSAL_LENGTHS)
    out_u = wp.zeros(64, dtype=wp.float32, device=device)

    @ct.kernel(num_worker_warps=2)
    def kernel_all_shared(out_a):
        ct.store(out_a, (0,), d_uni(1.5, 2.0))

    _launch(device, kernel_all_shared, (out_u,))
    np.testing.assert_allclose(out_u.numpy(), 3.5)


def test_mesh_query_scratch_stack(test, device):
    """Run point and iterator mesh queries at any tile length and block size, hint-independent."""
    ct = _import_ct()
    d = wpct.export_func(probe_mesh_sdf, num_worker_warps=NUM_WARPS, device=device)
    d_box = wpct.export_func(aabb_count, num_worker_warps=NUM_WARPS, device=device)
    # both kinds are declared for every tile length, and neither bakes the
    # hint: the iterator query's traversal stack comes from compiler scratch
    test.assertEqual(d.lengths, _cutile._UNIVERSAL_LENGTHS)
    test.assertEqual(d_box.lengths, _cutile._UNIVERSAL_LENGTHS)
    test.assertEqual(wpct.export_func(aabb_count, num_worker_warps=8, device=device).version, d_box.version)
    test.assertEqual(wpct.export_func(probe_mesh_sdf, num_worker_warps=8, device=device).version, d.version)
    # the scratch slab costs 128 bytes per worker thread per query family
    # (documented): a 32-warp block needs 128 KiB, more than some GPUs
    # allow per block, and that is reported at declaration, not as a fault
    smem_limit = device.max_shared_memory_per_block
    fits_1024 = 128 * 1024 <= smem_limit
    if not fits_1024:
        with test.assertRaisesRegex(ValueError, "compiler scratch .* shared memory per block"):
            wpct.export_func(aabb_count, num_worker_warps=32, device=device)
    # a larger hint grows the slab and keys the identity; the point query
    # needs no slab, so its identity ignores the hint
    test.assertNotEqual(wpct.export_func(aabb_count, num_worker_warps=16, device=device).version, d_box.version)
    test.assertEqual(wpct.export_func(probe_mesh_sdf, num_worker_warps=16, device=device).version, d.version)

    # the point query computes correctly at the hint length ...
    # a dense soup of small triangles: deep BVH traversals fill the
    # per-thread stacks, so an under-sized scratch slab faults instead of
    # passing by luck (a handful of triangles never leaves the top levels)
    rng = np.random.default_rng(13)
    ntri = 20000
    centers = rng.uniform(0.0, 1.0, (ntri, 3)).astype(np.float32)
    offsets = rng.uniform(-0.01, 0.01, (ntri, 3, 3)).astype(np.float32)
    tri_np = (centers[:, None, :] + offsets).reshape(-1, 3)
    mesh = wp.Mesh(
        points=wp.array(tri_np, dtype=wp.vec3, device=device),
        indices=wp.array(np.arange(ntri * 3, dtype=np.int32), dtype=int, device=device),
    )
    pts_np = rng.uniform(0.0, 1.0, (NUM_THREADS, 3)).astype(np.float32)
    pts_pad = wp.array(_pad_vec3(pts_np), dtype=wp.float32, device=device)
    out = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def kernel(mesh_id: ct.Constant[int], pts_a, out_a):
        p = ct.reshape(ct.load(pts_a, (0, 0), shape=(NUM_THREADS, VEC_PAD)), (NUM_THREADS * VEC_PAD,))
        r = d(mesh_id, p)
        ct.store(out_a, (0,), r)

    _launch(device, kernel, (mesh.id, pts_pad, out))
    ref = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    pts_vec = wp.array(pts_np, dtype=wp.vec3, device=device)
    wp.launch(probe_mesh_sdf_reference, dim=NUM_THREADS, inputs=[mesh.id, pts_vec, ref], device=device)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=1e-5, atol=1e-6)

    # the id may also arrive per invocation, as an int64 tile (Tile IR has
    # no unsigned dtypes; the 64 bits pass through unchanged)
    d_ids = wpct.export_func(probe_mesh_sdf, num_worker_warps=NUM_WARPS, device=device)
    ids_np = np.full(NUM_THREADS, mesh.id, dtype=np.uint64).view(np.int64)
    ids = wp.array(ids_np, dtype=wp.int64, device=device)
    out_ids = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def kernel_ids(ids_a, pts_a, out_a):
        p = ct.reshape(ct.load(pts_a, (0, 0), shape=(NUM_THREADS, VEC_PAD)), (NUM_THREADS * VEC_PAD,))
        ct.store(out_a, (0,), d_ids(ct.load(ids_a, (0,), shape=(NUM_THREADS,)), p))

    _launch(device, kernel_ids, (ids, pts_pad, out_ids))
    np.testing.assert_allclose(out_ids.numpy(), ref.numpy(), rtol=1e-5, atol=1e-6)

    # ... and at a shorter tile length on the same 128-worker kernel
    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def kernel_short(mesh_id: ct.Constant[int], pts_a, out_a):
        p = ct.reshape(ct.load(pts_a, (0, 0), shape=(32, VEC_PAD)), (32 * VEC_PAD,))
        r = d(mesh_id, p)
        ct.store(out_a, (0,), r)

    out32 = wp.zeros(32, dtype=wp.float32, device=device)
    _launch(device, kernel_short, (mesh.id, pts_pad, out32))
    np.testing.assert_allclose(out32.numpy(), ref.numpy()[:32], rtol=1e-5, atol=1e-6)

    # the iterator query on an 8-warp block with 256-element tiles: every
    # thread traverses at once, so the slab must be sized for the block,
    # not for the compile-time hint
    big_np = rng.uniform(0.0, 1.0, (256, 3)).astype(np.float32)
    big_pad = wp.array(_pad_vec3(big_np), dtype=wp.float32, device=device)
    big_vec = wp.array(big_np, dtype=wp.vec3, device=device)

    @ct.kernel(num_worker_warps=8)
    def kernel_box(mesh_id: ct.Constant[int], pts_a, out_a):
        p = ct.reshape(ct.load(pts_a, (0, 0), shape=(256, VEC_PAD)), (256 * VEC_PAD,))
        r = d_box(mesh_id, p)
        ct.store(out_a, (0,), r)

    out_box = wp.zeros(256, dtype=wp.float32, device=device)
    _launch(device, kernel_box, (mesh.id, big_pad, out_box))
    ref_box = wp.zeros(256, dtype=wp.float32, device=device)
    wp.launch(aabb_count_reference, dim=256, inputs=[mesh.id, big_vec, ref_box], device=device)
    np.testing.assert_array_equal(out_box.numpy(), ref_box.numpy())

    # blocks beyond 8 warps: declare the larger block through num_worker_warps
    # (the 16-warp kernel overran the old fixed 256-thread slab with an
    # illegal address; now it is sized for the block)
    def run_big(warps):
        """Run the iterator query on a block of the given warp count, declared through num_worker_warps."""
        threads = warps * 32
        d_big = wpct.export_func(aabb_count, num_worker_warps=warps, device=device)
        big2_np = rng.uniform(0.0, 1.0, (threads, 3)).astype(np.float32)
        big2_pad = wp.array(_pad_vec3(big2_np), dtype=wp.float32, device=device)
        big2_vec = wp.array(big2_np, dtype=wp.vec3, device=device)

        @ct.kernel(num_worker_warps=warps)
        def kernel_big(mesh_id: ct.Constant[int], pts_a, out_a):
            p = ct.reshape(ct.load(pts_a, (0, 0), shape=(threads, VEC_PAD)), (threads * VEC_PAD,))
            ct.store(out_a, (0,), d_big(mesh_id, p))

        out_big = wp.zeros(threads, dtype=wp.float32, device=device)
        _launch(device, kernel_big, (mesh.id, big2_pad, out_big))
        ref_big = wp.zeros(threads, dtype=wp.float32, device=device)
        wp.launch(aabb_count_reference, dim=threads, inputs=[mesh.id, big2_vec, ref_big], device=device)
        np.testing.assert_array_equal(out_big.numpy(), ref_big.numpy())

    run_big(16)
    if fits_1024:
        run_big(32)

    # at block scope the slab is sized to the pinned block exactly
    d_box_block = wpct.export_func(aabb_count, num_worker_warps=8, scope="block", device=device)

    @ct.kernel(num_worker_warps=8)
    def kernel_box_block(mesh_id: ct.Constant[int], pts_a, out_a):
        p = ct.reshape(ct.load(pts_a, (0, 0), shape=(256, VEC_PAD)), (256 * VEC_PAD,))
        r = d_box_block(mesh_id, p)
        ct.store(out_a, (0,), r)

    out_box_block = wp.zeros(256, dtype=wp.float32, device=device)
    _launch(device, kernel_box_block, (mesh.id, big_pad, out_box_block))
    np.testing.assert_array_equal(out_box_block.numpy(), ref_box.numpy())


def test_bvh_and_nested_query_scratch_stacks(test, device):
    """Serve BVH sphere queries and nested mesh+BVH queries from separate scratch slabs on an 8-warp block."""
    ct = _import_ct()
    rng = np.random.default_rng(29)
    # a dense soup of small triangles and small boxes so traversals go deep
    ntri = 20000
    centers = rng.uniform(0.0, 1.0, (ntri, 3)).astype(np.float32)
    offsets = rng.uniform(-0.01, 0.01, (ntri, 3, 3)).astype(np.float32)
    tri_np = (centers[:, None, :] + offsets).reshape(-1, 3)
    mesh = wp.Mesh(
        points=wp.array(tri_np, dtype=wp.vec3, device=device),
        indices=wp.array(np.arange(ntri * 3, dtype=np.int32), dtype=int, device=device),
    )
    nbox = 20000
    box_c = rng.uniform(0.0, 1.0, (nbox, 3)).astype(np.float32)
    bvh = wp.Bvh(
        lowers=wp.array(box_c - 0.01, dtype=wp.vec3, device=device),
        uppers=wp.array(box_c + 0.01, dtype=wp.vec3, device=device),
    )
    pts_np = rng.uniform(0.0, 1.0, (256, 3)).astype(np.float32)
    pts_pad = wp.array(_pad_vec3(pts_np), dtype=wp.float32, device=device)
    pts_vec = wp.array(pts_np, dtype=wp.vec3, device=device)

    # every BVH query kind gets the scratch slab, sized for the block
    d_sph = wpct.export_func(bvh_sphere_count, num_worker_warps=NUM_WARPS, device=device)

    @ct.kernel(num_worker_warps=8)
    def kernel_sph(bvh_id: ct.Constant[int], pts_a, out_a):
        p = ct.reshape(ct.load(pts_a, (0, 0), shape=(256, VEC_PAD)), (256 * VEC_PAD,))
        ct.store(out_a, (0,), d_sph(bvh_id, p))

    out_sph = wp.zeros(256, dtype=wp.float32, device=device)
    _launch(device, kernel_sph, (bvh.id, pts_pad, out_sph))
    ref_sph = wp.zeros(256, dtype=wp.float32, device=device)
    wp.launch(bvh_sphere_count_reference, dim=256, inputs=[bvh.id, pts_vec, ref_sph], device=device)
    np.testing.assert_array_equal(out_sph.numpy(), ref_sph.numpy())
    test.assertGreater(ref_sph.numpy().max(), 0.0)

    # a BVH query nested in a mesh query: the two slabs are laid out
    # back to back, so the inner traversal cannot clobber the outer one
    d_nest = wpct.export_func(nested_mesh_bvh_count, num_worker_warps=NUM_WARPS, device=device)

    @ct.kernel(num_worker_warps=8)
    def kernel_nest(mesh_id: ct.Constant[int], bvh_id: ct.Constant[int], pts_a, out_a):
        p = ct.reshape(ct.load(pts_a, (0, 0), shape=(256, VEC_PAD)), (256 * VEC_PAD,))
        ct.store(out_a, (0,), d_nest(mesh_id, bvh_id, p))

    out_nest = wp.zeros(256, dtype=wp.float32, device=device)
    _launch(device, kernel_nest, (mesh.id, bvh.id, pts_pad, out_nest))
    ref_nest = wp.zeros(256, dtype=wp.float32, device=device)
    # 128-thread blocks keep the reference's two static stacks at 32 KiB
    # (see the kernel's module="unique" note)
    wp.launch(
        nested_mesh_bvh_count_reference,
        dim=256,
        inputs=[mesh.id, bvh.id, pts_vec, ref_nest],
        device=device,
        block_dim=128,
    )
    np.testing.assert_array_equal(out_nest.numpy(), ref_nest.numpy())
    test.assertGreater(ref_nest.numpy().max(), 1000.0)

    # two foreign functions of different query families in one kernel, with
    # no barrier between the calls: each wrapper binds only the slab it
    # owns, so a warp still re-reading the mesh slab pointer inside the
    # repeated mesh query never observes the BVH-only call's binding
    d_rep = wpct.export_func(aabb_count_repeat, num_worker_warps=NUM_WARPS, device=device)

    @ct.kernel(num_worker_warps=8)
    def kernel_pair(mesh_id: ct.Constant[int], bvh_id: ct.Constant[int], pts_a, out_a):
        p = ct.reshape(ct.load(pts_a, (0, 0), shape=(256, VEC_PAD)), (256 * VEC_PAD,))
        ct.store(out_a, (0,), d_rep(mesh_id, p) + d_sph(bvh_id, p))

    out_pair = wp.zeros(256, dtype=wp.float32, device=device)
    for _ in range(4):
        _launch(device, kernel_pair, (mesh.id, bvh.id, pts_pad, out_pair))
    ref_pair = wp.zeros(256, dtype=wp.float32, device=device)
    wp.launch(
        aabb_sphere_pair_reference,
        dim=256,
        inputs=[mesh.id, bvh.id, pts_vec, ref_pair],
        device=device,
        block_dim=128,
    )
    np.testing.assert_array_equal(out_pair.numpy(), ref_pair.numpy())


def test_module_build_options_honored(test, device):
    """Compile the foreign function with its module's fast_math setting, key the cache on it, and warn on fuse_fp=False."""
    if wp.config.llvm_cuda:
        # the wp.launch reference is built by Clang, which ignores fast_math; the foreign build is NVRTC
        test.skipTest("fast_math comparison needs the NVRTC reference path")
    ct = _import_ct()
    module = sin_probe.module
    # large arguments expose the approximate __sinf that fast_math selects
    x_np = np.linspace(1.0e3, 1.0e4, NUM_THREADS, dtype=np.float32)
    x = wp.array(x_np, dtype=wp.float32, device=device)
    out = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    ref = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)

    def run(fast_math):
        """Build and call the foreign function under the given fast_math option; return (version, results, reference)."""
        wp.set_module_options({"fast_math": fast_math}, module=module)
        d = wpct.export_func(sin_probe, num_worker_warps=NUM_WARPS, device=device)

        @ct.kernel(num_worker_warps=NUM_THREADS // 32)
        def kernel(x_a, out_a):
            ct.store(out_a, (0,), d(ct.load(x_a, (0,), shape=(NUM_THREADS,))))

        _launch(device, kernel, (x, out))
        wp.launch(sin_probe_reference, dim=NUM_THREADS, inputs=[x, ref], device=device)
        return d.version, out.numpy().copy(), ref.numpy().copy()

    original = module.options.get("fast_math", False)
    try:
        v_fast, out_fast, ref_fast = run(True)
        v_prec, out_prec, ref_prec = run(False)
    finally:
        wp.set_module_options({"fast_math": original}, module=module)
    # each setting matches wp.launch of the same function under the same option ...
    np.testing.assert_array_equal(out_fast, ref_fast)
    np.testing.assert_array_equal(out_prec, ref_prec)
    # ... the precise build matches NumPy, the fast one visibly does not,
    # and the tilelib identity tracks the option
    np.testing.assert_allclose(out_prec, np.sin(x_np), rtol=0.0, atol=1e-4)
    test.assertGreater(np.abs(out_fast - np.sin(x_np)).max(), 2e-4)
    test.assertNotEqual(v_fast, v_prec)

    # fuse_fp=False cannot be guaranteed: contraction is decided when
    # tileiras links the kernel, so declaring under it warns (Warp routes
    # warnings through its logger, so capture them there)
    class _RecordingLogger:
        """Collect Warp warning messages."""

        def __init__(self):
            """Start with no recorded messages."""
            self.messages = []

        def debug(self, message):
            """Ignore debug output."""

        def info(self, message):
            """Ignore info output."""

        def warning(self, message, category=None, stacklevel=1):
            """Record a warning message."""
            self.messages.append(message)

        def error(self, message):
            """Ignore error output."""

    recorder = _RecordingLogger()
    original_logger = wp.get_logger()
    had_fuse_fp = "fuse_fp" in module.options
    original_fuse_fp = module.options.get("fuse_fp")
    try:
        wp.set_module_options({"fuse_fp": False}, module=module)
        wp.set_logger(recorder)
        # the warning is emitted once per process; start from a clean slate
        with unittest.mock.patch.object(warp._src.logger, "_warnings_seen", set()):
            wpct.export_func(sin_probe, num_worker_warps=NUM_WARPS, device=device)
    finally:
        wp.set_logger(original_logger)
        if had_fuse_fp:
            wp.set_module_options({"fuse_fp": original_fuse_fp}, module=module)
        else:
            module.options.pop("fuse_fp", None)
    test.assertTrue(any("fuse_fp=False is not honored" in m for m in recorder.messages))


def test_module_preamble_and_build_dependencies(test, device):
    """Compile with the module's extra_cuda_preamble and rebuild when a registered build-dependency header changes."""
    ct = _import_ct()
    module = preamble_probe.module
    # a wp.func_native dependency that needs the module's preamble define
    x_np = np.arange(NUM_THREADS, dtype=np.float32)
    x = wp.array(x_np, dtype=wp.float32, device=device)
    out = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    ref = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    original = module.options.get("extra_build_options")
    try:
        wp.set_module_options(
            {"extra_build_options": wp.ModuleBuildOptions(extra_cuda_preamble="#define WP_CUTILE_TEST_SEVEN 7.0f\n")},
            module=module,
        )
        d = wpct.export_func(preamble_probe, num_worker_warps=NUM_WARPS, device=device)

        @ct.kernel(num_worker_warps=NUM_THREADS // 32)
        def kernel(x_a, out_a):
            ct.store(out_a, (0,), d(ct.load(x_a, (0,), shape=(NUM_THREADS,))))

        _launch(device, kernel, (x, out))
        wp.launch(preamble_probe_reference, dim=NUM_THREADS, inputs=[x, ref], device=device)
    finally:
        wp.set_module_options({"extra_build_options": original}, module=module)
    np.testing.assert_array_equal(out.numpy(), ref.numpy())
    np.testing.assert_array_equal(out.numpy(), x_np + 7.0)

    # the same define now comes from a header registered as a build dependency
    tmp = tempfile.mkdtemp(prefix="wp_cutile_dep_test_")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    header = os.path.join(tmp, "wp_cutile_dep.h")
    x.zero_()

    def run(value):
        """Write the header with ``value``, re-register it, build, and return (version, foreign result)."""
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
        d = wpct.export_func(preamble_probe, num_worker_warps=NUM_WARPS, device=device)

        @ct.kernel(num_worker_warps=NUM_THREADS // 32)
        def kernel(x_a, out_a):
            ct.store(out_a, (0,), d(ct.load(x_a, (0,), shape=(NUM_THREADS,))))

        _launch(device, kernel, (x, out))
        return d.version, out.numpy().copy()

    original = module.options.get("extra_build_options")
    try:
        v7, out7 = run(7.0)
        v11, out11 = run(11.0)
    finally:
        wp.set_module_options({"extra_build_options": original}, module=module)
    np.testing.assert_array_equal(out7, 7.0)
    # the same source with a changed header must not reuse the old tilelib
    test.assertNotEqual(v11, v7)
    np.testing.assert_array_equal(out11, 11.0)


def test_struct_shared_across_tilelibs(test, device):
    """Compose two foreign functions that share a Warp struct inside one kernel (producer -> consumer)."""
    ct = _import_ct()
    d_make = wpct.export_func(make_params, num_worker_warps=NUM_WARPS, device=device)
    d_apply = wpct.export_func(apply_params, num_worker_warps=NUM_WARPS, device=device)
    rng = np.random.default_rng(11)
    x_np = rng.uniform(0.5, 2.0, NUM_THREADS).astype(np.float32)
    y_np = rng.standard_normal((NUM_THREADS, 3)).astype(np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)
    ys_pad = wp.array(_pad_vec3(y_np), dtype=wp.float32, device=device)
    out = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)

    # both tilelibs instantiate BridgeParams and its helper functions, which
    # must link into one kernel without multiply-defined symbols
    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def kernel(xs_a, ys_a, out_a):
        x = ct.load(xs_a, (0,), shape=(NUM_THREADS,))
        y = ct.reshape(ct.load(ys_a, (0, 0), shape=(NUM_THREADS, VEC_PAD)), (NUM_THREADS * VEC_PAD,))
        ct.store(out_a, (0,), d_apply(d_make(x), y))

    _launch(device, kernel, (xs, ys_pad, out))
    # reference: apply_params(make_params(x), y) = length(y + (x, 0, -x)) * x * 0.5
    expected = np.linalg.norm(y_np + np.stack([x_np, np.zeros_like(x_np), -x_np], axis=1), axis=1) * x_np * 0.5
    np.testing.assert_allclose(out.numpy(), expected.astype(np.float32), rtol=1e-5, atol=1e-6)


def test_keyword_parameter_names(test, device):
    """Call functions whose parameter names are C++ keywords or collide with generated operand names."""
    ct = _import_ct()
    d = wpct.export_func(keyword_params, num_worker_warps=NUM_WARPS, device=device)
    rng = np.random.default_rng(5)
    x_np = rng.standard_normal(NUM_THREADS).astype(np.float32)
    y_np = rng.standard_normal(NUM_THREADS).astype(np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)
    ys = wp.array(y_np, dtype=wp.float32, device=device)
    out = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def kernel(xs_a, ys_a, out_a):
        x = ct.load(xs_a, (0,), shape=(NUM_THREADS,))
        y = ct.load(ys_a, (0,), shape=(NUM_THREADS,))
        ct.store(out_a, (0,), d(x, y))

    _launch(device, kernel, (xs, ys, out))
    np.testing.assert_allclose(out.numpy(), x_np * 2.0 + y_np, rtol=1e-6, atol=1e-6)

    # an array parameter and a scalar named after one of its generated elements coexist
    d_c = wpct.export_func(colliding_names, num_worker_warps=NUM_WARPS, device=device)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def kernel_c(ptr: ct.Constant[int], n: ct.Constant[int], ys_a, out_a):
        ct.store(out_a, (0,), d_c((ptr, n), ct.load(ys_a, (0,), shape=(NUM_THREADS,))))

    _launch(device, kernel_c, (*wpct.array_pointer_args(xs), ys, out))
    np.testing.assert_allclose(out.numpy(), x_np[0] + y_np, rtol=1e-6, atol=1e-6)


def test_mat_uniform_vec_return(test, device):
    """Marshal a broadcast matrix input and per-invocation vector returns at several tile lengths."""
    ct = _import_ct()
    rng = np.random.default_rng(7)
    d_mv = wpct.export_func(mat_vec, device=device)

    m_np = rng.standard_normal((3, 3)).astype(np.float32)
    m_pad = np.zeros(16, dtype=np.float32)
    m_pad[:9] = m_np.reshape(-1)  # row-major, padded to the pow2 tile width
    ms = wp.array(m_pad, dtype=wp.float32, device=device)

    def run(length):
        """Call the shared declaration with width-4 tiles of the given length."""
        v_np = rng.standard_normal((length, 3)).astype(np.float32)
        vs = wp.array(_pad_vec3(v_np), dtype=wp.float32, device=device)
        # poison the destination so the zero-filled pad lanes are proven
        out = wp.full(length * VEC_PAD, 777.0, dtype=wp.float32, device=device)

        @ct.kernel(num_worker_warps=NUM_THREADS // 32)
        def kernel(ms_a, vs_a, out_a):
            m_t = ct.load(ms_a, (0,), shape=(16,))
            v_t = ct.reshape(ct.load(vs_a, (0, 0), shape=(length, VEC_PAD)), (length * VEC_PAD,))
            r = d_mv(m_t, v_t)
            ct.store(out_a, (0,), r)

        _launch(device, kernel, (ms, vs, out))
        got = out.numpy().reshape(length, VEC_PAD)
        np.testing.assert_allclose(got[:, :3], v_np @ m_np.T, rtol=1e-5, atol=1e-5)
        # padding lanes of the returned vec3 are zero-filled by the wrapper
        np.testing.assert_allclose(got[:, 3], 0.0)

    # non-fallback lengths exercise the width-4 dispatch constants
    # (compare against length * VEC_PAD, not length)
    for length in (64, NUM_THREADS, 512):
        run(length)

    # a uniform vector takes a single (width,) tile, broadcast to every invocation
    d_blend_u = wpct.export_func(blend, device=device)
    y_one = np.array([0.3, -1.2, 2.0], dtype=np.float32)
    y_pad = wp.array(_pad_vec3(y_one[None, :])[0], dtype=wp.float32, device=device)
    x_np = rng.standard_normal(NUM_THREADS).astype(np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)
    out_u = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def kernel_u(xs_a, y_a, out_a):
        ct.store(
            out_a, (0,), d_blend_u(ct.load(xs_a, (0,), shape=(NUM_THREADS,)), ct.load(y_a, (0,), shape=(VEC_PAD,)))
        )

    _launch(device, kernel_u, (xs, y_pad, out_u))
    ref_u = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    ys_same = wp.array(np.tile(y_one, (NUM_THREADS, 1)), dtype=wp.vec3, device=device)
    wp.launch(blend_reference, dim=NUM_THREADS, inputs=[xs, ys_same, ref_u], device=device)
    np.testing.assert_allclose(out_u.numpy(), ref_u.numpy(), rtol=1e-5, atol=1e-6)


def test_array_operands(test, device):
    """Exercise uniform array gather, atomic mutation, 2d indexing, and wrong-length tuple rejection."""
    ct = _import_ct()
    rng = np.random.default_rng(11)

    # section 1: uniform 1d array + per-thread index + uniform scalar
    # (raw pointers travel as rank-0 int64 tiles: Tile IR dtypes are signed only)
    data_np = rng.standard_normal(1000).astype(np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    idx_np = rng.integers(0, 1000, NUM_THREADS).astype(np.int32)
    idx = wp.array(idx_np, dtype=wp.int32, device=device)
    d_gather = wpct.export_func(gather_scale, num_worker_warps=NUM_WARPS, device=device)
    data_ptr, data_len = wpct.array_pointer_args(data)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def k_gather(ptr: ct.Constant[int], length: ct.Constant[int], idxs, out):
        # 1:1 call: the array parameter takes its (address, length) tuple,
        # plain numbers materialize as broadcast tiles automatically
        i_t = ct.load(idxs, (0,), shape=(NUM_THREADS,))
        r = d_gather((ptr, length), i_t, 2.5)
        ct.store(out, (0,), r)

    out = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    _launch(device, k_gather, (data_ptr, data_len, idx, out))
    np.testing.assert_allclose(out.numpy(), data_np[idx_np] * np.float32(2.5), rtol=1e-6)

    # section 2: array mutation through the foreign call (histogram via atomic_add)
    hist = wp.zeros(16, dtype=wp.float32, device=device)
    bins_np = rng.integers(0, 16, NUM_THREADS).astype(np.int32)
    bins = wp.array(bins_np, dtype=wp.int32, device=device)
    d_acc = wpct.export_func(accumulate, num_worker_warps=NUM_WARPS, device=device)
    hist_ptr, hist_len = wpct.array_pointer_args(hist)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def k_acc(ptr: ct.Constant[int], length: ct.Constant[int], bins_arr, out):
        # explicit tiles also pass through unchanged inside the tuple
        ptr_t = ct.full((), ptr, dtype=ct.int64)
        len_t = ct.full((), length, dtype=ct.int32)
        b_t = ct.load(bins_arr, (0,), shape=(NUM_THREADS,))
        v_t = ct.full((NUM_THREADS,), 1.0, dtype=ct.float32)
        r = d_acc((ptr_t, len_t), b_t, v_t)
        ct.store(out, (0,), r)

    prev = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    _launch(device, k_acc, (hist_ptr, hist_len, bins, prev))
    expected_hist = np.bincount(bins_np, minlength=16).astype(np.float32)
    np.testing.assert_allclose(hist.numpy(), expected_hist)

    # the call's side effect survives when its result is discarded: foreign
    # calls carry a memory effect, so the unused-result call is not eliminated
    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def k_acc_discard(ptr: ct.Constant[int], length: ct.Constant[int], bins_arr):
        b_t = ct.load(bins_arr, (0,), shape=(NUM_THREADS,))
        v_t = ct.full((NUM_THREADS,), 1.0, dtype=ct.float32)
        d_acc((ptr, length), b_t, v_t)

    _launch(device, k_acc_discard, (hist_ptr, hist_len, bins))
    np.testing.assert_allclose(hist.numpy(), 2.0 * expected_hist)

    # section 3: 2d array expands to pointer + two extents
    m_np = rng.standard_normal((32, 8)).astype(np.float32)
    m = wp.array(m_np, dtype=wp.float32, device=device)
    ii_np = rng.integers(0, 32, NUM_THREADS).astype(np.int32)
    jj_np = rng.integers(0, 8, NUM_THREADS).astype(np.int32)
    ii = wp.array(ii_np, dtype=wp.int32, device=device)
    jj = wp.array(jj_np, dtype=wp.int32, device=device)
    d_2d = wpct.export_func(read2d, num_worker_warps=NUM_WARPS, device=device)
    m_ptr, m_d0, m_d1 = wpct.array_pointer_args(m)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def k_2d(ptr: ct.Constant[int], d0: ct.Constant[int], d1: ct.Constant[int], iis, jjs, out):
        i_t = ct.load(iis, (0,), shape=(NUM_THREADS,))
        j_t = ct.load(jjs, (0,), shape=(NUM_THREADS,))
        r = d_2d((ptr, d0, d1), i_t, j_t)
        ct.store(out, (0,), r)

    out4 = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    _launch(device, k_2d, (m_ptr, m_d0, m_d1, ii, jj, out4))
    np.testing.assert_allclose(out4.numpy(), m_np[ii_np, jj_np], rtol=1e-6)

    # section 4: a wrong-length array tuple fails at trace time instead of
    # silently dropping trailing values
    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def k_long(ptr: ct.Constant[int], length: ct.Constant[int], idxs, out):
        i_t = ct.load(idxs, (0,), shape=(NUM_THREADS,))
        r = d_gather((ptr, length, 999), i_t, 2.5)
        ct.store(out, (0,), r)

    with test.assertRaisesRegex(ct.TileError, "expects a tuple of 2 plain values"):
        _launch(device, k_long, (data_ptr, data_len, idx, out))

    # a non-integer tuple element is rejected by name instead of being
    # truncated by the int32 materialization
    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def k_float_extent(ptr: ct.Constant[int], idxs, out):
        i_t = ct.load(idxs, (0,), shape=(NUM_THREADS,))
        r = d_gather((ptr, 2.5), i_t, 2.5)
        ct.store(out, (0,), r)

    with test.assertRaisesRegex(ct.TileError, "element 1 of parameter 'data' \\(wp.array\\) expects an integer in"):
        _launch(device, k_float_extent, (data_ptr, idx, out))


def test_struct_uniform(test, device):
    """Broadcast a mixed-field struct blob to every invocation."""
    ct = _import_ct()
    p = BridgeParams()
    p.scale = 3.0
    p.offset = wp.vec3(0.5, -1.0, 2.0)
    p.exponent = 2
    blob_np = wpct.encode_struct(p)
    blob = wp.array(blob_np, dtype=wp.int32, device=device)
    blob_w = len(blob_np)

    rng = np.random.default_rng(7)
    y_np = rng.standard_normal((NUM_THREADS, 3)).astype(np.float32)
    ys = wp.array(_pad_vec3(y_np), dtype=wp.float32, device=device)
    d_params = wpct.export_func(apply_params, num_worker_warps=NUM_WARPS, device=device)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def k_params(blob_arr, ys_arr, out):
        p_t = ct.load(blob_arr, (0,), shape=(blob_w,))
        y_t = ct.reshape(ct.load(ys_arr, (0, 0), shape=(NUM_THREADS, VEC_PAD)), (NUM_THREADS * VEC_PAD,))
        r = d_params(p_t, y_t)
        ct.store(out, (0,), r)

    out = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    _launch(device, k_params, (blob, ys, out))
    # expected: length(y + offset) * scale * 0.5^exponent
    v = y_np + np.array([0.5, -1.0, 2.0], dtype=np.float32)
    expected = np.linalg.norm(v, axis=1).astype(np.float32) * 3.0 * 0.25
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    # a struct with an 8-byte-aligned field: its blob is declared with a
    # two-word alignment and the field offsets survive the trip
    q = WideAlignParams()
    q.a = 1.5
    q.b = 7
    q_np = wpct.encode_struct(q)
    q_blob = wp.array(q_np, dtype=wp.int32, device=device)
    x_np = rng.standard_normal(NUM_THREADS).astype(np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)
    d_wide = wpct.export_func(wide_params_sum, num_worker_warps=NUM_WARPS, device=device)
    q_w = len(q_np)

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def k_wide(q_arr, xs_arr, out_a):
        ct.store(out_a, (0,), d_wide(ct.load(q_arr, (0,), shape=(q_w,)), ct.load(xs_arr, (0,), shape=(NUM_THREADS,))))

    out_w = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    _launch(device, k_wide, (q_blob, xs, out_w))
    np.testing.assert_allclose(out_w.numpy(), x_np + 1.5 + 7.0, rtol=1e-6, atol=1e-6)


def test_block_scope_reduction(test, device):
    """Run a block-cooperative reduction at block scope and reject a mismatched calling block at compile time."""
    ct = _import_ct()
    d_sum = wpct.export_func(block_sum, num_worker_warps=NUM_WARPS, scope="block", device=device)
    # block scope pins the declaration: the block size is baked in
    test.assertEqual(d_sum.lengths, (NUM_THREADS,))

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def kernel(xs, out):
        x = ct.load(xs, (0,), shape=(NUM_THREADS,))
        r = d_sum(x)
        ct.store(out, (0,), r)

    rng = np.random.default_rng(5)
    x_np = rng.standard_normal(NUM_THREADS).astype(np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)
    out = wp.zeros(NUM_THREADS, dtype=wp.float32, device=device)
    _launch(device, kernel, (xs, out))

    expected = np.full(NUM_THREADS, x_np.sum(dtype=np.float64), dtype=np.float32)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-4)

    # a kernel whose block differs from the pinned size is rejected when
    # tileiras compiles it, before any launch, and the device stays usable
    @ct.kernel(num_worker_warps=2 * NUM_THREADS // 32)
    def kernel_wrong_block(xs, out):
        x = ct.load(xs, (0,), shape=(NUM_THREADS,))
        ct.store(out, (0,), d_sum(x))

    with test.assertRaises(ct.TileCompilerExecutionError):
        _launch(device, kernel_wrong_block, (xs, out))
    test.assertEqual(wp.zeros(4, dtype=wp.float32, device=device).numpy().sum(), 0.0)


def test_tilelib_caching(test, device):
    """Reuse the cached .tilelib for an identical declaration without re-entering the packer."""
    # a private cache directory so the test never depends on, or pollutes,
    # artifacts another process (or test) may be linking against
    tmp = tempfile.mkdtemp(prefix="wp_cutile_tilelib_cache_test_")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    with unittest.mock.patch.object(_cutile, "_cache_dir", return_value=tmp):
        d1 = wpct.export_func(scale_add, num_worker_warps=NUM_WARPS, device=device)
        test.assertTrue(os.path.exists(d1.path))
        test.assertEqual(os.path.dirname(d1.path), tmp)

        # a cache hit must not re-enter the packing path at all
        with unittest.mock.patch.object(
            _cutile, "_pack_tilelib", side_effect=AssertionError("cache miss: _pack_tilelib was re-invoked")
        ):
            d2 = wpct.export_func(scale_add, num_worker_warps=NUM_WARPS, device=device)
        test.assertEqual(d2.path, d1.path)
        test.assertEqual(d2.symbol, d1.symbol)
        # redeclarations are equal handles
        test.assertEqual(d2, d1)
        # the traced call's source name is keyed by content (cuTile embeds
        # it in the kernel bytecode, so a per-process name would defeat its
        # cross-process cache) and differs between distinct declarations
        test.assertEqual(d2.invoke.__code__.co_filename, d1.invoke.__code__.co_filename)

        # compiling a backward-enabled kernel (or a wp.grad() call) that uses
        # the function leaves adjoint flags set on it process-wide; the
        # foreign build stays forward-only, so the declaration's identity
        # does not depend on what else the process compiled
        baseline = _cutile._codegen_warp_source(scale_add, NUM_THREADS)
        with (
            unittest.mock.patch.object(scale_add.adj, "used_by_backward_kernel", True),
            unittest.mock.patch.object(scale_add.adj, "force_adjoint_codegen", True),
        ):
            test.assertEqual(_cutile._codegen_warp_source(scale_add, NUM_THREADS), baseline)
            test.assertEqual(wpct.export_func(scale_add, num_worker_warps=NUM_WARPS, device=device).version, d1.version)
        # the identity follows every declaration choice a user can make
        d_block_scope = wpct.export_func(scale_add, scope="block", device=device)
        test.assertNotEqual(d_block_scope.version, d1.version)
        test.assertNotEqual(
            wpct.export_func(scale_add, scope="block", num_worker_warps=2, device=device).version,
            wpct.export_func(scale_add, scope="block", device=device).version,
        )
        # the repr states the call contract: the signature in Warp terms and the tile lengths served
        text = repr(d1)
        for fragment in (
            "scale_add(",
            "x: wp.float32",
            "y: wp.float32",
            "-> wp.float32",
            str(_cutile._UNIVERSAL_LENGTHS),
        ):
            test.assertIn(fragment, text)


def test_shared_argument_inference(test, device):
    """Infer shared vs per-invocation arguments from each call site, mixing patterns and runtime parameters in one kernel."""
    ct = _import_ct()
    d = wpct.export_func(scale_add, device=device)
    d_g = wpct.export_func(gather_scale, device=device)
    rng = np.random.default_rng(11)
    x_np = rng.standard_normal(NUM_THREADS).astype(np.float32)
    y_np = rng.standard_normal(NUM_THREADS).astype(np.float32)
    xs = wp.array(x_np, dtype=wp.float32, device=device)
    ys = wp.array(y_np, dtype=wp.float32, device=device)
    idx = wp.array(np.arange(NUM_THREADS, dtype=np.int32)[::-1].copy(), dtype=wp.int32, device=device)
    outs = [wp.zeros(NUM_THREADS, dtype=wp.float32, device=device) for _ in range(5)]

    @ct.kernel(num_worker_warps=NUM_THREADS // 32)
    def kernel(xs_a, ys_a, s: float, ptr: ct.Constant[int], n: int, idx_a, o0, o1, o2, o3, o4):
        x = ct.load(xs_a, (0,), shape=(NUM_THREADS,))
        y = ct.load(ys_a, (0,), shape=(NUM_THREADS,))
        i = ct.load(idx_a, (0,), shape=(NUM_THREADS,))
        ct.store(o0, (0,), d(x, y))  # both per-invocation
        ct.store(o1, (0,), d(x, 0.5))  # y shared: a literal
        ct.store(o2, (0,), d(x, s))  # y shared: a runtime kernel parameter
        ct.store(o3, (0,), d(2.0, y=y))  # x shared, y by keyword
        ct.store(o4, (0,), d_g((ptr, n), i, s))  # array with a runtime extent, shared scale

    _launch(device, kernel, (xs, ys, 1.5, xs.ptr, NUM_THREADS, idx, *outs))
    np.testing.assert_allclose(outs[0].numpy(), x_np * 2.0 + np.sin(y_np), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(outs[1].numpy(), x_np * 2.0 + np.sin(0.5), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(outs[2].numpy(), x_np * 2.0 + np.sin(1.5), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(outs[3].numpy(), 4.0 + np.sin(y_np), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(outs[4].numpy(), x_np[::-1] * 1.5, rtol=1e-6, atol=1e-6)


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
    "test_per_thread_scalar_vec_call",
    test_per_thread_scalar_vec_call,
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
    "test_thread_scope_universal_lengths",
    test_thread_scope_universal_lengths,
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
    "test_struct_shared_across_tilelibs",
    test_struct_shared_across_tilelibs,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_keyword_parameter_names",
    test_keyword_parameter_names,
    devices=cutile_test_devices,
)
add_function_test(
    TestCutileKernels,
    "test_mat_uniform_vec_return",
    test_mat_uniform_vec_return,
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
    "test_struct_uniform",
    test_struct_uniform,
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
