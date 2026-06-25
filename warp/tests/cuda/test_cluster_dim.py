# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

import numpy as np

import warp as wp
import warp._src.codegen as codegen
from warp._src.context import ModuleBuilder
from warp.tests.unittest_utils import add_function_test, assert_np_equal, get_cuda_test_devices, get_test_devices


def hopper_devices():
    """CUDA test devices that support thread block clusters (compute capability >= 9.0)."""
    return [d for d in get_cuda_test_devices() if d.arch >= 90]


def cuda_kernel_source(kernel) -> str:
    """Return the generated CUDA ``.cu`` source for *kernel*.

    Builds the module's adjoints via ModuleBuilder so codegen state (e.g.
    ``fun_def_lineno``) is populated, then invokes codegen directly.
    """
    options = kernel.module.resolve_options(wp.config) | kernel.options
    ModuleBuilder(kernel.module, options)
    return codegen.codegen_kernel(kernel, device="cuda", options=options)


# A plain kernel (no cluster_dim) used to probe a device's maximum cluster size.
# Kernels with __cluster_dims__ baked in only report their declared size, so the
# probe must be unclustered to reveal the device's true limit.
@wp.kernel(enable_backward=False, module="unique")
def query_probe(a: wp.array[int]):
    a[wp.tid()] = 0


# Device intrinsics used by the cluster-formation probe. %cluster_ctarank and
# %cluster_nctarank only exist on compute capability >= 9.0.
@wp.func_native(
    """
    unsigned int v;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(v));
    return v;
    """
)
def cluster_ctarank() -> wp.uint32: ...


@wp.func_native(
    """
    unsigned int v;
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(v));
    return v;
    """
)
def cluster_nctarank() -> wp.uint32: ...


def make_cluster_probe(cluster_dim: int, grid_stride: bool):
    """A clustered kernel where each leading CTA records its rank and cluster size."""

    @wp.kernel(cluster_dim=cluster_dim, grid_stride=grid_stride, enable_backward=False, module="unique")
    def probe(rank: wp.array[wp.uint32], size: wp.array[wp.uint32]):
        tid = wp.tid()
        if tid % wp.block_dim() == 0:
            bx = wp.uint32(tid // wp.block_dim())
            rank[bx] = cluster_ctarank()
            size[bx] = cluster_nctarank()

    return probe


# 2 and 8 are portable; 16 is non-portable and only valid on devices that
# report support for it. Each test skips sizes above the device's reported max.
# Probes are built for both launch shapes (grid-stride loop and lean 3D grid) so
# clusters are exercised on both paths.
CLUSTER_PROBES = {(cd, gs): make_cluster_probe(cd, gs) for cd in (2, 8, 16) for gs in (True, False)}


# Functional kernels for the cross-device launch, max_blocks, and APIC tests.
@wp.kernel(cluster_dim=2, enable_backward=False, module="unique")
def fill_index(a: wp.array[int]):
    i = wp.tid()
    a[i] = i


@wp.kernel(launch_bounds=128, cluster_dim=2, enable_backward=False, module="unique")
def fill_index_bounded(a: wp.array[int]):
    i = wp.tid()
    a[i] = i


@wp.kernel(cluster_dim=4, enable_backward=False, module="unique")
def fill_ones(a: wp.array[int]):
    a[wp.tid()] = 1


@wp.kernel(cluster_dim=2, enable_backward=False, module="unique")
def cluster_scale(a: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 3.0


# Lean (grid_stride=False) clustered kernel for the empty-launch regression below.
@wp.kernel(cluster_dim=2, grid_stride=False, enable_backward=False, module="unique")
def cluster_lean_fill(a: wp.array[int]):
    a[wp.tid()] = 1


def run_cluster_probe(test, probe, cluster_total, n_clusters, device, block_dim=32):
    """Launch *probe* and verify every CTA sees the requested cluster size and a
    valid rank within its cluster."""
    n_blocks = n_clusters * cluster_total
    rank = wp.zeros(n_blocks, dtype=wp.uint32, device=device)
    size = wp.zeros(n_blocks, dtype=wp.uint32, device=device)

    wp.launch(probe, dim=n_blocks * block_dim, inputs=[rank, size], device=device, block_dim=block_dim)

    rank_np = rank.numpy()
    size_np = size.numpy()

    assert_np_equal(size_np, np.full(n_blocks, cluster_total, dtype=np.uint32))
    for c in range(n_clusters):
        ranks = sorted(int(x) for x in rank_np[c * cluster_total : (c + 1) * cluster_total])
        test.assertEqual(ranks, list(range(cluster_total)), f"cluster {c} ranks should be a permutation of 0..N")


# Device-parametrized tests (registered at the bottom via add_function_test).


def test_cluster_kernel_runs_on_all_devices(test, device):
    # cluster_dim is silently ignored on CPU and sub-Hopper CUDA; clustered
    # kernels -- including a launch_bounds + cluster_dim combination -- must
    # still run and produce correct results everywhere. Block counts are kept
    # cluster-aligned (a multiple of cluster_dim) since Hopper+ rejects padded
    # launch shapes; on CPU and sub-Hopper the alignment is irrelevant.
    n = 256
    a = wp.zeros(n, dtype=int, device=device)
    # cluster_dim=2, block_dim=128 -> 2 blocks (aligned).
    wp.launch(fill_index, dim=n, inputs=[a], device=device, block_dim=128)
    assert_np_equal(a.numpy(), np.arange(n, dtype=np.int32))

    b = wp.zeros(n, dtype=int, device=device)
    # cluster_dim=2, block_dim=128 -> 2 blocks (aligned).
    wp.launch(fill_index_bounded, dim=n, inputs=[b], device=device, block_dim=128)
    assert_np_equal(b.numpy(), np.arange(n, dtype=np.int32))


def test_clusters_form_with_requested_size(test, device):
    max_cluster_dim = wp.get_cuda_max_cluster_dim(query_probe, device)
    test.assertGreaterEqual(max_cluster_dim, 2)
    for (cluster_total, grid_stride), probe in CLUSTER_PROBES.items():
        if cluster_total > max_cluster_dim:
            continue
        with test.subTest(cluster_dim=cluster_total, grid_stride=grid_stride):
            run_cluster_probe(test, probe, cluster_total, n_clusters=2, device=device)


def test_unaligned_cluster_launch_rejected(test, device):
    # A clustered launch whose natural block count (ceil(dim / block_dim)) is not
    # a multiple of cluster_dim must be rejected rather than padded: padded CTAs
    # would skip the kernel body yet still join their cluster, breaking native
    # distributed shared memory / cluster barriers. fill_ones is cluster_dim=4;
    # dim=160, block_dim=32 -> 5 natural blocks, not a multiple of 4.
    a = wp.zeros(160, dtype=int, device=device)
    with test.assertRaisesRegex(ValueError, "multiple of cluster_dim"):
        wp.launch(fill_ones, dim=160, inputs=[a], device=device, block_dim=32)

    # An aligned shape (dim=128 -> 4 blocks) launches cleanly.
    b = wp.zeros(128, dtype=int, device=device)
    wp.launch(fill_ones, dim=128, inputs=[b], device=device, block_dim=32)
    assert_np_equal(b.numpy(), np.ones(128, dtype=np.int32))


def test_max_blocks_cluster_rounding(test, device):
    # max_blocks caps an *aligned* launch grid into a grid-stride loop. The grid
    # is rounded *down* to a valid cluster multiple (never padded up); every
    # launched CTA still runs the body via the grid-stride loop, so all elements
    # are covered. block_dim=256, dim=2048 -> natural grid 8 (aligned, cluster_dim
    # =4); max_blocks=6 rounds down to 4 blocks (one cluster_dim=4 cluster).
    a = wp.zeros(2048, dtype=int, device=device)
    wp.launch(fill_ones, dim=2048, inputs=[a], device=device, block_dim=256, max_blocks=6)
    assert_np_equal(a.numpy(), np.ones(2048, dtype=np.int32))

    # max_blocks below cluster_dim cannot form even one cluster -> clear error.
    b = wp.zeros(2048, dtype=int, device=device)
    with test.assertRaisesRegex(ValueError, "max_blocks.*cluster_dim"):
        wp.launch(fill_ones, dim=2048, inputs=[b], device=device, block_dim=256, max_blocks=3)


def test_cluster_hooks_use_loaded_module_arch(test, device):
    # Regression: the first kernel-hook lookup must classify cluster support from
    # the arch the loaded binary was compiled for (frozen on ModuleExec at load),
    # not from the current global config, which can change between load and first
    # launch. We load on a cluster-capable device, then simulate the global
    # compile target dropping below sm90; hook lookup must not be fooled into
    # rejecting the already-loaded sm90+ module.
    @wp.kernel(cluster_dim=2, enable_backward=False, module="unique")
    def clustered_copy(out: wp.array[int]):
        out[wp.tid()] = wp.tid()

    block_dim = 32
    module_exec = clustered_copy.module.load(device, block_dim)
    test.assertIsNotNone(module_exec)
    test.assertIsNotNone(module_exec.compile_arch)
    test.assertGreaterEqual(module_exec.compile_arch, 90)

    original = type(device).get_cuda_compile_arch
    try:
        # Pretend a later config change retargets compiles below sm90.
        type(device).get_cuda_compile_arch = lambda self: 75
        hooks = module_exec.get_kernel_hooks(clustered_copy)  # must not raise
    finally:
        type(device).get_cuda_compile_arch = original

    test.assertEqual(hooks.cluster_dim, 2)


def test_clustered_launch_set_dim_revalidates(test, device):
    # A recorded clustered Launch re-validates cluster alignment on set_dim, so a
    # relaunch with a misaligned shape raises a clear Python error rather than
    # surfacing only as an opaque native CUDA error at launch time. fill_ones is
    # cluster_dim=4, block_dim=32.
    a = wp.zeros(256, dtype=int, device=device)
    launch = wp.launch(fill_ones, dim=128, inputs=[a], device=device, block_dim=32, record_cmd=True)
    launch.launch()  # 128/32 = 4 blocks (aligned)

    # 160/32 = 5 blocks: not a multiple of cluster_dim=4 -> rejected on set_dim.
    with test.assertRaisesRegex(ValueError, "multiple of cluster_dim"):
        launch.set_dim(160)

    # Re-aligning to 256/32 = 8 blocks works.
    launch.set_dim(256)
    launch.launch()
    assert_np_equal(a.numpy(), np.ones(256, dtype=np.int32))


def test_clustered_lean_set_dim_zero(test, device):
    # A recorded clustered lean (grid_stride=False) launch resized to zero work items must be a
    # no-op, not a CUDA error. Zero blocks passes the cluster alignment check, but the empty-grid
    # fallback uses gridDim.x=1, which is not a multiple of cluster_dim; the native launch must be
    # skipped entirely for an empty grid. cluster_dim=2, block_dim=32, dim=64 -> 2 blocks (aligned).
    a = wp.zeros(64, dtype=int, device=device)
    cmd = wp.launch(cluster_lean_fill, dim=64, inputs=[a], device=device, block_dim=32, record_cmd=True)
    cmd.set_dim(0)
    cmd.launch()  # empty clustered lean launch: must not raise
    wp.synchronize_device(device)
    assert_np_equal(a.numpy(), np.zeros(64, dtype=np.int32))  # nothing ran


def test_clustered_kernel_in_cuda_graph(test, device):
    # A clustered launch must form clusters at the requested size when captured
    # into a standard (non-APIC) CUDA graph and replayed -- distinct from the
    # APIC capture path covered above. A rank probe verifies every CTA sees the
    # right cluster size and a valid rank after graph replay.
    block_dim = 32
    cluster_dim = 2
    n_clusters = 2
    probe = CLUSTER_PROBES[(cluster_dim, True)]
    n_blocks = n_clusters * cluster_dim
    rank = wp.zeros(n_blocks, dtype=wp.uint32, device=device)
    size = wp.zeros(n_blocks, dtype=wp.uint32, device=device)

    # Preload so no module load happens during capture.
    probe.module.load(device, block_dim)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        wp.launch(probe, dim=n_blocks * block_dim, inputs=[rank, size], device=device, block_dim=block_dim)
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    assert_np_equal(size.numpy(), np.full(n_blocks, cluster_dim, dtype=np.uint32))
    for c in range(n_clusters):
        ranks = sorted(int(x) for x in rank.numpy()[c * cluster_dim : (c + 1) * cluster_dim])
        test.assertEqual(ranks, list(range(cluster_dim)), f"cluster {c} ranks should be a permutation of 0..N")

    # max_blocks grid-stride truncation through a captured graph: dim=2048,
    # block_dim=256 -> natural grid 8 (aligned, cluster_dim=4); max_blocks=6
    # rounds down to 4 blocks and the grid-stride loop covers all elements.
    out = wp.zeros(2048, dtype=int, device=device)
    fill_ones.module.load(device, 256)
    with wp.ScopedCapture(device=device, force_module_load=False) as strided:
        wp.launch(fill_ones, dim=2048, inputs=[out], device=device, block_dim=256, max_blocks=6)
    wp.capture_launch(strided.graph)
    wp.synchronize_device(device)
    assert_np_equal(out.numpy(), np.ones(2048, dtype=np.int32))


def test_apic_save_load_preserves_clusters(test, device):
    # A clustered kernel must survive APIC capture/save/load: cluster_dim is
    # serialized into the APIC record and replayed on load.
    n = 512
    block_dim = 32
    a = wp.array(np.arange(n, dtype=np.float32), dtype=float, device=device)
    out = wp.zeros(n, dtype=float, device=device)

    # Preload so no module load happens during graph capture.
    module_exec = cluster_scale.module.load(device, block_dim)
    module_exec.get_kernel_hooks(cluster_scale)

    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(cluster_scale, dim=n, inputs=[a, out], device=device, block_dim=block_dim)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cluster_save_load")
        wp.capture_save(capture.graph, path, inputs={"a": a}, outputs={"out": out})

        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)

        result = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("out", result)
        assert_np_equal(result.numpy(), np.arange(n, dtype=np.float32) * 3.0, tol=1e-6)


def test_max_cluster_dim_unsupported_returns_one(test, device):
    # get_cuda_max_cluster_dim reports 1 on devices that cannot form clusters
    # (CPU and sub-Hopper CUDA).
    if not (device.is_cuda and device.arch >= 90):
        test.assertEqual(wp.get_cuda_max_cluster_dim(query_probe, device), 1)


def test_get_cuda_max_cluster_dim_preserves_module_block_dim(test, device):
    # Regression: the query loads the module at a probe block_dim, which mutates
    # module.options["block_dim"] as a side effect. It must restore the prior
    # value so later launches that rely on the default are unaffected.
    @wp.kernel(module="unique")
    def k(a: wp.array[int]):
        a[wp.tid()] = 0

    prior = k.module.options["block_dim"]
    wp.get_cuda_max_cluster_dim(k, device, block_dim=512)
    test.assertEqual(k.module.options["block_dim"], prior)


class TestClusterDim(unittest.TestCase):
    """``cluster_dim`` kernel option: validation, codegen, and the sub-sm90 guard.

    Device-dependent behavior (cross-device launches, Hopper+ cluster formation,
    APIC replay, driver queries) is registered below via ``add_function_test``.
    """

    def test_cluster_dim_accepted(self):
        # Default: no option key is added, so existing kernels are unaffected.
        @wp.kernel
        def k_default(a: wp.array[int]):
            a[wp.tid()] = 0

        self.assertNotIn("cluster_dim", k_default.options)

        # Decorator value is stored.
        @wp.kernel(cluster_dim=4)
        def k_decorated(a: wp.array[int]):
            a[wp.tid()] = 0

        self.assertEqual(k_decorated.options["cluster_dim"], 4)

        # Module option value is honored at launch (no-op on CPU, but must run).
        @wp.kernel(module="unique")
        def k_module(a: wp.array[int]):
            i = wp.tid()
            a[i] = i

        k_module.module.options["cluster_dim"] = 8
        a = wp.zeros(4, dtype=int, device="cpu")
        wp.launch(k_module, dim=4, inputs=[a], device="cpu")
        assert_np_equal(a.numpy(), np.arange(4, dtype=np.int32))

    def test_invalid_cluster_dim_rejected(self):
        # Decorator validates eagerly: out-of-range values raise at decoration.
        for bad in (0, -1, 17):
            with self.subTest(cluster_dim=bad), self.assertRaises(ValueError):

                @wp.kernel(cluster_dim=bad)
                def k(a: wp.array[int]):
                    a[wp.tid()] = 0

        # cluster_dim is a 1D CTA count, not a shape tuple.
        with self.assertRaises(TypeError):

            @wp.kernel(cluster_dim=(2, 1, 1))
            def k_tuple(a: wp.array[int]):
                a[wp.tid()] = 0

        # A bad module-level override surfaces a clear error at launch.
        for bad in (0, 17):
            with self.subTest(module_option=bad):

                @wp.kernel(module="unique")
                def k_module(a: wp.array[int]):
                    a[wp.tid()] = 0

                k_module.module.options["cluster_dim"] = bad
                a = wp.zeros(1, dtype=int, device="cpu")
                with self.assertRaisesRegex(ValueError, "cluster_dim"):
                    wp.launch(k_module, dim=1, inputs=[a], device="cpu")

    def test_codegen_emits_cluster_macro(self):
        # No macro for the default or the explicit no-op cluster_dim=1.
        @wp.kernel(module="unique")
        def k_default(a: wp.array[int]):
            a[wp.tid()] = 0

        self.assertNotIn("WP_CLUSTER_DIMS(", cuda_kernel_source(k_default))

        @wp.kernel(cluster_dim=1, module="unique")
        def k_one(a: wp.array[int]):
            a[wp.tid()] = 0

        self.assertNotIn("WP_CLUSTER_DIMS(", cuda_kernel_source(k_one))

        # A non-trivial value emits the macro on both forward and backward
        # signatures, alongside any launch bounds.
        @wp.kernel(launch_bounds=128, cluster_dim=2, enable_backward=True, module="unique")
        def k_two(a: wp.array[float]):
            i = wp.tid()
            a[i] = float(i)

        src = cuda_kernel_source(k_two)
        self.assertEqual(src.count("WP_CLUSTER_DIMS(2, 1, 1)"), 2)
        self.assertIn("__launch_bounds__(128)", src)

        # The module-option path emits the macro too.
        @wp.kernel(module="unique")
        def k_module(a: wp.array[int]):
            a[wp.tid()] = 0

        k_module.module.options["cluster_dim"] = 4
        self.assertIn("WP_CLUSTER_DIMS(4, 1, 1)", cuda_kernel_source(k_module))

    def test_aot_cluster_dim_below_sm90_errors(self):
        # Compiling a clustered kernel for an arch < sm90 must hard-error rather
        # than emit a binary with the cluster attribute silently stripped by the
        # WP_CLUSTER_DIMS guard. The AOT-by-arch path has no backing device, so
        # it exercises the "dropped" branch directly (the same compile-path check
        # also fires for JIT loads onto a cluster-capable device compiled below
        # sm90, e.g. with warp.config.ptx_target_arch < 90).
        @wp.kernel(cluster_dim=2, module="unique")
        def k(a: wp.array[float]):
            i = wp.tid()
            a[i] = a[i] * 2.0

        with tempfile.TemporaryDirectory() as module_dir:
            with self.assertRaisesRegex(RuntimeError, "cluster_dim"):
                wp.compile_aot_module(k.module, device=None, arch=80, module_dir=module_dir)

            # A cluster-capable target compiles cleanly.
            wp.compile_aot_module(k.module, device=None, arch=90, module_dir=module_dir)

    def test_aot_stale_live_clustered_kernel_errors(self):
        # Regression: the sub-sm90 AOT guard must scan the same live unique-kernel
        # set codegen emits, not module.kernels (latest per key). A kernel
        # redefined with the same key leaves the latest entry unclustered while an
        # older *live* clustered kernel still generates WP_CLUSTER_DIMS. Guarding
        # only module.kernels would let that AOT-compile for sm < 90 with the
        # cluster attribute silently stripped.
        mod_name = "test_cluster_stale_live"

        def define(clustered: bool):
            if clustered:

                @wp.kernel(cluster_dim=2, module=mod_name)
                def k(a: wp.array[int]):
                    a[wp.tid()] = 0
            else:

                @wp.kernel(module=mod_name)
                def k(a: wp.array[int]):
                    a[wp.tid()] = 0

            return k

        clustered_kernel = define(clustered=True)  # registered first, clustered
        plain_kernel = define(clustered=False)  # same key, overwrites module.kernels

        # Preconditions: same key; module.kernels holds only the latest (plain)
        # while both remain live.
        self.assertEqual(clustered_kernel.key, plain_kernel.key)
        module = clustered_kernel.module
        self.assertIs(module.kernels[plain_kernel.key], plain_kernel)
        live = module._get_live_kernels()
        self.assertIn(clustered_kernel, live)
        self.assertIn(plain_kernel, live)

        with tempfile.TemporaryDirectory() as module_dir:
            with self.assertRaisesRegex(RuntimeError, "cluster_dim"):
                wp.compile_aot_module(module, device=None, arch=80, module_dir=module_dir)


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()
hopper = hopper_devices()

add_function_test(
    TestClusterDim, "test_cluster_kernel_runs_on_all_devices", test_cluster_kernel_runs_on_all_devices, devices=devices
)
add_function_test(
    TestClusterDim,
    "test_max_cluster_dim_unsupported_returns_one",
    test_max_cluster_dim_unsupported_returns_one,
    devices=devices,
)
add_function_test(
    TestClusterDim,
    "test_get_cuda_max_cluster_dim_preserves_module_block_dim",
    test_get_cuda_max_cluster_dim_preserves_module_block_dim,
    devices=cuda_devices,
)
add_function_test(
    TestClusterDim, "test_clusters_form_with_requested_size", test_clusters_form_with_requested_size, devices=hopper
)
add_function_test(TestClusterDim, "test_max_blocks_cluster_rounding", test_max_blocks_cluster_rounding, devices=hopper)
add_function_test(
    TestClusterDim, "test_unaligned_cluster_launch_rejected", test_unaligned_cluster_launch_rejected, devices=hopper
)
add_function_test(
    TestClusterDim,
    "test_cluster_hooks_use_loaded_module_arch",
    test_cluster_hooks_use_loaded_module_arch,
    devices=hopper,
)
add_function_test(
    TestClusterDim,
    "test_clustered_launch_set_dim_revalidates",
    test_clustered_launch_set_dim_revalidates,
    devices=hopper,
)
add_function_test(TestClusterDim, "test_clustered_lean_set_dim_zero", test_clustered_lean_set_dim_zero, devices=hopper)
add_function_test(
    TestClusterDim, "test_clustered_kernel_in_cuda_graph", test_clustered_kernel_in_cuda_graph, devices=hopper
)
add_function_test(
    TestClusterDim, "test_apic_save_load_preserves_clusters", test_apic_save_load_preserves_clusters, devices=hopper
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
