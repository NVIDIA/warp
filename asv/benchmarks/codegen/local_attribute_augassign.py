# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compilation and adjoints for repeated local attribute updates.

Unlike array-slot and subscript benchmarks, these exercise ``value.y +=``
and ``value.p +=`` after loading a differentiable aggregate. Compilation
includes Python code generation, native compilation, and loading, with warm
compiler headers but no kernel or CUDA compiler cache hits.
"""

import os

import warp as wp

from ..benchmarks_utils import CudaEventTimer, capture_launches, setup_once


@wp.kernel(module="unique", enable_backward=True, module_options={"max_unroll": 64})
def vector_updates(a: wp.array[wp.vec3], b: wp.array[float], out: wp.array[wp.vec3]):
    i = wp.tid()
    value = a[i]
    delta = b[i]
    for _ in range(64):
        value.y += delta
    out[i] = value


@wp.kernel(module="unique", enable_backward=True, module_options={"max_unroll": 64})
def translation_updates(a: wp.array[wp.transform], b: wp.array[float], out: wp.array[wp.transform]):
    i = wp.tid()
    value = a[i]
    delta = b[i]
    for _ in range(64):
        value.p += wp.vec3(delta)
    out[i] = value


@wp.kernel(module="unique", enable_backward=True)
def prime_compiler(a: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    value = a[wp.tid()]
    value[1] += 1.0
    out[wp.tid()] = value


class CompileModule:
    params = (["cpu", "cuda:0"], ["vec3", "transform"])
    param_names = ["device", "aggregate"]
    # Each call recompiles. More calls per sample multiply that cost.
    # Four repeats with ASV's inherited two rounds provide eight raw samples;
    # calibrated against the 10% CI threshold without overriding rounds.
    number = 1
    repeat = 4

    @setup_once
    def _initialize(self, device, aggregate):
        # ASV's spawn process imports this module before initializing Warp.
        # Warp cache misses alone do not bypass NVRTC's driver-backed cache.
        if device.startswith("cuda"):
            os.environ["CUDA_CACHE_DISABLE"] = "1"
        wp.init()
        self.module = (vector_updates if aggregate == "vec3" else translation_updates).module
        wp.load_module(module=prime_compiler.module, device=device)

    def setup(self, device, aggregate):
        self._cache_kernels = wp.config.cache_kernels
        wp.config.cache_kernels = False
        self._initialize(device, aggregate)
        self.module.unload()

    def teardown(self, device, aggregate):
        self.module.unload()
        wp.config.cache_kernels = self._cache_kernels

    def time_compile(self, device, aggregate):
        wp.load_module(module=self.module, device=device)


@wp.kernel(module="unique", enable_backward=True, module_options={"max_unroll": 1})
def translation_loop(a: wp.array[wp.transform], b: wp.array[float], out: wp.array[wp.transform]):
    i = wp.tid()
    value = a[i]
    delta = b[i]
    for _ in range(64):
        value.p += wp.vec3(delta)
    out[i] = value


class RunBackwardCPU:
    # Millisecond launches need no batching; retain the same eight samples.
    number = 1
    repeat = 4

    @setup_once
    def setup(self):
        wp.init()
        n = 1 << 15
        self.a = wp.ones(n, dtype=wp.transform, device="cpu", requires_grad=True)
        self.b = wp.full(n, 0.001, dtype=float, device="cpu", requires_grad=True)
        # Each adjoint must see the same nonzero seed, including after warmup.
        self.out = wp.zeros_like(self.a, retain_grad=True)
        wp.launch(translation_loop, n, inputs=[self.a, self.b], outputs=[self.out], device="cpu")
        self.out.grad.fill_(1.0)
        self.command = wp.launch(
            translation_loop,
            n,
            inputs=[self.a, self.b],
            outputs=[self.out],
            adj_inputs=[self.a.grad, self.b.grad],
            adj_outputs=[self.out.grad],
            adjoint=True,
            device="cpu",
            record_cmd=True,
        )

    def time_translation(self):
        # Only gradients accumulate; primal inputs and the seeded output stay
        # unchanged. CPU launches complete synchronously.
        self.command.launch()


vec32 = wp.types.vector(length=32, dtype=wp.float32)


@wp.kernel(module="unique", enable_backward=True, module_options={"max_unroll": 1})
def wide_vector_loop(a: wp.array[vec32], b: wp.array[float], out: wp.array[vec32]):
    i = wp.tid()
    value = a[i]
    delta = b[i]
    for _ in range(64):
        value.y += delta
    out[i] = value


class RunWideBackwardCPU:
    """Measure the wide-vector adjoint cost of repeated local updates."""

    # One call met the precision target; four repeats with inherited rounds
    # provide eight raw samples without extra batching.
    number = 1
    repeat = 4

    @setup_once
    def setup(self):
        wp.init()
        n = 1 << 15
        self.a = wp.ones(n, dtype=vec32, device="cpu", requires_grad=True)
        self.b = wp.full(n, 0.001, dtype=float, device="cpu", requires_grad=True)
        self.out = wp.zeros_like(self.a, retain_grad=True)
        wp.launch(wide_vector_loop, n, inputs=[self.a, self.b], outputs=[self.out], device="cpu")
        self.out.grad.fill_(1.0)
        self.command = wp.launch(
            wide_vector_loop,
            n,
            inputs=[self.a, self.b],
            outputs=[self.out],
            adj_inputs=[self.a.grad, self.b.grad],
            adj_outputs=[self.out.grad],
            adjoint=True,
            device="cpu",
            record_cmd=True,
        )

    def time_vector(self):
        self.command.launch()


class RunBackwardCUDA:
    """Time a graph of 32 transform adjoint launches on CUDA."""

    timer = CudaEventTimer()
    # Calibration gives a ~2 ms replay, keeping enqueue cost a small share.
    launches_per_sample = 32
    # Calibrated at one replay per sample and eight samples over two rounds.
    number = 1
    repeat = 4

    @setup_once
    def setup(self):
        wp.init()
        n = 1 << 18
        self.a = wp.ones(n, dtype=wp.transform, device="cuda:0", requires_grad=True)
        self.b = wp.full(n, 0.001, dtype=float, device="cuda:0", requires_grad=True)
        self.out = wp.zeros_like(self.a, retain_grad=True)
        wp.launch(translation_loop, n, inputs=[self.a, self.b], outputs=[self.out], device="cuda:0")
        self.out.grad.fill_(1.0)
        self.command = wp.launch(
            translation_loop,
            n,
            inputs=[self.a, self.b],
            outputs=[self.out],
            adj_inputs=[self.a.grad, self.b.grad],
            adj_outputs=[self.out.grad],
            adjoint=True,
            device="cuda:0",
            record_cmd=True,
        )
        self.graph = capture_launches(self.command.launch, self.launches_per_sample)
        wp.capture_launch(self.graph)
        wp.synchronize_device("cuda:0")

    def time_translation(self):
        wp.capture_launch(self.graph)
