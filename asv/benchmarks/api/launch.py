# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import warp as wp

wp.set_module_options({"enable_backward": False})

N = 8192


@wp.kernel
def inc_kernel(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] + 1.0


class KernelLaunch:
    number = 1000  # Number of measurements to make between a single setup and teardown

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.load_module("cuda:0")
        self.test_array = wp.zeros(N, dtype=float, device="cuda:0")
        self.stream = wp.Stream("cuda:0")
        self.cmd = wp.launch(inc_kernel, (N,), inputs=[self.test_array], record_cmd=True)
        wp.synchronize_device("cuda:0")

    def teardown(self):
        wp.synchronize_device("cuda:0")

    def time_standard_launch(self):
        """Time a standard kernel launch.

        A synchronize at the end of the function is intentionally omitted.
        """

        wp.launch(inc_kernel, (N,), inputs=[self.test_array])

    def time_launch_on_stream(self):
        """Time a kernel launch on a specified stream.

        A synchronize at the end of the function is intentionally omitted.
        """

        wp.launch(inc_kernel, (N,), inputs=[self.test_array], stream=self.stream)

    def time_launch_object(self):
        """Time a kernel launch from a stored launch object.

        A synchronize at the end of the function is intentionally omitted.
        """

        self.cmd.launch()


@wp.struct
class Sz:
    a: wp.array(dtype=float)
    b: wp.array(dtype=float)
    c: wp.array(dtype=float)
    x: float
    y: float
    z: float
    u: wp.vec3
    v: wp.vec3
    w: wp.vec3


@wp.kernel
def ksz(s: Sz):
    tid = wp.tid()  # noqa: F841


@wp.kernel
def kz(
    a: wp.array(dtype=float),
    b: wp.array(dtype=float),
    c: wp.array(dtype=float),
    x: float,
    y: float,
    z: float,
    u: wp.vec3,
    v: wp.vec3,
    w: wp.vec3,
):
    tid = wp.tid()  # noqa: F841


@wp.struct
class S0:
    pass


@wp.kernel
def ks0(s: S0):
    tid = wp.tid()  # noqa: F841


@wp.kernel
def k0():
    tid = wp.tid()  # noqa: F841


class KernelLaunchParameters:
    number = 1000

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.load_module("cuda:0")

        n = 1
        self.a = wp.zeros(n, dtype=float, device="cuda:0")
        self.b = wp.zeros(n, dtype=float, device="cuda:0")
        self.c = wp.zeros(n, dtype=float, device="cuda:0")
        self.x = 17.0
        self.y = 42.0
        self.z = 99.0
        self.u = wp.vec3(1, 2, 3)
        self.v = wp.vec3(10, 20, 30)
        self.w = wp.vec3(100, 200, 300)

        sz = Sz()
        sz.a = self.a
        sz.b = self.b
        sz.c = self.c
        sz.x = self.x
        sz.y = self.y
        sz.z = self.z
        sz.u = self.u
        sz.v = self.v
        sz.w = self.w
        self.sz = sz

        self.s0 = S0()

        wp.synchronize_device("cuda:0")

    def teardown(self):
        wp.synchronize_device("cuda:0")

    def time_direct_full(self):
        wp.launch(
            kz, dim=1, inputs=[self.a, self.b, self.c, self.x, self.y, self.z, self.u, self.v, self.w], device="cuda:0"
        )

    def time_struct_full(self):
        wp.launch(ksz, dim=1, inputs=[self.sz], device="cuda:0")

    def time_direct_empty(self):
        wp.launch(k0, dim=1, inputs=[], device="cuda:0")

    def time_struct_empty(self):
        wp.launch(ks0, dim=1, inputs=[self.s0], device="cuda:0")


class GraphLaunch:
    repeat = 10
    number = 1000

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.load_module("cuda:0")
        self.test_array = wp.zeros(N, dtype=float, device="cuda:0")
        self.stream = wp.Stream("cuda:0")

        # capture graph
        with wp.ScopedCapture(device="cuda:0") as capture:
            wp.launch(inc_kernel, (N,), inputs=[self.test_array])

        self.graph = capture.graph

        # Warmup
        for _ in range(5):
            wp.capture_launch(self.graph)

        wp.synchronize_device("cuda:0")

    def teardown(self):
        wp.synchronize_device("cuda:0")

    def time_ten_graph(self):
        for _ in range(10):
            wp.capture_launch(self.graph)

    def time_ten_graph_on_stream(self):
        for _ in range(10):
            wp.capture_launch(self.graph, stream=self.stream)
