# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

###########################################################################
# Benchmarks for kernel launches with different types of args
###########################################################################

import warp as wp


@wp.struct
class S0:
    pass


@wp.struct
class Sf:
    x: float
    y: float
    z: float


@wp.struct
class Sv:
    u: wp.vec3
    v: wp.vec3
    w: wp.vec3


@wp.struct
class Sm:
    M: wp.mat33
    N: wp.mat33
    O: wp.mat33


@wp.struct
class Sa:
    a: wp.array(dtype=float)
    b: wp.array(dtype=float)
    c: wp.array(dtype=float)


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
def k0():
    tid = wp.tid()  # noqa: F841


@wp.kernel
def kf(x: float, y: float, z: float):
    tid = wp.tid()  # noqa: F841


@wp.kernel
def kv(u: wp.vec3, v: wp.vec3, w: wp.vec3):
    tid = wp.tid()  # noqa: F841


@wp.kernel
def km(M: wp.mat33, N: wp.mat33, O: wp.mat33):
    tid = wp.tid()  # noqa: F841


@wp.kernel
def ka(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
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


@wp.kernel
def ks0(s: S0):
    tid = wp.tid()  # noqa: F841


@wp.kernel
def ksf(s: Sf):
    tid = wp.tid()  # noqa: F841


@wp.kernel
def ksv(s: Sv):
    tid = wp.tid()  # noqa: F841


@wp.kernel
def ksm(s: Sm):
    tid = wp.tid()  # noqa: F841


@wp.kernel
def ksa(s: Sa):
    tid = wp.tid()  # noqa: F841


@wp.kernel
def ksz(s: Sz):
    tid = wp.tid()  # noqa: F841


wp.clear_kernel_cache()

devices = wp.get_devices()
num_launches = 100000

for device in devices:
    with wp.ScopedDevice(device):
        print(f"\n=================== Device '{device}' ===================")

        wp.force_load(device)

        n = 1
        a = wp.zeros(n, dtype=float)
        b = wp.zeros(n, dtype=float)
        c = wp.zeros(n, dtype=float)
        x = 17.0
        y = 42.0
        z = 99.0
        u = wp.vec3(1, 2, 3)
        v = wp.vec3(10, 20, 30)
        w = wp.vec3(100, 200, 300)
        M = wp.mat33()
        N = wp.mat33()
        O = wp.mat33()

        s0 = S0()

        sf = Sf()
        sf.x = x
        sf.y = y
        sf.z = z

        sv = Sv()
        sv.u = u
        sv.v = v
        sv.w = w

        sm = Sm()
        sm.M = M
        sm.N = N
        sm.O = O

        sa = Sa()
        sa.a = a
        sa.b = b
        sa.c = c

        sz = Sz()
        sz.a = a
        sz.b = b
        sz.c = c
        sz.x = x
        sz.y = y
        sz.z = z
        sz.u = u
        sz.v = v
        sz.w = w

        tk0 = wp.ScopedTimer("k0")
        tkf = wp.ScopedTimer("kf")
        tkv = wp.ScopedTimer("kv")
        tkm = wp.ScopedTimer("km")
        tka = wp.ScopedTimer("ka")
        tkz = wp.ScopedTimer("kz")

        ts0 = wp.ScopedTimer("s0")
        tsf = wp.ScopedTimer("sf")
        tsv = wp.ScopedTimer("sv")
        tsm = wp.ScopedTimer("sm")
        tsa = wp.ScopedTimer("sa")
        tsz = wp.ScopedTimer("sz")

        wp.synchronize_device()

        with tk0:
            for _ in range(num_launches):
                wp.launch(k0, dim=1, inputs=[])

        wp.synchronize_device()

        with tkf:
            for _ in range(num_launches):
                wp.launch(kf, dim=1, inputs=[x, y, z])

        wp.synchronize_device()

        with tkv:
            for _ in range(num_launches):
                wp.launch(kv, dim=1, inputs=[u, v, w])

        wp.synchronize_device()

        with tkm:
            for _ in range(num_launches):
                wp.launch(km, dim=1, inputs=[M, N, O])

        wp.synchronize_device()

        with tka:
            for _ in range(num_launches):
                wp.launch(ka, dim=1, inputs=[a, b, c])

        wp.synchronize_device()

        with tkz:
            for _ in range(num_launches):
                wp.launch(kz, dim=1, inputs=[a, b, c, x, y, z, u, v, w])

        # structs

        wp.synchronize_device()

        with ts0:
            for _ in range(num_launches):
                wp.launch(ks0, dim=1, inputs=[s0])

        wp.synchronize_device()

        with tsf:
            for _ in range(num_launches):
                wp.launch(ksf, dim=1, inputs=[sf])

        wp.synchronize_device()

        with tsv:
            for _ in range(num_launches):
                wp.launch(ksv, dim=1, inputs=[sv])

        wp.synchronize_device()

        with tsm:
            for _ in range(num_launches):
                wp.launch(ksm, dim=1, inputs=[sm])

        wp.synchronize_device()

        with tsa:
            for _ in range(num_launches):
                wp.launch(ksa, dim=1, inputs=[sa])

        wp.synchronize_device()

        with tsz:
            for _ in range(num_launches):
                wp.launch(ksz, dim=1, inputs=[sz])

        wp.synchronize_device()

        timers = [
            [tk0, ts0],
            [tkf, tsf],
            [tkv, tsv],
            [tkm, tsm],
            [tka, tsa],
            [tkz, tsz],
        ]

        print("--------------------------------")
        print("| args |    direct |    struct |")
        print("--------------------------------")
        for tk, ts in timers:
            print(f"|  {tk.name}  |{tk.elapsed:10.0f} |{ts.elapsed:10.0f} |")
        print("--------------------------------")
