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

import math

from warp.types import constant

__all__ = [
    "E",
    "e",
    "INF",
    "inf",
    "LOG2E",
    "log2e",
    "LOG10E",
    "log10e",
    "LN2",
    "ln2",
    "LN10",
    "ln10",
    "NAN",
    "nan",
    "PHI",
    "phi",
    "PI",
    "pi",
    "HALF_PI",
    "half_pi",
    "TAU",
    "tau",
]

E = e = constant(2.71828182845904523536)  # e
LOG2E = log2e = constant(1.44269504088896340736)  # log2(e)
LOG10E = log10e = constant(0.43429448190325182765)  # log10(e)
LN2 = ln2 = constant(0.69314718055994530942)  # ln(2)
LN10 = ln10 = constant(2.30258509299404568402)  # ln(10)
PHI = phi = constant(1.61803398874989484820)  # golden constant
PI = pi = constant(3.14159265358979323846)  # pi
HALF_PI = half_pi = constant(1.57079632679489661923)  # half pi
TAU = tau = constant(6.28318530717958647692)  # 2 * pi

INF = inf = constant(math.inf)

NAN = nan = constant(float("nan"))
