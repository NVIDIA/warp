# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from warp.types import constant

__all__ = [
    "E",
    "e",
    "LOG2E",
    "log2e",
    "LOG10E",
    "log10e",
    "LN2",
    "ln2",
    "LN10",
    "ln10",
    "PHI",
    "phi",
    "PI",
    "pi",
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
TAU = tau = constant(6.28318530717958647692)  # 2 * pi
