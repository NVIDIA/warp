# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest
from typing import List, Tuple

import warp as wp
from warp.tests.unittest_utils import *


def test_context_type_str(test, device):
    assert wp.context.type_str(List[int]) == "List[int]"
    assert wp.context.type_str(List[float]) == "List[float]"

    assert wp.context.type_str(Tuple[int]) == "Tuple[int]"
    assert wp.context.type_str(Tuple[float]) == "Tuple[float]"
    assert wp.context.type_str(Tuple[int, float]) == "Tuple[int, float]"
    assert wp.context.type_str(Tuple[int, ...]) == "Tuple[int, ...]"


class TestContext(unittest.TestCase):
    pass


add_function_test(TestContext, "test_context_type_str", test_context_type_str)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
