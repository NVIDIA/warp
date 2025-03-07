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
