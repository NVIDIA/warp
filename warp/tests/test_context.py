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

import warp as wp


class TestContext(unittest.TestCase):
    def test_context_type_str(self):
        self.assertEqual(wp._src.context.type_str(list[int]), "list[int]")
        self.assertEqual(wp._src.context.type_str(list[float]), "list[float]")

        self.assertEqual(wp._src.context.type_str(tuple[int]), "tuple[int]")
        self.assertEqual(wp._src.context.type_str(tuple[float]), "tuple[float]")
        self.assertEqual(wp._src.context.type_str(tuple[int, float]), "tuple[int, float]")
        self.assertEqual(wp._src.context.type_str(tuple[int, ...]), "tuple[int, ...]")

    def test_resolve_supported_ptx_arch_exact_match(self):
        """Target arch is in the supported set — returned as-is."""
        resolve = wp._src.context._resolve_supported_ptx_arch
        self.assertEqual(resolve(75, {70, 75, 80, 86}), 75)
        self.assertEqual(resolve(86, {70, 75, 80, 86}), 86)

    def test_resolve_supported_ptx_arch_clamp_up(self):
        """Target arch missing — lowest supported arch >= target is chosen."""
        resolve = wp._src.context._resolve_supported_ptx_arch
        # 75 not in set, next above is 80
        self.assertEqual(resolve(75, {70, 80, 86}), 80)
        # 72 not in set, next above is 75
        self.assertEqual(resolve(72, {60, 75, 80}), 75)

    def test_resolve_supported_ptx_arch_fallback_to_max(self):
        """All supported archs are below the target — highest supported is returned."""
        resolve = wp._src.context._resolve_supported_ptx_arch
        self.assertEqual(resolve(90, {70, 75, 80}), 80)

    def test_resolve_supported_ptx_arch_single_element(self):
        """Only one supported arch available."""
        resolve = wp._src.context._resolve_supported_ptx_arch
        self.assertEqual(resolve(75, {80}), 80)
        self.assertEqual(resolve(90, {80}), 80)
        self.assertEqual(resolve(80, {80}), 80)

    def test_resolve_supported_ptx_arch_empty_raises(self):
        """Empty supported set must raise an explicit error."""
        resolve = wp._src.context._resolve_supported_ptx_arch
        with self.assertRaises(ValueError):
            resolve(75, set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
