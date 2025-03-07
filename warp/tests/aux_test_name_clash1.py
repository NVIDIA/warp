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


# test clashes with identical struct from another module
@wp.struct
class SameStruct:
    x: float


# test clashes with identically named but different struct from another module
@wp.struct
class DifferentStruct:
    v: float


# test clashes with identical function from another module
@wp.func
def same_func():
    return 99


# test clashes with identically named but different function from another module
@wp.func
def different_func():
    return 17
