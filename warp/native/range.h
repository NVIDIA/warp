/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

namespace wp
{

// All iterable types should implement 3 methods:
//
// T iter_next(iter)       - returns the current value and moves iterator to next state 
// int iter_cmp(iter)      - returns 0 if finished
// iter iter_reverse(iter) - return an iterator of the same type representing the reverse order
//
// iter_next() should also be registered as a built-in hidden function so that code-gen
// can call it and generate the appropriate variable storage

// represents a built-in Python range() loop
struct range_t
{
    CUDA_CALLABLE range_t()
        : start(0),
          end(0),
          step(0),
          i(0)
    {}

    int start;
    int end;
    int step;
    
    int i;
};

CUDA_CALLABLE inline range_t range(int end)
{
    range_t r;
    r.start = 0;
    r.end = end;
    r.step = 1;
    
    r.i = r.start;

    return r;
}

CUDA_CALLABLE inline range_t range(int start, int end)
{
    range_t r;
    r.start = start;
    r.end = end;
    r.step = 1;
    
    r.i = r.start;

    return r;
}

CUDA_CALLABLE inline range_t range(int start, int end, int step)
{
    range_t r;
    r.start = start;
    r.end = end;
    r.step = step;
    
    r.i = r.start;

    return r;
}


CUDA_CALLABLE inline void adj_range(int end, int adj_end, range_t& adj_ret) {}
CUDA_CALLABLE inline void adj_range(int start, int end, int adj_start, int adj_end, range_t& adj_ret) {}
CUDA_CALLABLE inline void adj_range(int start, int end, int step, int adj_start, int adj_end, int adj_step, range_t& adj_ret) {}


CUDA_CALLABLE inline int iter_next(range_t& r)
{
    int iter = r.i;

    r.i += r.step;
    return iter;
}

CUDA_CALLABLE inline bool iter_cmp(const range_t& r)
{
    // implements for-loop comparison to emulate Python range() loops with negative arguments
    if (r.step == 0)
        // degenerate case where step == 0
        return false;
    if (r.step > 0)
        // normal case where step > 0
        return r.i < r.end;
    else
        // reverse case where step < 0
        return r.i > r.end;
}

CUDA_CALLABLE inline range_t iter_reverse(const range_t& r)
{
    // generates a reverse range, equivalent to reversed(range())
    range_t rev;

    if (r.step > 0)
    {
        rev.start = r.start + int((r.end - r.start - 1) / r.step) * r.step;
    }
    else
    {
        rev.start = r.start + int((r.end - r.start + 1) / r.step) * r.step;
    }

    rev.end = r.start - r.step;
    rev.step = -r.step;

    rev.i = rev.start;
    
    return rev;
}

CUDA_CALLABLE inline void adj_iter_reverse(const range_t& r, range_t& adj_r, range_t& adj_ret)
{
}

} // namespace wp
