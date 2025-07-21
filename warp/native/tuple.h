/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

template <typename... Types>
struct tuple_t;

template <>
struct tuple_t<>
{

    static constexpr int size() { return 0; }

    // Base case: empty tuple.
    template <typename Callable>
    void apply(Callable&&) const { }
};

template <typename Head, typename... Tail>
struct tuple_t<Head, Tail...>
{
    Head head;
    tuple_t<Tail...> tail;

    CUDA_CALLABLE inline tuple_t() {}
    CUDA_CALLABLE inline tuple_t(Head h, Tail... t) : head(h), tail(t...) {}

    static constexpr int size() { return 1 + tuple_t<Tail...>::size(); }

    // Applies a callable to each element.
    template <typename Callable>
    void apply(Callable&& func) const
    {
        func(head);        // Apply the callable to the current element.
        tail.apply(func);  // Recursively process the rest of the tuple.
    }
};

// Tuple constructor.
template <typename... Args>
CUDA_CALLABLE inline tuple_t<Args...>
tuple(
    Args... args
)
{
    return tuple_t<Args...>(args...);
}

// Helper to extract a value from the tuple.
// Can be replaced with simpler member function version when our CPU compiler
// backend supports constexpr if statements.
template <int N, typename Head, typename... Tail>
struct tuple_get
{
    static CUDA_CALLABLE inline const auto&
    value(
        const tuple_t<Head, Tail...>& t
    )
    {
        return tuple_get<N - 1, Tail...>::value(t.tail);
    }
};

// Specialization for the base case N == 0. Simply return the head of the tuple.
template <typename Head, typename... Tail>
struct tuple_get<0, Head, Tail...>
{
    static CUDA_CALLABLE inline const auto&
    value(
        const tuple_t<Head, Tail...>& t
    )
    {
        return t.head;
    }
};

template <int Index, typename... Args>
CUDA_CALLABLE inline auto
extract(
    const tuple_t<Args...>& t
)
{
    return tuple_get<Index, Args...>::value(t);
}

template <typename... Args>
CUDA_CALLABLE inline int
len(
    const tuple_t<Args...>& t
)
{
    return t.size();
}

template <typename... Args>
CUDA_CALLABLE inline void
adj_len(
    const tuple_t<Args...>& t,
    tuple_t<Args...>& adj_t,
    int adj_ret
)
{
}

template <typename... Args>
CUDA_CALLABLE inline void
print(
    const tuple_t<Args...>& t
)
{
    t.apply([&](auto a) { print(a); });
}

template <typename... Args>
CUDA_CALLABLE inline void
adj_print(
    const tuple_t<Args...>& t,
    tuple_t<Args...>& adj_t
)
{
    adj_t.apply([&](auto a) { print(a); });
}

CUDA_CALLABLE inline tuple_t<>
add(
    const tuple_t<>& a,
    const tuple_t<>& b
)
{
    return tuple_t<>();
}

template <typename Head, typename... Tail>
CUDA_CALLABLE inline tuple_t<Head, Tail...>
add(
    const tuple_t<Head, Tail...>& a,
    const tuple_t<Head, Tail...>& b
)
{
    tuple_t<Head, Tail...> out;
    out.head = add(a.head, b.head);
    out.tail = add(a.tail, b.tail);
    return out;
}

CUDA_CALLABLE inline void
adj_add(
    const tuple_t<>& a,
    const tuple_t<>& b,
    tuple_t<>& adj_a,
    tuple_t<>& adj_b,
    const tuple_t<>& adj_ret
)
{
}

template <typename Head, typename... Tail>
CUDA_CALLABLE inline void
adj_add(
    const tuple_t<Head, Tail...>& a,
    const tuple_t<Head, Tail...>& b,
    tuple_t<Head, Tail...>& adj_a,
    tuple_t<Head, Tail...>& adj_b,
    const tuple_t<Head, Tail...>& adj_ret
)
{
    adj_add(a.head, b.head, adj_a.head, adj_b.head, adj_ret.head);
    adj_add(a.tail, b.tail, adj_a.tail, adj_b.tail, adj_ret.tail);
}

} // namespace wp
