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

 // The Apache 2 License

// Copyright 2023 Anka He Chen
// 
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.You may obtain a copy of the License at
// http ://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed under the 
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, 
// either express or implied.See the License for the specific language governing permissions
// and limitations under the License.
// 
// Source: https://github.com/AnkaChan/Gaia/blob/main/Simulator/Modules/GraphColoring/ColoringAlgorithms.cpp
//         https://github.com/AnkaChan/Gaia/blob/main/Simulator/Modules/GraphColoring/ColoringAlgorithms.h
//         https://github.com/AnkaChan/Gaia/blob/main/Simulator/Modules/GraphColoring/Graph.h



#include "warp.h"

#include <iostream>
#include <vector>
#include <array>
#include <queue>
#include <queue>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <numeric>

#define SHRINK_GRAPH_PER_PERCENTAGE (5)
#define NODE_WEIGHTS_PREALLOC (64)
#define WEIGHT_BUCKET_PREALLOC (512)

namespace wp
{

    struct Graph
    {
        Graph(int num_nodes_in, const wp::array_t<int>& edges)
            : num_nodes(num_nodes_in)
        {
            node_offsets.resize(num_nodes + 1, 0);
            node_colors.resize(num_nodes, -1);

           std::vector<int> node_degrees(num_nodes, 0);

            // count degrees
            for (size_t edge_idx = 0; edge_idx < edges.shape[0]; edge_idx++)
            {
                int e0 = *address(edges, edge_idx, 0);
                int e1 = *address(edges, edge_idx, 1);
                node_degrees[e0] += 1;
                node_degrees[e1] += 1;
            }
            
            int offset = 0;
            for (size_t node = 0; node < num_nodes; node++)
            {
                offset += node_degrees[node];
                node_offsets[node + 1] = offset;
            }

            // fill adjacency list
            std::vector<int> node_adjacency_fill_count(num_nodes, 0);
            graph_flatten.resize(offset, -1);
            for (size_t edge_idx = 0; edge_idx < edges.shape[0]; edge_idx++)
            {
                int e0 = *address(edges, edge_idx, 0);
                int e1 = *address(edges, edge_idx, 1);

                int fill_count_e0 = node_adjacency_fill_count[e0];
                graph_flatten[node_offsets[e0] + fill_count_e0] = e1;

                int fill_count_e1 = node_adjacency_fill_count[e1];
                graph_flatten[node_offsets[e1] + fill_count_e1] = e0;

                node_adjacency_fill_count[e0] = fill_count_e0 + 1;
                node_adjacency_fill_count[e1] = fill_count_e1 + 1;
            }

        }

        int get_node_neighbor(int node, int neighbor_index) const {
            return graph_flatten[node_offsets[node] + neighbor_index];
        }

        int get_node_degree(int node) const {
            return node_offsets[node + 1] - node_offsets[node];
        }


        int num_nodes;
        std::vector<int> graph_flatten;
        std::vector<int> node_offsets;
        std::vector<int> node_colors;
    };

void convert_to_color_groups(const int num_colors, const std::vector<int>& node_colors, std::vector<std::vector<int>>& color_groups)
{
    color_groups.resize(num_colors);

    for (int node_idx = 0; node_idx < node_colors.size(); node_idx++) {
        int color = node_colors[node_idx];
        color_groups[color].push_back(node_idx);
    }
}

float find_largest_smallest_groups(const std::vector<std::vector<int>>& color_groups, int& biggest_group, int& smallest_group)
{
    if (color_groups.size() == 0)
    {
        biggest_group = -1;
        smallest_group = -1;

        return 1;
    }

    size_t max_size = color_groups[0].size();
    biggest_group = 0;
    size_t min_size = color_groups[0].size();
    smallest_group = 0;

    for (size_t color = 0; color < color_groups.size(); color++)
    {
        if (max_size < color_groups[color].size()) {
            biggest_group = color;
            max_size = color_groups[color].size();
        }

        if (min_size > color_groups[color].size())
        {
            smallest_group = color;
            min_size = color_groups[color].size();
        }
    }

    return float(color_groups[biggest_group].size()) / float(color_groups[smallest_group].size());
}

bool color_changeable(const Graph& graph, int node, int target_color){
    // loop through node and see if it has target color
    for (size_t i = 0; i < graph.get_node_degree(node); i++)
    {
        int nei_node_idx = graph.get_node_neighbor(node, i);
        if (graph.node_colors[nei_node_idx] == target_color)
        {
            return false;
        }
    }
    return true;
}

int find_changeable_node_in_category(
    const Graph& graph, 
    const std::vector<std::vector<int>>& color_groups, 
    int source_color, 
    int target_color
)
{
    auto& source_group = color_groups[source_color];
    for (size_t node_idx = 0; node_idx < source_group.size(); node_idx++)
    {
        if (color_changeable(graph, source_group[node_idx], target_color)) {
            return node_idx;
        }
    }
    return -1;
}

void change_color(int color, int node_idx_in_group, int target_color, std::vector<int>& node_colors, std::vector<std::vector<int>>& color_groups)
{
    int node_idx = color_groups[color][node_idx_in_group];
    node_colors[node_idx] = target_color;

    if (color_groups.size())
    {
        // O(1) erase
        std::swap(color_groups[color][node_idx_in_group], color_groups[color].back());
        color_groups[color].pop_back();

        color_groups[target_color].push_back(node_idx);
    }
}

float balance_color_groups(float target_max_min_ratio, 
    Graph& graph, 
    std::vector<std::vector<int>>& color_groups)
{
    float max_min_ratio = -1.f;

    do
    {
        int biggest_group = -1, smallest_group = -1;
        float prev_max_min_ratio = max_min_ratio;
        max_min_ratio = find_largest_smallest_groups(color_groups, biggest_group, smallest_group);

        if (prev_max_min_ratio > 0 && prev_max_min_ratio < max_min_ratio) {
            return max_min_ratio;
        }

        // graph is not optimizable anymore or target ratio reached
        if (color_groups[biggest_group].size() - color_groups[smallest_group].size() <= 2 
            || max_min_ratio < target_max_min_ratio)
        {
            return max_min_ratio;
        }

        // find a available vertex from the biggest category to move to the smallest category
        int changeable_color_group_idx = biggest_group;
        int changeable_node_idx = find_changeable_node_in_category(graph, color_groups, biggest_group, smallest_group);
        if (changeable_node_idx == -1)
        {
            for (size_t color = 0; color < color_groups.size(); color++)
            {
                if (color == biggest_group || color == smallest_group)
                {
                    continue;
                }

                changeable_node_idx = find_changeable_node_in_category(graph, color_groups, color, smallest_group);

                if (changeable_node_idx != -1)
                {
                    changeable_color_group_idx = color;

                    break;
                }
            }
        }


        if (changeable_node_idx == -1)
        {
            // fprintf(stderr, "The graph is not optimizable anymore, terminated with a max/min ratio: %f without reaching the target ratio: %f\n", max_min_ratio, target_max_min_ratio);
            return max_min_ratio;
        }
        // change the color of changeable_color_idx in group changeable_color_group_idx to 
        change_color(changeable_color_group_idx, changeable_node_idx, smallest_group, graph.node_colors, color_groups);


    } while (max_min_ratio > target_max_min_ratio);

    return max_min_ratio;
}

int graph_coloring_ordered_greedy(const std::vector<int>& order, Graph& graph)
{
    // greedy coloring
    int max_color = -1;
    int num_colored = 0;
    std::vector<bool> color_used;
    color_used.reserve(128);

    for (size_t i = 0; i < order.size(); i++)
    {
        int node = order[i];

        // first one
        if (max_color == -1)
        {
            ++max_color;
            graph.node_colors[node] = max_color;
        }
        else {
            color_used.resize(max_color + 1);

            for (int color_counter = 0; color_counter < color_used.size(); color_counter++)
            {
                color_used[color_counter] = false;
            }

            // see its neighbor's color
            for (int nei_counter = 0; nei_counter < graph.get_node_degree(node); nei_counter++)
            {
                int nei_node_idx = graph.get_node_neighbor(node, nei_counter);
                if (graph.node_colors[nei_node_idx] >= 0)
                {
                    color_used[graph.node_colors[nei_node_idx]] = true;
                }
            }

            // find the minimal usable color
            int min_usable_color = -1;
            for (int color_counter = 0; color_counter < color_used.size(); color_counter++)
            {
                if (!color_used[color_counter]) {
                    min_usable_color = color_counter;
                    break;
                }
            }
            if (min_usable_color == -1)
            {
                ++max_color;
                graph.node_colors[node] = max_color;
            }
            else
            {
                graph.node_colors[node] = min_usable_color;
            }
        }

        num_colored++;
    }
    return (max_color + 1);
}

class NodeWeightBuckets
{
public:
    NodeWeightBuckets(int num_nodes)
        : node_weights(num_nodes, 0), node_indices_in_bucket(num_nodes, -1)
    {
        weight_buckets.resize(NODE_WEIGHTS_PREALLOC);
        for (size_t i = 1; i < weight_buckets.size(); i++)
        {
            weight_buckets[i].reserve(WEIGHT_BUCKET_PREALLOC);
        }
        max_weight = 0;
    }

    int get_node_weight(int node_idx) 
    {
        return node_weights[node_idx];
    }

    void add_node(int weight, int node_idx)
    {
        if (weight >= weight_buckets.size()) 
        {
            weight_buckets.resize(weight + 1);
        }
        
        node_indices_in_bucket[node_idx] = weight_buckets[weight].size();
        node_weights[node_idx] = weight;
        weight_buckets[weight].push_back(node_idx);

        if (max_weight < weight)
        {
            max_weight = weight;
        }
    }

    int pop_node_with_max_weight() {
        int node_with_max_weight = weight_buckets[max_weight].front();
        node_indices_in_bucket[node_with_max_weight] = -1;

        // we pop the first element so it has a breadth-first like behavior, which is better than depth-first
        if (weight_buckets[max_weight].size() > 1)
        {
            node_indices_in_bucket[weight_buckets[max_weight].back()] = 0;
            weight_buckets[max_weight][0] = weight_buckets[max_weight].back();
        }
        weight_buckets[max_weight].pop_back();
        // mark node deleted
        node_weights[node_with_max_weight] = -1;

        if (weight_buckets[max_weight].size() == 0)
            // we need to update max_weight because weight_buckets[max_weight] became empty
        {
            int new_max_weight = 0;
            for (int bucket_idx = max_weight - 1; bucket_idx >= 0; bucket_idx--)
            {
                if (weight_buckets[bucket_idx].size())
                {
                    new_max_weight = bucket_idx;
                    break;
                }
            }

            max_weight = new_max_weight;
        }
        // mark deleted
        return node_with_max_weight;
    }

    void increase_node_weight(int node_idx)
    {
        int weight = node_weights[node_idx];
        assert(weight < weight_buckets.size());
        int node_idx_in_bucket = node_indices_in_bucket[node_idx];
        assert(node_idx_in_bucket < weight_buckets[weight].size());

        // swap index with the last element
        node_indices_in_bucket[weight_buckets[weight].back()] = node_idx_in_bucket;
        // O(1) erase
        weight_buckets[weight][node_idx_in_bucket] = weight_buckets[weight].back();
        weight_buckets[weight].pop_back();

        add_node(weight + 1, node_idx);
    }

    bool empty()
    {
        return max_weight <= 0 && weight_buckets[0].size() == 0;
    }


private:
    int max_weight;
    std::vector<std::vector<int>> weight_buckets;
    std::vector<int> node_indices_in_bucket;
    std::vector<int> node_weights;
};

// Pereira, F. M. Q., & Palsberg, J. (2005, November). Register allocation via coloring of chordal graphs. In Asian Symposium on Programming Languages and Systems (pp. 315-329). Berlin, Heidelberg: Springer Berlin Heidelberg.
int graph_coloring_mcs_vector(Graph& graph)
{
    // Initially set the weight of each node to 0
    std::vector<int> ordering;
    ordering.reserve(graph.num_nodes);

    NodeWeightBuckets weight_buckets(graph.num_nodes);
    // add the first node
    weight_buckets.add_node(0, 0);

    for (int node_idx = 0; node_idx < graph.num_nodes; node_idx++)
    {
        // this might look like it's O(N^2) but this only happens once per connected components
        if (weight_buckets.empty())
        {
            int non_negative_node = -1;
            for (size_t i = 0; i < graph.num_nodes; i++)
            {
                if (weight_buckets.get_node_weight(i) >= 0) {
                    non_negative_node = i;
                    break;
                }
            }
            assert(weight_buckets.get_node_weight(non_negative_node) == 0);
            weight_buckets.add_node(0, non_negative_node);
        }

        int max_node = weight_buckets.pop_node_with_max_weight();

        // Add highest weight node to the queue and increment all of its neighbors weights by 1
        ordering.push_back(max_node);

        for (unsigned j = 0; j < graph.get_node_degree(max_node); j++) {
            int neighbor_node = graph.get_node_neighbor(max_node, j);
            int old_weight = weight_buckets.get_node_weight(neighbor_node);

            if (old_weight == 0)
                // 0-weighted node is not in buckets by default
            {
                weight_buckets.add_node(old_weight + 1, neighbor_node);

            }
            else if (old_weight > 0) {
                weight_buckets.increase_node_weight(neighbor_node);
            }
            // skip neighbor nodes with negative weight because they are visited
        }
    }

    return graph_coloring_ordered_greedy(ordering, graph);
}

int next_node(const int num_nodes, const std::vector<int>& degrees)
{
    int node_min_degrees = -1;
    int min_degree = num_nodes + 1;
    for (size_t node_idx = 0; node_idx < degrees.size(); node_idx++)
    {
        if (degrees[node_idx] == -1)
        {
            continue;
        }
        if (min_degree > degrees[node_idx]) {
            min_degree = degrees[node_idx];
            node_min_degrees = node_idx;
        }
    }
    return node_min_degrees;
}

void reduce_degree(int node_idx, Graph& graph, std::vector<int>& degrees)
{
    degrees[node_idx] = -1;
    for (size_t nei_node_counter = 0; nei_node_counter < graph.get_node_degree(node_idx); nei_node_counter++)
    {
        int nei_node_idx = graph.get_node_neighbor(node_idx, nei_node_counter);

        if (degrees[nei_node_idx] != -1)
        {
            degrees[nei_node_idx]--;
        }
    }
}


// Fratarcangeli, Marco, and Fabio Pellacini. "Scalable partitioning for parallel position based dynamics." Computer Graphics Forum. Vol. 34. No. 2. 2015.
int graph_coloring_degree_ordered_greedy(Graph& graph)
{
    // initialize the degree
    std::vector<int> degrees(graph.num_nodes, 0);
    for (int node_idx = 0; node_idx < graph.num_nodes; node_idx++) {
        degrees[node_idx] = graph.get_node_degree(node_idx);
    }

    // order them in a descending order
    std::vector<int> ordering(graph.num_nodes);
    std::iota(std::begin(ordering), std::end(ordering), 0);
    std::sort(std::begin(ordering), std::end(ordering),
        [&degrees](const auto& lhs, const auto& rhs)
        {
            return degrees[lhs] > degrees[rhs];
        }
    );

    return graph_coloring_ordered_greedy(ordering, graph);
}

int graph_coloring_naive_greedy(Graph& graph)
{
    std::vector<int> ordering(graph.num_nodes);
    std::iota(std::begin(ordering), std::end(ordering), 0);
    return graph_coloring_ordered_greedy(ordering, graph);
}
}
using namespace wp;

extern "C"
{
    int wp_graph_coloring(int num_nodes, wp::array_t<int> edges, int algorithm, wp::array_t<int> node_colors)
    {
        if (node_colors.ndim != 1 || node_colors.shape[0] != num_nodes)
        {
            fprintf(stderr, "The node_colors array must have the preallocated shape of (num_nodes,)!\n");
            return -1;
        }

        if (edges.ndim != 2)
        {
            fprintf(stderr, "The edges array must have 2 dimensions!\n");
            return -1;
        }

        if (num_nodes == 0)
        {
            fprintf(stderr, "Empty graph!\n");
            return -1;
        }

        // convert to a format that coloring algorithm can recognize

        Graph graph(num_nodes, edges);

        int num_colors = -1;
        switch (algorithm)
        {
        case 0:
            // mcs algorithm
            num_colors = graph_coloring_mcs_vector(graph);
            break;
        case 1:
            // greedy
            num_colors = graph_coloring_degree_ordered_greedy(graph);
            break;
        //case 2:
        //    // mcs algorithm
        //    num_colors = graph_coloring_mcs_set(graph);
        //    break;
        //case 3:
        //    // naive greedy
        //    num_colors = graph_coloring_naive_greedy(graph);
        //    break;
        default:
            fprintf(stderr, "Unrecognized coloring algorithm number: %d!\n", algorithm);
            return -1;
            break;
        }

        // copy the color info back
        memcpy(node_colors.data, graph.node_colors.data(), num_nodes * sizeof(int));

        return num_colors;
    }

    float wp_balance_coloring(int num_nodes, wp::array_t<int> edges, int num_colors, 
        float target_max_min_ratio, wp::array_t<int> node_colors)
    {
        Graph graph(num_nodes, edges);
        // copy the color info to graph
        memcpy(graph.node_colors.data(), node_colors.data, num_nodes * sizeof(int));
        if (num_colors > 1) {
            std::vector<std::vector<int>> color_groups;
            convert_to_color_groups(num_colors, graph.node_colors, color_groups);

            float max_min_ratio = balance_color_groups(target_max_min_ratio, graph, color_groups);
            memcpy(node_colors.data, graph.node_colors.data(), num_nodes * sizeof(int));

            return max_min_ratio;
        }
        else
        {
            return 1.f;
        }
    }
}
