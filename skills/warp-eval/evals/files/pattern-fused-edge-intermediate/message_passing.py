# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


def aggregate(
    node_features: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    messages = node_features[source] * edge_weight[:, None]
    transformed = torch.tanh(messages)
    output = torch.zeros_like(node_features)
    output.index_add_(0, destination, transformed)
    return output
