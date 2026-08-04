# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch


def closest_face(points: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Return one candidate face index per point."""
    points_np = points.detach().cpu().numpy()
    vertices_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    centers = vertices_np[faces_np].mean(axis=1)
    result = np.empty(len(points_np), dtype=np.int32)
    for point_index, point in enumerate(points_np):
        distance_squared = np.sum((centers - point) ** 2, axis=1)
        result[point_index] = int(np.argmin(distance_squared))
    return torch.from_numpy(result).to(points.device)
