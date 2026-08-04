# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def adaptive_filter(samples: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    output = np.empty_like(samples)
    state = np.float32(0.0)
    for index, sample in enumerate(samples):
        state = np.tanh(alpha * sample + beta * state * state)
        output[index] = state
    return output
