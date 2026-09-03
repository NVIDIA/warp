#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

asv continuous --append-samples --interleave-rounds --no-only-changed main $(git rev-parse HEAD 2>/dev/null)
