# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os


def get_source_directory():
    return os.path.realpath(os.path.dirname(__file__))


def get_asset_directory():
    return os.path.join(get_source_directory(), "assets")
