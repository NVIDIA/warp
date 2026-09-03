#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

SCRIPT_DIR=$(dirname ${BASH_SOURCE})
exec "$SCRIPT_DIR/packman/python.sh" $SCRIPT_DIR/repoman/repoman.py "$@"
