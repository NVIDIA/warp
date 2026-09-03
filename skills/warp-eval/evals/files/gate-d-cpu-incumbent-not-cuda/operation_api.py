# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

class QueryIndex:
    def query(self, inputs, mode="all", return_metadata=False):
        """Return matching entries for each input."""
        raise NotImplementedError

    def query_first(self, inputs):
        """Return the first matching entry for each input."""
        raise NotImplementedError
