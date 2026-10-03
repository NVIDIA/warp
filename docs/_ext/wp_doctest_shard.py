# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sphinx.ext.doctest import DocTestBuilder


class ShardedDocTestBuilder(DocTestBuilder):
    """Run doctests for one deterministic shard of Sphinx documents."""

    name = "doctest-shard"

    def write_documents(self, docnames: set[str]) -> None:
        shard_index = self.config.warp_doctest_shard_index
        shard_count = self.config.warp_doctest_shard_count
        if shard_count < 1:
            raise ValueError("Doctest shard count must be at least 1")
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError("Doctest shard index must be within the shard count")
        if shard_count > len(docnames):
            raise ValueError("Doctest shard count cannot exceed the number of Sphinx documents")

        super().write_documents(set(sorted(docnames)[shard_index::shard_count]))


def setup(app):
    """Register the sharded doctest builder."""
    app.add_config_value("warp_doctest_shard_index", -1, "env", types=frozenset({int}))
    app.add_config_value("warp_doctest_shard_count", 0, "env", types=frozenset({int}))
    app.add_builder(ShardedDocTestBuilder)
    return {"parallel_read_safe": True}
