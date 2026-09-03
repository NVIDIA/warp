# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import sphinx.util.logging
from sphinx.ext.doctest import DocTestBuilder

logger = sphinx.util.logging.getLogger(__name__)


class ShardedDocTestBuilder(DocTestBuilder):
    """Run doctests only for documents named in a shard manifest."""

    name = "doctest-shard"

    def write_documents(self, docnames: set[str]) -> None:
        manifest = Path(self.config.warp_doctest_shard_manifest)
        if not manifest.is_file():
            raise RuntimeError(f"Doctest shard manifest does not exist: {manifest}")

        selected_docnames = set(manifest.read_text(encoding="utf-8").splitlines())
        unknown_docnames = selected_docnames - self.env.found_docs
        if unknown_docnames:
            logger.warning("doctest shard contains unknown documents: %s", ", ".join(sorted(unknown_docnames)))

        super().write_documents(docnames & selected_docnames)


def setup(app):
    """Register the sharded doctest builder."""
    app.add_config_value("warp_doctest_shard_manifest", "", "env", types=frozenset({str}))
    app.add_builder(ShardedDocTestBuilder)
    return {"parallel_read_safe": True}
