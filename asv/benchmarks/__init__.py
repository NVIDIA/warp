# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

# Give each benchmarked commit its own kernel cache. Builds of the same Warp
# version share the default cache directory, and module hashes cover kernel
# source but not native headers or code generation, so a commit could load
# kernels another commit compiled. asv sets these variables in every benchmark
# process, and this package is imported before any benchmark initializes Warp.
_commit = os.environ.get("ASV_COMMIT")
_env_dir = os.environ.get("ASV_ENV_DIR")
if _commit and _env_dir:
    os.environ["WARP_CACHE_PATH"] = os.path.join(_env_dir, "warp-cache", _commit)
