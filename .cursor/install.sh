#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the Warp repository.
#
# The Cloud Agent VM is CPU-only (no NVIDIA GPU), so the native library is built
# with --no-cuda. This produces the warp-clang CPU backend used to JIT-compile
# kernels for the "cpu" device.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

# 1. Ensure uv is available (pinned tooling comes from .python-version / uv.lock).
if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --user --upgrade uv
fi

# 2. Install the pinned CPython interpreter (uv reads .python-version).
uv python install

# 3. Build the native libraries (CPU-only) BEFORE syncing the project.
#    setup.py aborts if warp/bin is empty ("run build_lib.py first"), so the
#    editable install in step 4 requires the binaries to already exist. Use
#    --no-project so this step does not try to build the warp-lang package
#    itself (which would hit the same empty-warp/bin failure).
uv run --no-project --with numpy build_lib.py --no-cuda

# 4. Sync the project environment with dev + examples dependencies.
uv sync --extra dev
