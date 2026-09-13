#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

expected_uv_version="${EXPECTED_UV_VERSION:-0.12.13}"
expected_cuda_version="${EXPECTED_CUDA_VERSION:?EXPECTED_CUDA_VERSION must be set}"
expected_cuda_platform="${EXPECTED_CUDA_PLATFORM:?EXPECTED_CUDA_PLATFORM must be set}"
expected_python_dir="${UV_PYTHON_INSTALL_DIR:-}"

if [[ "${WARP_CUDA_VERSION:-}" != "$expected_cuda_version" ]]; then
    echo "Expected CUDA Toolkit ${expected_cuda_version}, got: ${WARP_CUDA_VERSION:-<unset>}" >&2
    exit 1
fi

for cuda_path_var in CUDA_HOME CUDA_PATH WARP_CUDA_PATH; do
    if [[ "${!cuda_path_var:-}" != "/opt/cuda" ]]; then
        echo "${cuda_path_var} must be /opt/cuda, got: ${!cuda_path_var:-<unset>}" >&2
        exit 1
    fi
done

uv run --no-project --no-python-downloads --python 3.12 \
    python /workspace/tools/ci/cuda_toolkit.py activate \
    --version "$expected_cuda_version" \
    --platform "$expected_cuda_platform" \
    --cuda-path /opt/cuda

echo "CUDA Toolkit ${expected_cuda_version}: /opt/cuda"

if [[ "$expected_python_dir" != "/opt/uv/python" ]]; then
    echo "UV_PYTHON_INSTALL_DIR must be /opt/uv/python, got: ${expected_python_dir:-<unset>}" >&2
    exit 1
fi

if [[ "${UV_PYTHON_PREFERENCE:-}" != "only-managed" ]]; then
    echo "UV_PYTHON_PREFERENCE must be only-managed, got: ${UV_PYTHON_PREFERENCE:-<unset>}" >&2
    exit 1
fi

actual_uv_version="$(uv --version)"
if [[ "$actual_uv_version" != "uv ${expected_uv_version} "* ]]; then
    echo "Expected uv ${expected_uv_version}, got: ${actual_uv_version}" >&2
    exit 1
fi

python_requests=(
    "3.10:3.10.21:gil"
    "3.11:3.11.16:gil"
    "3.12:3.12.14:gil"
    "3.13:3.13.15:gil"
    "3.14:3.14.7:gil"
    "3.14t:3.14.7:freethreaded"
)

for entry in "${python_requests[@]}"; do
    IFS=: read -r request expected_version expected_variant <<< "$entry"

    python_path="$(uv python find --no-python-downloads "$request")"
    if [[ "$python_path" != "$expected_python_dir/"* ]]; then
        echo "Python ${request} resolved outside ${expected_python_dir}: ${python_path}" >&2
        exit 1
    fi

    actual="$(
        uv run --no-project --no-python-downloads --python "$request" \
            python -c 'import sys, sysconfig; print(".".join(map(str, sys.version_info[:3])), int(bool(sysconfig.get_config_var("Py_GIL_DISABLED"))))'
    )"

    expected_gil_disabled=0
    if [[ "$expected_variant" == "freethreaded" ]]; then
        expected_gil_disabled=1
    fi

    if [[ "$actual" != "$expected_version $expected_gil_disabled" ]]; then
        echo "Python ${request} reported ${actual}; expected ${expected_version} ${expected_gil_disabled}" >&2
        exit 1
    fi

    echo "Python ${request}: ${python_path} (${actual})"
done
