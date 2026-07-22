#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs inside a manylinux container: build, deploy, package, and check one
# Linux LLVM SDK. Required env: LLVM_VERSION, BUNDLE_REVISION, PROFILE
# (e.g. linux-x86_64), IMAGE_DIGEST, OUTPUT_DIR (host-mounted),
# CONAN_VERSION, CMAKE_PIN, NINJA_PIN.
set -euo pipefail

: "${CONAN_VERSION:?CONAN_VERSION must be set}"
: "${CMAKE_PIN:?CMAKE_PIN must be set}"
: "${NINJA_PIN:?NINJA_PIN must be set}"

# pipx puts console scripts on ~/.local/bin regardless of which Python the
# manylinux image exposes; plain pip's script location varies by image.
pipx install "conan==${CONAN_VERSION}"
pipx install "cmake==${CMAKE_PIN}"
pipx install "ninja==${NINJA_PIN}"
export PATH="$HOME/.local/bin:$PATH"

profile_gcc=$(awk -F= '/^\[/{s=($0=="[settings]")} s && $1=="compiler.version"{print $2}' "tools/llvm/profiles/${PROFILE}")
actual_gcc=$(gcc -dumpversion | cut -d. -f1)
if [[ "$profile_gcc" != "$actual_gcc" ]]; then
  echo "FAIL: profile ${PROFILE} pins gcc ${profile_gcc} but the container ships gcc ${actual_gcc}" >&2
  echo "Update the profile (and the asset-name ABI segment) together with the image digest." >&2
  exit 1
fi

conan profile detect --force
conan create tools/llvm --version "${LLVM_VERSION}" \
  -pr:h "tools/llvm/profiles/${PROFILE}" -pr:b default

conan install --requires "clang-warp/${LLVM_VERSION}" \
  -pr:h "tools/llvm/profiles/${PROFILE}" -pr:b default \
  --deployer tools/llvm/deployers/llvm_sdk.py --deployer-folder _sdk_deploy

python3 tools/llvm/check_sdk.py --sdk-dir _sdk_deploy/llvm-sdk --platform "${PROFILE}"

python3 tools/llvm/package_sdk.py \
  --sdk-dir _sdk_deploy/llvm-sdk \
  --profile "tools/llvm/profiles/${PROFILE}" \
  --llvm-version "${LLVM_VERSION}" \
  --bundle-revision "${BUNDLE_REVISION}" \
  --output-dir "${OUTPUT_DIR}" \
  --recipe-sha "${GITHUB_SHA:-unknown}" \
  --image-digest "${IMAGE_DIGEST}" \
  --conan-version "${CONAN_VERSION}" \
  --toolchain-info "$(gcc --version | head -1)"
