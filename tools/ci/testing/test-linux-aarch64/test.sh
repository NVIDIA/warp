#!/bin/bash
set -e

SCRIPT_DIR=$(dirname ${BASH_SOURCE})
"$SCRIPT_DIR/../../../../repo.sh" build --fetch-only $@

PYTHON="$SCRIPT_DIR/../../../../_build/target-deps/python/python"

echo "Installing test dependencies"
#$PYTHON -m pip install matplotlib
#$PYTHON -m pip install usd-core
#$PYTHON -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
#$PYTHON -m pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "Installing Warp to Python"
$PYTHON -m pip install -e "$SCRIPT_DIR/../../../../."

echo "Running tests"
$PYTHON -m warp.tests
