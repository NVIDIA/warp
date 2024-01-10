#!/bin/bash
set -e

SCRIPT_DIR=$(dirname ${BASH_SOURCE})
"$SCRIPT_DIR/../../../../repo.sh" build --fetch-only $@

PYTHON="$SCRIPT_DIR/../../../../_build/target-deps/python/python"
CUDA_BIN="$SCRIPT_DIR/../../../../_build/target-deps/cuda/bin"

# Make sure ptxas can be run by JAX
export PATH="$CUDA_BIN:$PATH"

echo "Installing test dependencies"

if [ -n "$TEAMCITY_VERSION" ]; then
    echo "##teamcity[blockOpened name='Installing test dependencies']"
fi

$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install matplotlib
$PYTHON -m pip install usd-core
$PYTHON -m pip install coverage
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    $PYTHON -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu115
    $PYTHON -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

if [ -n "$TEAMCITY_VERSION" ]; then
    echo "##teamcity[blockClosed name='Installing test dependencies']"
fi

echo "Installing Warp to Python"
$PYTHON -m pip install -e "$SCRIPT_DIR/../../../../."

echo "Running tests"
$PYTHON -m warp.tests
