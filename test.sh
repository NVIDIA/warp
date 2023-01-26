#!/bin/bash
set -e

SCRIPT_DIR=$(dirname ${BASH_SOURCE})
"$SCRIPT_DIR/repo.sh" build --fetch-only --no-docker $@ 

echo "Installing test dependencies"
./_build/target-deps/python/python -m pip install matplotlib
./_build/target-deps/python/python -m pip install usd-core
./_build/target-deps/python/python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
./_build/target-deps/python/python -m pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "Installing Warp to Python"
./_build/target-deps/python/python -m pip install -e .

echo "Running tests"
./_build/target-deps/python/python warp/tests/test_all.py
