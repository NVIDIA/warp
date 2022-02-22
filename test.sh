#!/bin/bash
set -e
#SCRIPT_DIR=$(dirname ${BASH_SOURCE})
#source "$SCRIPT_DIR/repo.sh" build --fetch-only $@ || exit $?

#"$SCRIPT_DIR/repo.sh" build --fetch-only --no-docker $@ 
#cd _build/packages

echo "Installing test dependencies"
./_build/target-deps/python/python -m pip install matplotlib
./_build/target-deps/python/python -m pip install usd-core

echo "Installing Warp to Python"
../target-deps/python/python -m pip install -e .

echo "Running tests"
../target-deps/python/python tests/test_all.py