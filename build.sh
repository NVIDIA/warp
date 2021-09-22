#!/bin/bash
#set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE})
#source "$SCRIPT_DIR/repo.sh" build --fetch-only $@ || exit $?

"$SCRIPT_DIR/repo.sh" build --fetch-only --no-docker $@ 

python --version
python3 --version

# pip deps
python -m pip install numpy

python build_lib.py
