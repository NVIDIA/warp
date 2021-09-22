#!/bin/bash
set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE})
#source "$SCRIPT_DIR/repo.sh" build --fetch-only $@ || exit $?
source "$SCRIPT_DIR/repo.sh" build --fetch-only $@ 

# pip deps
python -m pip install numpy

python build_lib.py
