#!/bin/bash
#set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE})
#source "$SCRIPT_DIR/repo.sh" build --fetch-only $@ || exit $?

"$SCRIPT_DIR/repo.sh" build --fetch-only --no-docker $@ 

# pip deps
python3 -m pip install numpy

python3 build_lib.py
