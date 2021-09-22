#!/bin/bash
#set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE})
#source "$SCRIPT_DIR/repo.sh" build --fetch-only $@ || exit $?
echo "here"
"$SCRIPT_DIR/repo.sh" build --fetch-only --no-docker $@ 

echo "post"

# pip deps
python -m pip install numpy

python build_lib.py
