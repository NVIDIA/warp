#!/bin/bash
set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE})
source "$SCRIPT_DIR/repo.sh" build --fetch-only $@ || exit $?

python build_lib.py
