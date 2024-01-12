#!/bin/bash

set -e

SCRIPT_DIR=$(dirname ${BASH_SOURCE})
cd "$SCRIPT_DIR"

# Set OMNI_REPO_ROOT early so `repo` bootstrapping can target the repository
# root when writing out Python dependencies.
OMNI_REPO_ROOT="$( cd "$(dirname "$0")" ; pwd -P )" exec "tools/packman/python.sh" tools/repoman/repoman.py "$@"
