#!/bin/bash

set -e

SCRIPT_DIR=$(dirname ${BASH_SOURCE})
exec "$SCRIPT_DIR/packman/python.sh" $SCRIPT_DIR/repoman/repoman.py "$@"
