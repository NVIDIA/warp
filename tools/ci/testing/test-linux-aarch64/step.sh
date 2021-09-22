#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(dirname "${BASH_SOURCE}")"

# tests
"$SCRIPT_DIR/../../../../repo.sh" test --config release --from-package
