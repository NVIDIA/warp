#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(dirname "${BASH_SOURCE}")"

# Verify formatting
# "$SCRIPT_DIR/../../../../format_code.sh" --verify

# Full rebuild
"$SCRIPT_DIR/../../../../build.sh" -x

# Package all
"$SCRIPT_DIR/../../../../tools/package.sh" -a -c release

# Package all
"$SCRIPT_DIR/../../../../tools/package.sh" -a -c debug

# publish artifacts to teamcity
echo "##teamcity[publishArtifacts '_build/packages']"


