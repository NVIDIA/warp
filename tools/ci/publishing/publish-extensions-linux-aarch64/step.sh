#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(dirname "${BASH_SOURCE}")"

"$SCRIPT_DIR/../../../../repo.sh" build --fetch-only

"$SCRIPT_DIR/../../../../repo.sh" publish_exts -c release --from-package $*

"$SCRIPT_DIR/../../../../repo.sh" publish_exts -c debug --from-package $*