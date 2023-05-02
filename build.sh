#!/bin/bash
set -e

SCRIPT_DIR=$(dirname ${BASH_SOURCE})
#source "$SCRIPT_DIR/repo.sh" build --fetch-only $@ || exit $?

"$SCRIPT_DIR/repo.sh" build --fetch-only $@ 

# host deps
# tools/packman/packman pull -p linux-x86_64 deps/host-deps.packman.xml
# tools/packman/packman pull -p linux-x86_64 deps/target-deps.packman.xml

# pip deps
./_build/target-deps/python/python -m pip install numpy

if [[ "$OSTYPE" == "darwin"* ]]; then
    ./_build/target-deps/python/python build_lib.py
else
    # build with docker for increased compatibility
    ./_build/host-deps/linbuild/linbuild.sh -- ./_build/target-deps/python/python build_lib.py --cuda_path="_build/target-deps/cuda"
fi