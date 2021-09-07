# This script is an 'offline' build of the core warp runtime libraries
# designed to be executed as part of CI / developer workflows, not 
# as part of the user runtime (since it requires CUDA toolkit, etc)

import os
import sys
import subprocess

import warp as wp


wp.init()

function_ref = open("docs/modules/functions.rst","w")

wp.print_builtins(function_ref)

function_ref.close()

# run Sphinx build
try:
    subprocess.check_output("make.bat html", cwd="docs", shell=True)
except subprocess.CalledProcessError as e:
    print(e.output.decode())
    raise(e)

print("Finished")