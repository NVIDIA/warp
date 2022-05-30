import os
import sys
import subprocess

import warp as wp


wp.init()

### docs

# disable sphinx color output
os.environ["NO_COLOR"] = "1"

function_ref = open("docs/modules/functions.rst","w")
wp.print_builtins(function_ref)
function_ref.close()

# run Sphinx build
try:
    if os.name == 'nt':
        subprocess.check_output("make.bat html", cwd="docs", shell=True)
    else:
        subprocess.check_output("make html", cwd="docs", shell=True)
except subprocess.CalledProcessError as e:
    print(e.output.decode())
    raise(e)


### generate stubs for autocomplete
stub_file = open("warp/stubs.py","w")
wp.export_stubs(stub_file)
stub_file.close()

print("Finished")
