import os
import sys
import subprocess

import warp as wp


wp.init()

### docs

# disable sphinx color output
os.environ["NO_COLOR"] = "1"

with open("docs/modules/functions.rst","w") as function_ref:
    wp.print_builtins(function_ref)

# run Sphinx build
try:
    if os.name == 'nt':
        subprocess.check_output("make.bat html", cwd="docs", shell=True)
    else:
        subprocess.run("make clean", cwd="docs", shell=True)
        subprocess.check_output("make html", cwd="docs", shell=True)
except subprocess.CalledProcessError as e:
    print(e.output.decode())
    raise(e)


### generate stubs for autocomplete
stub_file = open("warp/stubs.py","w")
wp.export_stubs(stub_file)
stub_file.close()

# code formatting
subprocess.run([sys.executable, "-m", "black", "warp/stubs.py"])

print("Finished")
