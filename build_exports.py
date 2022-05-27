import os
import sys
import subprocess

import warp as wp

# this script generates a header that can be used to bind
# builtins e.g.: quat_inverse(), etc to Python through
# ctypes, this could allow calling all builtins
# from Python

wp.init()

f = open("warp/native/exports.h","w")
wp.export_builtins(f)
f.close()

print("Finished")
