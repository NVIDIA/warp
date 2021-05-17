# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import oglang as og


@og.kernel
def basic(x: og.array(dtype=float)):
    
    tid = og.tid()

    og.store(x, tid, float(tid)*1.0)


device = "cuda"
n = 32

x = og.zeros(n, dtype=float, device="cuda")

og.launch(
    kernel=basic, 
    dim=n, 
    inputs=[x], 
    device=device)

print(x.to("cpu").numpy())

# redefine kernel
@og.kernel
def basic(x: og.array(dtype=float)):
    
    tid = og.tid()

    og.store(x, tid, float(tid)*2.0)
    

og.launch(
    kernel=basic, 
    dim=n, 
    inputs=[x], 
    device=device)

print(x.to("cpu").numpy())