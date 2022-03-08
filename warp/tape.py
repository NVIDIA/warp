# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import warp as wp

class Tape:

    def __init__(self):

        self.gradients = {}
        self.launches = []

    def __enter__(self):      
        if (wp.context.runtime.tape != None):
            raise RuntimeError("Warp: Error, entering a tape while one is already active")

        wp.context.runtime.tape = self

    def __exit__(self, exc_type, exc_value, traceback):
        if (wp.context.runtime.tape == None):
            raise RuntimeError("Warp: Error, ended tape capture, but tape not present")            

        wp.context.runtime.tape = None

    # record a kernel launch on the tape
    def record(self, kernel, dim, inputs, outputs, device):   
        self.launches.append([kernel, dim, inputs, outputs, device])


    # adj_outputs is a mapping from output tensor -> adjoint of the output
    # after running backward the gradients of tensors may be retrieved by:
    #
    #  adj_tensor = tape.gradients[tensor]
    #
    def backward(self, loss: wp.array=None, grads: dict=None):

        # if scalar loss is specified then allocate 
        # a 'seed' array for it, with gradient of one
        if (loss):
            self.gradients[loss] = wp.array(np.ones(1), dtype=wp.float32, device=loss.device)

        # insert any user specified gradients (e.g.: from Torch)
        if (grads):
            self.gradients.update(grads)

        # run launches backwards
        for launch in reversed(self.launches):

            kernel = launch[0]
            dim = launch[1]
            inputs = launch[2]
            outputs = launch[3]
            device = launch[4]

            adj_inputs = []
            adj_outputs = []

            # lookup adjoint inputs
            for a in inputs:
                adj_inputs.append(self.get_adjoint(a))

            # lookup adjoint outputs, todo: only allocate outputs if necessary
            for a in outputs:
                adj_outputs.append(self.get_adjoint(a))

            wp.launch(
                kernel=kernel, 
                dim=dim, 
                inputs=inputs, 
                outputs=outputs,
                adj_inputs=adj_inputs,
                adj_outputs=adj_outputs,
                device=device,
                adjoint=True)
            

    # returns the adjoint version of a tensor used in the computation
    def get_adjoint(self, a):
        
        if isinstance(a, wp.array) == False:
            # if input is a simple type (e.g.: float, vec3, etc) just return a value copy
            return a

        elif a in self.gradients:
            # try and find adjoint array in map
            return self.gradients[a]
                    
        elif wp.type_is_int(a.dtype) or a.requires_grad == False:
            # otherwise if input is an array that is integer typed or doesn't require grad then return null array
            return None

        else:
            # otherwise allocate a zero array for the array adjoint
            adj = wp.zeros_like(a)
            self.gradients[a] = adj
            return adj


    def reset(self):
        
        self.launches = []

        for a in self.gradients.values():
            a.zero_()
