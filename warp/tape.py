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
        self.const_gradients = set()
        self.launches = []

        self.loss = None


    def __enter__(self):      
        if (wp.context.runtime.tape != None):
            raise RuntimeError("Warp: Error, entering a tape while one is already active")

        wp.context.runtime.tape = self

    def __exit__(self, exc_type, exc_value, traceback):
        if (wp.context.runtime.tape == None):
            raise RuntimeError("Warp: Error, ended tape capture, but tape not present")            

        wp.context.runtime.tape = None

    # adj_outputs is a mapping from output tensor -> adjoint of the output
    # after running backward the gradients of tensors may be retrieved by:
    #
    #  adj_tensor = tape.gradients[tensor]
    #
    def backward(self, loss: wp.array=None, grads: dict=None):

        # if scalar loss is specified then initialize
        # a 'seed' array for it, with gradient of one
        if loss:
            
            if loss.size > 1 or wp.types.type_length(loss.dtype) > 1:
                raise RuntimeError("Can only return gradients for scalar loss functions.")

            if loss.requires_grad == False:
                raise RuntimeError("Scalar loss arrays should have requires_grad=True set before calling Tape.backward()")

            # set the seed grad to 1.0
            loss.grad.fill_(1.0)
            
        # simply apply dict grads to objects
        # this is just for backward compat. with
        # existing code before we added wp.array.grad attribute
        if grads:
            for a, g in grads.items():
                a.grad = g
                self.const_gradients.add(a)

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


    # record a kernel launch on the tape
    def record(self, kernel, dim, inputs, outputs, device):
        self.launches.append([kernel, dim, inputs, outputs, device])


    # returns the adjoint of a kernel parameter
    def get_adjoint(self, a):


        if isinstance(a, wp.array) == False and isinstance(a, wp.codegen.StructInstance) == False:
            # if input is a simple type (e.g.: float, vec3, etc) then
            # no gradient needed (we only return gradients through arrays and structs)
            return a

        elif isinstance(a, wp.array) and a.grad:
            # keep track of all gradients used by the tape (for zeroing)
            # ignore the scalar loss since we don't want to clear its grad
            self.gradients[a] = a.grad
            return a.grad

        elif isinstance(a, wp.codegen.StructInstance):
            adj = wp.codegen.StructInstance(a._struct_)
            for name in a.__dict__:
                if name.startswith("_"):
                    continue
                if isinstance(a._struct_.vars[name].type, wp.array):
                    arr = getattr(a, name)
                    if arr.grad:
                        grad = self.gradients[arr] = arr.grad
                    else:
                        grad = None
                    setattr(adj, name, grad)
                else:
                    setattr(adj, name, a.__dict__[name])

            self.gradients[a] = adj
            return adj

        return None

    def reset(self):
        
        self.launches = []
        self.zero()

    def zero(self):

        for a, g in self.gradients.items():
            if a not in self.const_gradients:
                g.zero_()

