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
        if wp.context.runtime.tape is not None:
            raise RuntimeError("Warp: Error, entering a tape while one is already active")

        wp.context.runtime.tape = self

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if wp.context.runtime.tape is None:
            raise RuntimeError("Warp: Error, ended tape capture, but tape not present")

        wp.context.runtime.tape = None

    # adj_outputs is a mapping from output tensor -> adjoint of the output
    # after running backward the gradients of tensors may be retrieved by:
    #
    #  adj_tensor = tape.gradients[tensor]
    #
    def backward(self, loss: wp.array = None, grads: dict = None):
        # if scalar loss is specified then initialize
        # a 'seed' array for it, with gradient of one
        if loss:
            if loss.size > 1 or wp.types.type_length(loss.dtype) > 1:
                raise RuntimeError("Can only return gradients for scalar loss functions.")

            if loss.requires_grad == False:
                raise RuntimeError(
                    "Scalar loss arrays should have requires_grad=True set before calling Tape.backward()"
                )

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
            if callable(launch):
                launch()

            else:
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
                    adjoint=True,
                )

    # record a kernel launch on the tape
    def record_launch(self, kernel, dim, inputs, outputs, device):
        self.launches.append([kernel, dim, inputs, outputs, device])

    # records a custom function for the backward pass, can be any
    # Callable python object. Callee should also pass arrays that
    # take part in the function for gradient tracking.
    def record_func(self, backward, arrays):
        self.launches.append(backward)

        for a in arrays:
            if isinstance(a, wp.array) and a.grad:
                self.gradients[a] = a.grad
            else:
                raise RuntimeError(
                    f"Array {a} is not of type wp.array or is missing a gradient array. Set array parameter requires_grad=True during instantiation."
                )

    # returns the adjoint of a kernel parameter
    def get_adjoint(self, a):
        if not wp.types.is_array(a) and not isinstance(a, wp.codegen.StructInstance):
            # if input is a simple type (e.g.: float, vec3, etc) then
            # no gradient needed (we only return gradients through arrays and structs)
            return a

        elif wp.types.is_array(a) and a.grad:
            # keep track of all gradients used by the tape (for zeroing)
            # ignore the scalar loss since we don't want to clear its grad
            self.gradients[a] = a.grad
            return a.grad

        elif isinstance(a, wp.codegen.StructInstance):
            adj = a._cls()
            for name, _ in a._cls.ctype._fields_:
                if name.startswith("_"):
                    continue
                if isinstance(a._cls.vars[name].type, wp.array):
                    arr = getattr(a, name)
                    if arr.grad:
                        grad = self.gradients[arr] = arr.grad
                    else:
                        grad = None
                    setattr(adj, name, grad)
                else:
                    setattr(adj, name, getattr(a, name))

            self.gradients[a] = adj
            return adj

        return None

    def reset(self):
        self.launches = []
        self.zero()

    def zero(self):
        for a, g in self.gradients.items():
            if a not in self.const_gradients:
                if isinstance(a, wp.codegen.StructInstance):
                    for name in g._cls.vars:
                        if isinstance(g._cls.vars[name].type, wp.array) and g._cls.vars[name].requires_grad:
                            getattr(g, name).zero_()
                else:
                    g.zero_()
