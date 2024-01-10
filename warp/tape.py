# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp


class Tape:
    """
    Record kernel launches within a Tape scope to enable automatic differentiation.
    Gradients can be computed after the operations have been recorded on the tape via
    ``tape.backward()``.

    Example
    -------

    .. code-block:: python

        tape = wp.Tape()

        # forward pass
        with tape:
            wp.launch(kernel=compute1, inputs=[a, b], device="cuda")
            wp.launch(kernel=compute2, inputs=[c, d], device="cuda")
            wp.launch(kernel=loss, inputs=[d, l], device="cuda")

        # reverse pass
        tape.backward(l)

    Gradients can be accessed via the ``tape.gradients`` dictionary, e.g.:

    .. code-block:: python

        print(tape.gradients[a])

    """

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
        """
        Evaluate the backward pass of the recorded operations on the tape.
        A single-element array ``loss`` or a dictionary of arrays ``grads``
        can be provided to assign the incoming gradients for the reverse-mode
        automatic differentiation pass.

        Args:
            loss (wp.array): A single-element array that holds the loss function value whose gradient is to be computed
            grads (dict): A dictionary of arrays that map from Warp arrays to their incoming gradients

        """
        # if scalar loss is specified then initialize
        # a 'seed' array for it, with gradient of one
        if loss:
            if loss.size > 1 or wp.types.type_length(loss.dtype) > 1:
                raise RuntimeError("Can only return gradients for scalar loss functions.")

            if not loss.requires_grad:
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
                max_blocks = launch[2]
                inputs = launch[3]
                outputs = launch[4]
                device = launch[5]

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
                    max_blocks=max_blocks,
                )

    # record a kernel launch on the tape
    def record_launch(self, kernel, dim, max_blocks, inputs, outputs, device):
        self.launches.append([kernel, dim, max_blocks, inputs, outputs, device])

    def record_func(self, backward, arrays):
        """
        Records a custom function to be executed only in the backward pass.

        Args:
            backward (Callable): A callable Python object (can be any function) that will be executed in the backward pass.
            arrays (list): A list of arrays that are used by the function for gradient tracking.
        """
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
        """
        Clear all operations recorded on the tape and zero out all gradients.
        """
        self.launches = []
        self.zero()

    def zero(self):
        """
        Zero out all gradients recorded on the tape.
        """
        for a, g in self.gradients.items():
            if a not in self.const_gradients:
                if isinstance(a, wp.codegen.StructInstance):
                    for name in g._cls.vars:
                        if isinstance(g._cls.vars[name].type, wp.array) and g._cls.vars[name].requires_grad:
                            getattr(g, name).zero_()
                else:
                    g.zero_()
