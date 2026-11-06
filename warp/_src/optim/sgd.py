# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import warp as wp
from warp._src.types import float_types, type_repr, type_scalar_type


@wp.kernel
def sgd_step_kernel(
    g: wp.array(dtype=Any),
    b: wp.array(dtype=Any),
    lr: Any,
    momentum: Any,
    damping: Any,
    weight_decay: Any,
    nesterov: int,
    t: int,
    params: wp.array(dtype=Any),
):
    i = wp.tid()
    gt = g[i]
    if weight_decay != type(weight_decay)(0.0):
        gt += weight_decay * params[i]
    if momentum != type(momentum)(0.0):
        bt = b[i]
        if t > 0:
            bt = momentum * bt + (type(damping)(1.0) - damping) * gt
        else:
            bt = gt
        if nesterov == 1:
            gt += momentum * bt
        else:
            gt = bt
        b[i] = bt
    params[i] = params[i] - lr * gt


def _sgd_scalar_type(dtype):
    scalar_type = type_scalar_type(dtype)
    if scalar_type not in float_types:
        raise TypeError(f"SGD parameters must have a floating-point dtype, got {type_repr(dtype)}")
    return scalar_type


class SGD:
    """Stochastic Gradient Descent (SGD) optimizer with optional momentum.

    This optimizer implements gradient descent with support for momentum,
    Nesterov accelerated gradient, and weight decay (L2 regularization).

    The interface is similar to `PyTorch's torch.optim.SGD
    <https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_.

    Args:
        params: List of :class:`warp.array` objects to optimize. Can be ``None``
            and set later via :meth:`set_params`. Arrays may use any
            floating-point scalar, vector, or matrix dtype; the update is computed
            in the parameter's scalar precision.
        lr: Learning rate (step size).
        momentum: Momentum factor for accelerating SGD in relevant directions.
        dampening: Dampening factor applied to the momentum.
        weight_decay: Weight decay coefficient (L2 regularization).
        nesterov: Whether to use Nesterov momentum. Requires ``momentum > 0``
            and ``dampening = 0``.
    """

    def __init__(self, params=None, lr=0.001, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False):
        self.b = []  # momentum buffer
        self.set_params(params)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.t = 0

    def set_params(self, params):
        """Set parameters to optimize and allocate momentum buffers.

        Args:
            params: List of :class:`warp.array` objects to optimize, or ``None``.
        """
        has_params = params is not None and isinstance(params, list) and len(params) > 0
        # Check every dtype before touching any state, so a rejected list leaves the optimizer unchanged.
        scalar_types = [_sgd_scalar_type(param.dtype) for param in params] if has_params else []
        self.params = params
        if has_params:
            if len(self.b) != len(params):
                self.b = [None] * len(params)
            for i in range(len(params)):
                param = params[i]
                scalar_type = scalar_types[i]
                if self.b[i] is None or self.b[i].shape != param.shape or self.b[i].dtype != param.dtype:
                    self.b[i] = wp.zeros_like(param)
                elif self.b[i].device != param.device:
                    self.b[i] = self.b[i].to(param.device)
                # Overload the kernel for each parameter so we can precompile the SGD kernel
                if param is not None:
                    wp.overload(
                        sgd_step_kernel,
                        {
                            "g": param,
                            "b": param,
                            "lr": scalar_type,
                            "momentum": scalar_type,
                            "damping": scalar_type,
                            "weight_decay": scalar_type,
                            "params": param,
                        },
                    )

    def reset_internal_state(self):
        """Reset momentum buffers and timestep to zero."""
        for b_i in self.b:
            b_i.zero_()
        self.t = 0

    def step(self, grad):
        """Apply one SGD step using the provided gradients.

        Args:
            grad: List of gradient arrays matching ``params``.
        """
        if self.params is None:
            raise RuntimeError("SGD parameters must be set before calling step(), got None")
        for i in range(len(self.params)):
            SGD.step_detail(
                grad[i],
                self.b[i],
                self.lr,
                self.momentum,
                self.dampening,
                self.weight_decay,
                self.nesterov,
                self.t,
                self.params[i],
            )
        self.t = self.t + 1

    @staticmethod
    def step_detail(g, b, lr, momentum, dampening, weight_decay, nesterov, t, params):
        """Apply an SGD update to a single parameter array.

        Args:
            g: Gradient array.
            b: Momentum buffer.
            lr: Learning rate.
            momentum: Momentum factor.
            dampening: Momentum dampening factor.
            weight_decay: Weight decay coefficient.
            nesterov: Whether to use Nesterov momentum.
            t: Current step index.
            params: Parameter array to update in-place.
        """
        if params.dtype != g.dtype:
            raise TypeError(
                f"SGD gradient dtype must match parameter dtype {type_repr(params.dtype)}, got {type_repr(g.dtype)}"
            )
        if params.dtype != b.dtype:
            raise TypeError(
                f"SGD momentum buffer dtype must match parameter dtype {type_repr(params.dtype)}, "
                f"got {type_repr(b.dtype)}"
            )
        if params.shape != g.shape:
            raise ValueError(f"SGD gradient shape must match parameter shape {params.shape}, got {g.shape}")
        scalar_type = _sgd_scalar_type(params.dtype)
        kernel_inputs = (
            g,
            b,
            scalar_type(lr),
            scalar_type(momentum),
            scalar_type(dampening),
            scalar_type(weight_decay),
            int(nesterov),
            t,
            params,
        )
        wp.launch(sgd_step_kernel, dim=len(params), inputs=kernel_inputs, device=params.device)
