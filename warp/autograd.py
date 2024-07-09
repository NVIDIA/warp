# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import itertools
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

import warp as wp

__all__ = [
    "jacobian",
    "jacobian_fd",
    "gradcheck",
    "gradcheck_tape",
    "plot_kernel_jacobians",
]


def gradcheck(
    function: wp.Kernel,
    dim: Tuple[int],
    inputs: Sequence,
    outputs: Sequence,
    *,
    eps=1e-4,
    atol=1e-3,
    rtol=1e-2,
    raise_exception=True,
    input_output_mask: List[Tuple[Union[str, int], Union[str, int]]] = None,
    device: wp.context.Devicelike = None,
    max_blocks=0,
    max_inputs_per_var=-1,
    max_outputs_per_var=-1,
    plot_relative_error=False,
    plot_absolute_error=False,
    show_summary: bool = True,
) -> bool:
    """
    Checks whether the autodiff gradient of a Warp kernel matches finite differences.
    Fails if the relative or absolute errors between the autodiff and finite difference gradients exceed the specified tolerance, or if the autodiff gradients contain NaN values.

    The kernel function and its adjoint version are launched with the given inputs and outputs, as well as the provided ``dim`` and ``max_blocks`` arguments (see :func:`warp.launch` for more details).

    Note:
        This function only supports Warp kernels whose input arguments precede the output arguments.

        Only Warp arrays with ``requires_grad=True`` are considered for the Jacobian computation.

        Structs arguments are not yet supported by this function to compute Jacobians.

    Args:
        function: The Warp kernel function, decorated with the ``@wp.kernel`` decorator.
        dim: The number of threads to launch the kernel, can be an integer, or a Tuple of ints.
        inputs: List of input variables.
        outputs: List of output variables.
        eps: The finite-difference step size.
        atol: The absolute tolerance for the gradient check.
        rtol: The relative tolerance for the gradient check.
        raise_exception: If True, raises a `ValueError` if the gradient check fails.
        input_output_mask: List of tuples specifying the input-output pairs to compute the Jacobian for. Inputs and outputs can be identified either by their integer indices of where they appear in the kernel input/output arguments, or by the respective argument names as strings. If None, computes the Jacobian for all input-output pairs.
        device: The device to launch on (optional)
        max_blocks: The maximum number of CUDA thread blocks to use.
        max_inputs_per_var: Maximum number of input dimensions over which to evaluate the Jacobians for the input-output pairs. Evaluates all input dimensions if value <= 0.
        max_outputs_per_var: Maximum number of output dimensions over which to evaluate the Jacobians for the input-output pairs. Evaluates all output dimensions if value <= 0.
        plot_relative_error: If True, visualizes the relative error of the Jacobians in a plot (requires ``matplotlib``).
        plot_absolute_error: If True, visualizes the absolute error of the Jacobians in a plot (requires ``matplotlib``).
        show_summary: If True, prints a summary table of the gradient check results.

    Returns:
        True if the gradient check passes, False otherwise.
    """

    assert isinstance(function, wp.Kernel), "The function argument must be a Warp kernel"

    jacs_fd = jacobian_fd(
        function,
        dim=dim,
        inputs=inputs,
        outputs=outputs,
        input_output_mask=input_output_mask,
        device=device,
        max_blocks=max_blocks,
        max_inputs_per_var=max_inputs_per_var,
        eps=eps,
        plot_jacobians=False,
    )

    jacs_ad = jacobian(
        function,
        dim=dim,
        inputs=inputs,
        outputs=outputs,
        input_output_mask=input_output_mask,
        device=device,
        max_blocks=max_blocks,
        max_outputs_per_var=max_outputs_per_var,
        plot_jacobians=False,
    )

    relative_error_jacs = {}
    absolute_error_jacs = {}

    if show_summary:
        summary = []
        summary_header = ["Input", "Output", "Max Abs Error", "Max Rel Error", "Pass"]

        class FontColors:
            OKGREEN = "\033[92m"
            WARNING = "\033[93m"
            FAIL = "\033[91m"
            ENDC = "\033[0m"

    success = True
    for (input_i, output_i), jac_fd in jacs_fd.items():
        jac_ad = jacs_ad[input_i, output_i]
        if plot_relative_error or plot_absolute_error:
            jac_rel_error = wp.empty_like(jac_fd)
            jac_abs_error = wp.empty_like(jac_fd)
            flat_jac_fd = scalarize_array_1d(jac_fd)
            flat_jac_ad = scalarize_array_1d(jac_ad)
            flat_jac_rel_error = scalarize_array_1d(jac_rel_error)
            flat_jac_abs_error = scalarize_array_1d(jac_abs_error)
            wp.launch(
                compute_error_kernel,
                dim=len(flat_jac_fd),
                inputs=[flat_jac_ad, flat_jac_fd, flat_jac_rel_error, flat_jac_abs_error],
                device=jac_fd.device,
            )
            relative_error_jacs[(input_i, output_i)] = jac_rel_error
            absolute_error_jacs[(input_i, output_i)] = jac_abs_error
        cut_jac_fd = jac_fd.numpy()
        cut_jac_ad = jac_ad.numpy()
        if max_outputs_per_var > 0:
            cut_jac_fd = cut_jac_fd[:max_outputs_per_var]
            cut_jac_ad = cut_jac_ad[:max_outputs_per_var]
        if max_inputs_per_var > 0:
            cut_jac_fd = cut_jac_fd[:, :max_inputs_per_var]
            cut_jac_ad = cut_jac_ad[:, :max_inputs_per_var]
        grad_matches = np.allclose(cut_jac_ad, cut_jac_fd, atol=atol, rtol=rtol)
        success = success and grad_matches
        if not grad_matches:
            if raise_exception:
                raise ValueError(
                    f"Gradient check failed for kernel {function.key}, input {input_i}, output {output_i}: "
                    f"finite difference and autodiff gradients do not match"
                )
            elif not show_summary:
                return False
        isnan = np.any(np.isnan(cut_jac_ad))
        success = success and not isnan
        if isnan:
            if raise_exception:
                raise ValueError(
                    f"Gradient check failed for kernel {function.key}, input {input_i}, output {output_i}: "
                    f"gradient contains NaN values"
                )
            elif not show_summary:
                return False

        if show_summary:
            max_abs_error = np.abs(cut_jac_ad - cut_jac_fd).max()
            max_rel_error = np.abs((cut_jac_ad - cut_jac_fd) / (cut_jac_fd + 1e-8)).max()
            if isnan:
                pass_str = FontColors.FAIL + "NaN" + FontColors.ENDC
            elif grad_matches:
                pass_str = FontColors.OKGREEN + "PASS" + FontColors.ENDC
            else:
                pass_str = FontColors.FAIL + "FAIL" + FontColors.ENDC
            input_name = function.adj.args[input_i].label
            output_name = function.adj.args[len(inputs) + output_i].label
            summary.append([input_name, output_name, f"{max_abs_error:.7e}", f"{max_rel_error:.7e}", pass_str])

    if show_summary:
        print_table(summary_header, summary)
        if not success:
            print(FontColors.FAIL + f"Gradient check for kernel {function.key} failed" + FontColors.ENDC)
        else:
            print(FontColors.OKGREEN + f"Gradient check for kernel {function.key} passed" + FontColors.ENDC)
    if plot_relative_error:
        plot_kernel_jacobians(
            relative_error_jacs,
            function,
            inputs,
            outputs,
            title=f"{function.key} kernel Jacobian relative error",
        )
    if plot_absolute_error:
        plot_kernel_jacobians(
            absolute_error_jacs,
            function,
            inputs,
            outputs,
            title=f"{function.key} kernel Jacobian absolute error",
        )

    return success


def gradcheck_tape(
    tape: wp.Tape,
    *,
    eps=1e-4,
    atol=1e-3,
    rtol=1e-2,
    raise_exception=True,
    input_output_masks: Dict[str, List[Tuple[Union[str, int], Union[str, int]]]] = None,
    blacklist_kernels: List[str] = None,
    whitelist_kernels: List[str] = None,
    max_inputs_per_var=-1,
    max_outputs_per_var=-1,
    plot_relative_error=False,
    plot_absolute_error=False,
    show_summary: bool = True,
) -> bool:
    """
    Checks whether the autodiff gradients for kernels recorded on the Warp tape match finite differences.
    Fails if the relative or absolute errors between the autodiff and finite difference gradients exceed the specified tolerance, or if the autodiff gradients contain NaN values.

    Note:
        Only Warp kernels recorded on the tape are checked but not arbitrary functions that have been recorded, e.g. via :meth:`Tape.record_func`.

        Only Warp arrays with ``requires_grad=True`` are considered for the Jacobian computation.

        Structs arguments are not yet supported by this function to compute Jacobians.

    Args:
        tape: The Warp tape to perform the gradient check on.
        eps: The finite-difference step size.
        atol: The absolute tolerance for the gradient check.
        rtol: The relative tolerance for the gradient check.
        raise_exception: If True, raises a `ValueError` if the gradient check fails.
        input_output_masks: Dictionary of input-output masks for each kernel in the tape, mapping from kernel keys to input-output masks. Inputs and outputs can be identified either by their integer indices of where they appear in the kernel input/output arguments, or by the respective argument names as strings. If None, computes the Jacobian for all input-output pairs.
        blacklist_kernels: List of kernel keys to exclude from the gradient check.
        whitelist_kernels: List of kernel keys to include in the gradient check. If not empty or None, only kernels in this list are checked.
        max_blocks: The maximum number of CUDA thread blocks to use.
        max_inputs_per_var: Maximum number of input dimensions over which to evaluate the Jacobians for the input-output pairs. Evaluates all input dimensions if value <= 0.
        max_outputs_per_var: Maximum number of output dimensions over which to evaluate the Jacobians for the input-output pairs. Evaluates all output dimensions if value <= 0.
        plot_relative_error: If True, visualizes the relative error of the Jacobians in a plot (requires ``matplotlib``).
        plot_absolute_error: If True, visualizes the absolute error of the Jacobians in a plot (requires ``matplotlib``).
        show_summary: If True, prints a summary table of the gradient check results.

    Returns:
        True if the gradient check passes for all kernels on the tape, False otherwise.
    """
    if input_output_masks is None:
        input_output_masks = {}
    if blacklist_kernels is None:
        blacklist_kernels = []
    else:
        blacklist_kernels = set(blacklist_kernels)
    if whitelist_kernels is None:
        whitelist_kernels = []
    else:
        whitelist_kernels = set(whitelist_kernels)

    overall_success = True
    for launch in tape.launches:
        if not isinstance(launch[0], wp.Kernel):
            continue
        kernel, dim, max_blocks, inputs, outputs, device = launch[:6]
        if len(whitelist_kernels) > 0 and kernel.key not in whitelist_kernels:
            continue
        if kernel.key in blacklist_kernels:
            continue
        input_output_mask = input_output_masks.get(kernel.key)
        success = gradcheck(
            kernel,
            dim,
            inputs,
            outputs,
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=raise_exception,
            input_output_mask=input_output_mask,
            device=device,
            max_blocks=max_blocks,
            max_inputs_per_var=max_inputs_per_var,
            max_outputs_per_var=max_outputs_per_var,
            plot_relative_error=plot_relative_error,
            plot_absolute_error=plot_absolute_error,
            show_summary=show_summary,
        )
        overall_success = overall_success and success

    return overall_success


def get_struct_vars(x: wp.codegen.StructInstance):
    return {varname: getattr(x, varname) for varname, _ in x._cls.ctype._fields_}


def infer_device(xs: list):
    # retrieve best matching Warp device for a list of variables
    for x in xs:
        if isinstance(x, wp.array):
            return x.device
        elif isinstance(x, wp.codegen.StructInstance):
            for var in get_struct_vars(x).values():
                if isinstance(var, wp.array):
                    return var.device
    return wp.get_preferred_device()


def plot_kernel_jacobians(
    jacobians: Dict[Tuple[int, int], wp.array],
    kernel: wp.Kernel,
    inputs: Sequence,
    outputs: Sequence,
    show_plot=True,
    show_colorbar=True,
    scale_colors_per_submatrix=False,
    title: str = None,
    colormap: str = "coolwarm",
    log_scale=False,
):
    """
    Visualizes the Jacobians computed by :func:`jacobian` or :func:`jacobian_fd` in a combined image plot.
    Requires the ``matplotlib`` package to be installed.

    Args:
        jacobians: A dictionary of Jacobians, where the keys are tuples of input and output indices, and the values are the Jacobian matrices.
        kernel: The Warp kernel function, decorated with the ``@wp.kernel`` decorator.
        inputs: List of input variables.
        outputs: List of output variables.
        show_plot: If True, displays the plot via ``plt.show()``.
        show_colorbar: If True, displays a colorbar next to the plot (or a colorbar next to every submatrix if ).
        scale_colors_per_submatrix: If True, considers the minimum and maximum of each Jacobian submatrix separately for color scaling. Otherwise, uses the global minimum and maximum of all Jacobians.
        title: The title of the plot (optional).
        colormap: The colormap to use for the plot.
        log_scale: If True, uses a logarithmic scale for the matrix values shown in the image plot.

    Returns:
        The created Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

    jacobians = sorted(jacobians.items(), key=lambda x: (x[0][1], x[0][0]))
    jacobians = dict(jacobians)

    input_to_ax = {}
    output_to_ax = {}
    for i, j in jacobians.keys():
        if i not in input_to_ax:
            input_to_ax[i] = len(input_to_ax)
        if j not in output_to_ax:
            output_to_ax[j] = len(output_to_ax)

    num_rows = len(output_to_ax)
    num_cols = len(input_to_ax)
    if num_rows == 0 or num_cols == 0:
        return

    # determine the width and height ratios for the subplots based on the
    # dimensions of the Jacobians
    width_ratios = []
    height_ratios = []
    for i, input in enumerate(inputs):
        if not isinstance(input, wp.array) or not input.requires_grad:
            continue
        input_stride = input.dtype._length_
        for j in range(len(outputs)):
            if (i, j) not in jacobians:
                continue
            jac_wp = jacobians[(i, j)]
            width_ratios.append(jac_wp.shape[1] * input_stride)
            break

    for i, output in enumerate(outputs):
        if not isinstance(output, wp.array) or not output.requires_grad:
            continue
        for j in range(len(inputs)):
            if (j, i) not in jacobians:
                continue
            jac_wp = jacobians[(j, i)]
            height_ratios.append(jac_wp.shape[0])
            break

    fig, axs = plt.subplots(
        ncols=num_cols,
        nrows=num_rows,
        figsize=(7, 7),
        sharex="col",
        sharey="row",
        gridspec_kw={
            "wspace": 0.1,
            "hspace": 0.1,
            "width_ratios": width_ratios,
            "height_ratios": height_ratios,
        },
        subplot_kw={"aspect": 1},
        squeeze=False,
    )
    if title is None:
        title = f"{kernel.key} kernel Jacobian"
    fig.suptitle(title)
    fig.canvas.manager.set_window_title(title)

    if not scale_colors_per_submatrix:
        safe_jacobians = [jac.numpy().flatten() for jac in jacobians.values()]
        safe_jacobians = [jac[~np.isnan(jac)] for jac in safe_jacobians]
        safe_jacobians = [jac for jac in safe_jacobians if len(jac) > 0]
        if len(safe_jacobians) == 0:
            vmin = 0
            vmax = 0
        else:
            vmin = min([jac.min() for jac in safe_jacobians])
            vmax = max([jac.max() for jac in safe_jacobians])

    has_plot = np.ones((num_rows, num_cols), dtype=bool)
    for i in range(num_rows):
        for j in range(num_cols):
            if (j, i) not in jacobians:
                ax = axs[i, j]
                ax.axis("off")
                has_plot[i, j] = False

    jac_i = 0
    for (input_i, output_i), jac_wp in jacobians.items():
        input = inputs[input_i]
        output = outputs[output_i]
        if not isinstance(input, wp.array) or not input.requires_grad:
            continue
        if not isinstance(output, wp.array) or not output.requires_grad:
            continue

        input_name = kernel.adj.args[input_i].label
        output_name = kernel.adj.args[len(inputs) + output_i].label

        ax_i, ax_j = output_to_ax[output_i], input_to_ax[input_i]
        ax = axs[ax_i, ax_j]
        ax.tick_params(which="major", width=1, length=7)
        ax.tick_params(which="minor", width=1, length=4, color="gray")
        # ax.yaxis.set_minor_formatter('{x:.0f}')

        input_stride = input.dtype._length_
        output_stride = output.dtype._length_

        jac = jac_wp.numpy()
        # Jacobian matrix has output stride already multiplied to first dimension
        jac = jac.reshape(jac_wp.shape[0], jac_wp.shape[1] * input_stride)
        ax.xaxis.set_minor_formatter("")
        ax.yaxis.set_minor_formatter("")
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        # ax.set_xticks(np.arange(jac.shape[0]))
        # stride = jac.shape[1] // jacobians[jac_i].shape[1]
        # ax.xaxis.set_major_locator(MultipleLocator(input_stride))
        if input_stride > 1:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=1, steps=[input_stride]))
            ticks = FuncFormatter(lambda x, pos, input_stride=input_stride: "{0:g}".format(x // input_stride))
            ax.xaxis.set_major_formatter(ticks)
        # ax.xaxis.set_major_locator(FixedLocator(np.arange(0, jac.shape[1] + 1, input_stride)))
        # ax.xaxis.set_major_formatter('{x:.0f}')
        # ticks =  np.arange(jac_wp.shape[1] + 1)
        # ax.set_xticklabels(ticks)

        # ax.yaxis.set_major_locator(FixedLocator(np.arange(0, jac.shape[0] + 1, output_stride)))
        # ax.yaxis.set_major_formatter('{x:.0f}')
        # ax.yaxis.set_major_locator(MultipleLocator(output_stride))

        if output_stride > 1:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=1, steps=[output_stride]))
        max_y = jac_wp.shape[0]
        ticks = FuncFormatter(
            lambda y, pos, max_y=max_y, output_stride=output_stride: "{0:g}".format((max_y - y) // output_stride)
        )
        ax.yaxis.set_major_formatter(ticks)
        # divide by output stride to get the correct number of rows
        ticks = np.arange(jac_wp.shape[0] // output_stride + 1)
        # flip y labels to match the order of matrix rows starting from the top
        # ax.set_yticklabels(ticks[::-1])
        if scale_colors_per_submatrix:
            safe_jac = jac[~np.isnan(jac)]
            vmin = safe_jac.min()
            vmax = safe_jac.max()
        img = ax.imshow(
            np.log10(np.abs(jac) + 1e-8) if log_scale else jac,
            cmap=colormap,
            aspect="auto",
            interpolation="nearest",
            extent=[0, jac.shape[1], 0, jac.shape[0]],
            vmin=vmin,
            vmax=vmax,
        )
        if ax_i == len(outputs) - 1 or not has_plot[ax_i + 1 :, ax_j].any():
            # last plot of this column
            ax.set_xlabel(input_name)
        if ax_j == 0 or not has_plot[ax_i, :ax_j].any():
            # first plot of this row
            ax.set_ylabel(output_name)
        ax.grid(color="gray", which="minor", linestyle="--", linewidth=0.5)
        ax.grid(color="black", which="major", linewidth=1.0)

        if show_colorbar and scale_colors_per_submatrix:
            plt.colorbar(img, ax=ax, orientation="vertical", pad=0.02)

        jac_i += 1

    if show_colorbar and not scale_colors_per_submatrix:
        m = plt.cm.ScalarMappable(cmap=colormap)
        m.set_array([vmin, vmax])
        m.set_clim(vmin, vmax)
        plt.colorbar(m, ax=axs, orientation="vertical", pad=0.02)

    plt.tight_layout()
    if show_plot:
        plt.show()
    return fig


def scalarize_array_1d(arr):
    # convert array to 1D array with scalar dtype
    if arr.dtype in wp.types.scalar_types:
        return arr.flatten()
    elif arr.dtype in wp.types.vector_types:
        return wp.array(
            ptr=arr.ptr,
            shape=(arr.size * arr.dtype._length_,),
            dtype=arr.dtype._wp_scalar_type_,
            device=arr.device,
        )
    else:
        raise ValueError(
            f"Unsupported array dtype {arr.dtype}: array to be flattened must be a scalar/vector/matrix array"
        )


def scalarize_array_2d(arr):
    assert arr.ndim == 2
    # convert array to 2D array with scalar dtype
    if arr.dtype in wp.types.scalar_types:
        return arr
    elif arr.dtype in wp.types.vector_types:
        return wp.array(
            ptr=arr.ptr,
            shape=(arr.shape[0], arr.shape[1] * arr.dtype._length_),
            dtype=arr.dtype._wp_scalar_type_,
            device=arr.device,
        )
    else:
        raise ValueError(
            f"Unsupported array dtype {arr.dtype}: array to be flattened must be a scalar/vector/matrix array"
        )


def jacobian(
    kernel: wp.Kernel,
    dim: Tuple[int],
    inputs: Sequence,
    outputs: Sequence = None,
    input_output_mask: List[Tuple[Union[str, int], Union[str, int]]] = None,
    device: wp.context.Devicelike = None,
    max_blocks=0,
    max_outputs_per_var=-1,
    plot_jacobians=False,
) -> Dict[Tuple[int, int], wp.array]:
    """
    Computes the Jacobians of a Warp kernel launch for the provided selection of differentiable inputs to differentiable outputs.

    The kernel adjoint function is launched with the given inputs and outputs, as well as the provided ``dim`` and ``max_blocks`` arguments (see :func:`warp.launch` for more details).

    Note:
        This function only supports Warp kernels whose input arguments precede the output arguments.

        Only Warp arrays with ``requires_grad=True`` are considered for the Jacobian computation.

        Structs arguments are not yet supported by this function to compute Jacobians.

    Args:
        kernel: The Warp kernel function, decorated with the ``@wp.kernel`` decorator
        dim: The number of threads to launch the kernel, can be an integer, or a Tuple of ints
        inputs: List of input variables.
        outputs: List of output variables. If None, the outputs are inferred from the kernel argument flags.
        input_output_mask: List of tuples specifying the input-output pairs to compute the Jacobian for. Inputs and outputs can be identified either by their integer indices of where they appear in the kernel input/output arguments, or by the respective argument names as strings. If None, computes the Jacobian for all input-output pairs.
        device: The device to launch on (optional)
        max_blocks: The maximum number of CUDA thread blocks to use.
        max_outputs_per_var: Maximum number of output dimensions over which to evaluate the Jacobians for the input-output pairs. Evaluates all output dimensions if value <= 0.
        plot_jacobians: If True, visualizes the computed Jacobians in a plot (requires ``matplotlib``).

    Returns:
        A dictionary of Jacobians, where the keys are tuples of input and output indices, and the values are the Jacobian matrices.
    """
    if outputs is None:
        outputs = []
    if input_output_mask is None:
        input_output_mask = []
    arg_names = [arg.label for arg in kernel.adj.args]

    def resolve_arg(name):
        if isinstance(name, int):
            return name
        return arg_names.index(name)

    input_output_mask = [
        (resolve_arg(input_name), resolve_arg(output_name) - len(inputs))
        for input_name, output_name in input_output_mask
    ]
    input_output_mask = set(input_output_mask)

    if device is None:
        device = infer_device(inputs + outputs)

    tape = wp.Tape()
    tape.record_launch(kernel=kernel, dim=dim, max_blocks=max_blocks, inputs=inputs, outputs=outputs, device=device)

    jacobians = {}

    for input_i, output_i in itertools.product(range(len(inputs)), range(len(outputs))):
        if len(input_output_mask) > 0 and (input_i, output_i) not in input_output_mask:
            continue
        input = inputs[input_i]
        output = outputs[output_i]
        if not isinstance(input, wp.array) or not input.requires_grad:
            continue
        if not isinstance(output, wp.array) or not output.requires_grad:
            continue
        out_grad = scalarize_array_1d(output.grad)
        output_num = out_grad.shape[0]
        jacobian = wp.empty((output_num, input.size), dtype=input.dtype, device=input.device)
        jacobian.fill_(wp.nan)
        if max_outputs_per_var > 0:
            output_num = min(output_num, max_outputs_per_var)
        for i in range(output_num):
            tape.zero()
            if i > 0:
                set_element(out_grad, i - 1, 0.0)
            set_element(out_grad, i, 1.0)
            tape.backward()
            jacobian[i].assign(input.grad)
        output.grad.zero_()
        jacobians[input_i, output_i] = jacobian

    if plot_jacobians:
        plot_kernel_jacobians(
            jacobians,
            kernel,
            inputs,
            outputs,
        )

    return jacobians


def jacobian_fd(
    kernel: wp.Kernel,
    dim: Tuple[int],
    inputs: Sequence,
    outputs: Sequence = None,
    input_output_mask: List[Tuple[Union[str, int], Union[str, int]]] = None,
    device: wp.context.Devicelike = None,
    max_blocks=0,
    max_inputs_per_var=-1,
    eps=1e-4,
    plot_jacobians=False,
) -> Dict[Tuple[int, int], wp.array]:
    """
    Computes the finite-difference Jacobian of a Warp kernel launch for the provided selection of differentiable inputs to differentiable outputs.
    The method uses a central difference scheme to approximate the Jacobian.

    The kernel is launched multiple times in forward-only mode with the given inputs and outputs, as well as the provided ``dim`` and ``max_blocks`` arguments (see :func:`warp.launch` for more details).

    Note:
        This function only supports Warp kernels whose input arguments precede the output arguments.

        Only Warp arrays with ``requires_grad=True`` are considered for the Jacobian computation.

        Structs arguments are not yet supported by this function to compute Jacobians.

    Args:
        kernel: The Warp kernel function, decorated with the ``@wp.kernel`` decorator
        dim: The number of threads to launch the kernel, can be an integer, or a Tuple of ints
        inputs: List of input variables.
        outputs: List of output variables. If None, the outputs are inferred from the kernel argument flags.
        input_output_mask: List of tuples specifying the input-output pairs to compute the Jacobian for. Inputs and outputs can be identified either by their integer indices of where they appear in the kernel input/output arguments, or by the respective argument names as strings. If None, computes the Jacobian for all input-output pairs.
        device: The device to launch on (optional)
        max_blocks: The maximum number of CUDA thread blocks to use.
        max_inputs_per_var: Maximum number of input dimensions over which to evaluate the Jacobians for the input-output pairs. Evaluates all input dimensions if value <= 0.
        eps: The finite-difference step size.
        plot_jacobians: If True, visualizes the computed Jacobians in a plot (requires ``matplotlib``).

    Returns:
        A dictionary of Jacobians, where the keys are tuples of input and output indices, and the values are the Jacobian matrices.
    """
    if outputs is None:
        outputs = []
    if input_output_mask is None:
        input_output_mask = []
    arg_names = [arg.label for arg in kernel.adj.args]

    def resolve_arg(name):
        if isinstance(name, int):
            return name
        return arg_names.index(name)

    input_output_mask = [
        (resolve_arg(input_name), resolve_arg(output_name) - len(inputs))
        for input_name, output_name in input_output_mask
    ]
    input_output_mask = set(input_output_mask)

    if device is None:
        device = infer_device(inputs + outputs)

    jacobians = {}

    for input_i, output_i in itertools.product(range(len(inputs)), range(len(outputs))):
        if len(input_output_mask) > 0 and (input_i, output_i) not in input_output_mask:
            continue
        input = inputs[input_i]
        output = outputs[output_i]
        if not isinstance(input, wp.array) or not input.requires_grad:
            continue
        if not isinstance(output, wp.array) or not output.requires_grad:
            continue

        flat_input = scalarize_array_1d(input)

        left = wp.clone(output)
        right = wp.clone(output)
        flat_left = scalarize_array_1d(left)
        flat_right = scalarize_array_1d(right)

        left_outputs = outputs[:output_i] + [left] + outputs[output_i + 1 :]
        right_outputs = outputs[:output_i] + [right] + outputs[output_i + 1 :]

        input_num = flat_input.shape[0]
        jacobian = wp.empty((flat_left.size, input.size), dtype=input.dtype, device=input.device)
        jacobian.fill_(wp.nan)

        jacobian_scalar = scalarize_array_2d(jacobian)
        jacobian_t = jacobian_scalar.transpose()
        if max_inputs_per_var > 0:
            input_num = min(input_num, max_inputs_per_var)
        for i in range(input_num):
            set_element(flat_input, i, -eps, relative=True)
            wp.launch(kernel, dim=dim, max_blocks=max_blocks, inputs=inputs, outputs=left_outputs, device=device)

            set_element(flat_input, i, 2 * eps, relative=True)
            wp.launch(kernel, dim=dim, max_blocks=max_blocks, inputs=inputs, outputs=right_outputs, device=device)

            set_element(flat_input, i, -eps, relative=True)

            compute_fd(flat_left, flat_right, eps, jacobian_t[i])

        output.grad.zero_()
        jacobians[input_i, output_i] = jacobian

    if plot_jacobians:
        plot_kernel_jacobians(
            jacobians,
            kernel,
            inputs,
            outputs,
        )

    return jacobians


@wp.kernel(enable_backward=False)
def set_element_kernel(a: wp.array(dtype=Any), i: int, val: Any, relative: bool):
    if relative:
        a[i] += val
    else:
        a[i] = val


def set_element(a: wp.array(dtype=Any), i: int, val: Any, relative: bool = False):
    wp.launch(set_element_kernel, dim=1, inputs=[a, i, a.dtype(val), relative], device=a.device)


@wp.kernel(enable_backward=False)
def compute_fd_kernel(left: wp.array(dtype=Any), right: wp.array(dtype=Any), eps: Any, fd: wp.array(dtype=Any)):
    tid = wp.tid()
    fd[tid] = (right[tid] - left[tid]) / (2.0 * eps)


def compute_fd(left: wp.array(dtype=Any), right: wp.array(dtype=Any), eps: float, fd: wp.array(dtype=Any)):
    wp.launch(compute_fd_kernel, dim=len(left), inputs=[left, right, eps], outputs=[fd], device=left.device)


@wp.kernel(enable_backward=False)
def compute_error_kernel(
    jacobian_ad: wp.array(dtype=Any),
    jacobian_fd: wp.array(dtype=Any),
    relative_error: wp.array(dtype=Any),
    absolute_error: wp.array(dtype=Any),
):
    tid = wp.tid()
    ad = jacobian_ad[tid]
    fd = jacobian_fd[tid]
    relative_error[tid] = (ad - fd) / (ad + 1e-8)
    absolute_error[tid] = wp.abs(ad - fd)


def print_table(headers, cells):
    """
    Prints a table with the given headers and cells.

    Args:
        headers: List of header strings.
        cells: List of lists of cell strings.
    """
    import re

    def sanitized_len(s):
        return len(re.sub(r"\033\[\d+m", "", str(s)))

    col_widths = [max(sanitized_len(cell) for cell in col) for col in zip(headers, *cells)]
    for header, col_width in zip(headers, col_widths):
        print(f"{header:{col_width}}", end=" | ")
    print()
    print("-" * (sum(col_widths) + 3 * len(col_widths) - 1))
    for cell_row in cells:
        for cell, col_width in zip(cell_row, col_widths):
            print(f"{cell:{col_width}}", end=" | ")
        print()
