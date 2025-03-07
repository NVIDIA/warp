# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict, namedtuple
from typing import Dict, List

import warp as wp


class Tape:
    """
    Record kernel launches within a Tape scope to enable automatic differentiation.
    Gradients can be computed after the operations have been recorded on the tape via
    :meth:`Tape.backward()`.

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
        self.launches = []
        self.scopes = []

        self.loss = None

    def __enter__(self):
        wp.context.init()

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
                if a.grad is None:
                    a.grad = g
                else:
                    # ensure we can capture this backward pass in a CUDA graph
                    a.grad.assign(g)

        # run launches backwards
        for launch in reversed(self.launches):
            if callable(launch):
                launch()

            else:
                # kernel option takes precedence over module option
                enable_backward = launch[0].options.get("enable_backward")
                if enable_backward is False:
                    msg = f"Running the tape backwards may produce incorrect gradients because recorded kernel {launch[0].key} is configured with the option 'enable_backward=False'."
                    wp.utils.warn(msg)
                elif enable_backward is None:
                    enable_backward = launch[0].module.options.get("enable_backward")
                    if enable_backward is False:
                        msg = f"Running the tape backwards may produce incorrect gradients because recorded kernel {launch[0].key} is defined in a module with the option 'enable_backward=False' set."
                        wp.utils.warn(msg)

                kernel = launch[0]
                dim = launch[1]
                max_blocks = launch[2]
                inputs = launch[3]
                outputs = launch[4]
                device = launch[5]
                block_dim = launch[6]

                adj_inputs = []
                adj_outputs = []

                # lookup adjoint inputs
                for a in inputs:
                    adj_inputs.append(self.get_adjoint(a))

                # lookup adjoint outputs, todo: only allocate outputs if necessary
                for a in outputs:
                    adj_outputs.append(self.get_adjoint(a))

                if enable_backward:
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
                        block_dim=block_dim,
                    )

    # record a kernel launch on the tape
    def record_launch(self, kernel, dim, max_blocks, inputs, outputs, device, block_dim=0, metadata=None):
        if metadata is None:
            metadata = {}
        self.launches.append([kernel, dim, max_blocks, inputs, outputs, device, block_dim, metadata])

    def record_func(self, backward, arrays):
        """
        Records a custom function to be executed only in the backward pass.

        Args:
            backward (Callable): A callable Python object (can be any function) that will be executed in the backward pass.
            arrays (list): A list of arrays that are used by the backward function. The tape keeps track of these to be able to zero their gradients in Tape.zero()
        """
        self.launches.append(backward)

        for a in arrays:
            if isinstance(a, wp.array) and a.grad:
                self.gradients[a] = a.grad
            else:
                raise RuntimeError(
                    f"Array {a} is not of type wp.array or is missing a gradient array. Set array parameter requires_grad=True during instantiation."
                )

    def record_scope_begin(self, scope_name, metadata=None):
        """
        Begin a scope on the tape to group operations together. Scopes are only used in the visualization functions.
        """
        if metadata is None:
            metadata = {}
        self.scopes.append((len(self.launches), scope_name, metadata))

    def record_scope_end(self, remove_scope_if_empty=True):
        """
        End a scope on the tape.

        Args:
            remove_scope_if_empty (bool): If True, the scope will be removed if no kernel launches were recorded within it.
        """
        if remove_scope_if_empty and self.scopes[-1][0] == len(self.launches):
            self.scopes = self.scopes[:-1]
        else:
            self.scopes.append((len(self.launches), None, None))

    def _check_kernel_array_access(self, kernel, args):
        """Detect illegal inter-kernel write after read access patterns during launch capture"""
        adj = kernel.adj
        kernel_name = adj.fun_name
        filename = adj.filename
        lineno = adj.fun_lineno

        for i, arg in enumerate(args):
            if isinstance(arg, wp.array):
                arg_name = adj.args[i].label

                # we check write condition first because we allow (write --> read) within the same kernel
                if adj.args[i].is_write:
                    arg.mark_write(arg_name=arg_name, kernel_name=kernel_name, filename=filename, lineno=lineno)

                if adj.args[i].is_read:
                    arg.mark_read()

    # returns the adjoint of a kernel parameter
    def get_adjoint(self, a):
        if not wp.types.is_array(a) and not isinstance(a, wp.codegen.StructInstance):
            # if input is a simple type (e.g.: float, vec3, etc) or a non-Warp array,
            # then no gradient needed (we only return gradients through Warp arrays and structs)
            return None

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
        self.scopes = []
        self.zero()
        if wp.config.verify_autograd_array_access:
            self._reset_array_read_flags()

    def zero(self):
        """
        Zero out all gradients recorded on the tape.
        """
        for a, g in self.gradients.items():
            if isinstance(a, wp.codegen.StructInstance):
                for name in g._cls.vars:
                    if isinstance(g._cls.vars[name].type, wp.array) and g._cls.vars[name].requires_grad:
                        getattr(g, name).zero_()
            else:
                g.zero_()

    def _reset_array_read_flags(self):
        """
        Reset all recorded array read flags to False
        """
        for a in self.gradients:
            if isinstance(a, wp.array):
                a.mark_init()

    def visualize(
        self,
        filename: str = None,
        simplify_graph=True,
        hide_readonly_arrays=False,
        array_labels: Dict[wp.array, str] = None,
        choose_longest_node_name: bool = True,
        ignore_graph_scopes: bool = False,
        track_inputs: List[wp.array] = None,
        track_outputs: List[wp.array] = None,
        track_input_names: List[str] = None,
        track_output_names: List[str] = None,
        graph_direction: str = "LR",
    ) -> str:
        """
        Visualize the recorded operations on the tape as a `GraphViz diagram <https://graphviz.org/>`_.

        Example
        -------

        .. code-block:: python

            import warp as wp

            tape = wp.Tape()
            with tape:
                # record Warp kernel launches here
                wp.launch(...)

            dot_code = tape.visualize("tape.dot")

        This function creates a GraphViz dot file that can be rendered into an image using the GraphViz command line tool, e.g. via

        .. code-block:: bash

                dot -Tpng tape.dot -o tape.png

        Args:
            filename (str): The filename to save the visualization to (optional).
            simplify_graph (bool): If True, simplify the graph by detecting repeated kernel launch sequences and summarizing them in subgraphs.
            hide_readonly_arrays (bool): If True, hide arrays that are not modified by any kernel launch.
            array_labels (Dict[wp.array, str]): A dictionary mapping arrays to custom labels.
            choose_longest_node_name (bool): If True, the automatic name resolution will aim to find the longest name for each array in the computation graph.
            ignore_graph_scopes (bool): If True, ignore the scopes recorded on the tape when visualizing the graph.
            track_inputs (List[wp.array]): A list of arrays to track as inputs in the graph to ensure they are shown regardless of the `hide_readonly_arrays` setting.
            track_outputs (List[wp.array]): A list of arrays to track as outputs in the graph so that they remain visible.
            track_input_names (List[str]): A list of custom names for the input arrays to track in the graph (used in conjunction with `track_inputs`).
            track_output_names (List[str]): A list of custom names for the output arrays to track in the graph (used in conjunction with `track_outputs`).
            graph_direction (str): The direction of the graph layout (default: "LR").

        Returns:
            str: The dot code representing the graph.

        """
        if track_output_names is None:
            track_output_names = []
        if track_input_names is None:
            track_input_names = []
        if track_outputs is None:
            track_outputs = []
        if track_inputs is None:
            track_inputs = []
        if array_labels is None:
            array_labels = {}
        return visualize_tape_graphviz(
            self,
            filename,
            simplify_graph,
            hide_readonly_arrays,
            array_labels,
            choose_longest_node_name,
            ignore_graph_scopes,
            track_inputs,
            track_outputs,
            track_input_names,
            track_output_names,
            graph_direction,
        )


class TapeVisitor:
    def emit_array_node(self, arr: wp.array, label: str, active_scope_stack: List[str], indent_level: int):
        pass

    def emit_kernel_launch_node(
        self, kernel: wp.Kernel, kernel_launch_id: str, launch_data: dict, rendered: bool, indent_level: int
    ):
        pass

    def emit_edge_array_kernel(self, arr: wp.array, kernel_launch_id: str, kernel_input_id: int, indent_level: int):
        pass

    def emit_edge_kernel_array(self, kernel_launch_id: str, kernel_output_id: int, arr: wp.array, indent_level: int):
        pass

    def emit_edge_array_array(self, src: wp.array, dst: wp.array, indent_level: int):
        pass

    def emit_scope_begin(self, active_scope_id: int, active_scope_name: str, metadata: dict, indent_level: int):
        pass

    def emit_scope_end(self, indent_level: int):
        pass


def get_struct_vars(x: wp.codegen.StructInstance):
    return {varname: getattr(x, varname) for varname, _ in x._cls.ctype._fields_}


class GraphvizTapeVisitor(TapeVisitor):
    def __init__(self):
        self.graphviz_lines = []
        self.indent_str = "\t"
        self.scope_classes = {}
        self.max_indent = 0
        # mapping from array pointer to kernel:port ID
        self.pointer_to_port = {}
        # set of inserted edges between kernel:port IDs
        self.edges = set()
        # set of inserted array nodes
        self.array_nodes = set()

    @staticmethod
    def sanitize(s):
        return (
            s.replace("\n", " ")
            .replace('"', " ")
            .replace("'", " ")
            .replace("[", "&#91;")
            .replace("]", "&#93;")
            .replace("`", "&#96;")
            .replace(":", "&#58;")
            .replace("\\", "\\\\")
            .replace("/", "&#47;")
            .replace("(", "&#40;")
            .replace(")", "&#41;")
            .replace(",", "")
            .replace("{", "&#123;")
            .replace("}", "&#125;")
            .replace("<", "&#60;")
            .replace(">", "&#62;")
        )

    @staticmethod
    def dtype2str(dtype):
        type_str = str(dtype)
        if hasattr(dtype, "key"):
            type_str = dtype.key
        elif "'" in type_str:
            type_str = type_str.split("'")[1]
        return type_str

    def emit_array_node(self, arr: wp.array, label: str, active_scope_stack: List[str], indent_level: int):
        if arr.ptr in self.array_nodes:
            return
        if arr.ptr in self.pointer_to_port:
            return
        self.array_nodes.add(arr.ptr)
        color = "lightgray"
        if arr.requires_grad:
            color = "#76B900"
        options = [
            f'label="{label}"',
            "shape=ellipse",
            "style=filled",
            f'fillcolor="{color}"',
        ]
        chart_indent = self.indent_str * indent_level
        arr_id = f"arr{arr.ptr}"
        type_str = self.dtype2str(arr.dtype)
        # type_str = self.sanitize(type_str)
        # class_name = "array" if not arr.requires_grad else "array_grad"
        # self.graphviz_lines.append(chart_indent + f'{arr_id}(["`{label}`"]):::{class_name}')
        tooltip = f"Array {label} / ptr={arr.ptr}, shape={str(arr.shape)}, dtype={type_str}, requires_grad={arr.requires_grad}"
        options.append(f'tooltip="{tooltip}"')
        # self.graphviz_lines.append(chart_indent + f'click {arr_id} callback "{tooltip}"')
        # self.max_indent = max(self.max_indent, indent_level)
        self.graphviz_lines.append(f"{chart_indent}{arr_id} [{','.join(options)}];")

    def emit_kernel_launch_node(
        self, kernel: wp.Kernel, kernel_launch_id: str, launch_data: dict, rendered: bool, indent_level: int
    ):
        if not rendered:
            return
        chart_indent = self.indent_str * indent_level

        table = []
        table.append(
            '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" border="0" cellspacing="2" cellpadding="4" bgcolor="#888888" gradientangle="0">'
        )
        table.append(f'<TR><TD BGCOLOR="#ffffffaa" colspan="2" align="center"><b>{kernel.key}</b></TD></TR>')
        num_inputs = len(launch_data["inputs"])
        num_outputs = len(launch_data["outputs"])
        nrows = max(num_inputs, num_outputs)
        nargs = len(kernel.adj.args)
        for i in range(nrows):
            row = []
            if i < num_inputs:
                arg = kernel.adj.args[i]
                port_id = f"in_{i}"
                if isinstance(arg.type, wp.array):
                    tooltip = f"array: dtype={self.dtype2str(arg.type.dtype)}"
                else:
                    tooltip = f"dtype={self.sanitize(self.dtype2str(arg.type))}"
                row.append(
                    f'<TD PORT="{port_id}" BGCOLOR="#BBBBBB" align="left" title="{tooltip}"><font color="black">{arg.label}</font></TD>'
                )
                launch_data["inputs"][i]
                # if var is not None and isinstance(var, wp.array):
                #     self.pointer_to_port[var.ptr] = f"{kernel_launch_id}:{port_id}"
            else:
                row.append('<TD BORDER="0"></TD>')
            # if i >= nargs - 1:
            #     row.append('<TD></TD>')
            #     table.append(f'<TR>{row[0]}{row[1]}</TR>')
            #     break
            if i < num_outputs and i + num_inputs < nargs:
                arg = kernel.adj.args[i + num_inputs].label
                port_id = f"out_{i}"
                row.append(
                    f'<TD PORT="{port_id}" BGCOLOR="#BBBBBB" align="right"><font color="black">{arg}</font></TD>'
                )
                launch_data["outputs"][i]
                # if var is not None and isinstance(var, wp.array):
                #     self.pointer_to_port[var.ptr] = f"{kernel_launch_id}:{port_id}"
            else:
                row.append('<TD BORDER="0"></TD>')
            table.append(f"<TR>{row[0]}{row[1]}</TR>")
        table.append("</TABLE>")

        label = f"{chart_indent}\n".join(table)
        node_attrs = f"label=<{label}>"
        if "caller" in launch_data:
            caller = launch_data["caller"]
            node_attrs += f',tooltip="{self.sanitize(caller["file"])}:{caller["lineno"]} ({caller["func"]})"'

        self.graphviz_lines.append(f"{chart_indent}{kernel_launch_id} [{node_attrs}];")

    def emit_edge_array_kernel(self, arr_ptr: int, kernel_launch_id: str, kernel_input_id: int, indent_level: int):
        chart_indent = self.indent_str * indent_level
        if arr_ptr in self.pointer_to_port:
            arr_id = self.pointer_to_port[arr_ptr]
        elif arr_ptr in self.array_nodes:
            arr_id = f"arr{arr_ptr}"
        else:
            return
        target_id = f"{kernel_launch_id}:in_{kernel_input_id}"
        if (arr_id, target_id) in self.edges:
            return
        self.edges.add((arr_id, target_id))
        self.graphviz_lines.append(f"{chart_indent}{arr_id} -> {target_id}")

    def emit_edge_kernel_array(self, kernel_launch_id: str, kernel_output_id: int, arr_ptr: int, indent_level: int):
        chart_indent = self.indent_str * indent_level
        if arr_ptr in self.pointer_to_port:
            arr_id = self.pointer_to_port[arr_ptr]
        elif arr_ptr in self.array_nodes:
            arr_id = f"arr{arr_ptr}"
        else:
            return
        source_id = f"{kernel_launch_id}:out_{kernel_output_id}"
        if (source_id, arr_id) in self.edges:
            return
        self.edges.add((source_id, arr_id))
        self.graphviz_lines.append(f"{chart_indent}{source_id} -> {arr_id};")

    def emit_edge_array_array(self, src: wp.array, dst: wp.array, indent_level: int):
        chart_indent = self.indent_str * indent_level
        src_id = f"arr{src.ptr}"
        dst_id = f"arr{dst.ptr}"
        if (src_id, dst_id) in self.edges:
            return
        self.edges.add((src_id, dst_id))
        self.graphviz_lines.append(f'{chart_indent}{src_id} -> {dst_id} [color="#0072B9",constraint=false];')

    def emit_scope_begin(self, active_scope_id: int, active_scope_name: str, metadata: dict, indent_level: int):
        chart_indent = self.indent_str * indent_level
        scope_key = f"cluster{active_scope_id}"
        scope_class = metadata.get("type", "scope")
        self.graphviz_lines.append(f"{chart_indent}subgraph {scope_key} {{")
        chart_indent += self.indent_str
        self.graphviz_lines.append(f'{chart_indent}style="rounded,filled";')
        if scope_class == "scope":
            self.graphviz_lines.append(f'{chart_indent}fillcolor="#76B90022";')
            self.graphviz_lines.append(f'{chart_indent}pencolor="#76B900";')
        else:
            self.graphviz_lines.append(f'{chart_indent}fillcolor="#0072B922";')
            self.graphviz_lines.append(f'{chart_indent}pencolor="#0072B9";')
        self.graphviz_lines.append(f"{chart_indent}label=<<b>{active_scope_name}</b>>;\n")

    def emit_scope_end(self, indent_level: int):
        chart_indent = self.indent_str * indent_level
        self.graphviz_lines.append(f"{chart_indent}}};\n")


class ArrayStatsVisitor(TapeVisitor):
    ArrayState = namedtuple("ArrayState", ["mean", "std", "min", "max"])

    def __init__(self):
        self.array_names = {}
        self.launch_data = {}
        self.launches = []
        self.array_value_stats = []
        self.array_grad_stats = []

    def emit_array_node(self, arr: wp.array, label: str, active_scope_stack: List[str], indent_level: int):
        if arr.device.is_capturing:
            raise RuntimeError("Cannot record arrays while graph capturing is active.")
        self.array_names[arr.ptr] = label

    def emit_kernel_launch_node(
        self, kernel: wp.Kernel, kernel_launch_id: str, launch_data: dict, rendered: bool, indent_level: int
    ):
        self.launch_data[kernel_launch_id] = launch_data
        self.launches.append(kernel_launch_id)
        value_stats = {}
        grad_stats = {}
        for output in launch_data["outputs"]:
            if isinstance(output, wp.array):
                arr_np = output.numpy()
                value_stats[output.ptr] = self.ArrayState(
                    mean=arr_np.mean(), std=arr_np.std(), min=arr_np.min(), max=arr_np.max()
                )
        for input in launch_data["inputs"]:
            if isinstance(input, wp.array) and input.requires_grad and input.grad is not None:
                arr_np = input.grad.numpy()
                grad_stats[input.ptr] = self.ArrayState(
                    mean=arr_np.mean(), std=arr_np.std(), min=arr_np.min(), max=arr_np.max()
                )
        self.array_value_stats.append(value_stats)
        self.array_grad_stats.insert(0, grad_stats)


Launch = namedtuple(
    "Launch", ["id", "kernel", "dim", "max_blocks", "inputs", "outputs", "device", "block_dim", "metadata"]
)
RepeatedSequence = namedtuple("RepeatedSequence", ["start", "end", "repetitions"])


def visit_tape(
    tape: Tape,
    visitor: TapeVisitor,
    simplify_graph=True,
    hide_readonly_arrays=False,
    array_labels: Dict[wp.array, str] = None,
    choose_longest_node_name: bool = True,
    ignore_graph_scopes: bool = False,
    track_inputs: List[wp.array] = None,
    track_outputs: List[wp.array] = None,
    track_input_names: List[str] = None,
    track_output_names: List[str] = None,
):
    if track_output_names is None:
        track_output_names = []
    if track_input_names is None:
        track_input_names = []
    if track_outputs is None:
        track_outputs = []
    if track_inputs is None:
        track_inputs = []
    if array_labels is None:
        array_labels = {}

    def get_launch_id(launch):
        kernel = launch[0]
        suffix = ""
        if len(launch) > 7:
            metadata = launch[7]
            # calling function helps to identify unique launches
            if "caller" in metadata:
                caller = metadata["caller"]
                suffix = str(hash(caller["file"] + caller["func"] + str(caller["lineno"])))
        return f"{kernel.module.name}.{kernel.key}{suffix}"

    # exclude function calls, only consider kernel launches
    kernel_launches = []
    kernel_scopes = []

    next_scope_id = 0
    id_offset = 0
    for i, launch in enumerate(tape.launches):
        if isinstance(launch, list):
            kernel_launches.append(launch)
        else:
            id_offset -= 1
        while next_scope_id < len(tape.scopes) and i == tape.scopes[next_scope_id][0]:
            scope = tape.scopes[next_scope_id]
            # update scope launch index to account for removed function calls
            new_scope = (scope[0] + id_offset, *scope[1:])
            kernel_scopes.append(new_scope)
            next_scope_id += 1

    launch_structs = [
        Launch(
            id=get_launch_id(launch),
            kernel=launch[0],
            dim=launch[1],
            max_blocks=launch[2],
            inputs=launch[3],
            outputs=launch[4],
            device=launch[5],
            block_dim=launch[6],
            metadata=launch[7] if len(launch) > 7 else {},
        )
        for launch in kernel_launches
    ]
    launch_ids = [get_launch_id(launch) for launch in kernel_launches]

    def get_repeating_sequences(sequence: List[str]):
        # yield all consecutively repeating subsequences in descending order of length
        for length in range(len(sequence) // 2 + 1, 0, -1):
            for start in range(len(sequence) - length):
                if sequence[start : start + length] == sequence[start + length : start + 2 * length]:
                    # we found a sequence that repeats at least once
                    candidate = RepeatedSequence(start, start + length, 2)
                    if length == 1:
                        # this repetition cannot be made up of smaller repetitions
                        yield candidate

                    # check if this sequence is made up entirely of smaller repetitions
                    for sl in range(1, length // 2 + 1):
                        # loop over subsequence lengths and check if they repeat
                        subseq = sequence[start : start + sl]
                        if all(
                            sequence[start + i * sl : start + (i + 1) * sl] == subseq for i in range(1, length // sl)
                        ):
                            rep_count = length // sl + 1
                            # check whether there are more repetitions beyond the previous end
                            for cstart in range(start + length, len(sequence) - sl, sl):
                                if sequence[cstart : cstart + sl] != subseq:
                                    break
                                rep_count += 1
                            candidate = RepeatedSequence(start, start + sl, rep_count)
                            yield candidate
                            break

    def process_sequence(sequence: List[str]) -> RepeatedSequence:
        # find the longest contiguous repetition in the sequence
        if len(sequence) < 2:
            return None

        for r in get_repeating_sequences(sequence):
            rlen = r.end - r.start
            rseq = sequence[r.start : r.end]
            # ensure that the repetitions of this subsequence immediately follow each other
            candidates = defaultdict(int)  # mapping from start index to number of repetitions
            curr_start = r.start
            i = r.end
            while i + rlen <= len(sequence):
                if sequence[i : i + rlen] == rseq:
                    candidates[curr_start] += 1
                    i += rlen
                else:
                    try:
                        curr_start = sequence.index(rseq, i)
                        i = curr_start + rlen
                    except ValueError:
                        break

            if len(candidates) > 0:
                start, reps = max(candidates.items(), key=lambda x: x[1])
                return RepeatedSequence(start, start + rlen, reps + 1)

        return None

    repetitions = []

    def find_sequences(sequence):
        # recursively find repetitions in sequence
        nonlocal repetitions

        if len(sequence) == 0:
            return

        # find LRS in current sequence
        longest_rep = process_sequence(sequence)
        if longest_rep is None:
            return

        # process sequence up until the current LRS
        find_sequences(sequence[: longest_rep.start])

        # process repeated sequence
        rstr = sequence[longest_rep.start : longest_rep.end]
        if longest_rep.repetitions >= 2:
            repetitions.append(longest_rep)

        find_sequences(rstr)

        # process remaining sequence
        rlen = longest_rep.end - longest_rep.start
        reps = longest_rep.repetitions
        end_idx = longest_rep.start + (reps + 1) * rlen
        if end_idx < len(sequence):
            find_sequences(sequence[end_idx:])

        return

    find_sequences(launch_ids)

    wrap_around_connections = set()

    # mapping from array ptr to already existing array in a repetition
    array_repeated = {}

    array_to_launch = defaultdict(list)
    launch_to_array = defaultdict(list)

    if simplify_graph:
        # mappings from unique launch string to index of first occurrence and vice versa
        launch_to_index = {}
        index_to_launch = {}

        # new arrays of launches, scopes without repetitions
        launches = []
        scopes = []

        def find_scope_end(scope_i):
            opened_scopes = 0
            for i in range(scope_i, len(kernel_scopes)):
                scope = kernel_scopes[i]
                if scope[1] is not None:
                    opened_scopes += 1
                else:
                    opened_scopes -= 1
                if opened_scopes == 0:
                    return scope[0]
            return len(kernel_scopes)

        def process_launches(kernel_launches, start_i, end_i, rep_i, scope_i, skipped_i):
            nonlocal \
                launches, \
                scopes, \
                launch_to_index, \
                index_to_launch, \
                wrap_around_connections, \
                launch_to_array, \
                array_to_launch
            i = start_i  # index of current launch
            opened_scopes = 0
            while i < end_i:
                launch_id = launch_ids[i]

                while (
                    scope_i < len(kernel_scopes)
                    and i >= kernel_scopes[scope_i][0]
                    and kernel_scopes[scope_i][1] is None
                ):
                    # add any missing closing scopes before we go into a repeating sequence
                    scope = kernel_scopes[scope_i]
                    if opened_scopes >= 1:
                        scopes.append((scope[0] - skipped_i, *scope[1:]))
                    scope_i += 1
                    opened_scopes -= 1

                # keep track of the mapping between arrays and kernel launch arguments
                for arg_i, arg in enumerate(kernel_launches[i].inputs + kernel_launches[i].outputs):
                    if isinstance(arg, wp.array):
                        array_to_launch[arg.ptr].append((launch_id, arg_i))
                        launch_to_array[(launch_id, arg_i)].append(arg)

                # handle repetitions
                if rep_i < len(repetitions):
                    rep = repetitions[rep_i]
                    if i == rep.start:
                        rep_len = rep.end - rep.start
                        after_rep = rep.start + rep.repetitions * rep_len
                        # check if there is a scope that matches the entire repetition
                        skip_adding_repetition_scope = False
                        for scope_j in range(scope_i, len(kernel_scopes)):
                            scope = kernel_scopes[scope_j]
                            if scope[0] > rep.start:
                                break
                            if scope[0] == rep.start and scope[1] is not None:
                                # check if this scope also ends at the end of the repetition
                                scope_end = find_scope_end(scope_j)
                                if scope_end == after_rep:
                                    # replace scope details
                                    kernel_scopes[scope_j] = (
                                        rep.start,
                                        f"{scope[1]} (repeated {rep.repetitions}x)",
                                        {"type": "repeated", "count": rep.repetitions},
                                    )
                                    skip_adding_repetition_scope = True
                                    break

                        if not skip_adding_repetition_scope:
                            # add a new scope marking this repetition
                            scope_name = f"repeated {rep.repetitions}x"
                            scopes.append((len(launches), scope_name, {"type": "repeated", "count": rep.repetitions}))

                        # process repetition recursively to handle nested repetitions
                        process_launches(kernel_launches, rep.start, rep.end, rep_i + 1, scope_i, skipped_i)

                        if not skip_adding_repetition_scope:
                            # close the scope we just added marking this repetition
                            scopes.append((len(launches), None, None))

                        # collect all output arrays from the first iteration
                        output_arrays = {}
                        for j in range(i, i + rep_len):
                            launch = kernel_launches[j]
                            launch_id = launch_ids[j]
                            for k, arg in enumerate(launch.outputs):
                                arg_i = k + len(launch.inputs)
                                if isinstance(arg, wp.array):
                                    output_arrays[arg.ptr] = arg
                                    array_to_launch[arg.ptr].append((launch_id, arg_i))

                        # find out which output arrays feed back as inputs to the next iteration
                        # so we can add them as wrap-around connections
                        for j in range(i + rep_len, i + 2 * rep_len):
                            launch = kernel_launches[j]
                            launch_id = launch_ids[j]
                            for arg_i, arg in enumerate(launch.inputs):
                                if isinstance(arg, wp.array) and arg.ptr in output_arrays:
                                    first_encountered_var = launch_to_array[(launch_id, arg_i)][0]
                                    # print(array_to_launch[arg.ptr])
                                    # array_to_launch[arg.ptr].append((launch_id, arg_i))
                                    # launch_to_array[(launch_id, arg_i)].append(arg)
                                    src_launch = array_to_launch[arg.ptr][-1]
                                    src_arr = launch_to_array[src_launch][-1]
                                    wrap_around_connections.add((src_arr.ptr, first_encountered_var.ptr))

                        # map arrays appearing as launch arguments in following repetitions to their first occurrence
                        skip_len = rep.repetitions * rep_len
                        for j in range(i + rep_len, i + skip_len):
                            launch = kernel_launches[j]
                            launch_id = launch_ids[j]
                            for arg_i, arg in enumerate(launch.inputs + launch.outputs):
                                if isinstance(arg, wp.array):
                                    array_repeated[arg.ptr] = launch_to_array[(launch_id, arg_i)][0].ptr

                        # skip launches during these repetitions
                        i += skip_len
                        skipped_i += skip_len - rep_len
                        rep_i += 1

                        # skip scopes during the repetitions
                        while scope_i < len(kernel_scopes) and i > kernel_scopes[scope_i][0]:
                            scope_i += 1

                        continue

                # add launch
                launch = kernel_launches[i]
                launch_id = launch_ids[i]
                if launch_id not in launch_to_index:
                    # we encountered an unseen kernel
                    j = len(launch_to_index)
                    launch_to_index[launch_id] = j
                    index_to_launch[j] = launch_id
                    launches.append(launch)

                while scope_i < len(kernel_scopes) and i >= kernel_scopes[scope_i][0]:
                    # add scopes encompassing the kernels we added so far
                    scope = kernel_scopes[scope_i]
                    if scope[1] is not None:
                        scopes.append((scope[0] - skipped_i, *scope[1:]))
                        opened_scopes += 1
                    else:
                        if opened_scopes >= 1:
                            # only add closing scope if there was an opening scope
                            scopes.append((scope[0] - skipped_i, *scope[1:]))
                        opened_scopes -= 1
                    scope_i += 1

                i += 1

            # close any remaining open scopes
            for _ in range(opened_scopes):
                scopes.append((end_i - skipped_i, None, None))

        process_launches(launch_structs, 0, len(launch_structs), 0, 0, 0)

        # end of simplify_graph
    else:
        launches = launch_structs
        scopes = kernel_scopes

    node_labels = {}
    inserted_arrays = {}  # mapping from array ptr to array
    kernel_launch_count = defaultdict(int)
    # array -> list of kernels that modify it
    manipulated_nodes = defaultdict(list)

    indent_level = 0

    input_output_ptr = set()
    for input in track_inputs:
        input_output_ptr.add(input.ptr)
    for output in track_outputs:
        input_output_ptr.add(output.ptr)

    def add_array_node(x: wp.array, name: str, active_scope_stack=None):
        if active_scope_stack is None:
            active_scope_stack = []
        nonlocal node_labels
        if x in array_labels:
            name = array_labels[x]
        if x.ptr in node_labels:
            if x.ptr not in input_output_ptr:
                # update name unless it is an input/output array
                if choose_longest_node_name:
                    if len(name) > len(node_labels[x.ptr]):
                        node_labels[x.ptr] = name
                else:
                    node_labels[x.ptr] = name
            return

        visitor.emit_array_node(x, name, active_scope_stack, indent_level)
        node_labels[x.ptr] = name
        inserted_arrays[x.ptr] = x

    for i, x in enumerate(track_inputs):
        if i < len(track_input_names):
            name = track_input_names[i]
        else:
            name = f"input_{i}"
        add_array_node(x, name)
    for i, x in enumerate(track_outputs):
        if i < len(track_output_names):
            name = track_output_names[i]
        else:
            name = f"output_{i}"
        add_array_node(x, name)
    # add arrays which are output of a kernel (used to simplify the graph)
    computed_nodes = set()
    for output in track_outputs:
        computed_nodes.add(output.ptr)
    active_scope_stack = []
    active_scope = None
    active_scope_id = -1
    active_scope_kernels = {}
    if not hasattr(tape, "scopes"):
        ignore_graph_scopes = True
    if not ignore_graph_scopes and len(scopes) > 0:
        active_scope = scopes[0]
        active_scope_id = 0
    for launch_id, launch in enumerate(launches):
        if active_scope is not None:
            if launch_id == active_scope[0]:
                if active_scope[1] is None:
                    # end previous scope
                    indent_level -= 1
                    visitor.emit_scope_end(indent_level)
                    active_scope_stack = active_scope_stack[:-1]
                else:
                    # begin new scope
                    active_scope_stack.append(f"scope{active_scope_id}")
                    visitor.emit_scope_begin(active_scope_id, active_scope[1], active_scope[2], indent_level)
                    indent_level += 1
            # check if we are in the next scope now
            while (
                not ignore_graph_scopes
                and active_scope_id < len(scopes) - 1
                and launch_id == scopes[active_scope_id + 1][0]
            ):
                active_scope_id += 1
                active_scope = scopes[active_scope_id]
                active_scope_kernels = {}
                if active_scope[1] is None:
                    # end previous scope
                    indent_level -= 1
                    visitor.emit_scope_end(indent_level)
                    active_scope_stack = active_scope_stack[:-1]
                else:
                    # begin new scope
                    active_scope_stack.append(f"scope{active_scope_id}")
                    visitor.emit_scope_begin(active_scope_id, active_scope[1], active_scope[2], indent_level)
                    indent_level += 1

        kernel = launch.kernel
        launch_data = {
            "id": launch_id,
            "dim": launch.dim,
            "inputs": launch.inputs,
            "outputs": launch.outputs,
            "stack_trace": "",
            "kernel_launch_count": kernel_launch_count[kernel.key],
        }
        launch_data.update(launch.metadata)

        rendered = not hide_readonly_arrays or ignore_graph_scopes or kernel.key not in active_scope_kernels
        if rendered:
            active_scope_kernels[kernel.key] = launch_id

        if not ignore_graph_scopes and hide_readonly_arrays:
            k_id = f"kernel{active_scope_kernels[kernel.key]}"
        else:
            k_id = f"kernel{launch_id}"

        visitor.emit_kernel_launch_node(kernel, k_id, launch_data, rendered, indent_level)

        # loop over inputs and outputs to add them to the graph
        input_arrays = []
        for id, x in enumerate(launch.inputs):
            name = kernel.adj.args[id].label
            if isinstance(x, wp.array):
                if x.ptr is None:
                    continue
                # if x.ptr in array_to_launch and len(array_to_launch[x.ptr]) > 1:
                #     launch_arg_i = array_to_launch[x.ptr]
                #     actual_input = launch_to_array[launch_arg_i][0]
                #     visitor.emit_edge_array_kernel(actual_input.ptr, k_id, id, indent_level)
                if not hide_readonly_arrays or x.ptr in computed_nodes or x.ptr in input_output_ptr:
                    xptr = x.ptr
                    if xptr in array_repeated:
                        xptr = array_repeated[xptr]
                    else:
                        add_array_node(x, name, active_scope_stack)
                    # input_arrays.append(x.ptr)
                    visitor.emit_edge_array_kernel(xptr, k_id, id, indent_level)
            elif isinstance(x, wp.codegen.StructInstance):
                for varname, var in get_struct_vars(x).items():
                    if isinstance(var, wp.array):
                        if not hide_readonly_arrays or var.ptr in computed_nodes or var.ptr in input_output_ptr:
                            add_array_node(var, f"{name}.{varname}", active_scope_stack)
                            input_arrays.append(var.ptr)
                            xptr = var.ptr
                            if xptr in array_repeated:
                                xptr = array_repeated[xptr]
                            visitor.emit_edge_array_kernel(xptr, k_id, id, indent_level)
        output_arrays = []
        for id, x in enumerate(launch.outputs):
            name = kernel.adj.args[id + len(launch.inputs)].label
            if isinstance(x, wp.array) and x.ptr is not None:
                add_array_node(x, name, active_scope_stack)
                output_arrays.append(x.ptr)
                computed_nodes.add(x.ptr)
                visitor.emit_edge_kernel_array(k_id, id, x.ptr, indent_level)
            elif isinstance(x, wp.codegen.StructInstance):
                for varname, var in get_struct_vars(x).items():
                    if isinstance(var, wp.array):
                        add_array_node(var, f"{name}.{varname}", active_scope_stack)
                        output_arrays.append(var.ptr)
                        computed_nodes.add(var.ptr)
                        visitor.emit_edge_kernel_array(k_id, id, var.ptr, indent_level)

        for output_x in output_arrays:
            # track how many kernels modify each array
            manipulated_nodes[output_x].append(kernel.key)

        kernel_launch_count[kernel.key] += 1

    # close any open scopes
    for _ in range(len(active_scope_stack)):
        indent_level -= 1
        visitor.emit_scope_end(indent_level)

    # add additional edges between arrays
    for src, dst in wrap_around_connections:
        if src == dst or src not in inserted_arrays or dst not in inserted_arrays:
            continue
        visitor.emit_edge_array_array(inserted_arrays[src], inserted_arrays[dst], indent_level)


def visualize_tape_graphviz(
    tape: Tape,
    filename: str,
    simplify_graph=True,
    hide_readonly_arrays=False,
    array_labels: Dict[wp.array, str] = None,
    choose_longest_node_name: bool = True,
    ignore_graph_scopes: bool = False,
    track_inputs: List[wp.array] = None,
    track_outputs: List[wp.array] = None,
    track_input_names: List[str] = None,
    track_output_names: List[str] = None,
    graph_direction: str = "LR",
):
    if track_output_names is None:
        track_output_names = []
    if track_input_names is None:
        track_input_names = []
    if track_outputs is None:
        track_outputs = []
    if track_inputs is None:
        track_inputs = []
    if array_labels is None:
        array_labels = {}
    visitor = GraphvizTapeVisitor()
    visit_tape(
        tape,
        visitor,
        simplify_graph,
        hide_readonly_arrays,
        array_labels,
        choose_longest_node_name,
        ignore_graph_scopes,
        track_inputs,
        track_outputs,
        track_input_names,
        track_output_names,
    )

    chart = "\n".join(visitor.graphviz_lines)
    code = f"""digraph " " {{
    graph [fontname="Helvetica,Arial,sans-serif",tooltip=" "];
    node [style=rounded,shape=plaintext,fontname="Helvetica,Arial,sans-serif", margin="0.05,0.02", width=0, height=0, tooltip=" "];
    edge [fontname="Helvetica,Arial,sans-serif",tooltip=" "];
    rankdir={graph_direction};

{chart}
}}
"""

    if filename is not None:
        with open(filename, "w") as f:
            f.write(code)

    return code
