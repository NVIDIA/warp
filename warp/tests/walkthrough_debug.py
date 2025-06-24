# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

####################################################################################################
#
# This file demonstrates step-through debugging support of the C++ code generated for a Warp kernel
# running on the CPU.
#
# This is not a unit test; it should be run interactively.
#
# For a fully integrated experience use Visual Studio Code and install the "Python C++ Debugger"
# and "CodeLLDB" extensions. Add the following configurations to your .vscode/launch.json file:
#

"""
{
    "name": "Warp Debugger",
    "type": "pythoncpp",
    "request": "launch",
    "pythonLaunchName": "Python: Current File",
    "cppAttachName": "(lldb) Attach",
},
{
    "name": "Python: Current File",
    "type": "debugpy",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal",
    "stopOnEntry": false,
},
{
    "name": "(lldb) Attach",
    "type": "lldb",
    "request": "attach",
},
"""

#
# Then run this .py file using the "Warp Debugger" configuration.
#
# Check out the following resources for more information about launch configurations and
# troubleshooting common VSCode debugger integration issues:
# • https://vscode-docs.readthedocs.io/en/stable/editor/debugging/#launch-configurations
# • https://code.visualstudio.com/docs/cpp/cpp-debug#_debugging
#
####################################################################################################

import warp as wp

# The init() function prints the directory of the kernel cache which contains the .cpp files
# generated from Warp kernels. You can put breakpoints in these C++ files through Visual Studio Code,
# but it's generally more convenient to use wp.breakpoint(). See the example below.
wp.init()

# Enable kernels to be compiled with debug info and disable optimizations
wp.config.mode = "debug"

# Make sure Warp was built with `build_lib.py --mode=debug`
assert wp.context.runtime.core.wp_is_debug_enabled(), "Warp must be built in debug mode to enable debugging kernels"


@wp.kernel
def example_breakpoint(n: int):
    a = int(0)

    for _i in range(0, n):
        if a == 5:
            # Your debugger should halt at the C++ code corresponding with the next line,
            # namely a call to the __debugbreak() intrinsic function.
            wp.breakpoint()

            break

        a += 1

    wp.expect_eq(a, 5)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    wp.launch(example_breakpoint, dim=1, inputs=[10], device="cpu")
