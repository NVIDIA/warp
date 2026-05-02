// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// METH_FASTCALL wrappers for performance-critical native functions.
// Embedded in warp.dll and loaded as a Python extension module via importlib.
#include "warp.h"

#include <Python.h>

// Cached at PyInit time instead of using PyExc_TypeError directly. PyExc_TypeError
// is a data import; data symbols are eagerly resolved by the OS loader on every
// platform, which prevents warp.dll/so/dylib from being loaded by non-Python C++
// hosts. Looking it up via PyImport_AddModule + PyObject_GetAttrString uses only
// function imports, which are lazily bound and never resolved unless Python is
// actually present in the process.
static PyObject* g_python_type_error = nullptr;

static PyObject* fastcall_float_to_half_bits(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 1) {
        PyErr_SetString(g_python_type_error, "float_to_half_bits() takes exactly 1 argument");
        return nullptr;
    }

    double value = PyFloat_AsDouble(args[0]);
    if (value == -1.0 && PyErr_Occurred())
        return nullptr;

    uint16_t bits = wp_float_to_half_bits(static_cast<float>(value));
    return PyLong_FromUnsignedLong(bits);
}

static PyObject* fastcall_half_bits_to_float(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 1) {
        PyErr_SetString(g_python_type_error, "half_bits_to_float() takes exactly 1 argument");
        return nullptr;
    }

    unsigned long bits = PyLong_AsUnsignedLongMask(args[0]);
    if (PyErr_Occurred())
        return nullptr;

    float value = wp_half_bits_to_float(static_cast<uint16_t>(bits));
    return PyFloat_FromDouble(static_cast<double>(value));
}

// Method names use the wp_ prefix to match the ctypes bindings on runtime.core.
// At init time, these override the ctypes versions so all call sites use the
// faster path transparently. If the module fails to load, the ctypes versions
// remain in place as a fallback.
static PyMethodDef fastcall_methods[] = {
    { "wp_float_to_half_bits", reinterpret_cast<PyCFunction>(fastcall_float_to_half_bits), METH_FASTCALL,
      "Convert a float to float16 bit pattern" },
    { "wp_half_bits_to_float", reinterpret_cast<PyCFunction>(fastcall_half_bits_to_float), METH_FASTCALL,
      "Convert a float16 bit pattern to float" },
    { nullptr, nullptr, 0, nullptr },
};

static PyModuleDef fastcall_module = {
    PyModuleDef_HEAD_INIT, "_warp_fastcall", "Warp METH_FASTCALL native bindings", -1, fastcall_methods,
};

extern "C" WP_API PyObject* PyInit__warp_fastcall()
{
    PyObject* builtins = PyImport_AddModule("builtins");
    if (!builtins)
        return nullptr;

    g_python_type_error = PyObject_GetAttrString(builtins, "TypeError");
    if (!g_python_type_error)
        return nullptr;

    return PyModule_Create(&fastcall_module);
}
