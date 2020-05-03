#ifndef THINNING_LIBRARY_H
#define THINNING_LIBRARY_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

static
PyObject *zhang_and_suen_binary_thinning(PyObject *_, PyObject *args);

#endif //THINNING_LIBRARY_H
