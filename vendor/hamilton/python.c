#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "qrsdet.h"
#include <stdbool.h>

#define REFINE_MAX_LOOKBEHIND 8

int QRSDet(int datum, int init);

size_t *hamilton_detect_qrs(int16_t *data, size_t data_len, size_t *qrs_count) {
	size_t peaks_len = 16;
	size_t *peaks = (size_t*)malloc(peaks_len*sizeof(size_t));
	size_t next_peak_index = 0;
	if (!peaks) return NULL;

	// prerun - adapts to the graph
	for (npy_intp i = 0; i < data_len; i++) {
		QRSDet(data[i], i == 0);
	}

	for (npy_intp i = 0; i < data_len; i++) {
		int delay = QRSDet(data[i], 0);
		if (delay) {
			if (next_peak_index >= peaks_len) {
				peaks_len *= 2;
				peaks = (size_t*)realloc(peaks, peaks_len*sizeof(size_t));
				if (!peaks) return NULL;
			}
			int data_index = i - delay;
			// safeguard! :)
			data_index = data_index < data_len ? data_index : data_len;
			peaks[next_peak_index] = data_index;
			
			next_peak_index++;
		}
	}

	*qrs_count = next_peak_index;
	return peaks;
}

void refine_peaks(int16_t *data, size_t data_len, size_t *peaks, size_t peak_count) {
	for (size_t i = 0; i < peak_count; i++) {
		// perform lookaround for local maxima
		size_t *peak = &peaks[i];
		size_t best_index = *peak;
		int16_t best_value = data[best_index];
		size_t lb = best_index - REFINE_MAX_LOOKBEHIND;
		lb = lb > 0 ? lb : 0;
		size_t j = best_index - 1;
		while (j >= lb) {
			if (data[j] > best_value) {
				*peak = j;
				best_index = j;
				best_value = data[best_index];
			}
			j--;
		}
	}
}

static PyObject* detect_r_peaks(PyObject *self, PyObject *args) {
	PyObject *input_array;

	if (!PyArg_ParseTuple(args, "O", &input_array) || !PyArray_Check(input_array)) {
		PyErr_SetString(PyExc_TypeError, "Expected a single numpy array as argument.");
		return NULL;
	}

	PyArrayObject *np_array = (PyArrayObject*)PyArray_FROM_OTF(input_array, NPY_INT16, NPY_ARRAY_IN_ARRAY);
	if (!np_array) {
		PyErr_SetString(PyExc_ValueError, "The numpy array must be of dtype int16.");
		return NULL;
	}

	if (PyArray_NDIM(np_array) != 1) {
		PyErr_SetString(PyExc_ValueError, "The numpy array must be 1d.");
		return NULL;
	}

	npy_int16 *data = (npy_int16*)PyArray_DATA(np_array);
	npy_intp data_len = PyArray_SIZE(np_array);

	size_t peak_count;
	size_t *peaks = hamilton_detect_qrs(data, data_len, &peak_count);
	refine_peaks(data, data_len, peaks, peak_count);

	PyObject *py_peaks = PyList_New(peak_count);
	if (!py_peaks) return NULL;
	for (size_t i = 0; i < peak_count; i++) {
		PyObject *p = PyLong_FromUnsignedLong(peaks[i]);
		PyList_SET_ITEM(py_peaks, i, p);
	}

	free(peaks);
	
	return py_peaks;
}

static PyMethodDef methods[] = {
    {"detect_r_peaks", detect_r_peaks, METH_VARARGS, "This method detects all QRS complexes within a numpy array."},
    {NULL, NULL, 0, NULL} // Sentinel
};


static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "hamilton",
    "Module for hamilton pre-processing.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_hamilton(void) {
	import_array();
    return PyModule_Create(&module);
}
