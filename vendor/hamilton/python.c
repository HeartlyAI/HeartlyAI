#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "qrsdet.h"
#include <stdbool.h>

#define REFINE_MAX_LOOKBEHIND 8
#define SIGN(x) ((x > 0) - (x < 0))

int QRSDet(int datum, int init);

void refine_peaks(int16_t *data, size_t data_len, ssize_t *peaks, size_t peak_count) {
	int absmax = 0;
	int maxsign = 0;
	for (size_t i = 0; i < data_len; i++) {
		int absval = abs(data[i]);
		if (absval > absmax) {
			absmax = absval;
			maxsign = SIGN(data[i]);
		}
	}


	for (size_t i = 0; i < peak_count; i++) {
		// perform lookaround for local extremum
		ssize_t *peak = &peaks[i];
		if (*peak == -1) continue;

		size_t lb = *peak - REFINE_MAX_LOOKBEHIND;
		lb = lb > 1 ? lb : 1;
		size_t j = *peak;
		while (j >= lb) {
			int lgrad = data[j] - data[j-1];
			int rgrad = data[j + 1] - data[j];
			if (SIGN(lgrad) != SIGN(rgrad) && maxsign == SIGN(data[j])) {
				*peak = j;
				break;
			}
			j--;
		}
	}
}

ssize_t *hamilton_detect_r_peaks(int16_t *data, ssize_t data_len, size_t *qrs_count) {
	size_t peaks_len = 16;
	ssize_t *peaks = (ssize_t*)malloc(peaks_len*sizeof(ssize_t));
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
				peaks = (ssize_t*)realloc(peaks, peaks_len*sizeof(ssize_t));
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
	refine_peaks(data, data_len, peaks, next_peak_index);
	return peaks;
}

void find_local_extrema(int16_t *data, size_t data_len, ssize_t *peaks, size_t peaks_len, ssize_t *extrema, int dir) {
	for (size_t i = 0; i < peaks_len; i++) {
		ssize_t peak = peaks[i];
		if (peak == -1) {
			extrema[i] = -1;
			continue;
		}
		int prev_gradient_sign = 0;
		ssize_t found_extrema = -1;
		size_t bound = dir < 0 ? 0 : data_len - 1;
		for (size_t j = peak + dir; j != bound ; j += dir) {
			int gradient = data[j] - data[j - dir];
 			int gradient_sign = SIGN(gradient);
			if (gradient_sign != prev_gradient_sign && prev_gradient_sign != 0) {
				found_extrema = j - dir;
				break;
			}
			prev_gradient_sign = gradient_sign;
		}
		extrema[i] = found_extrema;
	}
}

bool from_args_1d_int16_numpy_array(PyObject *args, npy_int16 **data, npy_intp *data_len) {
	PyObject *input_array;

	if (!PyArg_ParseTuple(args, "O", &input_array) || !PyArray_Check(input_array)) {
		PyErr_SetString(PyExc_TypeError, "Expected a single numpy array as argument.");
		return false;
	}

	PyArrayObject *np_array = (PyArrayObject*)PyArray_FROM_OTF(input_array, NPY_INT16, NPY_ARRAY_IN_ARRAY);
	if (!np_array) {
		PyErr_SetString(PyExc_ValueError, "The numpy array must be of dtype int16.");
		return false;
	}

	if (PyArray_NDIM(np_array) != 1) {
		PyErr_SetString(PyExc_ValueError, "The numpy array must be 1d.");
		return false;
	}

	*data = (npy_int16*)PyArray_DATA(np_array);
	*data_len = PyArray_SIZE(np_array);
	return true;
}

PyObject* indices_to_PyList(ssize_t *indices, size_t indices_len) {
	PyObject *py_list = PyList_New(indices_len);
	if (!py_list) return NULL;
	for (size_t i = 0; i < indices_len; i++) {
		PyObject *p = PyLong_FromLongLong(indices[i]);
		PyList_SET_ITEM(py_list, i, p);
	}
	return py_list;
}

static PyObject* detect_r_peaks(PyObject *self, PyObject *args) {
	npy_int16 *data;
	npy_intp data_len;
	if (!from_args_1d_int16_numpy_array(args, &data, &data_len)) return NULL;

	size_t peak_count;
	ssize_t *peaks = hamilton_detect_r_peaks(data, data_len, &peak_count);

	PyObject *py_peaks = indices_to_PyList(peaks, peak_count);

	free(peaks);
	return py_peaks;
}

static PyObject *detect_qrs(PyObject *self, PyObject *args) {
	npy_int16 *data;
	npy_intp data_len;
	if (!from_args_1d_int16_numpy_array(args, &data, &data_len)) return NULL;

	size_t peak_count;
	ssize_t *r = hamilton_detect_r_peaks(data, data_len, &peak_count);
	ssize_t *q = (ssize_t*)malloc(sizeof(ssize_t)*peak_count);
	ssize_t *s = (ssize_t*)malloc(sizeof(ssize_t)*peak_count);
	find_local_extrema(data, data_len, r, peak_count, q, -1);
	find_local_extrema(data, data_len, r, peak_count, s, 1);
	
	PyObject *py_list = PyList_New(3);
	PyObject *py_q = indices_to_PyList(q, peak_count);
	PyObject *py_r = indices_to_PyList(r, peak_count);
	PyObject *py_s = indices_to_PyList(s, peak_count);
	PyList_SET_ITEM(py_list, 0, py_q);
	PyList_SET_ITEM(py_list, 1, py_r);
	PyList_SET_ITEM(py_list, 2, py_s);

	free(s);
	free(q);
	free(r);

	return py_list;
}


static PyMethodDef methods[] = {
    {"detect_r_peaks", detect_r_peaks, METH_VARARGS, "This method detects all R peaks within a numpy array."},
    {"detect_qrs", detect_qrs, METH_VARARGS, "This method detects all QRS complexes within a numpy array."},
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
