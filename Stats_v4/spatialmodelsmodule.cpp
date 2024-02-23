#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <vector>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <iostream>

#include <python3.6/Python.h>
#include <python3.6/numpy/arrayobject.h>

#include "spatialmodelsmodule.hpp"


static PyObject* radiation_model(PyObject *self, PyObject *args, PyObject *kwargs) {

    // Create PyObjects into which data is parsed.
    PyObject* input_list = NULL;
    PyObject* output_list = NULL;
    PyObject* distance_matrix = NULL;
    BOOL normalisation_int = 0;

    // Create keywords list.
    static char *kwlist[] = {(char*)"", (char*)"", (char*)"normalisation", (char*)"output_list", NULL};

    // Parse the arguments into C++ variables and check for success.
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|pO", kwlist, &input_list, &distance_matrix, &normalisation_int, &output_list)) {
        // Returning NULL tells Python there's been an error.
        return NULL;
    }
    if (input_list == NULL || distance_matrix == NULL) {
      return NULL;
    }

    bool normalisation = normalisation_int;

    // Check objects are arrays.
    if (!PyArray_Check(input_list)) {
        PyErr_SetString(PyExc_TypeError, "Input input_list is not a NumPy array.");
        return NULL;
    }
    if (!PyArray_Check(distance_matrix)) {
        PyErr_SetString(PyExc_TypeError, "Input distance_matrix is not a NumPy array.");
        return NULL;
    }

    // Check the data type of the arrays.
    if (PyArray_TYPE((PyArrayObject*)input_list) != NPY_DOUBLE) {
        PyErr_WarnEx(PyExc_Warning, "Data type of input_list is not float64.", 1);
    }
    if (PyArray_TYPE((PyArrayObject*)distance_matrix) != NPY_DOUBLE) {
        PyErr_WarnEx(PyExc_Warning, "Data type of distance_matrix is not float64.", 1);
    }

    // Check that the arrays are C contiguous.
    if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)input_list)) {
        PyErr_SetString(PyExc_ValueError, "Array input_list is not C contiguous.");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)distance_matrix)) {
        PyErr_SetString(PyExc_ValueError, "Array distance_matrix is not C contiguous.");
        return NULL;
    }

    // Check that the dimensions of the input arrays are correct.
    if (PyArray_NDIM((PyArrayObject*)input_list) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array input_list must be 1-dimensional.");
        return NULL;
    }
    if (PyArray_NDIM((PyArrayObject*)distance_matrix) != 2) {
        PyErr_SetString(PyExc_ValueError, "Array distance_matrix must be 2-dimensional.");
        return NULL;
    }

    const unsigned long distance_size = PyArray_Size(distance_matrix);
    const unsigned long input_size = PyArray_Size(input_list);

    // Check that the size of distance_matrix is equal to the square of the size of input_list.
    if (input_size * input_size != distance_size) {
        PyErr_SetString(PyExc_ValueError, "The distance_matrix array must be a square matrix with rows of size equal to the size of input_list.");
        return NULL;
    }

    unsigned long int output_size;
    double * output_list_ptr;

    if (output_list != NULL) {
        if (PyArray_TYPE((PyArrayObject*)output_list) != NPY_DOUBLE) {
            PyErr_WarnEx(PyExc_Warning, "Data type of output_list is not float64.", 1);
        }
        if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)output_list)) {
            PyErr_SetString(PyExc_ValueError, "Array output_list is not C continguous.");
            return NULL;
        }
        if (PyArray_NDIM((PyArrayObject*)output_list) != 1) {
            PyErr_SetString(PyExc_ValueError, "Array output_list must be 1-dimensional.");
            return NULL;
        }

        if(!normalisation){
            PyErr_WarnEx(PyExc_Warning, "The radiation model with separate source and target lists is always normalised, so ignoring normalisation=False.", 1);
        }

        output_size = PyArray_Size(output_list);

        output_list_ptr = (double *)PyArray_GETPTR1((PyArrayObject*)output_list, 0);

        if (input_size != output_size) {
            PyErr_SetString(PyExc_ValueError, "The input_list array must be the same size as the output_list array.");
            return NULL;
        }
    }


    // Create pointers to access array data.
    double* input_list_ptr = (double *)PyArray_GETPTR1((PyArrayObject*)input_list, 0);
    double* distance_matrix_ptr = (double *)PyArray_GETPTR2((PyArrayObject*)distance_matrix, 0, 0);

    // Create a new array to store the output values and a pointer to access the array.
    npy_intp* distance_matrix_dims = PyArray_DIMS((PyArrayObject*)distance_matrix);
    PyObject* values = (PyObject*)PyArray_SimpleNew(2, distance_matrix_dims, NPY_DOUBLE);
    double* values_ptr = (double *)PyArray_GETPTR2((PyArrayObject*)values, 0, 0);

    // Create a C array for the intervening_matrix (s_ij in Simini et al.).
    std::vector<double> intervening_matrix(distance_size);

    // Calculate the intervening_matrix.
    for (unsigned long i = 0; i < input_size; i++) {
        for (unsigned long j = 0; j < input_size; j++) {
            double distance = distance_matrix_ptr[i * input_size + j];
            double population = 0;

            for (unsigned long k = 0; k < input_size; k++) {
                if (distance_matrix_ptr[i * input_size + k] < distance) {
                    population += input_list_ptr[k];
                }
            }

            intervening_matrix[i * input_size + j] = population;
        }
    }

    std::vector<double> normalisation_value(input_size);
    double input_list_sum = 0;

    // Implement the definition of the model.
    if (output_list==NULL) {
        // If normalisation, change normalisation_value away from 1.
        if (normalisation) {
            for (unsigned long i = 0; i < input_size; i++) {
                input_list_sum += input_list_ptr[i];
            }

            for (unsigned long i = 0; i < input_size; i++) {
                normalisation_value[i] = 1 - input_list_ptr[i] / input_list_sum;
            }
        } else {
            for (unsigned long i = 0; i < input_size; i++) {
                normalisation_value[i] = 1;
            }
        }

        for (unsigned long i = 0; i < input_size; i++) {
            for (unsigned long j = 0; j < input_size; j++) {
                values_ptr[i * input_size + j] = input_list_ptr[i] * (input_list_ptr[i] * input_list_ptr[j])
                    / ((input_list_ptr[i] + intervening_matrix[i * input_size + j])
                    * (input_list_ptr[i] + input_list_ptr[j] + intervening_matrix[i * input_size + j])
                    * normalisation_value[i]);
            }
        }
    } else {
        for (unsigned long i = 0; i < input_size; i++) {
            input_list_sum += input_list_ptr[i];
        }
        for (unsigned long i = 0; i < input_size; i++) {
            for(unsigned long j = 0; j < input_size; j++) {
                double normalisation = input_list_sum/(input_list_sum - input_list_ptr[i]);
                double quotient = ((output_list_ptr[i] * input_list_ptr[i] * input_list_ptr[j]) /
                    ((input_list_ptr[i] + intervening_matrix[i * input_size + j]) *
                    (input_list_ptr[i] + intervening_matrix[i * input_size + j] + input_list_ptr[j])));
                values_ptr[i * input_size + j] = normalisation * quotient;
            }
        }
    }

    return values;

}

// Reminder - Complete docstring and add in exp for constraints.
static PyObject* gravity_model(PyObject *self, PyObject *args, PyObject *kwargs) {

    // Create PyObjects into which data is parsed.
    PyObject* input_list = NULL;
    PyObject* output_list = NULL;
    PyObject* distance_matrix = NULL;
    PyObject* deterrence_func = NULL;
    double input_exp = 1;
    double output_exp = 1;
    BOOL production_constrained_int = 0;
    BOOL doubly_constrained_int = 0;
    double threshold = -1;

    // Create keywords list.
    static char* kwlist[] = {(char*)"", (char*)"", (char*)"", (char*)"",
                             (char*)"input_exp", (char*)"output_exp",
                             (char*)"production_constrained", (char*)"doubly_constrained", (char*)"threshold", NULL};

    // Parse the arguments into C++ variables and check for success.
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|ddppd", kwlist,
                                     &input_list, &output_list, &distance_matrix,
                                     &deterrence_func, &input_exp, &output_exp,
                                     &production_constrained_int, &doubly_constrained_int, &threshold)) {
        // Returning NULL tells Python there's been an error.
        return NULL;
    }
    if (input_list == NULL || output_list == NULL || distance_matrix == NULL || deterrence_func == NULL) {
      return NULL;
    }

    bool production_constrained = production_constrained_int;
    bool doubly_constrained = doubly_constrained_int;


    // Check the data type of the arrays.
    if (PyArray_TYPE((PyArrayObject*)input_list) != NPY_DOUBLE) {
        PyErr_WarnEx(PyExc_Warning, "Data type of input_list is not float64.", 1);
    }
    if (PyArray_TYPE((PyArrayObject*)output_list) != NPY_DOUBLE) {
        PyErr_WarnEx(PyExc_Warning, "Data type of output_list is not float64.", 1);
    }
    if (PyArray_TYPE((PyArrayObject*)distance_matrix) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "Data type of distance_matrix is not float64.");
        return NULL;
    }

    // Check that the arrays are C contiguous.
    if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)input_list)) {
        PyErr_SetString(PyExc_ValueError, "Array input_list is not C continguous.");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)output_list)) {
        PyErr_SetString(PyExc_ValueError, "Array output_list is not C continguous.");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)distance_matrix)) {
        PyErr_SetString(PyExc_ValueError, "Array distance_matrix is not C continguous.");
        return NULL;
    }

    // Check that the dimensions of the input arrays are correct.
    if (PyArray_NDIM((PyArrayObject*)input_list) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array input_list must be 1-dimensional.");
        return NULL;
    }
    if (PyArray_NDIM((PyArrayObject*)output_list) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array output_list must be 1-dimensional.");
        return NULL;
    }
    if (PyArray_NDIM((PyArrayObject*)distance_matrix) != 2) {
        PyErr_SetString(PyExc_ValueError, "Array distance_matrix must be 2-dimensional.");
        return NULL;
    }

    const unsigned long int distance_size = PyArray_Size(distance_matrix);
    const unsigned long int input_size = PyArray_Size(input_list);
    const unsigned long int output_size = PyArray_Size(output_list);

    // Check that the size of distance_matrix is equal to the square of the size of input_list.
    if (input_size != output_size) {
        PyErr_SetString(PyExc_ValueError, "The input_list array must be the same size as the output_list array.");
        return NULL;
    }

    // Check that the size of distance_matrix is equal to the square of the size of input_list.
    if (input_size * output_size != distance_size) {
        PyErr_SetString(PyExc_ValueError, "The distance_matrix array must be a square matrix with rows of size equal to the size of input_list.");
        return NULL;
    }

    // Check that deterrence_func is a callable object.
    if (!(PyCallable_Check(deterrence_func))) {
        PyErr_SetString(PyExc_TypeError, "The deterrence_func object is not callable.");
        return NULL;
    }

    // Check whether a threshold has been set.
    if (threshold != -1 && !doubly_constrained) {
        PyErr_WarnEx(PyExc_Warning, "A convergence threshold has been provided but the doubly constrained model is not being used.", 1);
    }

    // Create pointers to access array data.
    double* input_list_ptr = (double *)PyArray_GETPTR1((PyArrayObject*)input_list, 0);
    double* output_list_ptr = (double *)PyArray_GETPTR1((PyArrayObject*)output_list, 0);
    double* distance_matrix_ptr = (double *)PyArray_GETPTR2((PyArrayObject*)distance_matrix, 0, 0);

    // Create a new array to store the output values and a pointer to access the array.
    npy_intp* distance_matrix_dims = PyArray_DIMS((PyArrayObject*)distance_matrix);
    PyObject* values = (PyObject*)PyArray_SimpleNew(2, distance_matrix_dims, NPY_DOUBLE);
    double* values_ptr = (double *)PyArray_GETPTR2((PyArrayObject*)values, 0, 0);

    std::vector<double> deterrence_matrix(distance_size);

    // Calculate the deterrence function.
    for (unsigned long i = 0; i < input_size; i++) {
        for (unsigned long j = 0; j < input_size; j++) {
            // Call deterrence_func.
            PyObject* deterrence_args = Py_BuildValue("(d)", distance_matrix_ptr[i * input_size + j]);
            PyObject* deterrence_value = PyObject_CallObject(deterrence_func, deterrence_args);
            if (deterrence_value == NULL) {
                if (PyErr_Occurred() == NULL) {
                    PyErr_SetString(PyExc_TypeError, "There was an error calling deterrence_func. Ensure it has the correct signiature.");
                }
                return NULL;
            }
            deterrence_matrix[i * input_size + j] = PyFloat_AsDouble(deterrence_value);
            Py_DECREF(deterrence_args);
            Py_DECREF(deterrence_value);
        }
    }

    // Implement input_exp.
    if (input_exp != 1) {
        for (unsigned long i = 0; i < input_size; i++) {
            input_list_ptr[i] = std::pow(input_list_ptr[i], input_exp);
        }
    }

    // Implement output_exp.
    if (output_exp != 1) {
        for (unsigned long i = 0; i < input_size; i++) {
            output_list_ptr[i] = std::pow(output_list_ptr[i], output_exp);
        }
    }


    // Implement the definition of the gravity model.
    for (unsigned long i = 0; i < input_size; i++) {
        for (unsigned long j = 0; j < input_size; j++) {
            // Calculate the output value.
            values_ptr[i * input_size + j] = output_list_ptr[i] * input_list_ptr[j] * deterrence_matrix[i * input_size + j];
        }
    }

    // Implement doubly constrained model.
    if (doubly_constrained) {

        // Check that production_constrained has not also been set.
        if (production_constrained) {
            PyErr_WarnEx(PyExc_Warning, "Inputs doubly_constrained and production_constrained are both true. Implementing doubly_constrained model.", 1);
        }

        // Check that a threshold has been provided.
        if (threshold == -1) {
            PyErr_SetString(PyExc_ValueError, "A convergence threshold must be provided for the doubly constrained model.");
            return NULL;
        }

        // Check that the threshold is positive.
        if (threshold < 0) {
            PyErr_SetString(PyExc_ValueError, "The convergence threshold must be positive.");
            return NULL;
        }

        // Check that the sum of I_j is equal to the sum of O_i. This is necessary for a solution to exist.
        double input_check = 0;
        double output_check = 0;

        for (unsigned long i = 0; i < input_size; i++) {
            input_check += input_list_ptr[i];
            output_check += output_list_ptr[i];
        }

        if (input_check != output_check){
            PyErr_SetString(PyExc_ValueError, "For doubly constrained gravity model, the sum of values in input_list must equal the sum of values in output_list.");
            return NULL;
        }


        // Create normalisation vectors,
        std::vector<double> normalisation_a(input_size);
        std::vector<double> normalisation_b(input_size);

        // Create vector for the new values of F_{ij}.
        std::vector<double> new_values(distance_size);

        // Fill normalisation vectors.
        for (unsigned long i = 0; i < input_size; i++) {
            normalisation_a[i] = threshold;
            normalisation_b[i] = threshold;
        }

        // Set the initial error and previous_error.
        double error = threshold + 1;
        double previous_error = 0;

        while (error > threshold) {

            // Calculate B_j.
            for (unsigned long i = 0; i < input_size; i++) {
                double sum = 0;
                for (unsigned long j = 0; j < input_size; j++) {
                    sum += normalisation_a[j] * output_list_ptr[j] * deterrence_matrix[j * input_size + i];
                }
                normalisation_b[i] = 1/sum;
            }

            // Calculate A_i.
            for (unsigned long i = 0; i < input_size; i++) {
                double sum = 0;
                for (unsigned long j = 0; j < input_size; j++) {
                    sum += normalisation_b[j] * input_list_ptr[j] * deterrence_matrix[i * input_size + j];
                }
                normalisation_a[i] = 1/sum;
            }

            error = 0;

            for (unsigned long i = 0; i < input_size; i++) {
                for (unsigned long j = 0; j < input_size; j++) {
                    // Calculate the new output value.
                    new_values[i * input_size + j] = values_ptr[i * input_size + j] * normalisation_a[i] * normalisation_b[j];
                }
            }

            // Calculate new error.
            for (unsigned long i = 0; i < input_size; i++) {
                double input_sum = 0;
                double output_sum = 0;
                for (unsigned long j = 0; j < input_size; j++) {
                    output_sum += new_values[i * input_size + j];
                    input_sum += new_values[j * input_size + i];
                }
                error += std::abs(input_list_ptr[i] - input_sum) + std::abs(output_list_ptr[i] - output_sum);
            }

            // Check that the result is meaningful.
            if((std::isnan(error) || std::isinf(error)) || error == previous_error) {
                char error_string[100];
                snprintf(error_string, 100, "Constraints failed to converge. Total error reached: %e.", error);
                PyErr_WarnEx(PyExc_RuntimeWarning, error_string, 1);
                break;
            }

            // Store previous_error
            previous_error = error;

        }

        // Store calculated values.
        for (unsigned long i = 0; i < input_size; i++) {
            for (unsigned long j = 0; j < input_size; j++) {
                values_ptr[i * input_size + j] = new_values[i * input_size + j];
            }
        }

    } else if (production_constrained) {  // Implement production constrained model.

        std::vector<double> normalisation(input_size);
        for (unsigned long i = 0; i < input_size; i++) {
            double sum = 0;
            for (unsigned long j=0; j < input_size; j++) {
                sum += input_list_ptr[j] * deterrence_matrix[i * input_size + j];
            }
            normalisation[i] = 1/sum;
        }
        for (unsigned long i = 0; i < input_size; i++) {
            for (unsigned long j = 0; j < input_size; j++) {
                // Calculate the output value.
                values_ptr[i * input_size + j] *= normalisation[i];
            }
        }
    }

    return values;
}


// Define a list of methods to be accessed by Python.
static PyMethodDef methods[] = {
    {"radiation_model", (PyCFunction)radiation_model, METH_VARARGS | METH_KEYWORDS, radiation_model__doc__},
    {"gravity_model", (PyCFunction)gravity_model, METH_VARARGS | METH_KEYWORDS, gravity_model__doc__},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

// Define the module as a whole.
PyDoc_STRVAR(
    module__doc__,
    "Implements the calculation of various spatial distribution models in C++.\n"
);
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "spatialmodels",
    module__doc__,
    -1,
    methods
 };

// Define the initialisation function for the module.
PyMODINIT_FUNC PyInit_spatialmodels(void){
    import_array();
    return PyModule_Create(&module);
}
