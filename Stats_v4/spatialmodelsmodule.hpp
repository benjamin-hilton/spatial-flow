#ifndef SPATIALMODELSMODULE_H
#define SPATIALMODELSMODULE_H

typedef int BOOL;

PyDoc_STRVAR(
    radiation_model__doc__,
    "radiation_model(population_list, distance_matrix, normalisation=False)\n"
    "--\n"
    "\n"
    "Calculates radiation model values.\n"
    "Equation (2), Simini et al. (2012), with N_c/N = 1.\n"
    "\n"
    "Arguments\n"
    "----------\n"
    "    population_list: A 1-D NumPy float array containing populations at\n"
    "        each site.\n"
    "    distance_matrix: A 2-D square NumPy float array of size equal to the\n"
    "        square of the size of population_list, where distance_matrix[i][j] is\n"
    "        the distance between site i (with population population_list[i]) and\n"
    "        site j (with population population_list[j]).\n"
    "    normalisation: A Python boolean which turns on the normalisation factor\n"
    "        for the radiation model equal to 1/(1-n_i/N), where N is the total \n"
    "        population (not included in Simini et al. (2012)).\n"
    "\n"
    "Returns\n"
    "----------\n"
    "    values: A 2-D square NumPy array containing the radiation model values\n"
    "    between every pair of points. The value in values[i][j] corresponds to the\n"
    "    distance in distance_matrix[i][j].\n"
);
static PyObject* radiation_model(PyObject *self, PyObject *args, PyObject *kwargs);

PyDoc_STRVAR(
    gravity_model__doc__,
    "gravity_model(input_list, output_list, distance_matrix, deterrence_func, input_exp=1, output_exp=1, production_constrained=False, doubly_constrained=False, threshold=-1)\n"
    "--\n"
    "\n"
    "Calculates gravity model values.\n"
    "Equation is F_{ij} =  I_i^a O_j^b * f(r_{ij}), where I_i is the ith element\n"
    "of input_list, O_j is the jth element of output_list, a is input_exp, b is\n"
    "output_exp, r_{ij} is the ijth element of distance_matrix and f(r) is\n"
    "deterrence_func.\n"
    "\n"
    "Arguments\n"
    "----------\n"
    "    input_list: A 1-D NumPy float array containing values for each site.\n"
    "        Must be the same size as output_list."
    "    output_list: A 1-D NumPy float array containing values for each site.\n"
    "        Must be the same size as output_list."
    "    distance_matrix: A 2-D square NumPy float array of size equal to the\n"
    "        square of the size of input_list, where distance_matrix[i][j] is the\n"
    "        distance between site i (corresponding to input_list[i]) and site j\n"
    "        (corresponding to output_list[j]).\n"
    "    deterrence_func: A callable Python object with exactly one argument. Takes\n"
    "        a value from the distance_matrix array and returns a float."
    "    input_exp: The power to which values from input_list are raised. Must be a\n"
    "        float.\n"
    "    output_exp: The power to which values from output_list are raised. Must be\n"
    "        a float.\n"
    "    production_constrained: A Python boolean which turns on the production\n"
    "        constrained gravity model (sum_j F_{ij} = O_i).\n"
    "    doubly_constrained: A Python boolean which turns on the doubly constraned\n"
    "        gravity model (sum_j F_{ij} = O_i and sum_i F_{ij} = I_j).\n"
    "    threshold: The conversion threshold for the doubly constrained model.\n"
    "        The algorithm will stop when total error is less than threshold.\n"
    "        Error is calculated as the summed differences between the sum of the\n"
    "        row and column elements and the target sums, i.e. sum_i(sum_j F_{ij}\n"
    "        - O_i) + sum_j (sum_i F_{ij} - I_j).\n"
    "\n"
    "Returns\n"
    "----------\n"
    "    values: A 2-D square NumPy array containing the gravity model values\n"
    "    between every pair of points. The value in values[i][j] corresponds to the\n"
    "    distance in distance_matrix[i][j].\n"
);
static PyObject* gravity_model(PyObject *self, PyObject *args, PyObject *kwargs);

#endif //SPATIALMODELSMODULE_H
