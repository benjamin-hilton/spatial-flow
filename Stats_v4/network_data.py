import numpy as np
import warnings


class DataObject:
    """
    Class containing the data modelling a flow across a network.

    Methods:
        __init__ -- Initialises the object.
        add_flow_matrix -- Adds a flow matrix (flows between each site).

    Member variables
        _input_list -- Stores the I_{i} values for the model as a numpy array.
        _output_list -- Stores the O_{j} values for the model as a numpy array.
        _distance_matrix -- Stores the distances between every site as a numpy
             array.
        flow_matrices -- Stores a dictionary of flow matrices as numpy arrays.
    """

    def __init__(self, distance_matrix, input_list,
                 output_list=None, observed_matrix=None):

        if not isinstance(input_list, np.ndarray):
            raise TypeError("Input input_list is not a NumPy array.")

        if not isinstance(distance_matrix, np.ndarray):
            raise TypeError("Input distance_matrix is not a NumPy array.")

        if input_list.size ** 2 != distance_matrix.size:
            raise ValueError("The size of distance_matrix must be equal to the \
                              square of the size of input_list.")

        self._input_list = input_list

        if output_list is not None:
            if not isinstance(output_list, np.ndarray):
                raise TypeError("Input output_list is not a NumPy array.")
            elif output_list.size != input_list.size:
                raise ValueError("The size of output_list must be equal to the \
                                  size of input_list.")
            self._output_list = output_list
        else:
            self._output_list = self._input_list

        self._distance_matrix = distance_matrix

        self.observed_matrix = observed_matrix

        self.flow_matrices = {}

        self.stats = {}

    def add_flow_matrix(self, flow_matrix, model_name):

        if not isinstance(flow_matrix, np.ndarray):
            raise TypeError("Input flow_matrix is not a NumPy array.")

        if model_name in self.flow_matrices:
            warnings.warn("There is already a matrix labelled with "
                          + model_name + ". Overwriting this matrix.",
                          RuntimeWarning)

        if flow_matrix.shape == self._distance_matrix.shape:
            self.flow_matrices[model_name] = flow_matrix
        else:
            raise ValueError("The shape of flow_matrix must be equal to the \
                              shape of distance_matrix.")

    def distance_matrix(self):

        return self._distance_matrix

    def input_list(self):

        return self._input_list

    def output_list(self):

        return self._output_list
