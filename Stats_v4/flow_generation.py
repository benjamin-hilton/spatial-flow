import numpy as np
import warnings
import scipy.spatial as spsp
import scipy.stats as sps
import statistical_test as st
import network_data as nd


class DistributionModel:
    """
    Wrapper class for a function which generates flows.

    Methods:
        __init__ -- Initialises the object.
        __call__ -- Calls the function.

    Member variables:
        gen_func -- The function which generates flows.
        twoargs -- True if gen_func takes two input_lists.
    """

    def __init__(self, gen_func, twoargs, ignore_diag=True, fit=False,
                 param_no=0):

        if not callable(gen_func):
            raise TypeError("Input gen_func must be a callable object.")

        if param_no < 0:
            raise ValueError("Input param_no must be positive.")

        if fit and param_no == 0:
            raise ValueError("Models cannot run a fitting algorithm \
                (fit is True) with 0 parameters (param_no is 0).")

        if (not fit) and param_no != 0:
            raise ValueError("Models cannot have fitting parameters (param_no \
                is 0) without running a fitting algorithm (fit is False).")

        self.gen_func = gen_func
        self.twoargs = twoargs
        self.fit = fit
        self.ignore_diag = ignore_diag
        self.param_no = param_no

    def __call__(self, *args):

        return self.gen_func(*args)


class FlowGenerator(nd.DataObject):

    def __init__(self, distance_matrix, input_list, output_list=None,
                 ignore_diag=True, set_diag=np.nan, observed_matrix=None):

        nd.DataObject.__init__(self, distance_matrix, input_list, output_list)

        if output_list is None:
            self.twoargs = False
        else:
            self.twoargs = True

        self.ignore_diag = ignore_diag
        self.set_diag = set_diag

        self.models = {}

        self.fit_info = {}

        self.observed_matrix = observed_matrix

    def add_model(self, model, model_name):

        if model_name in self.flow_matrices:
            raise ValueError("There is already a model labelled with "
                             + model_name + ".")

        if not isinstance(model, DistributionModel):

            warnings.warn("This model is not a DistributionModel. Assumed \
                           attributes may not be correct.", RuntimeWarning)
            self.models[model_name] = DistributionModel(model, self.twoargs)

        else:
            if self.twoargs == model.twoargs:
                if not(not(model.ignore_diag) or self.ignore_diag):
                    # If model.ignore_diag is true then self.ignore_diag must
                    # also be true. But if model.ignore_diag is not true it
                    # doesn't matter.
                    warnings.warn("This model ignores diagonal elements but \
                                   this class in general does not.",
                                  RuntimeWarning)
                else:
                    self.models[model_name] = model
                    self.fit_info[model_name] = {}
                    self.stats[model_name] = {}
            else:
                raise ValueError("The model takes the wrong number of \
                                  input_list arguments.")

    def generate_flow(self, model_name):

        if model_name in self.flow_matrices:
            warnings.warn("There is already a matrix labelled with "
                          + model_name + ". Overwriting this matrix.",
                          RuntimeWarning)

        if self.models[model_name].twoargs:
            flow_matrix = self.models[model_name](self._input_list,
                                                  self._output_list,
                                                  self._distance_matrix)
        else:
            flow_matrix = self.models[model_name](self._input_list,
                                                  self._distance_matrix)

        if self.models[model_name].fit:
            self.fit_info[model_name]["likelihood"] = flow_matrix[1]
            self.fit_info[model_name]["parameters"] = flow_matrix[2]
            self.fit_info[model_name]["error"] = flow_matrix[3]
            flow_matrix = flow_matrix[0]

        if self.ignore_diag:
            np.fill_diagonal(flow_matrix, self.set_diag)

        self.add_flow_matrix(flow_matrix, model_name)

    def calculate_BIC(self, model_name, dist_function):
        self.stats[model_name]["BIC"] = st.BIC(
            dist_function,
            self.observed_matrix,
            self.flow_matrices[model_name],
            number_of_params=self.models[model_name].param_no)

    def calculate_all_BIC(self, dist_function):
        for key in self.stats:
            self.calculate_BIC(key, dist_function)

    def calculate_KS(self, model_name):
        self.stats[model_name]["KS"] = st.kolmogorov_smirnov(
            self.observed_matrix,
            self.flow_matrices[model_name])

    def calculate_all_KS(self):
        for key in self.stats:
            self.calculate_KS(key)

    def coarse_graining_err(self, area_data):
        self.area_data = area_data
        outer = np.add.outer(self.area_data, self.area_data)
        self.coarse_graining_err = np.sqrt(outer * \
            (16 * self._distance_matrix ** 2 * np.pi - outer) \
            / ((64 * self._distance_matrix ** 2) * (np.pi ** 2)))


class RandomFlowGenerator(FlowGenerator):

    def __init__(self, model,
                 length,
                 distance_param,
                 input_param=1.0,
                 output_param=0.0,
                 variant=1.0,
                 ignore_diag=True,
                 set_diag=np.nan,
                 observed_matrix=None):
        """
        Function which initialises the FlowGenerator class.

        Arguments:
            self -- The current FlowGenerator object.
            model -- A DistributionModel object used to generate flows.
                values for the model.
            length -- If any of the above matrices or lists is not given,
                the length of a replacement square matrix or list. Must be
                equal to the length of the above matrices or lists, if they are
                given.
            distance_param -- The edge length of the square plane used to
                generate a new distance_matrix.
            input_param -- The power of the power law distribution used to
                generate a new input_list.
            output_param -- The power of the power law distribution used to
                generate a new output_list. If 0, output_list is set as
                a pointer to input_list.
            variant -- A matrix of numbers by which the model will be
                multiplied.

        """

        coords = np.random.rand(length, 2) * distance_param

        self._distance_matrix = spsp.distance.squareform(
                                spsp.distance.pdist(coords))

        self._input_list = sps.powerlaw.rvs(input_param, size=length)

        if output_param != 0:
            self._output_list = sps.powerlaw.rvs(output_param, size=length)
        else:
            self._output_list = self.input_list

        FlowGenerator.__init__(self,
                               self._distance_matrix,
                               self._input_list,
                               self._output_list,
                               ignore_diag,
                               set_diag,
                               observed_matrix)

        self.variant = variant

    def add_model(self, model, model_name):

        FlowGenerator.add_model(model, model_name)

    def generate_flow(self, model_name):

        FlowGenerator.generate_flow(model_name)
        self.flow_matrices[model_name] *= self.variant
