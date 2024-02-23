import flow_generation as fg
import spatialmodels as sm
import scipy.optimize as spo
import statistical_test as st
import numpy as np


def fit_proportion(input_list, distance_matrix, model, observed_matrix,
                   initial_arg, output_list=None, ignore_diag=True,
                   kwargs=None):

    if kwargs['method'] != "L-BFGS-B":
        raise ValueError("Minimisation algorithm must be L-BFGS-B.")

    if model.fit:
        raise ValueError("The input model already implements a fitting \
            algorithm.")

    def func(input_list, distance_matrix, model, observed_matrix, initial_arg,
             output_list=None, ignore_diag=True, kwargs=None):

        if model.twoargs:
            if output_list is None:
                raise ValueError("This model requires an output_list.")
            else:
                modelled_matrix = model(input_list, output_list,
                                        distance_matrix)
        else:
            modelled_matrix = model(input_list, distance_matrix)

        print(modelled_matrix)

        def likelihood(arg):

            likelihood = st.maximum_likelihood(st.poisson_hurdle_term,
                                               observed_matrix,
                                               arg * modelled_matrix,
                                               ignore_diag)

            print(likelihood, arg)

            return likelihood

        likelihood_0 = likelihood(initial_arg)

        if kwargs is not None:
            res = spo.minimize(lambda x: likelihood_0 - likelihood(x),
                               initial_arg,
                               **kwargs)
        else:
            res = spo.minimize(lambda x: likelihood_0 - likelihood(x),
                               initial_arg)

        arg = res['x']

        modelled_matrix *= arg

        likelihood = st.maximum_likelihood(st.poisson_hurdle_term,
                                           observed_matrix,
                                           modelled_matrix,
                                           ignore_diag)

        jac = res['jac']
        hess_inv = res['hess_inv'](np.ones(jac.size))
        error_arg = np.sqrt(hess_inv)
        error = np.sqrt(np.sum((jac * error_arg) ** 2))

        return modelled_matrix, likelihood, arg, error

    if model.twoargs:
        return fg.DistributionModel(lambda i, o, d:
                                    func(i, o, d, model,
                                         observed_matrix, initial_arg,
                                         output_list, ignore_diag, kwargs),
                                    twoargs=True, fit=True, param_no=1)
    else:
        return fg.DistributionModel(lambda i, d:
                                    func(i, d, model,
                                         observed_matrix, initial_arg,
                                         output_list, ignore_diag, kwargs),
                                    twoargs=False, fit=True, param_no=1)


radiation_model = fg.DistributionModel(lambda i, d:
                                       sm.radiation_model(i, d, False),
                                       False, param_no=0)
normed_radiation_model = fg.DistributionModel(lambda i, d:
                                              sm.radiation_model(i, d, True),
                                              False, param_no=0)


def gravity_fit(input_list, output_list, distance_matrix,
                deterrence_func, observed_matrix, initial_args,
                ignore_diag=True, kwargs=None, production_constrained=False,
                doubly_constrained=False, threshold=-1, fit_proportion=False):

    initial_args = np.array(initial_args)

    if kwargs['method'] != "L-BFGS-B":
        raise ValueError("Minimisation algorithm must be L-BFGS-B.")

    def likelihood(args_arr):

        if fit_proportion:
            prop_arg = args_arr[-1]
            args_arr = args_arr[:-1]

        modelled_matrix = sm.gravity_model(
            input_list, output_list, distance_matrix,
            lambda x: deterrence_func(x, args_arr),
            production_constrained=production_constrained,
            doubly_constrained=doubly_constrained,
            threshold=threshold)

        if fit_proportion:
            modelled_matrix *= prop_arg
            args_arr = np.append(args_arr, prop_arg)

        if ignore_diag:
            np.fill_diagonal(modelled_matrix, np.nan)

        likelihood = st.maximum_likelihood(st.poisson_hurdle_term,
                                           observed_matrix,
                                           modelled_matrix,
                                           ignore_diag)

        print("args_arr", args_arr, "; likelihood", likelihood)
        print(modelled_matrix)

        return likelihood

    likelihood_0 = likelihood(initial_args)

    if kwargs is not None:
        res = spo.minimize(lambda x: likelihood_0 - likelihood(x),
                           initial_args, **kwargs)
    else:
        res = spo.minimize(lambda x: likelihood_0 - likelihood(x),
                           initial_args)

    args = res['x']

    if fit_proportion:
        modelled_matrix = sm.gravity_model(
            input_list, output_list, distance_matrix,
            lambda x: deterrence_func(x, args[:-1]),
            production_constrained=production_constrained,
            doubly_constrained=doubly_constrained,
            threshold=threshold) * args[-1]
    else:
        modelled_matrix = sm.gravity_model(
            input_list, output_list, distance_matrix,
            lambda x: deterrence_func(x, args),
            production_constrained=production_constrained,
            doubly_constrained=doubly_constrained,
            threshold=threshold)

    likelihood = st.maximum_likelihood(st.poisson_hurdle_term,
                                       observed_matrix,
                                       modelled_matrix,
                                       ignore_diag)

    jac = res['jac']
    hess_inv = res['hess_inv'](np.ones(jac.size))
    error_arg = np.sqrt(hess_inv)
    error = np.sqrt(np.sum((jac * error_arg) ** 2))

    return modelled_matrix, likelihood, args, error


def exponential(distance, gamma=1):
    return np.exp(-gamma * distance)


def power(distance, gamma=2):
    if distance == 0:
        return 0
    else:
        return distance**(-gamma)


def tanner(distance, gamma_arr):
    return power(distance, gamma_arr[1]) * exponential(distance, gamma_arr[0])


def gravity_fit_exp_single_model(observed_matrix, gamma_0, kwargs=None,
                                 production_constrained=False,
                                 doubly_constrained=False, threshold=-1,
                                 fit_proportion=False):
    if fit_proportion:
        param_no = 2
    else:
        param_no = 1

    return fg.DistributionModel(lambda i, d:
                                gravity_fit(i, i, d,
                                            exponential,
                                            observed_matrix,
                                            gamma_0,
                                            True,
                                            kwargs,
                                            production_constrained,
                                            doubly_constrained,
                                            threshold,
                                            fit_proportion),
                                False,
                                fit=True, param_no=param_no)


def gravity_fit_power_single_model(observed_matrix, gamma_0, kwargs=None,
                                   production_constrained=False,
                                   doubly_constrained=False, threshold=-1,
                                   fit_proportion=False):
    if fit_proportion:
        param_no = 2
    else:
        param_no = 1

    return fg.DistributionModel(lambda i, d:
                                gravity_fit(i, i, d,
                                            power,
                                            observed_matrix,
                                            gamma_0,
                                            True,
                                            kwargs,
                                            production_constrained,
                                            doubly_constrained,
                                            threshold,
                                            fit_proportion),
                                False,
                                fit=True, param_no=param_no)


def gravity_fit_tanner_single_model(observed_matrix, gamma_0, kwargs=None,
                                    production_constrained=False,
                                    doubly_constrained=False, threshold=-1,
                                    fit_proportion=False):
    if fit_proportion:
        param_no = 3
    else:
        param_no = 2

    return fg.DistributionModel(lambda i, d:
                                gravity_fit(i, i, d,
                                            tanner,
                                            observed_matrix,
                                            gamma_0,
                                            True,
                                            kwargs,
                                            production_constrained,
                                            doubly_constrained,
                                            threshold,
                                            fit_proportion),
                                False,
                                fit=True, param_no=param_no)


def gravity_normed_exp_single_model(observed_matrix,
                                    production_constrained=False,
                                    doubly_constrained=False, threshold=-1):

    total = np.sum(observed_matrix) - np.trace(observed_matrix)

    def gravity_model(input_list, distance_matrix):
        modelled_matrix = sm.gravity_model(
            input_list, input_list, distance_matrix, exponential,
            production_constrained=production_constrained,
            doubly_constrained=doubly_constrained, threshold=threshold)

        return modelled_matrix / (np.nansum(modelled_matrix)
                                  - np.trace(modelled_matrix))

    return fg.DistributionModel(lambda i, d:
                                gravity_model(i, d) * total,
                                False)


def gravity_normed_power_single_model(observed_matrix,
                                      production_constrained=False,
                                      doubly_constrained=False, threshold=-1):

    total = np.sum(observed_matrix) - np.trace(observed_matrix)

    def gravity_model(input_list, distance_matrix):
        modelled_matrix = sm.gravity_model(
            input_list, input_list, distance_matrix, power,
            production_constrained=production_constrained,
            doubly_constrained=doubly_constrained, threshold=threshold)

        np.fill_diagonal(modelled_matrix, np.nan)

        return modelled_matrix / (np.nansum(modelled_matrix))

    return fg.DistributionModel(lambda i, d:
                                gravity_model(i, d) * total,
                                False)


def gravity_power_single_model(production_constrained=False,
                               doubly_constrained=False, threshold=-1):
    return fg.DistributionModel(lambda i, d:
                                sm.gravity_model(
                                    i, i, d, power,
                                    production_constrained=
                                    production_constrained,
                                    doubly_constrained=doubly_constrained,
                                    threshold=threshold),
                                False)


def gravity_fit_exp_double_model(observed_matrix, gamma_0, kwargs=None,
                                 production_constrained=False,
                                 doubly_constrained=False, threshold=-1,
                                 fit_proportion=False):
    if fit_proportion:
        param_no = 2
    else:
        param_no = 1

    return fg.DistributionModel(lambda i, o, d:
                                gravity_fit(i, o, d,
                                            exponential,
                                            observed_matrix,
                                            gamma_0,
                                            True,
                                            kwargs,
                                            production_constrained,
                                            doubly_constrained,
                                            threshold,
                                            fit_proportion),
                                False,
                                fit=True, param_no=param_no)


def gravity_fit_power_double_model(observed_matrix, gamma_0, kwargs=None,
                                   production_constrained=False,
                                   doubly_constrained=False, threshold=-1,
                                   fit_proportion=False):
    if fit_proportion:
        param_no = 2
    else:
        param_no = 1

    return fg.DistributionModel(lambda i, o, d:
                                gravity_fit(i, o, d,
                                            power,
                                            observed_matrix,
                                            gamma_0,
                                            True,
                                            kwargs,
                                            production_constrained,
                                            doubly_constrained,
                                            threshold,
                                            fit_proportion),
                                True,
                                fit=True, param_no=param_no)


def gravity_fit_tanner_double_model(observed_matrix, gamma_0, kwargs=None,
                                    production_constrained=False,
                                    doubly_constrained=False, threshold=-1,
                                    fit_proportion=False):
    if fit_proportion:
        param_no = 3
    else:
        param_no = 2

    return fg.DistributionModel(lambda i, o, d:
                                gravity_fit(i, o, d,
                                            tanner,
                                            observed_matrix,
                                            gamma_0,
                                            True,
                                            kwargs,
                                            production_constrained,
                                            doubly_constrained,
                                            threshold,
                                            fit_proportion),
                                True,
                                fit=True, param_no=param_no)


def gravity_normed_exp_double_model(observed_matrix,
                                    production_constrained=False,
                                    doubly_constrained=False, threshold=-1):

    total = np.sum(observed_matrix) - np.trace(observed_matrix)

    def gravity_model(input_list, distance_matrix):
        modelled_matrix = sm.gravity_model(
            input_list, input_list, distance_matrix, exponential,
            production_constrained=production_constrained,
            doubly_constrained=doubly_constrained, threshold=threshold)

        return modelled_matrix / (np.nansum(modelled_matrix)
                                  - np.trace(modelled_matrix))

    return fg.DistributionModel(lambda i, o, d:
                                gravity_model(i, d) * total,
                                True)


def gravity_normed_power_double_model(observed_matrix,
                                      production_constrained=False,
                                      doubly_constrained=False, threshold=-1):

    total = np.sum(observed_matrix) - np.trace(observed_matrix)

    def gravity_model(input_list, distance_matrix):
        modelled_matrix = sm.gravity_model(
            input_list, input_list, distance_matrix, power,
            production_constrained=production_constrained,
            doubly_constrained=doubly_constrained, threshold=threshold)

        np.fill_diagonal(modelled_matrix, np.nan)

        return modelled_matrix / (np.nansum(modelled_matrix))

    return fg.DistributionModel(lambda i, o, d:
                                gravity_model(i, d) * total,
                                True)


def gravity_power_double_model(production_constrained=False,
                               doubly_constrained=False, threshold=-1):
    return fg.DistributionModel(lambda i, o, d:
                                sm.gravity_model(
                                    i, o, d, power,
                                    production_constrained=
                                    production_constrained,
                                    doubly_constrained=doubly_constrained,
                                    threshold=threshold),
                                True)
