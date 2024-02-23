import numpy as np
import scipy.special as sps


def poisson_term(observed, modelled, variance):
    """
    Generates (zero truncated) Poisson distributed values for individual
    terms under the sum of a log-maximum likelihood calculation.

    Relies on equation (3.4) in Ilias Bamis' MSc - will need to be verified.

    Arguments:
        observed -- An observed datapoint
        modelled -- The value of the same point according to some model

    Returns:
        The term under the sum in a log-likelihood calculation corresponding to
        this particular point.

    """

    # Stirling's approximation
    result = observed * np.log(modelled) - modelled \
        - (observed * np.log(observed) - observed)

    return result


def poisson_gamma_term(observed, modelled, variance=1):
    """
    Generates Negative Binomial (Poisson-Gamma) values for individual terms
    under the sum of a log-maximum likelihood calculation.

    Relies on equation (3.6) in Ilias Bamis' MSc - will need to be verified.

    Arguments:
        observed -- An observed datapoint
        modelled -- The value of the same point according to some model
        variance -- The variance on the model-predicted data

    Returns:
        The term under the sum in a log-likelihood calculation corresponding to
        this particular point.
    """

    product = variance * modelled

    result = observed * np.log((product)/(1 + product)) \
        - (np.log(1 + product) / variance) \
        + np.log(sps.gamma(observed + 1/variance)) \
        - np.log(sps.gamma(observed + 1)) - np.log(sps.gamma(1/variance))

    return result


def poisson_hurdle_term(observed, modelled, variance=1):

    result = np.zeros(observed.shape)

    result[observed == 0] = 1/(1 + modelled[observed == 0])
    result[observed != 0] = modelled[observed != 0] \
        * np.log(observed[observed != 0]) - observed[observed != 0] \
        - (modelled[observed != 0] + 1) * np.log(modelled[observed != 0] + 1) \
        + modelled[observed != 0] + 1 \
        - np.log(1 - np.exp(-observed[observed != 0]))

    np.fill_diagonal(result, 0)

    return result


def generate_maximum_likelihood_matrix(dist_function, observed_matrix,
                                       modelled_matrix, variance_matrix=1,
                                       ignore_diag=True):
    """
    Generates a matrix of log-maximum likelihood values

    Arguments:
        dist_function -- The function by which the maximum likelihood is
                         calculated; an assumption of the underlying pdf.
        observed_matrix -- A matrix of observed datapoints
        modelled_matrix -- A corresponding matrix of modelled datapoints
        variance_matrix -- The variance(s) on the model-predicted data

    Returns:
        A matrix where each element is equal to the calculated likelihood for
        each corresponding element in the observed and modelled data.
    """

    maximum_likelihood_matrix = dist_function(observed_matrix, modelled_matrix,
                                              variance_matrix)

    if ignore_diag:
        np.fill_diagonal(maximum_likelihood_matrix, 0)

    return maximum_likelihood_matrix


def maximum_likelihood(dist_function, observed_matrix, modelled_matrix,
                       variance_matrix=1, ignore_diag=True):
    """
    Finds the value of the log-maximum likelihood for a given dataset, given an
    underlying pdf

    Arguments:
        dist_function -- The function by which the maximum likelihood is
                         calculated; an assumption of the underlying pdf
        observed_matrix -- A matrix of observed datapoints
        modelled_matrix -- A corresponding matrix of modelled datapoints
        variance_matrix -- The variance(s) on the model-predicted data

    Returns:
        The value of the maximum likelihood.
    """

    maximum_likelihood_matrix = generate_maximum_likelihood_matrix(
        dist_function, observed_matrix, modelled_matrix, variance_matrix,
        ignore_diag)

    return np.nansum(maximum_likelihood_matrix)


def BIC(dist_function, observed_matrix, modelled_matrix,
        variance_matrix=1, number_of_params=1):
    """
    Calculates Bayesian Information Criterion value for given dataset

    Arguments:
        dist_function -- The function by which the maximum likelihood is
                         calculated; an assumption of the underlying pdf
        observed_matrix -- A matrix of observed datapoints
        modelled_matrix -- A corresponding matrix of modelled datapoints
        variance_matrix -- The variance(s) on the model-predicted data
        number_of_params -- The number of parameters estimated by the model

    Returns:
        The value of the Bayesian Information Criterion for these datasets,
        given the supposed underlying distribution.
    """

    number_of_datapoints = observed_matrix.size

    maximum_likelihood_value = maximum_likelihood(
        dist_function, observed_matrix, modelled_matrix, variance_matrix)

    return np.log(number_of_datapoints) * number_of_params \
        - 2 * maximum_likelihood_value


def kolmogorov_smirnov(observed_matrix, modelled_matrix):
    return np.nanmax(np.abs(observed_matrix - modelled_matrix))
