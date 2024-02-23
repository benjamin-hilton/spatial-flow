import numpy as np
import matplotlib.pyplot as plt


def model_data(flow_matrices, data_flow_matrix, equal_axes=False):
    figures = {}
    for key in flow_matrices:
        figures[key] = plt.figure(key + " against Data")
        xx = np.linspace(0, np.min(np.array([np.max(data_flow_matrix),
                                            np.max(flow_matrices[key])])), 1e6)
        figures[key].gca().plot(xx, xx, 'r--')
        figures[key].gca().plot(data_flow_matrix.flatten(),
                                flow_matrices[key].flatten(), 'x')
        figures[key].gca().set_xlabel("Flow (data)")
        figures[key].gca().set_ylabel("Flow (model)")

        if equal_axes:
            figures[key].gca().set_aspect('equal', adjustable='box')

    return figures


def model_data_loglog(flow_matrices, data_flow_matrix):
    figures = {}
    for key in flow_matrices:
        figures[key] = plt.figure(key + " against Data (log-log)")
        xx = np.linspace(np.min(np.array([np.min(data_flow_matrix),
                                            np.min(flow_matrices[key])])),
                        np.mnin(np.array([np.max(data_flow_matrix),
                                            np.max(flow_matrices[key])])), 1e6)
        figures[key].gca().loglog(xx, xx, 'r--')
        figures[key].gca().loglog(data_flow_matrix.flatten(),
                                  flow_matrices[key].flatten(), 'x')
        figures[key].gca().set_xlabel("Flow (data)")
        figures[key].gca().set_ylabel("Flow (model)")

    return figures


def error_rank(flow_matrices, data_flow_matrix, semilogy=True):
    figures = {}
    for key in flow_matrices:
        figures[key] = plt.figure("Error in " + key + " against rank.")

        error = flow_matrices[key]/data_flow_matrix
        error = error.flatten()
        error = error[data_flow_matrix.flatten().argsort()][::-1]
        ranks = np.arange(1, 1 + data_flow_matrix.size)

        if semilogy:
            figures[key].gca().semilogy(ranks, error, "bx")
        else:
            figures[key].gca().plot(ranks, error, "b-")

        figures[key].gca().set_xlabel("Rank")
        figures[key].gca().set_ylabel("Flow (model) / Flow (data)")

    return figures
