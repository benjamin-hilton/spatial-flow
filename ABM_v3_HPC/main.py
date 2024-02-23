import numpy as np
import radiationabm as abm
import datetime
import sys

std_dev = float(sys.argv[1])

inputs = np.load("/rds/general/user/aps515/home/abm/US_Census_Import/sources.npy")
distances = np.load("/rds/general/user/aps515/home/abm/US_Census_Import/distances.npy")

"""
import scipy.spatial as spsp
inputs = np.array([9, 23, 2, 12, 7]).astype("float64")

coords = np.array([[2.3, 1.1], [4.3, 3.4], [7.8, 0.5], [4.6, 7.3], [1.2, 5.6]])
distances = spsp.distance.squareform(spsp.distance.pdist(coords))
"""

flow_matrix = abm.run_abm(inputs, distances, std_dev)

np.save("/rds/general/user/aps515/home/abm/outputs/outputs_" + sys.argv[1] + "/" + datetime.datetime.now().isoformat() + ".npy", flow_matrix)
