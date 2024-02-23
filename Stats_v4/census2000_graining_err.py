import numpy as np
import flow_generation as fg
import matplotlib.pyplot as plt
import statistical_test as st
import models
import matplotlib.pyplot as plt
import pickle

inputs = np.load("US_Census_Import/sources.npy")
distances = np.load("US_Census_Import/distances.npy")
data_flow_matrix = np.load("US_Census_Import/flow_matrix.npy")
area_data = np.load("US_Census_Import/area_data.npy")

census2000 = fg.FlowGenerator(distances, inputs)


census2000.coarse_graining_err(area_data)
print(np.nanmax(census2000.coarse_graining_err))
print(np.unravel_index(np.nanargmax(census2000.coarse_graining_err), census2000.coarse_graining_err.shape))
print(np.nanmax(census2000.coarse_graining_err/distances))
index = np.unravel_index(np.nanargmax(census2000.coarse_graining_err/distances), census2000.coarse_graining_err.shape)
print(index)
print(census2000.coarse_graining_err[index], distances[index])

print(np.nanmin(census2000.coarse_graining_err))
print(np.unravel_index(np.nanargmin(census2000.coarse_graining_err), census2000.coarse_graining_err.shape))
print(np.nanmin(census2000.coarse_graining_err/distances))
index = np.unravel_index(np.nanargmin(census2000.coarse_graining_err/distances), census2000.coarse_graining_err.shape)
print(index)
print(census2000.coarse_graining_err[index], distances[index])

err = census2000.coarse_graining_err.flatten()
err = err[~np.isnan(err)]
per_err = census2000.coarse_graining_err/distances
per_err = per_err.flatten()
per_err = per_err[~np.isnan(per_err)]

fig1 = plt.figure("Distribution of Coarse Graining Error")
fig1.gca().hist(err.flatten(), bins='auto', density=True)
fig1.gca().set_xlabel("Coarse Graining Error (miles)")
fig1.gca().set_ylabel("Frequency Density (per mile)")
fig1.savefig("coarse_graining_dist.png")
fig2 = plt.figure("Distribution of Percentage Coars Graining Error")
fig2.gca().hist(per_err.flatten()*100, bins='auto', density=True)
fig2.gca().set_xlabel("Coarse Graining Percentage Error")
fig2.gca().set_ylabel("Frequency Density")
fig2.savefig("coarse_graining_dist_p.png")
plt.show()
