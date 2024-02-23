import numpy as np
# import matplotlib.pyplot as plt
import pickle
import os


def gen_files(std_dev):

    files = os.listdir("/rds/general/user/aps515/home/abm/outputs/outputs_"
                       + str(std_dev))
    matrices = []

    for file in files:
        matrices.append(
            np.load("/rds/general/user/aps515/home/abm/outputs/outputs_"
                    + str(std_dev) + "/" + file))

    matrices = np.array(matrices)

    means = np.mean(matrices, axis=0)
    std_dev_matrix = np.std(matrices, axis=0)

    del matrices

    with open("/rds/general/user/aps515/home/abm/outputs/means_"
              + str(std_dev) + ".pkl", "wb") as file:
        pickle.dump(means, file)

    with open("/rds/general/user/aps515/home/abm/outputs/std_dev_"
              + str(std_dev) + ".pkl", "wb") as file:
        pickle.dump(std_dev_matrix, file)


"""
with open("radiation_model.pkl", "rb") as file:
    radiation_model = pickle.load(file)

fig1 = plt.figure("ABM vs Radiation Model Data")
fig1.gca().loglog(means, radiation_model, 'bx')
fig1.gca().set_ylabel("Flow from Radiation Model")
fig1.gca().set_xlabel("Mean Flow from ABM")
fig1.gca().set_aspect('equal', adjustable='box')

plt.show()
"""

for std_dev in [50000, 100000, 150000, 200000]:
    gen_files(std_dev)
