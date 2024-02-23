import numpy as np

residence_data = np.genfromtxt("2KRESCO_US.TXT", invalid_raise=False)

residence_data = residence_data[:, [0, 1, 4, 5, -1]]

new_residence_data = np.zeros((residence_data.shape[0], 3))

new_residence_data[:, 0] = residence_data[:, 0] * 1e3 + residence_data[:, 1]
new_residence_data[:, 1] = residence_data[:, 2] * 1e3 + residence_data[:, 3]
new_residence_data[:, 2] = residence_data[:, -1]

residence_data = new_residence_data
new_residence_data = None

# Remove AK
residence_data = residence_data[np.logical_or(residence_data[:, 0] < 2000,
                                              residence_data[:, 0] >= 3000)]
residence_data = residence_data[np.logical_or(residence_data[:, 1] < 2000,
                                              residence_data[:, 1] >= 3000)]

# Remove HI
residence_data = residence_data[np.logical_or(residence_data[:, 0] < 15000,
                                              residence_data[:, 0] >= 16000)]
residence_data = residence_data[np.logical_or(residence_data[:, 1] < 15000,
                                              residence_data[:, 1] >= 16000)]

# Remove non-states (e.g. Puerto Rico)
residence_data = residence_data[np.logical_and(residence_data[:, 0] < 56000,
                                               residence_data[:, 1] < 56000)]

indices = np.unique(residence_data[:, 0])

# Separate into a list of arrays, one array for each source site.
residence_data_list = [residence_data[residence_data[:, 0] == i]
                       for i in np.unique(residence_data[:, 0])]

sources = np.zeros(indices.size)

for i in range(indices.size):
    sources[i] = np.sum(residence_data_list[i][:, -1])

# Separate into a list of arrays, one array for each target site.
residence_data_list = [residence_data[residence_data[:, 1] == i]
                       for i in np.unique(residence_data[:, 0])]

targets = np.zeros(indices.size)
for i in range(indices.size):
    targets[i] = np.sum(residence_data_list[i][:, -1])

np.save("sources.npy", sources)
np.save("targets.npy", targets)

for i in range(indices.size):
    index = indices[i]
    diagonal = residence_data[np.logical_and(residence_data[:, 0] == index,
                                             residence_data[:, 1] == index)][0]
    sources[i] -= diagonal[2]
    targets[i] -= diagonal[2]

np.save("sources_fixed.npy", sources)
np.save("targets_fixed.npy", targets)

# residence_data = residence_data[np.lexsort((residence_data[:, 0],
#                                             residence_data[:, 1]))]
# residence_data = np.reshape(residence_data[:, 2],
#                             (indices.size, indices.size))

print(residence_data)

flow_matrix = np.zeros((indices.size, indices.size))

data_row = 0
while data_row < indices.size:
    for i in range(indices.size):
        for j in range(indices.size):
            if (residence_data[data_row][0] == indices[i]
               and residence_data[data_row][1] == indices[j]):
                flow_matrix[i][j] = residence_data[data_row][2]
                data_row += 1

np.save("flow_matrix.npy", flow_matrix)

distances = np.genfromtxt("sf12000countydistancemiles.csv", delimiter=',')[1:]

indices_old = np.unique(distances[:, 0])

# Remove AK
distances = distances[np.logical_or(distances[:, 0] < 2000,
                                    distances[:, 0] >= 3000)]
distances = distances[np.logical_or(distances[:, 2] < 2000,
                                    distances[:, 2] >= 3000)]

# Remove HI
distances = distances[np.logical_or(distances[:, 0] < 15000,
                                    distances[:, 0] >= 16000)]
distances = distances[np.logical_or(distances[:, 2] < 15000,
                                    distances[:, 2] >= 16000)]

# Remove non-staes (e.g. Puerto Rico)
distances = distances[np.logical_and(distances[:, 0] < 56000,
                                     distances[:, 2] < 56000)]


indices = np.unique(distances[:, 0])

np.save("indices.npy", indices)

diagonal_elements = np.array([indices, np.full(indices.size, 0), indices]).T
distances = np.vstack((distances, diagonal_elements))

distances = distances[np.lexsort((distances[:, 2], distances[:, 0]))]

distances = np.reshape(distances[:, 1], (indices.size, indices.size))

np.save("distances.npy", distances)

area_data = np.genfromtxt("county2k.txt", invalid_raise=False)[:, 2]
area_data *= 3.86102159e-7

area_data = np.vstack((indices_old, area_data)).T
print(area_data)

# Remove AK
area_data = area_data[np.logical_or(area_data[:, 0] < 2000,
                                    area_data[:, 0] >= 3000)]
# Remove HI

area_data = area_data[np.logical_or(area_data[:, 0] < 15000,
                                    area_data[:, 0] >= 16000)]

# Remove non-states (e.g. Puerto Rico)
area_data = area_data[area_data[:, 0] < 56000]

np.save("area_data.npy", area_data[:, 1])
