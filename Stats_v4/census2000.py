import numpy as np
import flow_generation as fg
# import model_plots as mp
import statistical_test as st
import models
import matplotlib.pyplot as plt
import pickle

inputs = np.load("US_Census_Import/sources.npy")
distances = np.load("US_Census_Import/distances.npy")
data_flow_matrix = np.load("US_Census_Import/flow_matrix.npy")
area_data = np.load("US_Census_Import/area_data.npy")

census2000 = fg.FlowGenerator(distances, inputs)

census2000.observed_matrix = data_flow_matrix

prop_kwargs = {"method": "L-BFGS-B",
               "bounds": [(0.0, 10.0)]}

census2000.add_model(models.fit_proportion(census2000.input_list,
                                           census2000.distance_matrix,
                                           models.radiation_model,
                                           census2000.observed_matrix,
                                           1,
                                           kwargs=prop_kwargs),
                     "Radiation Model")
census2000.add_model(models.fit_proportion(census2000.input_list,
                                           census2000.distance_matrix,
                                           models.gravity_normed_exp_single_model(data_flow_matrix),
                                           census2000.observed_matrix,
                                           1,
                                           kwargs=prop_kwargs),
                     "Unconstrained Gravity Model (Exponential, Normalised)")

spo_kwargs = {"method": "L-BFGS-B",
              "bounds": [(0.0, 10.0), (0.0, 10.0)]}

census2000.add_model(models.gravity_fit_exp_single_model(data_flow_matrix,
                                                         [4.75045974, 0.04618847],
                                                         spo_kwargs,
                                                         fit_proportion=True),
                     "Unconstrained Gravity Model (Exponential, Fit)")

census2000.add_model(models.fit_proportion(census2000.input_list,
                                           census2000.distance_matrix,
                                           models.gravity_normed_power_single_model(data_flow_matrix),
                                           census2000.observed_matrix,
                                           1,
                                           kwargs=prop_kwargs),
                     "Unconstrained Gravity Model (Power Law, Normalised)")

spo_kwargs = {"method": "L-BFGS-B",
              "bounds": [(0.0, 12.0), (0.0, 10.0)]}

census2000.add_model(models.gravity_fit_power_single_model(data_flow_matrix,
                                                           [9.0, 1.0],
                                                           spo_kwargs,
                                                           fit_proportion=True),
                     "Unconstrained Gravity Model (Power Law, Fit)")

spo_kwargs = {"method": "L-BFGS-B",
              "bounds": [(0.0, 20.0), (0.0, 20.0), (0.0, 10.0)]}

census2000.add_model(models.gravity_fit_tanner_single_model(data_flow_matrix,
                                                            [9.0, 9.0, 1.0],
                                                            spo_kwargs,
                                                            fit_proportion=True),
                     "Unconstrained Gravity Model (Tanner Function, Fit)")

census2000.add_model(models.fit_proportion(census2000.input_list,
                                           census2000.distance_matrix,
                                           models.gravity_normed_exp_single_model(data_flow_matrix),
                                           census2000.observed_matrix,
                                           1,
                                           kwargs=prop_kwargs),
                     "Unconstrained Gravity Model (Exponential, Normalised)")

spo_kwargs = {"method": "L-BFGS-B",
              "bounds": [(0.0, 100.0), (0.0, 10.0)]}

census2000.add_model(models.gravity_fit_exp_single_model(data_flow_matrix,
                                                         [9.0, 1.0],
                                                         spo_kwargs,
                                                         production_constrained=True,
                                                         fit_proportion=True),
                     "Production Constrained Gravity Model (Exponential, Fit)")

census2000.add_model(models.fit_proportion(census2000.input_list,
                                           census2000.distance_matrix,
                                           models.gravity_normed_power_single_model(data_flow_matrix,
                                                                                    production_constrained=True),
                                           census2000.observed_matrix,
                                           1,
                                           kwargs=prop_kwargs),
                     "Production Constrained Gravity Model (Power Law, Normalised)")

census2000.add_model(models.fit_proportion(census2000.input_list,
                                           census2000.distance_matrix,
                                           models.gravity_normed_exp_single_model(data_flow_matrix,
                                                                                  production_constrained=True),
                                           census2000.observed_matrix,
                                           1,
                                           kwargs=prop_kwargs),
                     "Production Constrained Gravity Model (Exponential, Normalised)")

spo_kwargs = {"method": "L-BFGS-B",
              "bounds": [(0.0, 100.0), (0.0, 10.0)]}

census2000.add_model(models.gravity_fit_power_single_model(data_flow_matrix,
                                                           [6.0, 1.0],
                                                           spo_kwargs,
                                                           production_constrained=True,
                                                           fit_proportion=True),
                     "Production Constrained Gravity Model (Power Law, Fit)")

spo_kwargs = {"method": "L-BFGS-B",
              "bounds": [(0.0, 20.0), (0.0, 20.0), (0.0, 10.0)]}

census2000.add_model(models.gravity_fit_tanner_single_model(data_flow_matrix,
                                                            [9.0, 9.0, 1.0],
                                                            spo_kwargs,
                                                            production_constrained=True,
                                                            fit_proportion=True),
                     "Production Constrained Gravity Model (Tanner Function, Fit)")

census2000.coarse_graining_err(area_data)
print(np.nanmax(census2000.coarse_graining_err))
print(np.unravel_index(np.nanargmax(census2000.coarse_graining_err), census2000.coarse_graining_err.shape))
print(np.nanmax(census2000.coarse_graining_err/distances))
index=np.unravel_index(np.nanargmax(census2000.coarse_graining_err/distances), census2000.coarse_graining_err.shape)
print(index)
print(census2000.coarse_graining_err[index], distances[index])

print(np.nanmin(census2000.coarse_graining_err))
print(np.unravel_index(np.nanargmin(census2000.coarse_graining_err), census2000.coarse_graining_err.shape))
print(np.nanmin(census2000.coarse_graining_err/distances))
index=np.unravel_index(np.nanargmin(census2000.coarse_graining_err/distances), census2000.coarse_graining_err.shape)
print(index)
print(census2000.coarse_graining_err[index], distances[index])

"""
print("Generating Radiation Model...")
census2000.generate_flow("Radiation Model")
print("Generating Unconstrained Gravity Model (Exponential, Fit)...")
census2000.generate_flow("Unconstrained Gravity Model (Exponential, Fit)")
print("Generating Unconstrained Gravity Model (Exponential, Normalised)...")
census2000.generate_flow("Unconstrained Gravity Model (Exponential, Normalised)")
print("Generating Unconstrained Gravity Model (Power Law, Normalised)...")
census2000.generate_flow("Unconstrained Gravity Model (Power Law, Normalised)")
print("Generating Unconstrained Gravity Model (Power Law, Fit)...")
census2000.generate_flow("Unconstrained Gravity Model (Power Law, Fit)")
print("Generating Unconstrained Gravity Model (Tanner Function, Fit)...")
census2000.generate_flow("Unconstrained Gravity Model (Tanner Function, Fit)")
print("Generating Production Constrained Gravity Model (Exponential, Fit)...")
census2000.generate_flow("Production Constrained Gravity Model (Exponential, Fit)")
print("Generating Production Constrained Gravity Model (Exponential, Normalised)...")
census2000.generate_flow("Production Constrained Gravity Model (Exponential, Normalised)")
print("Generating Production Constrained Gravity Model (Power Law, Normalised)...")
census2000.generate_flow("Production Constrained Gravity Model (Power Law, Normalised)")
print("Generating Production Constrained Gravity Model (Power Law, Fit)...")
census2000.generate_flow("Production Constrained Gravity Model (Power Law, Fit)")
print("Generating Production Constrained Gravity Model (Tanner Function, Fit)...")
census2000.generate_flow("Production Constrained Gravity Model (Tanner Function, Fit)")

print("Saving data...")
with open("census2000.pkl", "wb") as file:
    pickle.dump(census2000.flow_matrices, file)
    pickle.dump(census2000.fit_info, file)
    pickle.dump(census2000.stats, file)

print("Done.")

with open("census2000.pkl", "rb") as file:
    census2000.flow_matrices = pickle.load(file)
    census2000.fit_info = pickle.load(file)
    census2000.stats = pickle.load(file)

census2000.calculate_all_KS()
census2000.calculate_all_BIC(st.poisson_hurdle_term)

with open("census2000.pkl", "wb") as file:
    pickle.dump(census2000.flow_matrices, file)
    pickle.dump(census2000.fit_info, file)
    pickle.dump(census2000.stats, file)

# mp.model_data(census2000.flow_matrices, data_flow_matrix)
# mp.error_rank(census2000.flow_matrices, data_flow_matrix)
# plt.show()
"""

print(census2000.fit_info)
