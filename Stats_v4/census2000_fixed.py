import numpy as np
import flow_generation as fg
# import model_plots as mp
import statistical_test as st
import models
import matplotlib.pyplot as plt
import pickle

inputs = np.load("US_Census_Import/sources_fixed.npy")
distances = np.load("US_Census_Import/distances.npy")
data_flow_matrix = np.load("US_Census_Import/flow_matrix.npy")


census2000_fixed = fg.FlowGenerator(distances, inputs)

census2000_fixed.observed_matrix = data_flow_matrix

census2000_fixed.add_model(models.radiation_model,
                     "Radiation Model")

census2000_fixed.add_model(models.gravity_normed_exp_single_model(data_flow_matrix),
                     "Unconstrained Gravity Model (Exponential, Normalised)")

spo_kwargs = {"method": "L-BFGS-B",
              "bounds": [(0.0, 10.0)]}

census2000_fixed.add_model(models.gravity_fit_exp_single_model(data_flow_matrix,
                                                         4.75045974,
                                                         spo_kwargs,
                                                         fit_proportion=False),
                     "Unconstrained Gravity Model (Exponential, Fit)")

census2000_fixed.add_model(models.gravity_normed_power_single_model(data_flow_matrix),
                     "Unconstrained Gravity Model (Power Law, Normalised)")

spo_kwargs = {"method": "TNC",
              "bounds": [(0.0, 12.0)]}

census2000_fixed.add_model(models.gravity_fit_power_single_model(data_flow_matrix,
                                                           9.0,
                                                           spo_kwargs,
                                                           fit_proportion=False),
                     "Unconstrained Gravity Model (Power Law, Fit)")

spo_kwargs = {"method": "TNC",
              "bounds": [(0.0, 20.0), (0.0, 20.0)]}

census2000_fixed.add_model(models.gravity_fit_tanner_single_model(data_flow_matrix,
                                                            [9.0, 9.0],
                                                            spo_kwargs,
                                                            fit_proportion=False),
                     "Unconstrained Gravity Model (Tanner Function, Fit)")

census2000_fixed.add_model(models.gravity_normed_exp_single_model(data_flow_matrix),
                     "Unconstrained Gravity Model (Exponential, Normalised)")

spo_kwargs = {"method": "L-BFGS-B",
              "bounds": [(0.0, 100.0)]}

census2000_fixed.add_model(models.gravity_fit_exp_single_model(data_flow_matrix,
                                                         [9.0],
                                                         spo_kwargs,
                                                         production_constrained=True,
                                                         fit_proportion=False),
                     "Production Constrained Gravity Model (Exponential, Fit)")

census2000_fixed.add_model(models.gravity_normed_power_single_model(data_flow_matrix,
                                                              production_constrained=True),
                     "Production Constrained Gravity Model (Power Law, Normalised)")

census2000_fixed.add_model(models.gravity_normed_exp_single_model(data_flow_matrix,
                                                            production_constrained=True),
                     "Production Constrained Gravity Model (Exponential, Normalised)")

spo_kwargs = {"method": "TNC",
              "bounds": [(0.0, 100.0)]}

census2000_fixed.add_model(models.gravity_fit_power_single_model(data_flow_matrix,
                                                           6.0,
                                                           spo_kwargs,
                                                           production_constrained=True,
                                                           fit_proportion=False),
                     "Production Constrained Gravity Model (Power Law, Fit)")

spo_kwargs = {"method": "TNC",
              "bounds": [(0.0, 20.0), (0.0, 20.0)]}

census2000_fixed.add_model(models.gravity_fit_tanner_single_model(data_flow_matrix,
                                                            [9.0, 9.0],
                                                            spo_kwargs,
                                                            production_constrained=True,
                                                            fit_proportion=False),
                     "Production Constrained Gravity Model (Tanner Function, Fit)")
"""
print("Generating Radiation Model...")
census2000_fixed.generate_flow("Radiation Model")
print("Generating Unconstrained Gravity Model (Exponential, Fit)...")
census2000_fixed.generate_flow("Unconstrained Gravity Model (Exponential, Fit)")
print("Generating Unconstrained Gravity Model (Exponential, Normalised)...")
census2000_fixed.generate_flow("Unconstrained Gravity Model (Exponential, Normalised)")
print("Generating Unconstrained Gravity Model (Power Law, Normalised)...")
census2000_fixed.generate_flow("Unconstrained Gravity Model (Power Law, Normalised)")
print("Generating Unconstrained Gravity Model (Power Law, Fit)...")
census2000_fixed.generate_flow("Unconstrained Gravity Model (Power Law, Fit)")
print("Generating Unconstrained Gravity Model (Tanner Function, Fit)...")
census2000_fixed.generate_flow("Unconstrained Gravity Model (Tanner Function, Fit)")
print("Generating Production Constrained Gravity Model (Exponential, Fit)...")
census2000_fixed.generate_flow("Production Constrained Gravity Model (Exponential, Fit)")
print("Generating Production Constrained Gravity Model (Exponential, Normalised)...")
census2000_fixed.generate_flow("Production Constrained Gravity Model (Exponential, Normalised)")
print("Generating Production Constrained Gravity Model (Power Law, Normalised)...")
census2000_fixed.generate_flow("Production Constrained Gravity Model (Power Law, Normalised)")
print("Generating Production Constrained Gravity Model (Power Law, Fit)...")
census2000_fixed.generate_flow("Production Constrained Gravity Model (Power Law, Fit)")
print("Generating Production Constrained Gravity Model (Tanner Function, Fit)...")
census2000_fixed.generate_flow("Production Constrained Gravity Model (Tanner Function, Fit)")

print("Saving data...")
with open("census2000_fixed.pkl", "wb") as file:
    pickle.dump(census2000_fixed.flow_matrices, file)
    pickle.dump(census2000_fixed.fit_info, file)
    pickle.dump(census2000_fixed.stats, file)

print(census2000_fixed.fit_info["Radiation Model"])

print("Done.")

"""
with open("census2000_fixed.pkl", "rb") as file:
    census2000_fixed.flow_matrices = pickle.load(file)
    census2000_fixed.fit_info = pickle.load(file)
    census2000_fixed.stats = pickle.load(file)

census2000_fixed.calculate_all_KS()
census2000_fixed.calculate_all_BIC(st.poisson_hurdle_term)

with open("census2000_fixed.pkl", "wb") as file:
    pickle.dump(census2000_fixed.flow_matrices, file)
    pickle.dump(census2000_fixed.fit_info, file)
    pickle.dump(census2000_fixed.stats, file)

# mp.model_data(census2000_fixed.flow_matrices, data_flow_matrix)
# mp.error_rank(census2000_fixed.flow_matrices, data_flow_matrix)
# plt.show()
