import numpy as np
import os
# Change to the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Import each row of model_results.csv as arrays
model_results_data = np.loadtxt('model_results.csv', delimiter=',', skiprows=1)[:,1:]
tolerance = model_results_data[:, -2]
model_results_data = model_results_data[:, :-2]



# Import each row of validation_data.csv as arrays
validation_data = np.loadtxt('validation_data.csv', delimiter=',', skiprows=1)[:,1:]

data_header = np.loadtxt('validation_data.csv', delimiter=',', max_rows=1, dtype=str)

def shapecorrelation(phi_exp,phi_mod):
    phi_exp = np.asarray(phi_exp).reshape(-1, 1)
    phi_mod = np.asarray(phi_mod).reshape(-1, 1)
    numerator = (phi_exp.T @ phi_mod)[0, 0] ** 2
    denominator = (phi_exp.T @ phi_exp)[0, 0] * (phi_mod.T @ phi_mod)[0, 0]
    return numerator / denominator

for test_case in range(1,11):
    results = model_results_data[test_case-1]
    validation = validation_data[test_case-1]
    SC = shapecorrelation(results,validation)
    meanabsolutedeviation = 1/len(results) * np.sum(np.abs(results-validation))
    print(f"Test case {test_case}: SC = {SC:.3f},Mean absolute deviation = {meanabsolutedeviation:.3f}, Tolerance = {tolerance[test_case-1]:.3f}")