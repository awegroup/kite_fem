import matplotlib.pyplot as plt
import numpy as np
from kite_fem.FEMStructure import FEM_structure

# Define initial conditions and connectivity matrix
initial_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, 0.0, 0.0], [0, 0, 0], 1, False],[[2.0, 0.0, 0.0], [0, 0, 0], 1, True]] #[position, velocity (not used), mass (not used),fixed]
connectivity_matrix = [[0, 1, 1, 1, 0, 'default'], [1, 2, 1, 1, 0, 'default']] #[node1, node2, spring constant, damping constant (not used), initial length, springtype]
fe = np.zeros(len(initial_conditions)*6, dtype=float)
fe[7] = 10  # Force applied at node 2 in y-direction

# Create FEM structure and solve
noncompressive = FEM_structure(initial_conditions, connectivity_matrix)
ax1, fig1 = noncompressive.plot_3D(color='red', plot_forces_displacements=True, fe=fe)
noncompressive.solve(fe = fe, tolerance=1e-2, max_iterations=500, step_limit=0.2, relax_init=0.5,relax_update=0.95, k_update=1)
ax2, fig2 = noncompressive.plot_3D(color='blue')

# Plot the results
ax1.legend()
ax2.legend()
ax1.set_title("initial")
ax2.set_title("final")
plt.show()