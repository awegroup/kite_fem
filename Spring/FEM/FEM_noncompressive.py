import matplotlib.pyplot as plt
import numpy as np
from FEMStructure import FEM_structure

# Define initial conditions and connectivity matrix
initial_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, 0.0, 0.0], [0, 0, 0], 1, False],[[2.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, 10, 0.0], [0, 0, 0], 1, True]]
connectivity_matrix = [[0, 1, 1, 1, 0, 'default'], [1, 2, 1, 1, 0, 'default'], [1, 3, 1, 1, 10, 'noncompressive']]
fe = np.zeros(len(initial_conditions)*6, dtype=float)
fe[7] = 10  # Force applied at node 2 in y-direction

# Create FEM structure and solve
noncompressive = FEM_structure(initial_conditions, connectivity_matrix)
ax1, fig1 = noncompressive.plot_3D(color='red', plot_forces_displacements=True, fe=fe)
ax2, fig2 = noncompressive.plot_3D(color='blue', plot_forces_displacements=True, fe=-fe)

noncompressive.solve(fe = fe, tolerance=1e-2, max_iterations=50, limit_init=0.2, relax_init=0.5,relax_update=0.95, k_update=1)
ax3, fig3 = noncompressive.plot_3D(color='red')
noncompressive.solve(fe = -fe, tolerance=1e-2, max_iterations=50, limit_init=0.2, relax_init=0.5,relax_update=0.95, k_update=1)
ax4, fig4 = noncompressive.plot_3D(color='blue')
# Plot the results
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax1.set_title("initial, force applied such that noncompressive spring is slack")
ax2.set_title("initial, force applied such that noncompressive spring is tensioned")
ax3.set_title("final, force applied such that noncompressive spring is slack")
ax4.set_title("final, force applied such that noncompressive spring is tensioned")
plt.show()