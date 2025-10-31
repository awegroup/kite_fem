import matplotlib.pyplot as plt
import numpy as np
from kite_fem.FEMStructure import FEM_structure
from kite_fem.Plotting import plot_structure

# Define initial conditions and connectivity matrix
initial_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, 0.0, 0.0], [0, 0, 0], 1, False],[[2.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, 10, 0.0], [0, 0, 0], 1, True]] #[position, velocity (not used), mass (not used),fixed]
connectivity_matrix = [[0, 1, 1, 1, 0, 'default'], [1, 2, 1, 1, 0, 'default'], [1, 3, 1, 1, 10, 'noncompressive']] #[node1, node2, spring constant, damping constant (not used), initial length, springtype]
fe = np.zeros(len(initial_conditions)*6, dtype=float)
fe = np.zeros(len(initial_conditions)*6, dtype=float)
fe[7] = 10  # Force applied at node 2 in y-direction

# Create FEM structure and solve
noncompressive = FEM_structure(initial_conditions, connectivity_matrix)

ax1, fig1 = plot_structure(noncompressive, fe=fe, fe_magnitude=3)
ax2, fig2 = plot_structure(noncompressive, fe=-fe,fe_magnitude=3)

noncompressive.solve(fe = fe, tolerance=1e-2, max_iterations=300, step_limit=0.2, relax_init=0.5,relax_update=0.95, k_update=1)
ax3, fig3 = plot_structure(noncompressive, fe=fe,fe_magnitude=3)

noncompressive.solve(fe = -fe, tolerance=1e-2, max_iterations=300, step_limit=0.2, relax_init=0.5,relax_update=0.95, k_update=1)
ax4, fig4 = plot_structure(noncompressive,fe=-fe,fe_magnitude=3)

# Plot the results
ax1.set_title("initial, force applied such that noncompressive spring is slack")
ax2.set_title("initial, force applied such that noncompressive spring is tensioned")
ax3.set_title("final, force applied such that noncompressive spring is slack")
ax4.set_title("final, force applied such that noncompressive spring is tensioned")
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.show()