import matplotlib.pyplot as plt
import numpy as np
from FEMStructure import FEM_structure

# Define initial conditions and connectivity matrix
# initial_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, -1.0, 0.0], [0, 0, 0], 1, False],[[2.0, 0.0, 0.0], [0, 0, 0], 1, True]] #[position, velocity (not used), mass (not used),fixed]
# connectivity_matrix = [[0, 1, 100, 1, 5, 'pulley'], [1, 2, 100, 1, 5, 'pulley']] #[node1, node2, spring constant, damping constant (not used), initial length, springtype]
# # connectivity_matrix2 = [[0, 1, 1000, 1, 1, 'default'], [1, 2, 1000, 1, 1, 'default']] #[node1, node2, spring constant, damping constant (not used), initial length, springtype]


# Pulley = FEM_structure(initial_conditions, connectivity_matrix)
# fext = np.zeros(Pulley.N)
# fext[6+1] = -1
# fext[6] = 1
# ax1,fig1 = Pulley.plot_3D(color='blue', plot_forces_displacements=True, fe = fext)

# Pulley.solve(fe = fext, tolerance=1e-2, max_iterations=50, limit_init=0.2, relax_init=0.8,relax_update=0.95, k_update=1)
# ax2,fig2 = Pulley.plot_3D(color='red', plot_forces_displacements=True)
# ax3,fig3 = Pulley.plot_convergence()


initial_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, 0.0, 0.0], [0, 0, 0], 1, True],[[2.0, 0.0, 0.0], [0, 0, 0], 1, True],[[3.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, -1.0, 0.0], [0, 0, 0], 1, False],[[2.0, -1.0, 0.0], [0, 0, 0], 1, False],[[1.5, -2.0, 0.0], [0, 0, 0], 1, False]]
l0 = 3
connectivity_matrix = [[0,4,1,100,0,l0],[2,5,3,100,0,l0],[4,6,5,100,0,l0]]

Pulleys = FEM_structure(initial_conditions, pulley_matrix=connectivity_matrix)
fe = np.zeros(Pulleys.N)
fe[(Pulleys.num_nodes-1)*6+1] = -1
ax1,fig1 = Pulleys.plot_3D(color='blue', plot_forces_displacements=True, fe = fe)
Pulleys.solve(fe = fe, tolerance=1e-3, max_iterations=50, limit_init=0.05, relax_init=0.8,relax_update=0.95, k_update=1)
ax2,fig2 = Pulleys.plot_3D(color='red', plot_forces_displacements=True)
plt.legend()
plt.show()