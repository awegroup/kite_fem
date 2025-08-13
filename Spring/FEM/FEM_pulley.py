import matplotlib.pyplot as plt
import numpy as np
from FEMStructure import FEM_structure

# Define initial conditions and connectivity matrix
# initial_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, -1.5, 0.0], [0, 0, 0], 1, False],[[2.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, -2.5, 0.0], [0, 0, 0], 1, False]] #[position, velocity (not used), mass (not used),fixed]
# # connectivity_matrix = [[0, 1, 100, 1, 5, 'pulley'], [1, 2, 100, 1, 5, 'pulley']] #[node1, node2, spring constant, damping constant (not used), initial length, springtype]
# # connectivity_matrix2 = [[0, 1, 1000, 1, 1, 'default'], [1, 2, 1000, 1, 1, 'default']] #[node1, node2, spring constant, damping constant (not used), initial length, springtype]
# pulley_matrix = [[0,1,2,100,0,5]]
# spring_matrix = [[1,3,100,0,0.8,"default"]]
# Pulley = FEM_structure(initial_conditions, spring_matrix=spring_matrix, pulley_matrix=pulley_matrix)
# fext = np.zeros(Pulley.N)
# fext[3*6+1] = -1
# fext[3*6] = 10
# ax1,fig1 = Pulley.plot_3D(color='blue', plot_forces_displacements=True, fe = fext)
# ax1.legend()
# Pulley.solve(fe = fext, tolerance=1e-2, max_iterations=500, limit_init=0.2, relax_init=0.8,relax_update=0.95, k_update=1)
# ax2,fig2 = Pulley.plot_3D(color='red', plot_forces_displacements=True)
# ax3,fig3 = Pulley.plot_convergence()


# initial_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True],[[0.1, -2, 0.0], [0, 0, 0], 1, False]]
# spring_matrix = [[0,1,100,0,1,"default"]]
# spring = FEM_structure(initial_conditions, spring_matrix=spring_matrix)
# fe = np.zeros(spring.N)
# fe[1*6+1] = -10  # Force applied at node 1 in y-direction
# fe[1*6] = 10  # Force applied at node 1 in x-direction
# spring.solve(fe = fe, tolerance=1e-2, max_iterations=100, limit_init=0.2, relax_init=0.5,relax_update=0.95, k_update=10)
# ax1,fig1 = spring.plot_3D(color='blue', plot_forces_displacements=True, fe=fe)
# ax2,fig2 = spring.plot_convergence()
# ax1.legend()
# plt.show()


initial_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, 0.0, 0.0], [0, 0, 0], 1, True],[[2.0, 0.0, 0.0], [0, 0, 0], 1, True],[[3.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, -1.0, 0.0], [0, 0, 0], 1, False],[[2.0, -1.0, 0.0], [0, 0, 0], 1, False],[[1.5, -2.0, 0.0], [0, 0, 0], 1, False]]
l0 = 3
pulley_matrix = [[0,4,1,1000,0,l0],[2,5,3,1000,0,l0],[4,6,5,1000,0,l0]]
# spring_matrix = [[6,7,100,0,0.8,"noncompressive"]]
Pulleys = FEM_structure(initial_conditions, pulley_matrix = pulley_matrix)
fe = np.zeros(Pulleys.N)
fe[(Pulleys.num_nodes-1)*6+1] = -100
fe[(Pulleys.num_nodes-1)*6] = 50

ax1,fig1 = Pulleys.plot_3D(color='blue', plot_forces_displacements=True, fe = fe)
Pulleys.solve(fe = fe, tolerance=1e-3, max_iterations=5000, limit_init=0.3, relax_init=0.5,relax_update=0.95, k_update=1)
ax2,fig2 = Pulleys.plot_3D(color='blue', plot_forces_displacements=True)
ax3,fig3 = Pulleys.plot_convergence()

ax2.set_xlim([-.5,3.5])  # Equal aspect ratio for x, y, z
ax2.set_ylim([0.5, -3.5])
ax2.set_zlim([-1, 1])
plt.show()