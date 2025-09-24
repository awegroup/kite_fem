import matplotlib.pyplot as plt
import numpy as np
from kite_fem.FEMStructure import FEM_structure

# Define initial conditions and connectivity matrix
initial_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, 0.0, 0.0], [0, 0, 0], 1, True],[[2.0, 0.0, 0.0], [0, 0, 0], 1, True],[[3.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, -1.0, 0.0], [0, 0, 0], 1, False],[[2.0, -1.0, 0.0], [0, 0, 0], 1, False],[[1.5, -2.0, 0.0], [0, 0, 0], 1, False]]
l0 = 3
pulley_matrix = [[0,4,1,1000,0,l0],[2,5,3,1000,0,l0],[4,6,5,1000,0,l0]]

Pulleys = FEM_structure(initial_conditions, pulley_matrix = pulley_matrix)
fe = np.zeros(Pulleys.N)
fe[(Pulleys.num_nodes-1)*6+1] = -100
fe[(Pulleys.num_nodes-1)*6] = 50

ax1,fig1 = Pulleys.plot_3D(color='blue', plot_forces_displacements=True, fe = fe)
Pulleys.solve(fe = fe, tolerance=1e-3, max_iterations=5000, step_limit=0.3, relax_init=0.5,relax_update=0.95, k_update=1)
ax2,fig2 = Pulleys.plot_3D(color='blue', plot_forces_displacements=True)
ax3,fig3 = Pulleys.plot_convergence()
