import matplotlib.pyplot as plt
import numpy as np
from kite_fem.FEMStructure import FEM_structure
from kite_fem.Plotting import plot_structure, plot_convergence
# Define initial conditions and connectivity matrix
initial_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, 0.0, 0.0], [0, 0, 0], 1, True],[[2.0, 0.0, 0.0], [0, 0, 0], 1, True],[[3.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, -1.0, 0.0], [0, 0, 0], 1, False],[[2.0, -1.0, 0.0], [0, 0, 0], 1, False],[[1.5, -2.0, 0.0], [0, 0, 0], 1, False]]
l0 = 3
pulley_matrix = [[0,4,1,1000,0,l0],[2,5,3,1000,0,l0],[4,6,5,1000,0,l0]]

Pulleys = FEM_structure(initial_conditions, pulley_matrix = pulley_matrix)
fe = np.zeros(Pulleys.N)
fe[(Pulleys.num_nodes-1)*6+1] = -100
fe[(Pulleys.num_nodes-1)*6] = 50

ax1,fig1 = plot_structure(Pulleys,fe=fe, plot_displacements=True,fe_magnitude=0.35,plot_2d=True)
Pulleys.solve(fe = fe, tolerance=1e-3, max_iterations=5000, step_limit=0.3, relax_init=0.5,relax_update=0.95, k_update=1)
ax2,fig2 = plot_structure(Pulleys, fe=fe, fe_magnitude=0.35,plot_2d=True)
ax3,fig3 = plot_convergence(Pulleys)

ax1.set_title("Initial Configuration")
ax2.set_title("Deformed Configuration")
ax3.set_title("Convergence History")
ax1.legend()
ax2.legend()
plt.show()
