import matplotlib.pyplot as plt
import numpy as np
from kite_fem.FEMStructure import FEM_structure

initital_conditions = []
length = 10 #m
elements = 3

for i in range(elements+1):
    initital_conditions.append([[i*length/elements, 0.0, 0.0], [0, 0, 0], 1, True if i==0 else False])

load_param = 1#
E = 210e5 # Pa
I = 1.6e-5  # m^4
A = 0.1  # m^2
print("E*I", E*I)
L = length/elements #m

beam_matrix = []

for i in range(elements):
    beam_matrix.append([i, i+1, E, A, I])

steel_beam = FEM_structure(initital_conditions, beam_matrix=beam_matrix)
tip_load = load_param*E*I/(length**2)
fe = np.zeros(steel_beam.N)
print('tip_load', tip_load, "N")
fe[1::6][-1] = -tip_load
ax,fig = steel_beam.plot_3D(color= "blue")


steel_beam.solve(        fe=fe,
        max_iterations=1000,
        tolerance=1e-1,
        step_limit=0.5,
        relax_init=0.25,
        relax_update=0.95,
        k_update=1,
        I_stiffness=25
        )

ax,fig = steel_beam.plot_3D(ax=ax, fig=fig, color="red",plot_forces_displacements=False)
ax2,fig2 = steel_beam.plot_convergence()
ax.set_xlim([0, 10])
ax.set_ylim([-10, 0])
ax.set_zlim([-5, 5])


print("load_param", load_param)
print("w/L", -steel_beam.coords_rotations_current[1::6][-1]/length)
print("u/L", 1-steel_beam.coords_rotations_current[0::6][-1]/length)
print("theta", -steel_beam.coords_rotations_current[-1])
plt.show()
# k = steel_beam.KC0
# bu = steel_beam.bu
# k = k[bu, :][:, bu]
# f = fe[bu]
# u = np.zeros(steel_beam.N)
# u[bu] = np.linalg.solve(k.toarray(), f)
# print(u)