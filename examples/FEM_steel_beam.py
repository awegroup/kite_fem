import matplotlib.pyplot as plt
import numpy as np
from kite_fem.FEMStructure import FEM_structure

initital_conditions = []
length = 10 #m
elements =6

for i in range(elements+1):
    initital_conditions.append([[i*length/elements, 0.0, 0.0], [0, 0, 0], 1, True if i==0 else False])

load_param = 1#
E = 210e9 # Pa
I = 1.6e-5  # m^4
A = 0.01  # m^2
L = length/elements #m

beam_matrix = []

for i in range(elements):
    beam_matrix.append([i, i+1, E, A, I])

steel_beam = FEM_structure(initital_conditions, beam_matrix=beam_matrix)

fe = np.zeros(steel_beam.N)
tip_load = load_param*E*I/(length**2)
print('tip_load', tip_load, "N")
fe[1::6][-1] = -tip_load
ax,fig = steel_beam.plot_3D(color= "blue")


steel_beam.solve(        fe=fe,
        max_iterations=1,
        tolerance=1e-2,
        step_limit=0.05,
        relax_init=0.05,
        relax_update=0.95,
        k_update=1,
        I_stiffness=0
        )

ax,fig = steel_beam.plot_3D(ax=ax, fig=fig, color="red")
ax2,fig2 = steel_beam.plot_convergence()
plt.show()

print("fint", steel_beam.fi)
print("fext", steel_beam.fe)

print("bu", steel_beam.bu)

print(np.linalg.norm((steel_beam.fi-steel_beam.fe)[steel_beam.bu]))