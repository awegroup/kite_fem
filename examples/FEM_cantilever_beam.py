import matplotlib.pyplot as plt
import numpy as np
from kite_fem.FEMStructure import FEM_structure

initital_conditions = []
length = 10 #m
elements = 2

for i in range(elements+1):
    initital_conditions.append([[i*length/elements, 0.0, 0.0], [0, 0, 0], 1, True if i==0 else False])


load_param = 1.5#
E = 210e5 # Pa
I = 1.6e-5  # m^4
A = 1 # m^2
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
ax,fig = steel_beam.plot_3D(color= "blue",show_plot=False)

steel_beam.reset()

steel_beam.solve(        fe=fe,
        max_iterations=2000,
        tolerance=0.1,
        step_limit=2,
        relax_init=1,
        relax_update=0.95,
        k_update=1,
        I_stiffness=25
        )

ax,fig = steel_beam.plot_3D(ax=ax, fig=fig, color="red",plot_forces_displacements=False,show_plot=False)
# print(np.array(steel_beam.beam_elements[0].beam.probe.ue))
# print(steel_beam.residual_norm_history[-1])
steel_beam.reinitialise()
# print(steel_beam.fi)
steel_beam.reset()
# print(steel_beam.residual_norm_history[0])

# steel_beam.reset()

steel_beam.solve(   fe=fe ,   
        max_iterations=2000,
        tolerance=0.1,
        step_limit=2,
        relax_init=1,
        relax_update=0.95,
        k_update=1,
        I_stiffness=25
        )


ax,fig = steel_beam.plot_3D(ax=ax, fig=fig, color="green",plot_forces_displacements=False,show_plot=False)


print("Tip displacements")
print("load_param", load_param)
print("w/L", -steel_beam.coords_rotations_current[1::6][-1]/length)
print("u/L", 1-steel_beam.coords_rotations_current[0::6][-1]/length)
print("theta", -steel_beam.coords_rotations_current[-1])
plt.show()
