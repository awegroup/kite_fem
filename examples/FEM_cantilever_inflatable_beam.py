import matplotlib.pyplot as plt
import numpy as np
from kite_fem.FEMStructure import FEM_structure

def F_inflatablebeam(p, r, v):
    # Coefficients
    C1 = 6582.82
    C2 = -272.43
    C3 = 40852.38
    C4 = 14.31
    C5 = 271865251.42
    C6 = 215.93
    C7 = 14021.79
    C8 = -589.05
    
    # Numerator and denominator
    denom = (C1 * r + C2) * p**2 + (C3 * r**3 + C4)
    numer = (C5 * r**5 + C6) * p + (C7 * r + C8)
    
    # Formula
    result = denom * (1 - np.exp(-(numer / denom) * v))
    return result


initital_conditions = []
length = 10 #m
elements = 5


for i in range(elements+1):
    initital_conditions.append([[i*length/elements, 0.0, 0.0], [0, 0, 0], 1, True if i==0 else False])

load_param = 1.5#
t = 0.001
d = 0.18
r = d/2
p = 0.5
v= 0.03
L = length/elements #m

EI = F_inflatablebeam(p,r,v)*length/(3*v)
A = np.pi*(r**2 - (r-t)**2)  # m^2
I = (np.pi/4)*(r**4 - (r-t)**4)  # m^4
E = EI/I

print(E)

print("E*I", E*I)

beam_matrix = []

for i in range(elements):
    beam_matrix.append([i, i+1, E, A, I])

steel_beam = FEM_structure(initital_conditions, beam_matrix=beam_matrix)

tip_load = load_param*E*I/(length**2)
fe = np.zeros(steel_beam.N)
print('tip_load', tip_load, "N")
fe[1::6][-1] = -tip_load
ax,fig = steel_beam.plot_3D(color= "blue",show_plot=False)


steel_beam.solve(        fe=fe,
        max_iterations=1000,
        tolerance=0.1,
        step_limit=2,
        relax_init=1,
        relax_update=0.95,
        k_update=1,
        I_stiffness=25
        )

ax,fig = steel_beam.plot_3D(ax=ax, fig=fig, color="red",plot_forces_displacements=False,show_plot=False)
ax2,fig2 = steel_beam.plot_convergence()


print("Tip displacements")
print("load_param", load_param)
print("w/L", -steel_beam.coords_rotations_current[1::6][-1]/length)
print("u/L", 1-steel_beam.coords_rotations_current[0::6][-1]/length)
print("theta", -steel_beam.coords_rotations_current[-1])
plt.show()
