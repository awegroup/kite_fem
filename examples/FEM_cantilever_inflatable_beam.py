import matplotlib.pyplot as plt
import numpy as np
from kite_fem.FEMStructure import FEM_structure


def instiantiate(d,p):
    length  = 1  
    elements = 1
    initital_conditions = []
    for i in range(elements+1):
        initital_conditions.append([[i*length/elements, 0.0, 0.0], [0, 0, 0], 1, True if i==0 else False])
    beam_matrix = []
    for i in range(elements):
        beam_matrix.append([i, i+1, d, p])
    inflatable_beam = FEM_structure(initital_conditions, beam_matrix=beam_matrix)
    return inflatable_beam

def solve_tip_load(inflatable_beam,tip_load):
    fe = np.zeros(inflatable_beam.N)
    fe[1::6][-1] = -tip_load
    inflatable_beam.solve(        fe=fe,
            max_iterations=1000,
            tolerance=0.00001,
            step_limit=0.5,
            relax_init=1,
            relax_update=0.95,
            k_update=1,
            I_stiffness=25
            )
    deflection = -inflatable_beam.coords_current[-2]*1000
    return deflection


def solve_tip_moment(inflatable_beam,tip_moment):
    fe = np.zeros(inflatable_beam.N)
    fe[3::6][-1] = -tip_moment
    inflatable_beam.solve(        fe=fe,
            max_iterations=1000,
            tolerance=0.00001,
            step_limit=0.5,
            relax_init=1,
            relax_update=0.95,
            k_update=1,
            I_stiffness=25
            )
    rotation = -np.rad2deg(inflatable_beam.coords_rotations_current[-3])
    return rotation

pressures = [0.3,0.4,0.5]
diameters = [0.13,0.13,0.13]
inflatable_beams = []

for pressure,diameter in zip(pressures,diameters):
    inflatable_beam = instiantiate(diameter,pressure)
    inflatable_beams.append(inflatable_beam)

tip_loads = np.arange(5,35,10)
tip_moments = np.arange(0,60,10)
fig, ax = plt.subplots()
for inflatable_beam in inflatable_beams:
    deflections = []
    rotations = []
    for tip_load in tip_loads:
        deflection = solve_tip_load(inflatable_beam,tip_load)
        deflections.append(deflection)
    for tip_moment in tip_moments:
        rotation = solve_tip_moment(inflatable_beam,tip_moment)
        rotations.append(rotation)
    ax.scatter(deflections,tip_loads,marker="+")
    ax.scatter(rotations,tip_moments,marker="+")





plt.show()
    