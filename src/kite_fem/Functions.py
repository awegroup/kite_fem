from kite_fem.FEMStructure import FEM_structure
import numpy as np

def tensionbridles(kite: FEM_structure, canopy_nodes,offset,scale):
    initial_conditions= kite.initial_conditions
    pulley_matrix = kite.pulley_matrix
    spring_matrix= kite.spring_matrix
    beam_matrix = kite.beam_matrix
    initial_conditions_temp = initial_conditions.copy()
    height = np.max(kite.coords_init[2::6])
    for id, (pos,vel,mass,fixed) in enumerate(initial_conditions):
        if id in canopy_nodes:
            newpos = pos + [0,0,(scale-1)*height+offset]
            initial_conditions_temp[id] = (newpos, vel, mass, fixed)
        elif id and id != 0:
            newpos = pos + [0,0,pos[2]*(scale-1)+offset]
            initial_conditions_temp[id] = (newpos, vel, mass, fixed)

    kite = FEM_structure(initial_conditions_temp,spring_matrix,pulley_matrix,beam_matrix)
    return kite

# def distributekitemass()
