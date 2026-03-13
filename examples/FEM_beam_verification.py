import numpy as np
import matplotlib.pyplot as plt
from kite_fem.FEMStructure import FEM_structure
#bending
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

def v_collapse(p,r):
    # Coefficients
    C9 = 322.55
    C10 = 0.0239
    C11 = 5.3833
    C12 = 0.0461
    v = (C9*r**4+C10)*p+C11*r**2+C12
    return v

def torsion(p,r,phi):
    C13 = 1467
    C14 = 40.908
    C15 = -191.8
    C16 = 47.406
    C17 = -17703
    C18 = 358.05
    C19 = 0.0918
    c1 = ((C13*r+C14)*p+(C15*r+C16))
    c2 = ((C17*r**4)*np.log(p)+(C18*r**3+C19))
    T = c1*np.arctan(c2*np.deg2rad(phi))
    return T

def plotinflatablebeam(p,d,ls,ax):
    r = d/2
    v_max = v_collapse(p,r)
    v = np.linspace(0, v_max, 100)
    v_step = v[1] - v[0]
    F = F_inflatablebeam(p,r,v)
    v = np.append(v, v_max+v_step)
    F = np.append(F, 0)
    ax.plot(v*1000,F,color="black",linestyle=ls,linewidth=1.5)
    phi_max = 140
    phi = np.linspace(0, phi_max, 100)
    T = torsion(p, r, phi)
    ax.plot(phi, T,color="red",linestyle=ls,linewidth=1.5)
    return ax

p_lst = [0.3,0.5] #bar
d_lst = [0.16,0.16] #m
linestyles = ['-', '--','-','--']
fig, ax = plt.subplots(figsize=(4,4))


for p,d,linestyle in zip(p_lst,d_lst,linestyles):
    ax = plotinflatablebeam(p,d,linestyle,ax)


for mode in ["bending","torsion"]:
    if mode == "bending":
        color = "black"
    else:
        color = "red"
    for p,d,linestyle in zip(p_lst,d_lst,linestyles):
        plt.plot([], [], color=color, linestyle=linestyle, label=f"p={p}bar, d={d*100}cm, {mode}")



###Kitefem code
# Setup a 1 meter inflatable beam with properties from p_lst and d_lst
# Plot Tip load - Tip deflection and Tip moment - Tip rotation curves

def setup_system(d,p):
    initial_conditions = [[[0,0,0],[0,0,0],1,True],[[1,0,0],[0,0,0],1,False]]
    beam_matrix = [[0,1,d, p, 1]]
    beam = FEM_structure(initial_conditions=initial_conditions,beam_matrix=beam_matrix)
    return beam

beams = []
for p in p_lst:
    for d in d_lst:
        beam = setup_system(d,p)
        beams.append(beam)

##Set up force vector
## do a range of tip loads
## also do a range of tip moments

#solve all beams for tip loads and also seperately for tip moments (reset system inbetween)

#plot all results on plot