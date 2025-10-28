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


chord = 2.63 #m
span_total = 11.302
span = span_total/8 #m
nodes_per_beam = 6
nodes = nodes_per_beam**2+3
elements_per_beam = nodes_per_beam - 1

initial_conditions=[]

#canopy discretisation
for i in range(nodes_per_beam):
    for j in range(nodes_per_beam):
        initial_conditions.append([[j*chord/(nodes_per_beam-1),i*span/(nodes_per_beam-1),0],[0,0,0],1,False])

#fixed coord
initial_conditions.append([[chord/2,span/2,-5],[0,0,0],1,True])

#leading edge bridle points
x_bridle = chord/2/5*1
y_bridle = span/2/5*1

initial_conditions.append([[x_bridle,y_bridle,-1],[0,0,0],1,False])
initial_conditions.append([[x_bridle,span-y_bridle,-1],[0,0,0],1,False])


#springs canopy
n1s_canopy_x = []
n2s_canopy_x = []
n1s_beam_x = []
n2s_beam_x = []
n1s_canopy_y = []
n2s_canopy_y = []
n1s_beam_y = []
n2s_beam_y = []
n1s_tether = []
n2s_tether = []

# Create connectivity for each node to its right and upper neighbor (no duplicates)

for i in range(nodes_per_beam):
    for j in range(nodes_per_beam):
        idx = j * nodes_per_beam + i
        # Connect to right neighbor
        if i < nodes_per_beam - 1:
            right_idx = j * nodes_per_beam + (i + 1)
            if j==0 or j==nodes_per_beam-1:
                n1s_beam_x.append(idx)
                n2s_beam_x.append(right_idx)
            else:
                n1s_canopy_x.append(idx)
                n2s_canopy_x.append(right_idx)
        # Connect to upper neighbor
        if j < nodes_per_beam - 1:
            up_idx = (j + 1) * nodes_per_beam + i
            if i == 0:
                n1s_beam_y.append(idx)
                n2s_beam_y.append(up_idx)
            else:
                n1s_canopy_y.append(idx)
                n2s_canopy_y.append(up_idx)
        #connect to bridle nodes
        if j==0 and (i==0 or i ==1):
            n1s_tether.append(nodes-2)
            n2s_tether.append(idx)
        if j==nodes_per_beam-1 and (i==0 or i ==1):
            n1s_tether.append(nodes-1)
            n2s_tether.append(idx)
        #connect to fixed node

        if (j == 0 or j==nodes_per_beam-1) and i==nodes_per_beam-1:
            n1s_tether.append(nodes-3)
            n2s_tether.append(idx)

#bridle nodes to fixed node
n1s_tether.append(nodes-2)
n2s_tether.append(nodes-3)
n1s_tether.append(nodes-1)
n2s_tether.append(nodes-3)

spring_matrix = []
beam_matrix = []
#springs canopy
l0_x = chord/elements_per_beam*1.0
l0_y = span/elements_per_beam*1.0
k = 50000
c = 0
for n1,n2 in zip(n1s_canopy_x,n2s_canopy_x):
    spring_matrix.append([n1,n2,k,c,l0_x,"noncompressive"])
for n1,n2 in zip(n1s_canopy_y,n2s_canopy_y):
    spring_matrix.append([n1,n2,k,c,l0_y,"noncompressive"])


#fictional springs constraining the canopy from collapse
# n1_fictional =[]
# n2_fictional = []
# for i in range(nodes_per_beam-1):
#     n1_fictional.append(nodes_per_beam-1-i)
#     n2_fictional.append(nodes_per_beam**2-1-i)

# for n1,n2 in zip(n1_fictional,n2_fictional):
#     spring_matrix.append([n1,n2,50000,c,span,"default"])

#springs to fixed node
l0 = 0
k = 50000

for n1,n2 in zip(n1s_tether,n2s_tether):
    spring_matrix.append([n1,n2,k,c,0,"default"])

#beam
t = 0.001
d = 0.18
r = d/2
p = 0.2
v= 0.03
L_y = span/elements_per_beam 
L_x = chord/elements_per_beam
EI_x = F_inflatablebeam(p,r,v)*chord/(3*v)
EI_y = F_inflatablebeam(p,r,v)*span/(3*v)
A = np.pi*(r**2 - (r-t)**2)  # 
I = (np.pi/4)*(r**4 - (r-t)**4)  # m^4
E_x = EI_x/I
E_y = EI_y/I


for n1,n2 in zip(n1s_beam_x,n2s_beam_x):
    beam_matrix.append([n1, n2, p, d])
for n1,n2 in zip(n1s_beam_y,n2s_beam_y):
    beam_matrix.append([n1, n2, p, d])

canopy = FEM_structure(initial_conditions=initial_conditions,spring_matrix=spring_matrix,beam_matrix=beam_matrix)

# #set legnths of noncompressive beams
coords_init = canopy.coords_init
for i in range(len(n1s_tether)):
    l0 = canopy.spring_elements[-(i+1)].unit_vector(coords_init)[1]
    canopy.spring_elements[-(i+1)].l0 = l0
    
fe = np.zeros(nodes*6)    

def create_uniform_load(nodes_per_beam, chord, span, total_force):
    nodes = nodes_per_beam ** 2 + 3
    fe = np.zeros(nodes * 6)
    sigma = 0.3 * chord  # width of the Gaussian peak

    for j in range(nodes_per_beam):
        for i in range(nodes_per_beam):
            idx = j * nodes_per_beam + i
            x = i * chord / (nodes_per_beam - 1)
            y = j * span / (nodes_per_beam - 1)
            # Gaussian weight along x, uniform along y
            weight_x = np.exp(-0.5 * ((x - 0.35 * chord) / sigma) ** 2) + 0.3

            # Angle from vertical: 0 at center, 30 deg at edges
            y_centered = (y - span / 2) / (span / 2)  # -1 to 1
            angle = y_centered * (np.pi / 4)  # -30deg to +30deg

            # Decompose force into z and y
            fe_z = weight_x * np.cos(angle)
            fe_y = weight_x * np.sin(angle)

            fe[idx * 6 + 2] = fe_z
            fe[idx * 6 + 1] = fe_y

    # Normalize so total force sums to total_force (z-component)
    total_weight = np.sum(fe[2::6])
    if total_weight > 0:
        scale = -total_force / total_weight  # negative z-direction
        fe[2::6] *= scale
        fe[1::6] *= scale

    return -fe

# Example usage:
total_force = 300  # N, adjust as needed
fe = create_uniform_load(nodes_per_beam, chord, span, total_force)
# ax,fig = canopy.plot_3D(show_plot=False)
canopy.plot_3D(fe=fe, plot_forces_displacements=True,show_plot=False,plot_nodes=False)
canopy.solve(fe=fe,max_iterations = 1000 ,I_stiffness=15,tolerance=5,relax_init=0.5,step_limit = 0.1)
canopy.plot_convergence()
canopy.plot_3D(plot_nodes=False,plot_forces_displacements=True)
