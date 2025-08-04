import matplotlib.pyplot as plt
import numpy as np
from FEMStructure import FEM_structure

# Define initial conditions and connectivity matrix
initial_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, -1.0, 0.0], [0, 0, 0], 1, False],[[2.0, 0.0, 0.0], [0, 0, 0], 1, True]] #[position, velocity (not used), mass (not used),fixed]
connectivity_matrix = [[0, 1, 100, 1, 5, 'pulley'], [1, 2, 100, 1, 5, 'pulley']] #[node1, node2, spring constant, damping constant (not used), initial length, springtype]
# connectivity_matrix2 = [[0, 1, 1000, 1, 1, 'default'], [1, 2, 1000, 1, 1, 'default']] #[node1, node2, spring constant, damping constant (not used), initial length, springtype]

Pulley = FEM_structure(initial_conditions, connectivity_matrix)
fext = np.zeros(Pulley.N)
fext[6+1] = -1
fext[6] = 1
ax1,fig1 = Pulley.plot_3D(color='blue', plot_forces_displacements=True, fe = None)

Pulley.solve(fe = fext, tolerance=1e-2, max_iterations=50, limit_init=0.2, relax_init=0.8,relax_update=0.95, k_update=1)
ax2,fig2 = Pulley.plot_3D(color='red')
ax3,fig3 = Pulley.plot_convergence()



ncoords = Pulley.ncoords_current
x = ncoords[0::3]
y = ncoords[1::3]
z = ncoords[2::3]
elements = Pulley.spring_elements
l1 = elements[0].unit_vector(ncoords)[1]
l2 = elements[1].unit_vector(ncoords)[1]
ltotal = l1 + l2
print(ltotal)
print(Pulley.fi)
for i in range(3):
    print(x[i], y[i], z[i])

# value = -1
# for element in elements:
#     fi = element.spring_internal_forces(ncoords)
#     ax1.plot([x,x+fi[0]*value],[y,y+fi[1]*value],[z,z+fi[2]*value],color='green')
#     value*= -1

# # fint = np.zeros(Pulley.N)


# # u = ncoords

# # for i in range(Pulley.num_nodes):
# #     u = np.insert(u, (Pulley.num_nodes-i)*3, [0,0,0])
# # L=2



plt.show()


# element1 = elements[0]
# element2 = elements[1]

# def fint_pulley(ncoords, element1, element2):
#     unit_vect, l1 = element1.unit_vector(ncoords)
#     unit_vect2, l2 = element2.unit_vector(ncoords)
#     l0_pulley = element1.l0 + element2.l0
#     element1.l0 = l1 / (l1 + l2) * l0_pulley
#     element2.l0 = l2 / (l1 + l2) * l0_pulley
#     F11 = (l1-element1.l0) * element1.k
#     F22 = (l2-element2.l0) * element2.k
#     print(F1,F11)
#     print(F2,F22)


    
#     # fi1 = F1 * unit_vect
#     # fi2 = F2 * unit_vect2

#     # fi1 = np.append(fi1, [0, 0, 0])  # Append zeros for the z-component
#     # fi2 = np.append(fi2, [0, 0, 0])  # Append zeros for the z-component
    
    

# fint_pulley(ncoords_init, element1, element2)