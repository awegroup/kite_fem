import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from pyfe3d import Spring, SpringData, SpringProbe, DOF, INT, DOUBLE
import matplotlib.pyplot as plt

def smart_inverse(A, threshold=1e10):
    cond_number = np.linalg.cond(A)
    if cond_number < threshold:
        return np.linalg.inv(A)
    else:
        return np.linalg.pinv(A)

# Define spring properties
k = 1  # Stiffness of the first spring in N/m
F = 10   # Force in N
L = 1  # Length of each spring in m

# Node coordinates
ncoords = np.array([[0.0, 0.0, 0.0],  # Node at x = 0 m
                    [1, .1, 0.0], # Node at x = 1 m
                    [2, 0, 0.0]]) 

nids = np.array([0, 1, 2])  # Node IDs
nid_pos = {nid: i for i, nid in enumerate(nids)}
n1s = np.array([0, 1])
n2s = np.array([1, 2])
num_elements = len(nids)-1
ncoords_init = ncoords.flatten()  # Flattened coordinates for probe updates

# Initialize spring data and probe
springdata = SpringData()
springprobe = SpringProbe()

# Initialize global stiffness matrix arrays
KC0r = np.zeros(springdata.KC0_SPARSE_SIZE * num_elements, dtype=INT)  # Two springs
KC0c = np.zeros(springdata.KC0_SPARSE_SIZE * num_elements, dtype=INT)
KC0v = np.zeros(springdata.KC0_SPARSE_SIZE * num_elements, dtype=DOUBLE)
N = DOF * len(nids)  # Total DOFs


init_k_KC0 = 0
springs = []
for n1, n2 in zip(n1s, n2s):
        probe = SpringProbe()
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        spring = Spring(probe)
        spring.init_k_KC0 = init_k_KC0
        spring.n1 = n1
        spring.n2 = n2
        spring.c1 = DOF * pos1
        spring.c2 = DOF * pos2
        spring.kxe =  k # Spring stiffness
        xi = ncoords_init[n2*3] - ncoords_init[n1*3] 
        xj = ncoords_init[n2*3+1] - ncoords_init[n1*3+1]
        xk = ncoords_init[n2*3+2] - ncoords_init[n1*3+2]
        tmp = (xi**2 + xj**2 + xk**2)**0.5
        xi /= tmp
        xj /= tmp
        xk /= tmp
        spring.update_rotation_matrix(ncoords_init)
        print("xi, xj, xk:", xi, xj, xk)
        spring.update_KC0(KC0r, KC0c, KC0v)
        springs.append(spring)
        init_k_KC0 += springdata.KC0_SPARSE_SIZE

KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

bk = np.zeros(N, dtype=bool)
bk[0:DOF] = True  # Fix the first node (x = 0 m)
bk[DOF+2:DOF*2] = True  # Fix the z and rotational DOF of the second node (x = 1 m)
bk[DOF*2:DOF*3] = True  # Fix the third node (x = 2 m)
bu = ~bk

print("bu:", bu)
f = np.zeros(N)

f[DOF+1] = F    # Node 2, y

KC0uu = KC0[bu, :][:, bu]
fu = f[bu]
KC0uu = KC0uu.toarray()



print("KC0uu",KC0)

KC0uu_inv = smart_inverse(KC0uu)

uu = KC0uu_inv @ fu  # Solve for displacements
u = np.zeros(N)
u[bu] = uu  # Assign displacements to free DOFs

# Define tolerance and maximum iterations
tolerance = 1e-8
max_iterations = 100

# Initialize displacement vector
u = np.zeros(N)
uu = np.zeros_like(fu)

xyz = np.zeros_like(u, dtype=bool)
xyz[0::DOF] = xyz[1::DOF] = xyz[2::DOF] = True  
ncoords_current = ncoords_init.copy()

print("Initial Node Coordinates:", ncoords_init)

# Iterative solver
for iteration in range(max_iterations):
    # Compute the residual
    residual = fu - (KC0uu @ uu)

    # Check if the residual is below the tolerance
    residual_norm = np.linalg.norm(residual)
    if residual_norm < tolerance:
        print("Converged!")
        break

    # Update displacements 
    uu += smart_inverse(KC0uu) @ residual 
    
    u = np.zeros(N)
    u[bu] = uu  # Assign displacements to free DOFs
    print("Current Node Coordinates:", ncoords_current)

    ncoords_current = ncoords_init + u[xyz]
    
    KC0v *= 0
    for spring in springs:
        xi = ncoords_current[n2*3] - ncoords_current[n1*3] 
        xj = ncoords_current[n2*3+1] - ncoords_current[n1*3+1]
        xk = ncoords_current[n2*3+2] - ncoords_current[n1*3+2]
        tmp = (xi**2 + xj**2 + xk**2)**0.5
        xi /= tmp
        xj /= tmp
        xk /= tmp
        spring.update_rotation_matrix(ncoords_current)
        spring.update_KC0(KC0r, KC0c, KC0v)
        
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    KC0uu = KC0[bu, :][:, bu].toarray()  # Update reduced stiffness matrix
    

    
print("residual:", residual)
print("uu:", uu)
print("fu:", fu)
print( "KC0uu @ uu",KC0uu @ uu)
print("KC0uu:", KC0uu)
# Assign displacements to the global displacement vector

