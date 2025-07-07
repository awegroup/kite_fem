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
l0 = 1
# Node coordinates
ncoords = np.array([[0.0, 0.0, 0.0],  # Node at x = 0 m
                    [1, 2, 0.0], # Node at x = 1 m
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
        # xi = ncoords_init[n2*3] - ncoords_init[n1*3] 
        # xj = ncoords_init[n2*3+1] - ncoords_init[n1*3+1]
        # xk = ncoords_init[n2*3+2] - ncoords_init[n1*3+2]
        # tmp = (xi**2 + xj**2 + xk**2)**0.5
        # xi /= tmp
        # xj /= tmp
        # xk /= tmp
        spring.update_rotation_matrix(ncoords_init)
        spring.update_KC0(KC0r, KC0c, KC0v)
        springs.append(spring)
        init_k_KC0 += springdata.KC0_SPARSE_SIZE

KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

bk = np.zeros(N, dtype=bool)
bk[0:DOF+1] = True  # Fix the first node (x = 0 m)
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
tolerance = 1e-5
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
    #compute internal forces
    fi = np.zeros(N, dtype=DOUBLE)
    for spring in springs:
        c1 = spring.c1
        c2 = spring.c2
        n1 = spring.n1
        n2 = spring.n2
        xi = ncoords_current[n2*3] - ncoords_current[n1*3]
        xj = ncoords_current[n2*3+1] - ncoords_current[n1*3+1]
        xk = ncoords_current[n2*3+2] - ncoords_current[n1*3+2]
        l = (xi**2 + xj**2 + xk**2)**(1/2)
        fi_local = np.array([spring.kxe * (l - l0),0,0])  # Spring force
        R = np.array([[spring.r11,spring.r12,spring.r13],[spring.r21,spring.r22,spring.r23],[spring.r31,spring.r32,spring.r33]])
        fi_global = R @ fi_local  # Transform to global coordinates
        fi_global = np.append(fi_global,[0,0,0])
        bu1 = bu[spring.c1:spring.c1+DOF]
        bu2 = bu[spring.c2:spring.c2+DOF]
        fi[n1*DOF:(n1+1)*DOF] -= fi_global*bu1
        fi[n2*DOF:(n2+1)*DOF] += fi_global*bu2
        
    # Compute the residual
    residual = fu - fi[bu]

    # Check if the residual is below the tolerance
    residual_norm = np.linalg.norm(residual)
    if residual_norm < tolerance:
        print("Converged!")
        break

    # Update displacements 
    uu += (smart_inverse(KC0uu) @ residual )
    
    u = np.zeros(N)
    u[bu] = uu  # Assign displacements to free DOFs

    ncoords_current = ncoords_init + u[xyz]
    print("Current Node Coordinates:", ncoords_current)

    KC0v *= 0
    for spring in springs:
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


print(fi[bu])

y = ncoords_current[4]
x = 1
hyp = (x**2 + y**2)**0.5
F = k * (hyp - l0)  # Calculate the force in the spring
print("Force in one spring:", F)
Fy = F * (y / hyp)  # Vertical component of the force
print("Vertical component of the force:", Fy)

# fi = np.zeros(N, dtype=DOUBLE)
# for spring in springs:
#     print("spring:", spring)
#     print("-------------------------------------------")
#     c1 = spring.c1
#     c2 = spring.c2
#     n1 = spring.n1
#     n2 = spring.n2
#     xi = ncoords_current[n1*3] - ncoords_current[n1*3]
#     xj = ncoords_current[n2*3+1] - ncoords_current[n1*3+1]
#     xk = ncoords_current[n2*3+2] - ncoords_current[n1*3+2]
#     l = (xi**2 + xj**2 + xk**2)**(1/2)
#     fi_local = np.array([spring.kxe * (l - l0),0,0])  # Spring force
#     R = np.array([[spring.r11,spring.r12,spring.r13],[spring.r21,spring.r22,spring.r23],[spring.r31,spring.r32,spring.r33]])
#     fi_global = R @ fi_local  # Transform to global coordinates
#     fi_global = np.append(fi_global,[0,0,0])
    
#     bu1 = bu[spring.c1:spring.c1+DOF]
#     bu2 = bu[spring.c2:spring.c2+DOF]
#     print("bu1:", bu1)
#     print("bu2:", bu2)
#     print("fi_global:", fi_global)
#     print("c1:", c1, "c2:", c2)
#     fi[c1:c2] -= fi_global*bu1
#     fi[c2:c2+DOF] += fi_global*bu2
#     print("fi:", fi)
#     print("-------------------------------------------")
