import numpy as np
import warnings
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve,lsmr,MatrixRankWarning
from pyfe3d import Spring, SpringData, SpringProbe, DOF, INT, DOUBLE
import matplotlib.pyplot as plt

def spring_internal_forces(spring, fi,  ncoords_current,l0):
    """Calculate the internal forces in a spring based on current node coordinates."""
    n1 = spring.n1
    n2 = spring.n2
    xi = ncoords_current[n2*3] - ncoords_current[n1*3]
    xj = ncoords_current[n2*3+1] - ncoords_current[n1*3+1]
    xk = ncoords_current[n2*3+2] - ncoords_current[n1*3+2]
    l = (xi**2 + xj**2 + xk**2)**(1/2)
    unit_vector = np.array([xi, xj, xk]) / l
    f_s = spring.kxe * (l - l0)  # Spring force
    fi_global = f_s * unit_vector  # Transform to global coordinates
    fi_global = np.append(fi_global, [0, 0, 0]) #add rotational DOF's
    bu1 = bu[spring.c1:spring.c1+DOF]
    bu2 = bu[spring.c2:spring.c2+DOF]
    fi[n1*DOF:(n1+1)*DOF] -= fi_global*bu1
    fi[n2*DOF:(n2+1)*DOF] += fi_global*bu2
    return fi


""" TODO : Fix the offset issue
    TODO: check convergence for all dofs
""" 

# Define spring properties
k = 1  # Stiffness of the first spring in N/m
F = 1  # Force in N
l0 = 0

# convergence parameters, can be tuned for convergence
limit = .1  # Maximum displacement limit in m per iteration  
relaxation = 1  # Initial relaxation factor for displacements
relaxtion_factor = 0.95  # Factor to reduce relaxation if divergence occurs
offset = .01 # Introduce small offset if iteration is stuck due to numerical issues

# Define solver parameters
tolerance = 1e-1
max_iterations = 1000

# Node coordinates
ncoords = np.array([[0.0, 0.0, 0.0],  # Node at x = 0 m
                    [1,0 , 0.0], # Node at x = 1 m
                    [2, 0, 0.0]]) 

nids = np.array([0, 1, 2])  # Node IDs
nid_pos = {nid: i for i, nid in enumerate(nids)}
n1s = np.array([0, 1])
n2s = np.array([1, 2])
num_elements = len(nids)-1
ncoords_init = ncoords.flatten() 
ncoords_current = ncoords_init.copy()

print("Initial Node Coordinates:", ncoords_init)

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
        spring.update_rotation_matrix(ncoords_init)
        spring.update_KC0(KC0r, KC0c, KC0v)
        springs.append(spring)
        init_k_KC0 += springdata.KC0_SPARSE_SIZE

KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

#Define DOF's / boundary conditions

bk = np.ones(N, dtype=bool)
bk[DOF] = False  # Free DOF at Node 1, x
bk[DOF+1] = False  # Free DOF at Node 2, y
bk[DOF+2] = False  # Free DOF at Node 2, z
bu = ~bk

f = np.zeros(N)
f[DOF] = -.1*F    # Node 1, x
f[DOF+1] = F    # Node 2, y
f[DOF+2] = F    # Node 2, z

fu = f[bu]  # Free DOFs force vector


xyz = np.zeros(N, dtype=bool)
xyz[0::DOF] = xyz[1::DOF] = xyz[2::DOF] = True  

residual_prev = 0
residual = 0
u = np.zeros(N, dtype=DOUBLE)  # Global displacement vector
uu = u[bu]
uu_prev = uu.copy()  # Previous displacements
# Iterative solver
for iteration in range(max_iterations):
    #compute internal forces
    fi = np.zeros(N, dtype=DOUBLE)
    for spring in springs:
        fi = spring_internal_forces(spring, fi, ncoords_current,l0)


    # Compute the residual
    residual_prev = residual
    residual = fu - fi[bu]
    # Assign previous residual norm
    residual_norm_prev = np.linalg.norm(residual_prev)
    # Check if the residual is below the tolerance
    residual_norm = np.linalg.norm(residual)
    
    if residual_norm < tolerance:
        print("Converged in", iteration + 1, "iterations!")
        break
    
    # Check for divergence, relax solution if necessary
    if residual_norm_prev < residual_norm and iteration > 0:
        print("Diverging, relaxing solution, falling back on previous residual and deformation.")
        residual = residual_prev
        uu = uu_prev
        relaxation *= relaxtion_factor
        
    elif residual_norm_prev == residual_norm and iteration > 0:
        print("Solution stuck, adding offset.")
        uu += offset
        
    # Update stiffness matrix
    KC0v *= 0
    for spring in springs:
        spring.update_rotation_matrix(ncoords_current)
        spring.update_KC0(KC0r, KC0c, KC0v)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    KC0uu = KC0[bu, :][:, bu]
    
    # Update displacements. Attempting sparse solver first, and lsmr method if it fails
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=MatrixRankWarning)
            duu = spsolve(KC0uu, residual)
    except MatrixRankWarning:
        print("Matrix is singular, using LSMR solver instead.")
        duu = lsmr(KC0uu, residual)[0]
        
    #Use convergence criteria to limit displacements
    uu_prev = uu.copy()
    uu += np.clip(duu*relaxation, -limit, limit)

    u[bu] = uu  
    ncoords_current = ncoords_init + u[xyz]
    
    print("Current Node Coordinates:", ncoords_current)   
    
    
    
    



    
    

