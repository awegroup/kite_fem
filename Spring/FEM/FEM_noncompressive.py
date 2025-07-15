import numpy as np
import warnings
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve,lsmr,MatrixRankWarning
from pyfe3d import Spring, SpringData, SpringProbe, DOF, INT, DOUBLE
import matplotlib.pyplot as plt

def spring_internal_forces(spring, fi,  ncoords_current,l0,springtype):
    """Calculate the internal forces in a spring based on current node coordinates."""
    """ updates fi with the internal forces of the spring """
    n1 = spring.n1
    n2 = spring.n2
    xi = ncoords_current[n2*3] - ncoords_current[n1*3]
    xj = ncoords_current[n2*3+1] - ncoords_current[n1*3+1]
    xk = ncoords_current[n2*3+2] - ncoords_current[n1*3+2]
    l = (xi**2 + xj**2 + xk**2)**(1/2)
    unit_vector = np.array([xi, xj, xk]) / l
    if springtype == "noncompressive":
        if l < l0:
            spring.kxe = 0
        else:
            spring.kxe = k
    f_s = spring.kxe * (l - l0)  # Spring force
    fi_global = f_s * unit_vector  # Transform to global coordinates
    fi_global = np.append(fi_global, [0, 0, 0]) #add rotational DOF's
    bu1 = bu[spring.c1:spring.c1+DOF]
    bu2 = bu[spring.c2:spring.c2+DOF]
    fi[n1*DOF:(n1+1)*DOF] -= fi_global*bu1
    fi[n2*DOF:(n2+1)*DOF] += fi_global*bu2
    return fi

def unit_vector(ncoords ,spring):
        xi = ncoords[spring.c2//2 + 0] - ncoords[spring.c1//2 + 0]
        xj = ncoords[spring.c2//2 + 1] - ncoords[spring.c1//2 + 1]
        xk = ncoords[spring.c2//2 + 2] - ncoords[spring.c1//2 + 2]
        l = (xi**2 + xj**2 + xk**2)**0.5
        unit_vector = np.array([xi, xj, xk])/l
        return unit_vector,l
    
def solve_sparse_system(K, F):
    """Solve the sparse system using spsolve or lsmr."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=MatrixRankWarning)
            u = spsolve(K, F)
    except MatrixRankWarning:
        print("Matrix is singular, using LSMR solver instead.")
        u = lsmr(K, F)[0]
    return u

""" 
    TODO: First solve for higher tolerance, then reset convergence parameters and solve again with lower tolerance., relate limit to tolerance 
""" 



# Define spring properties
k = 1  # Stiffness of the first spring in N/m
F = 10  # Force in N
l0_springs = [0,0,10]
springtypes = ["normal", "normal", "noncompressive"]  # Types of springs

# convergence parameters, can be tuned for convergence
limit = .2 # Maximum displacement limit in m per iteration  
relaxation = 1  # Initial relaxation factor for displacements
relaxtion_factor = 1 # Factor to reduce relaxation if divergence occurs
offset = .1 # Introduce offset if iteration is stuck due to numerical issues

# Define solver parameters
tolerance = .1
max_iterations = 50

# Node coordinates
ncoords = np.array([[0, 0, 0],  # Node at x = 0 m
                    [1, 0, 0], # Node at x = 1 m
                    [2, 0, 0],
                    [1, 0,10]], dtype=DOUBLE) 

nids = np.array([0, 1, 2, 3])  # Node IDs
nid_pos = {nid: i for i, nid in enumerate(nids)}
n1s = np.array([0, 1, 1])
n2s = np.array([1, 2, 3])
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
        unit_vectors, l = unit_vector(ncoords_current, spring)
        xi, xj ,xk = unit_vectors[0], unit_vectors[1], unit_vectors[2]      
        vxyi, vxyj, vxyk =  unit_vectors[1], unit_vectors[2], unit_vectors[0] 
        spring.update_rotation_matrix(xi, xj, xk, vxyi, vxyj, vxyk)
        spring.update_KC0(KC0r, KC0c, KC0v)
        springs.append(spring)
        init_k_KC0 += springdata.KC0_SPARSE_SIZE

KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

#Define DOF's / boundary conditions

bk = np.ones(N, dtype=bool)
bk[DOF+1] = False  # Free DOF at Node 2, y
bk[DOF+2] = False  # Free DOF at Node 2, z
bk[DOF] = False  # Free DOF at Node 2, x
bu = ~bk

f = np.zeros(N)
f[DOF+1] = F  # Node 3, y
fu = f[bu]  # Free DOFs force vector


xyz = np.zeros(N, dtype=bool)
xyz[0::DOF] = xyz[1::DOF] = xyz[2::DOF] = True  
templist = []  # List to store offsets for stuck iterations
residual_prev = 0
residual = 0
u = np.zeros(N, dtype=DOUBLE)  # Global displacement vector
uu = u[bu]
uu_prev = uu.copy()  # Previous displacements
rng = np.random.default_rng(seed=42)  # seed for reproducibility
# Iterative solver
for iteration in range(max_iterations):
    #compute internal forces
    fi = np.zeros(N, dtype=DOUBLE)
    for i,spring in enumerate(springs):
        l0 = l0_springs[i]
        springtype = springtypes[i]
        fi = spring_internal_forces(spring, fi, ncoords_current,l0,springtype)


    # Compute the residual
    residual_prev = residual
    residual = fu - fi[bu]
    # Assign previous residual norm
    residual_norm_prev = np.linalg.norm(residual_prev)
    # Check if the residual is below the tolerance
    residual_norm = np.linalg.norm(residual)
    
    if residual_norm < tolerance:
        print("Converged in", iteration, "iterations!")
        break
    
    if np.abs(residual_norm_prev - residual_norm) < offset and iteration > 0:
        print("Solution stuck, adding offset.")
        temp = rng.uniform(low=-offset, high=offset, size=np.size(uu)) 
        templist.append(temp)
        uu += temp
        uu_prev += temp
        
    # Check for divergence, relax solution if necessary
    if residual_norm_prev < residual_norm and iteration > 0:
        print("Diverging, relaxing solution, falling back on previous residual and deformation.")
        residual = residual_prev
        residual_norm = residual_norm_prev
        uu = uu_prev
        relaxation *= relaxtion_factor
            
    # Update stiffness matrix
    KC0v *= 0
    for spring in springs:
        unit_vectors, l = unit_vector(ncoords_current, spring)
        xi, xj ,xk = unit_vectors[0], unit_vectors[1], unit_vectors[2]      
        vxyi, vxyj, vxyk =  unit_vectors[1], unit_vectors[2], unit_vectors[0] 
        spring.update_rotation_matrix(xi, xj, xk, vxyi, vxyj, vxyk)
        spring.update_KC0(KC0r, KC0c, KC0v)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    KC0uu = KC0[bu, :][:, bu]
    
    # Update displacements. Attempting sparse solver first, and lsmr method if it fails
    duu = solve_sparse_system(KC0uu, residual)
        
    #Use convergence criteria to limit displacements
    uu_prev = uu.copy()
    uu += np.clip(duu*relaxation, -limit, limit)

    u[bu] = uu  
    ncoords_current = ncoords_init + u[xyz]
    
    print("Current Node Coordinates:", ncoords_current)   
    
    
    
    



    

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# Plot initial configuration
for spring in springs:
    n1 = spring.n1
    n2 = spring.n2
    x_init = [ncoords_init[n1*3], ncoords_init[n2*3]]
    y_init = [ncoords_init[n1*3+1], ncoords_init[n2*3+1]]
    if n2 == 3:
        axes[0].plot(x_init, y_init, 'ro--', label=f'non-compressive spring [k=1 N/m, l0=10 m]')
    else:
        axes[0].plot(x_init, y_init, 'ro-', label=f'normal spring [k=1 N/m, l0=0 m]' if n1 == 0 else "")

axes[0].set_title("Initial Configuration")
axes[0].set_xlabel("x [m]")
axes[0].set_ylabel("y [m]")
axes[0].grid()
axes[0].legend(loc='upper right', bbox_to_anchor=(1, 0.95))

# Plot current configuration
for spring in springs:
    n1 = spring.n1
    n2 = spring.n2
    x_current = [ncoords_current[n1*3], ncoords_current[n2*3]]
    y_current = [ncoords_current[n1*3+1], ncoords_current[n2*3+1]]
    if n2 == 3:
        axes[1].plot(x_current, y_current, 'bo--', label=f'non-compressive spring [k=1 N/m, l0=10 m]')
    else:
        axes[1].plot(x_current, y_current, 'bo-', label=f'normal spring [k=1 N/m, l0=0 m]' if n1 == 0 else "")



axes[0].arrow(1, 0, 0, F/5, color='green', width=0.01, head_width=0.1, head_length=.3, length_includes_head=True, label='Force')
axes[1].arrow(1, ncoords_current[4], 0, F/5, color='green', width=0.01, head_width=0.1, head_length=.3, length_includes_head=True)

axes[1].set_title("Current Configuration")
axes[1].set_xlabel("x [m]")
axes[1].set_ylabel("y [m]")
axes[1].grid()
axes[1].legend(loc='upper right', bbox_to_anchor=(1, 0.95))

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
