import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg, spsolve
from pyfe3d.beamprop import BeamProp
from pyfe3d import Truss, TrussData, TrussProbe, DOF, INT, DOUBLE
import matplotlib.pyplot as plt

def smart_inverse(A, threshold=1e10):
    cond_number = np.linalg.cond(A)
    if cond_number < threshold:
        return np.linalg.inv(A)
    else:
        return np.linalg.pinv(A)
    
def setup_truss_element():
    # Geometry and material
    L = 1.0
    A = 1.0  # Cross-sectional area (arbitrary)
    E = 1.0  # Young's modulus (arbitrary)
    F = 10 # Force magnitude

    # Node coordinates (triangle)
    ncoords = np.array([
        [0.0, 0.0, 0.0],  # Node 0
        [L,   .1,   0.0],  # Node 1
        [2*L,   0,   0.0],  # Node 2
    ])
    nids = np.array([0, 1, 2])
    nid_pos = {nid: i for i, nid in enumerate(nids)}

    # Truss connectivity: (0-1), (1-2), (0-2)
    n1s = np.array([0, 1])
    n2s = np.array([1, 2])
    num_elements = len(n1s)

    # FEM arrays
    data = TrussData()
    prop = BeamProp()
    prop.A = A
    prop.E = E
    prop.G = 1.0  # Not used for truss, but required by BeamProp
    prop.intrho = 1.0  # Not used here


    KC0r = np.zeros(data.KC0_SPARSE_SIZE * num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE * num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE * num_elements, dtype=DOUBLE)
    N = DOF * len(nids)

    # Assemble truss elements
    init_k_KC0 = 0
    trusses = []
    ncoords_init = ncoords.flatten()
    print('ncoords_flatten', ncoords_init)
    for n1, n2 in zip(n1s, n2s):
        probe = TrussProbe()
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        truss = Truss(probe)
        truss.init_k_KC0 = init_k_KC0
        truss.n1 = n1
        truss.n2 = n2
        truss.c1 = DOF * pos1
        truss.c2 = DOF * pos2

        truss.update_rotation_matrix(ncoords_init)
        truss.length = 1

        # truss.update_probe_xe(ncoords_init)
        # truss.update_length()

        truss.update_KC0(KC0r, KC0c, KC0v, prop)
        trusses.append(truss)


        init_k_KC0 += data.KC0_SPARSE_SIZE
        


    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    print("KC0 shape:", KC0)
    # Boundary conditions: fix node 0 (all DOFs), fix node 1 in y
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



    print("KC0uu",KC0uu)


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
        for truss in trusses:
            xi = ncoords_current[n2*3] - ncoords_current[n1*3] 
            xj = ncoords_current[n2*3+1] - ncoords_current[n1*3+1]
            xk = ncoords_current[n2*3+2] - ncoords_current[n1*3+2]
            tmp = (xi**2 + xj**2 + xk**2)**0.5
            xi /= tmp
            xj /= tmp
            xk /= tmp
            truss.update_rotation_matrix(ncoords_current)
            # truss.update_probe_xe(ncoords_current)
            truss.length = 1
            truss.update_KC0(KC0r, KC0c, KC0v,prop)
            
        KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
        KC0uu = KC0[bu, :][:, bu].toarray()  # Update reduced stiffness matrix
        

        
    print("residual:", residual)
    print("uu:", uu)
    print("fu:", fu)
    print( "KC0uu @ uu",KC0uu @ uu)
    print("KC0uu:", KC0uu)

if __name__ == "__main__":
    setup_truss_element()