import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
from pyfe3d.beamprop import BeamProp
from pyfe3d import Truss, TrussData, TrussProbe, DOF, INT, DOUBLE
import matplotlib.pyplot as plt

def setup_truss_element():
    # Geometry and material
    L = 1.0
    A = 1.0  # Cross-sectional area (arbitrary)
    E = 1.0  # Young's modulus (arbitrary)
    F = 1.0  # Force magnitude

    # Node coordinates (triangle)
    ncoords = np.array([
        [0.0, 0.0, 0.0],  # Node 0
        [L,   0,   0.0],  # Node 1
        [L,   L,   0.0],  # Node 2
    ])
    nids = np.array([0, 1, 2])
    nid_pos = {nid: i for i, nid in enumerate(nids)}

    # Truss connectivity: (0-1), (1-2), (0-2)
    n1s = np.array([0, 1, 0])
    n2s = np.array([1, 2, 2])
    num_elements = len(n1s)

    # FEM arrays
    data = TrussData()
    probe = TrussProbe()
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
    ncoords_flatten = ncoords.flatten()
    for n1, n2 in zip(n1s, n2s):
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        truss = Truss(probe)
        truss.init_k_KC0 = init_k_KC0
        truss.n1 = n1
        truss.n2 = n2
        truss.c1 = DOF * pos1
        truss.c2 = DOF * pos2
        truss.update_length()

        truss.update_rotation_matrix(ncoords_flatten)
        truss.update_probe_xe(ncoords_flatten)
        truss.update_KC0(KC0r, KC0c, KC0v, prop)
        trusses.append(truss)
        init_k_KC0 += data.KC0_SPARSE_SIZE

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    # Boundary conditions: fix node 0 (all DOFs), fix node 1 in y
    bk = np.zeros(N, dtype=bool)
    bk[:3] = True  # Node 0: all DOFs
    bk[DOF] = True  # Node 1: y DOF
    bu = ~bk

    # Force vector: apply F in x and y at node 2
    f = np.zeros(N)
    f[2* DOF] = F     # Node 2, x
    f[2 * DOF + 1] = F  # Node 2, y

    # Solve
    KC0uu = KC0[bu, :][:, bu]
    fu = f[bu]
    uu, info = cg(KC0uu, fu, atol=1e-9)
    u = np.zeros(N)
    u[bu] = uu
    print("Displacements:", u)
    # Print displacements
    # print("Displacements:", u.reshape(-1, DOF))
    
    print(np.array(probe.ue))

    # Plot initial and deformed shape
    u_init_x = ncoords[:, 0]
    u_init_y = ncoords[:, 1]
    u_updated_x = u_init_x + u[0::DOF]
    u_updated_y = u_init_y + u[1::DOF]

    # Plot lines for truss elements
    for (i, j) in zip(n1s, n2s):
        plt.plot([u_init_x[i], u_init_x[j]], [u_init_y[i], u_init_y[j]], 'r-', label='Initial' if i == 0 and j == 1 else "")
        plt.plot([u_updated_x[i], u_updated_x[j]], [u_updated_y[i], u_updated_y[j]], 'b-', label='Deformed' if i == 0 and j == 1 else "")

    plt.plot(u_init_x, u_init_y, 'ro')
    plt.plot(u_updated_x, u_updated_y, 'bo')
    plt.axis('equal')
    plt.legend()
    plt.title("Truss FEM: Initial (red) and Deformed (blue)")
    plt.show()

if __name__ == "__main__":
    setup_truss_element()