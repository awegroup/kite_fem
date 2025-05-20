import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
from pyfe3d import Spring, SpringData, SpringProbe, DOF, INT, DOUBLE
import matplotlib.pyplot as plt


def setup_spring_element():
    # Define spring properties
    k1 = 1  # Stiffness of the first spring in N/m
    k2 = 1  # Stiffness of the second spring in N/m
    k3 = 1  # Stiffness of the third spring in N/m
    F = 1   # Force in N
    L = 1.0  # Length of each spring in m

    # Node coordinates
    ncoords = np.array([[0.0, 0.0, 0.0],  # Node at x = 0 m
                        [L, 0.0, 0.0], # Node at x = 1 m
                        [L, L, 0.0]])  # Node at x = 1 y =1 m
    nids = np.array([0, 1,2])  # Node IDs
    nid_pos = {nid: i for i, nid in enumerate(nids)}
    num_elements = len(nids)-1
    
    # Initialize spring data and probe
    springdata = SpringData()
    springprobe = SpringProbe()

    # Initialize global stiffness matrix arrays
    KC0r = np.zeros(springdata.KC0_SPARSE_SIZE * num_elements, dtype=INT)  # Two springs
    KC0c = np.zeros(springdata.KC0_SPARSE_SIZE * num_elements, dtype=INT)
    KC0v = np.zeros(springdata.KC0_SPARSE_SIZE * num_elements, dtype=DOUBLE)
    N = DOF * len(nids)  # Total DOFs

    # Create the first spring element (x = 0 to x = 1)
    spring1 = Spring(springprobe)
    spring1.init_k_KC0 = 0  # Start index for stiffness matrix
    spring1.n1 = 0  # Node at x = 0 m
    spring1.n2 = 1  # Node at x = 1 m
    spring1.c1 = 0 * DOF  # DOF for node 0
    spring1.c2 = 1 * DOF  # DOF for node 1
    spring1.kxe = k1   # Spring stiffness
    spring1.kye = spring1.kze = 0  # Spring stiffness
    spring1.krxe = spring1.krye = spring1.krze = 0  # No rotational stiffness
    spring1.update_KC0(KC0r, KC0c, KC0v)  # Assemble stiffness matrix

    # Create the second spring element (x = 1 to x = 2)
    spring2 = Spring(springprobe)
    spring2.init_k_KC0 = springdata.KC0_SPARSE_SIZE  # Start index for second spring
    spring2.n1 = 1  # Node at x = 1 m
    spring2.n2 = 2  # Node at x = 2 m
    spring2.c1 = 1 * DOF  # DOF for node 1
    spring2.c2 = 2 * DOF  # DOF for node 2
    spring2.kye = k2  # Spring stiffness
    spring2.kxe = spring2.kze = 0  # Spring stiffness
    spring2.krxe = spring2.krye = spring2.krze = 0  # No rotational stiffness
    spring2.update_KC0(KC0r, KC0c, KC0v)  # Assemble stiffness matrix

    # Create the third spring element (x = 1 to x = 2)
    spring3 = Spring(springprobe)
    spring3.init_k_KC0 = springdata.KC0_SPARSE_SIZE  # Start index for second spring
    spring3.n1 = 0  # Node at x = 1 m
    spring3.n2 = 2  # Node at x = 2 m
    spring3.c1 = 0 * DOF  # DOF for node 1
    spring3.c2 = 2 * DOF  # DOF for node 2
    spring3.kxe = spring3.kye = k3/(2**0.5)  # Spring stiffness
    spring3.kze = 0 # Spring stiffness
    spring3.krxe = spring3.krye = spring3.krze = 0  # No rotational stiffness
    spring3.update_KC0(KC0r, KC0c, KC0v)  # Assemble stiffness matrix

    # Create global stiffness matrix
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    print(KC0)
    # Apply boundary conditions
    bk = np.zeros(N, dtype=bool)  # Boundary condition array
    bk[:3] = True  # Fix node 0 (all DOFs)
    bk[DOF+1] = True  # Fix node 1 (y)
    bu = ~bk  # Free DOFs

    # Apply force vector
    f = np.zeros(N)
    f[2*DOF] = F  # Apply force at node 2 in x-direction
    f[2*DOF + 1] = F  # Apply force at node 2 in y-direction
    # f[DOF] = F
    # f[1 * DOF + 1] = F  # Apply force at node 1 in y-direction
    # f[DOF +1] = F  # Apply force at node 1 in y-direction

    # Solve for displacements
    KC0uu = KC0[bu, :][:, bu]
    fu = f[bu]
    uu, info = cg(KC0uu, fu, atol=1e-9)

    # Post-process results
    u = np.zeros(N)
    u[bu] = uu
    print(u)
    spring1.update_probe_ue(u)
    
    # spring1
    # spring2.update_probe_ue(u)
    
    print(np.array(springprobe.xe))

    u_init_x = ncoords[:,0]
    u_init_y = ncoords[:,1]
    u_init_z = ncoords[:,2]
    u_updated_x = u_init_x + u[0::DOF]
    u_updated_y = u_init_y + u[1::DOF]
    u_updated_z = u_init_z + u[2::DOF]

    
    
    plt.plot(u_init_x, u_init_y, 'r-', label='Initial Position')
    plt.plot([u_init_x[0],u_init_x[2]], [u_init_y[0],u_init_y[2]], 'r-')
    plt.plot(u_updated_x, u_updated_y, 'b-', label='Updated Position')
    plt.plot([u_updated_x[0],u_updated_x[2]], [u_updated_y[0],u_updated_y[2]], 'b-')

    plt.plot(u_init_x, u_init_y, 'ro')
    plt.plot(u_updated_x, u_updated_y, 'bo')
    plt.show()
if __name__ == "__main__":
    setup_spring_element()