import numpy as np
from pyfe3d import Spring, SpringData, SpringProbe, DOF, INT, DOUBLE
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

# class Structure:
#     def __init__(Self,Spring_elements):
#             springdata = SpringProbe()

            
    
class SpringElement:
    def __init__(self, n1, n2, init_k_KC0):
        #setting up PyFE3D spring element
        self.DOF = 6
        springprobe = SpringProbe()
        self.spring = Spring(springprobe)
        self.spring.init_k_KC0 = init_k_KC0
        self.spring.n1 = n1
        self.spring.n2 = n2
        self.spring.c1 = DOF * n1
        self.spring.c2 = DOF * n2
               
    def set_spring_properties(self, l0, k, springtype):
        self.l0 = l0 
        self.k = k
        self.spring.kxe = k  
        self.springtype = springtype  
    
    # def __determine_vector(self, ncoords):
    
    def update_KC0(self, KC0r, KC0c, KC0v, ncoords):
        self.spring.update_rotation_matrix(ncoords)
        self.spring.update_KC0(KC0r, KC0c, KC0v)
    
    def spring_internal_forces(self, ncoords):
        xi = ncoords[self.spring.n2*3] - ncoords[self.spring.n1*3]
        xj = ncoords[self.spring.n2*3+1] - ncoords[self.spring.n1*3+1]
        xk = ncoords[self.spring.n2*3+2] - ncoords[self.spring.n1*3+2]
        l = (xi**2 + xj**2 + xk**2)**(1/2)
        unit_vector = np.array([xi, xj, xk]) / l
        if self.springtype == "noncompressive":
            if l < self.l0:
                self.spring.kxe = 0
            else:
                self.spring.kxe = self.k
        f_s = self.spring.kxe * (l - self.l0) 
        fi_global = f_s * unit_vector  
        fi_global = np.append(fi_global, [0, 0, 0]) 
        return fi_global
    


if __name__ == "__main__":
    springdata = SpringData()
    init_k_KC0 = 0
    nids = np.array([0, 1])  # Node IDs
    num_elements = len(nids)-1
    KC0r = np.zeros(springdata.KC0_SPARSE_SIZE * num_elements, dtype=INT)  # Two springs
    KC0c = np.zeros(springdata.KC0_SPARSE_SIZE * num_elements, dtype=INT)
    KC0v = np.zeros(springdata.KC0_SPARSE_SIZE * num_elements, dtype=DOUBLE)
    N = DOF * len(nids)  # Total DOFs
    n1 = 0
    n2 = 1
    ncoords = np.array([[0, 0, 0],[1, 0, 0]],dtype=DOUBLE) 
    ncoords_init = ncoords.flatten()  # Flatten the coordinates for the spring element
    SpringElement1 = SpringElement(n1, n2, init_k_KC0)
    SpringElement1.set_spring_properties(l0=0, k=1, springtype="default")
    SpringElement1.update_KC0(KC0r, KC0c, KC0v, ncoords_init)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    bk = np.ones(N, dtype=bool)
    bk[DOF] = False 
    bu = ~bk
    f = np.zeros(N)
    f[DOF] = 2  
    KC0uu = KC0[bu, :][:, bu]
    fu = f[bu]  # Free DOFs force vector
    

    
    fi = np.zeros(N, dtype=DOUBLE)
    fi_global = SpringElement1.spring_internal_forces(ncoords_init)
    bu1 = bu[SpringElement1.spring.c1:SpringElement1.spring.c1+DOF]
    bu2 = bu[SpringElement1.spring.c2:SpringElement1.spring.c2+DOF]
    fi[SpringElement1.spring.n1*DOF:(SpringElement1.spring.n1+1)*DOF] -= fi_global*bu1
    fi[SpringElement1.spring.n2*DOF:(SpringElement1.spring.n2+1)*DOF] += fi_global*bu2
    
    print("Internal forces:", fi)
    
    residual = fu - fi[bu]  # Calculate the residual

    
    uu = spsolve(KC0uu, residual)  # Solve for displacements
    u = np.zeros(N)
    u[bu] = uu  # Fill in the displacements for free DOFs
    xyz = np.zeros(N, dtype=bool)  # Initialize xyz array
    xyz[0::DOF] = xyz[1::DOF] = xyz[2::DOF] = True  

    ncoords_current = ncoords_init.copy()
    ncoords_current += u[xyz]  # Update node coordinates with dis
    print(ncoords_current)