import numpy as np
from pyfe3d import Spring, SpringProbe

class SpringElement:
    def __init__(self, n1 : int, n2 : int, init_k_KC0 : int):
        self.DOF = 6
        springprobe = SpringProbe()
        self.spring = Spring(springprobe)
        self.spring.init_k_KC0 = init_k_KC0
        self.spring.n1 = n1
        self.spring.n2 = n2
        self.spring.c1 = self.DOF * n1
        self.spring.c2 = self.DOF * n2
        self.update_KC0v_only = 0
        
    def set_spring_properties(self, l0 : float, k : float, springtype : str, i_other_pulley: int = 0):
        self.l0 = l0 
        self.k = k
        self.spring.kxe = k  
        self.springtype = springtype.lower()
        if self.springtype == "pulley":
            self.i_other_pulley = i_other_pulley
        if self.springtype not in ("noncompressive", "default", "pulley"):
            raise ValueError("Invalid spring type. Choose from 'noncompressive', 'default', or 'pulley'.")
        
    def unit_vector(self, ncoords : np.ndarray):
        xi = ncoords[self.spring.c2//2 + 0] - ncoords[self.spring.c1//2 + 0]
        xj = ncoords[self.spring.c2//2 + 1] - ncoords[self.spring.c1//2 + 1]
        xk = ncoords[self.spring.c2//2 + 2] - ncoords[self.spring.c1//2 + 2]
        l = (xi**2 + xj**2 + xk**2)**0.5
        unit_vect = np.array([xi, xj, xk])/l
        return unit_vect,l
    
    def update_KC0(self, KC0r : np.ndarray, KC0c : np.ndarray, KC0v : np.ndarray, ncoords : np.ndarray):
        unit_vect = self.unit_vector(ncoords)[0]
        xi, xj ,xk = unit_vect[0], unit_vect[1], unit_vect[2]      
        vxyi, vxyj, vxyk =  unit_vect[1], unit_vect[2], unit_vect[0] 
        if xi == xj  and xj == xk: # Edge case, if all are the same then KC0 returns NaN's
            vxyi *= -1
        self.spring.update_rotation_matrix(xi, xj, xk, vxyi, vxyj, vxyk)
        self.spring.update_KC0(KC0r, KC0c, KC0v,self.update_KC0v_only)
        self.update_KC0v_only = 1
        return KC0r, KC0c, KC0v

    def spring_internal_forces(self, ncoords: np.ndarray, l_other_pulley:float = 0.0):
        unit_vector,l = self.unit_vector(ncoords)
        k_fi = self.k
        if self.springtype == "noncompressive" or self.springtype == "pulley":
            if l+l_other_pulley < (self.l0):
                self.spring.kxe = 0.01*self.k
                k_fi = 0.0
            else:
                self.spring.kxe = self.k
                k_fi = self.k
        f_s = k_fi * (l + l_other_pulley - self.l0)
        fi = f_s * unit_vector
        fi = np.append(fi, [0, 0, 0])
        return fi
