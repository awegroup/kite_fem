import numpy as np
from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamC, BeamCProbe

class BeamElement:
    def __init__(self, n1 : int, n2 : int, init_k_KC0 : int, N: int):
        self.DOF = 6
        beamprobe = BeamCProbe()
        self.beam = BeamC(beamprobe)
        self.prop = BeamProp()
        self.beam.init_k_KC0 = init_k_KC0
        self.beam.n1 = n1
        self.beam.n2 = n2
        self.beam.c1 = self.DOF * n1
        self.beam.c2 = self.DOF * n2
        self.update_KC0v_only = 0
        self.fi = np.zeros(N, dtype=np.float64)
        
    def set_beam_properties(self,E,A,I,L):
        self.prop.E = E
        G = E/(2*(1+0.3))
        self.prop.G = G
        self.prop.A = A
        # self.prop.Ay = A*0.886
        # self.prop.Az = A*0.886
        self.prop.Iyy = I
        self.prop.Izz = I
        # self.prop.Iyz = 0.0
        self.prop.J = 2*I
        # rho = 1
        # self.prop.intrho = A*rho
        # self.prop.intrhoy2 = I*rho
        # self.prop.intrhoz2 = I*rho 
        # self.prop.intrhoyz = 0
        self.beam.length = L
    
    # def update_inflatable_beam_properties(self, d, p,coords: np.ndarray):
        
    
    def unit_vector(self, coords : np.ndarray):
        xi = coords[self.beam.c2//2 + 0] - coords[self.beam.c1//2 + 0]
        xj = coords[self.beam.c2//2 + 1] - coords[self.beam.c1//2 + 1]
        xk = coords[self.beam.c2//2 + 2] - coords[self.beam.c1//2 + 2]
        l = (xi**2 + xj**2 + xk**2)**0.5
        unit_vect = np.array([xi, xj, xk])/l
        return unit_vect,l
    
    def __update_rotation_matrix(self, coords : np.ndarray):
        unit_vect = self.unit_vector(coords)[0]
        xi, xj ,xk = unit_vect[0], unit_vect[1], unit_vect[2]
        vxyi, vxyj, vxyk =  unit_vect[1], unit_vect[2], unit_vect[0]
        if xi == xj  and xj == xk:
            vxyi *= -1
        self.beam.update_rotation_matrix(vxyi, vxyj, vxyk, coords)
    
    def update_KC0(self, KC0r : np.ndarray, KC0c : np.ndarray, KC0v : np.ndarray, coords : np.ndarray):
        self.__update_rotation_matrix(coords)
        self.beam.update_KC0(KC0r, KC0c, KC0v,self.prop,self.update_KC0v_only)
        self.update_KC0v_only = 1
        return KC0r, KC0c, KC0v
    
    def reset(self):
        self.beam.probe.ue=np.array(self.beam.probe.ue)*0
        self.fi *=0

    def beam_internal_forces(self, coords_rotations, coords: np.ndarray, coords_previous: np.ndarray):
        # self.beam.probe.ue=np.array(self.beam.probe.ue)*0
        # self.__update_rotation_matrix(coords_previous)
        self.beam.update_probe_ue(coords_rotations)
        self.__update_rotation_matrix(coords)
        self.beam.update_fint(self.fi,self.prop)
        return self.fi.copy()
        
        
        