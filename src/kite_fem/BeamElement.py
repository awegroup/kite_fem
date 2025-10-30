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
        # self.alpha = -0.23
        # self.beta = 1.21
        # self.gamma = 0.875
        # self.alpha = -0.3
        # self.beta = 1.35
        # self.gamma = 0.85
        # self.alpha = -0.2
        # self.beta = 1.15
        # self.gamma = 0.95

        # gamma twice
        # self.alpha = -0.2
        # self.beta = 1.35
        # self.gamma = 0.85
        # gamma twice
        self.alpha = -0.35
        self.beta = 1.41
        self.gamma = 0.95


    def set_inflatable_beam_properties(self,d,p,L):
        self.r = 0.5*d
        self.t = 0.01*d
        self.A = 2*np.pi*self.r*self.t
        self.I = (np.pi/4)*(self.r**4 - (self.r - self.t)**4)
        
        # self.A = np.pi * self.r**2
        # self.I = np.pi * self.r**4 / 4.0
        
        self.J = self.I*2   
        self.prop.A = self.A
        self.prop.Ay = 0
        self.prop.Az = 0
        self.prop.Iyy = self.I
        self.prop.Izz = self.I
        self.prop.J = self.I*2
        self.p = p
        self.L = L
        self.beam.length = L
        self.update_inflatable_beam_properties()

    def updateparameter(self,alpha,beta,gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        

    def update_inflatable_beam_properties(self):
        #Bending
        deflection = self.get_beam_deflection()
        # bending_rotation = self.get_bending_rotation()
        # deflection = 2/3*bending_rotation
        C1 = 6582.82
        C2 = -272.43
        C3 = 40852.38
        C4 = 14.31
        C5 = 271865251.42
        C6 = 215.93
        C7 = 14021.79
        C8 = -589.05
        
        # alpha = -0.35
        # beta =1.5
        # delta  =0.95
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        # gamma = 0.9
        # alpha = -0.25
        # beta =1.3
        gamma  = 1

        power = 1
        power2 = 1
        
        denom = (C1 * self.r + C2) * self.p**2 + (C3 * self.r**3 + C4)
        numer = (C5 * self.r**5 + C6) * self.p + (C7 * self.r + C8)
        # F = denom * (1 - np.exp(-(numer / denom) * gamma*(deflection/(self.L**beta)))) *(1/self.L)**(2+alpha) 
        
        func = (1-np.exp(-(numer/denom)))*deflection / ((1-np.exp(-(numer/denom*deflection))*self.L**3))


        # F = denom * (1 - np.exp(-(numer / denom) * (deflection/self.L**beta)*gamma))*(gamma/self.L**(2+alpha))

        F = denom * (1 - np.exp(-(numer / denom) * (deflection)))
        EI = F*1/(3*deflection)

        if np.isnan(EI) or EI <=50:
            EI = 50


        self.E = EI/self.I
        self.prop.E = self.E
        
        #rotation
        rotation = self.get_beam_rotation()
        C13 = 1467
        C14 = 40.908
        C15 = -191.8
        C16 = 47.406
        C17 = -17703
        C18 = 358.05
        C19 = 0.0918
        c1 = ((C13*self.r+C14)*self.p+(C15*self.r+C16))
        c2 = ((C17*self.r**4)*np.log(self.p)+(C18*self.r**3+C19))

        #extrapolated rotation to 1m
        # rotation = (rotation/self.L)
        

        T = c1*np.arctan(c2*rotation)
        
        GJ = T*1/(rotation)
        if np.isnan(GJ) or GJ <=50:
            GJ = 50

        self.G = GJ/self.J
        self.prop.G = self.G
        
    def set_beam_properties(self,E,A,I,L):
        self.prop.E = E
        G = E/(2*(1+0.3))
        self.prop.G = G
        self.prop.A = A
        self.prop.Ay = 0
        self.prop.Az = 0
        self.prop.Iyy = I
        self.prop.Izz = I
        self.prop.Iyz = 0
        self.prop.J = 2*I
        self.beam.length = L
    
    def get_beam_deflection(self):
        tipdeflection = np.array(self.beam.probe.ue[7:9])
        basedeflection = np.array(self.beam.probe.ue[1:3])
        a = 0.5
        n = 1 + (1 - np.exp(-((self.L)/a)**2))
        tipdeflection_scaled = tipdeflection/self.L**n
        basedeflection_scaled = basedeflection/self.L**n
        deflection = np.linalg.norm(tipdeflection_scaled - basedeflection_scaled)
        return deflection
    
    def get_beam_rotation(self):
        tiprotation = np.array(self.beam.probe.ue[9])
        baserotation = np.array(self.beam.probe.ue[3])
        tiprotation_scaled = tiprotation/self.L
        baserotation_scaled = baserotation/self.L
        rotation = tiprotation_scaled-baserotation_scaled
        return rotation
    
    # def get_bending_rotation(self):
    #     tiprotation = np.array(self.beam.probe.ue[9:12])
    #     baserotation = np.array(self.beam.probe.ue[3:6])
    #     rotation = np.linalg.norm(tiprotation-baserotation)
    #     # rotation_norm = rotation/self.L
    #     return rotation
    
    # def update_inflatable_beam_properties(self, d, p,coords: np.ndarray):
    #     # Coefficients
    #     C1 = 6582.82
    #     C2 = -272.43
    #     C3 = 40852.38
    #     C4 = 14.31
    #     C5 = 271865251.42
    #     C6 = 215.93
    #     C7 = 14021.79
    #     C8 = -589.05
        
    #     # Numerator and denominator
    #     denom = (C1 * r + C2) * p**2 + (C3 * r**3 + C4)
    #     numer = (C5 * r**5 + C6) * p + (C7 * r + C8)
        
    #     # Formula
    #     result = denom * (1 - np.exp(-(numer / denom) * v))
    #     return result
    
    def unit_vector(self, coords : np.ndarray):
        xi = coords[self.beam.c2//2 + 0] - coords[self.beam.c1//2 + 0]
        xj = coords[self.beam.c2//2 + 1] - coords[self.beam.c1//2 + 1]
        xk = coords[self.beam.c2//2 + 2] - coords[self.beam.c1//2 + 2]
        l = (xi**2 + xj**2 + xk**2)**0.5
        unit_vect = np.array([xi, xj, xk])/l
        return unit_vect,l
    
    def update_rotation_matrix(self, coords : np.ndarray):
        unit_vect = self.unit_vector(coords)[0]
        xi, xj ,xk = unit_vect[0], unit_vect[1], unit_vect[2]
        vxyi, vxyj, vxyk =  unit_vect[1], unit_vect[2], unit_vect[0]
        if xi == xj  and xj == xk:
            vxyi *= -1
        self.beam.update_rotation_matrix(vxyi, vxyj, vxyk, coords)
    
    def update_KC0(self, KC0r : np.ndarray, KC0c : np.ndarray, KC0v : np.ndarray, coords : np.ndarray):
        self.beam.update_KC0(KC0r, KC0c, KC0v,self.prop,self.update_KC0v_only)
        self.update_KC0v_only = 1
        return KC0r, KC0c, KC0v
    
    def reset(self,coords,displacement):
        self.update_rotation_matrix(coords)
        self.beam.update_probe_ue(displacement)
        self.fi *=0
        self.update_inflatable_beam_properties()

    def beam_internal_forces(self, displacement, coords: np.ndarray):
        self.update_rotation_matrix(coords)
        self.beam.update_probe_ue(displacement)
        self.update_inflatable_beam_properties()
        self.fi *= 0
        self.beam.update_fint(self.fi,self.prop)
        
        return self.fi.copy()
        
        
        