from SpringElement import SpringElement
from pyfe3d import DOF, INT, DOUBLE, SpringData
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve,lsmr,lsqr ,MatrixRankWarning
import numpy as np
import matplotlib.pyplot as plt
import warnings

class FEM_structure:
    def __init__(self, initial_conditions, connectivity_matrix):
        self.num_nodes = len(initial_conditions)
        self.num_elements = len(connectivity_matrix)
        self.N = DOF * self.num_nodes
        self.xyz = np.zeros(self.N, dtype=bool) 
        self.xyz[0::DOF] = self.xyz[1::DOF] = self.xyz[2::DOF] = True  
        self.rng = np.random.default_rng(seed=42)  
        self.__setup_initial_conditions(initial_conditions)
        self.__setup_spring_elements(connectivity_matrix)

    def __setup_initial_conditions(self, initial_conditions):
        self.bu = np.zeros(self.N, dtype=bool)
        self.ncoords_init = np.zeros((self.num_nodes, 3), dtype=float)
        for id, (pos,vel,mass,fixed) in enumerate(initial_conditions):
            self.ncoords_init[id] = pos
            if fixed == False:
                self.bu[DOF*id:DOF*id+3] = True
        self.ncoords_init = self.ncoords_init.flatten()
        self.ncoords_current = self.ncoords_init.flatten()
        
    def __setup_spring_elements(self, connectivity_matrix):
        springdata = SpringData()
        self.KC0r = np.zeros(springdata.KC0_SPARSE_SIZE * self.num_elements, dtype=INT)  
        self.KC0c = np.zeros(springdata.KC0_SPARSE_SIZE * self.num_elements, dtype=INT)
        self.KC0v = np.zeros(springdata.KC0_SPARSE_SIZE * self.num_elements, dtype=DOUBLE)
        init_KC0 = 0
        self.spring_elements = []
        for (n1,n2,k,c, springtype) in connectivity_matrix:
            spring_element = SpringElement(n1, n2, init_KC0)
            if springtype == "noncompressive":
                l0 = 10
            else:
                l0 = 0
            spring_element.set_spring_properties(l0, k, springtype) 
            self.spring_elements.append(spring_element)
            init_KC0 += springdata.KC0_SPARSE_SIZE

    def update_stiffness_matrix(self):
        self.KC0v *= 0  
        nancount = 0
        nancountprev =0
        for spring_element in self.spring_elements:
            self.KC0r, self.KC0c, self.KC0v = spring_element.update_KC0(self.KC0r, self.KC0c, self.KC0v, self.ncoords_current)
            nancount = np.count_nonzero(np.isnan(self.KC0v))
            if nancount > nancountprev:
                print(f"NaN detected in stiffness matrix, element: {spring_element.spring.n1}-{spring_element.spring.n2}, count: {nancount}")
                nancountprev = nancount
                
            
        if np.isnan(self.KC0v).any():
            print("Nan detected in stiffness matrix, setting to 0.")
            self.KC0v = np.where(np.isnan(self.KC0v),  0, self.KC0v)
            
            
        self.KC0 = coo_matrix((self.KC0v, (self.KC0r, self.KC0c)), shape=(self.N, self.N)).tocsc()

    def update_internal_forces(self):
        self.fi = np.zeros(self.N, dtype=DOUBLE)
        for spring_element in self.spring_elements:
            fi_element = spring_element.spring_internal_forces(self.ncoords_current)
            bu1 = self.bu[spring_element.spring.c1:spring_element.spring.c1+DOF]
            bu2 = self.bu[spring_element.spring.c2:spring_element.spring.c2+DOF]
            self.fi[spring_element.spring.n1*DOF:(spring_element.spring.n1+1)*DOF] -= fi_element*bu1
            self.fi[spring_element.spring.n2*DOF:(spring_element.spring.n2+1)*DOF] += fi_element*bu2
            
    def solve(self, fe=None, max_iterations=1000, tolerance=1e-2,limit=1e-5,offset=0.001):
        if fe is None:
            fe = np.zeros(self.N, dtype=DOUBLE)
        self.fe = fe
        u = np.zeros(self.N, dtype=DOUBLE)
        uu = u[self.bu]
        residual = 0
        for iteration in range(max_iterations):
            self.update_internal_forces()
            self.update_stiffness_matrix()
            
            residual_prev = residual


            residual = (self.fe-self.fi)[self.bu]
            residual_norm_prev = np.linalg.norm(residual_prev)
            residual_norm = np.linalg.norm(residual)
            
            if residual_norm < tolerance:
                print(f"Converged after {iteration} iterations.")
                print("residual",residual)
                break
            
            # if np.abs(residual_norm_prev - residual_norm) < offset and iteration > 1:
            #     print("Solution stuck, adding offset.")
            #     random = self.rng.uniform(low=-offset, high=offset, size=np.size(uu)) 
            #     uu += random
                
            KC0uu = self.KC0[self.bu, :][:, self.bu]
            duu = self.__sparse_solve(KC0uu, residual)
            
            # if any(np.isnan(duu)):
            #     duu = np.where(np.isnan(duu), self.rng.uniform(-offset,offset,size=np.size(duu)), duu)  

            uu += np.clip(duu,-limit,limit) #TODO Add line search
            

            u[self.bu] = uu
            self.ncoords_current = self.ncoords_init + u[self.xyz]
            
    def __sparse_solve(self, K, F):
        # try:
        #     with warnings.catch_warnings():
        #         warnings.filterwarnings("error", category=MatrixRankWarning)
        #         u = spsolve(K, F)
        # except MatrixRankWarning:
        #     print("Matrix is singular, using LSQR solver instead.")
        #     u = lsqr(K, F)[0]
        return lsqr(K, F)[0]
    
    def plot(self,color,ax=None,fig=None):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')        
        ax.scatter(self.ncoords_current[0::DOF//2], self.ncoords_current[1::DOF//2], self.ncoords_current[2::DOF//2], color= color)
        
        for spring_element in self.spring_elements:
            n1 = spring_element.spring.n1
            n2 = spring_element.spring.n2
            ax.plot([self.ncoords_current[n1*DOF//2], self.ncoords_current[n2*DOF//2]],
                    [self.ncoords_current[n1*DOF//2+1], self.ncoords_current[n2*DOF//2+1]],
                    [self.ncoords_current[n1*DOF//2+2], self.ncoords_current[n2*DOF//2+2]], color=color)
        
        #todo show internal forces/
        #TODO show external forces
        
        # self.update_internal_forces()
        # ax.quiver
        
        return ax,fig
    
if __name__ == "__main__":
    initial_conditions = [[[0.0, 0, 0.0], [0, 0, 0], 1, True], [[2.5, 0, 1.25], [0, 0, 0], 1, True], [[5.0, 0, 2.5], [0, 0, 0], 1, True], [[7.5, 0, 3.75], [0, 0, 0], 1, True], [[10.0, 0, 5.0], [0, 0, 0], 1, True], [[1.25, 1.25, 2.5], [0, 0, 0], 1, False], [[3.75, 1.25, 2.5], [0, 0, 0], 1, False], [[6.25, 1.25, 2.5], [0, 0, 0], 1, False], [[8.75, 1.25, 2.5], [0, 0, 0], 1, False], [[0.0, 2.5, 1.25], [0, 0, 0], 1, True], [[2.5, 2.5, 2.5], [0, 0, 0], 1, False], [[5.0, 2.5, 2.5], [0, 0, 0], 1, False], [[7.5, 2.5, 2.5], [0, 0, 0], 1, False], [[10.0, 2.5, 3.75], [0, 0, 0], 1, True], [[1.25, 3.75, 2.5], [0, 0, 0], 1, False], [[3.75, 3.75, 2.5], [0, 0, 0], 1, False], [[6.25, 3.75, 2.5], [0, 0, 0], 1, False], [[8.75, 3.75, 2.5], [0, 0, 0], 1, False], [[0.0, 5.0, 2.5], [0, 0, 0], 1, True], [[2.5, 5.0, 2.5], [0, 0, 0], 1, False], [[5.0, 5.0, 2.5], [0, 0, 0], 1, False], [[7.5, 5.0, 2.5], [0, 0, 0], 1, False], [[10.0, 5.0, 2.5], [0, 0, 0], 1, True], [[1.25, 6.25, 2.5], [0, 0, 0], 1, False], [[3.75, 6.25, 2.5], [0, 0, 0], 1, False], [[6.25, 6.25, 2.5], [0, 0, 0], 1, False], [[8.75, 6.25, 2.5], [0, 0, 0], 1, False], [[0.0, 7.5, 3.75], [0, 0, 0], 1, True], [[2.5, 7.5, 2.5], [0, 0, 0], 1, False], [[5.0, 7.5, 2.5], [0, 0, 0], 1, False], [[7.5, 7.5, 2.5], [0, 0, 0], 1, False], [[10.0, 7.5, 1.25], [0, 0, 0], 1, True], [[1.25, 8.75, 2.5], [0, 0, 0], 1, False], [[3.75, 8.75, 2.5], [0, 0, 0], 1, False], [[6.25, 8.75, 2.5], [0, 0, 0], 1, False], [[8.75, 8.75, 2.5], [0, 0, 0], 1, False], [[0.0, 10.0, 5.0], [0, 0, 0], 1, True], [[2.5, 10.0, 3.75], [0, 0, 0], 1, True], [[5.0, 10.0, 2.5], [0, 0, 0], 1, True], [[7.5, 10.0, 1.25], [0, 0, 0], 1, True], [[10.0, 10.0, 0.0], [0, 0, 0], 1, True]]
    final_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True], [[2.5, 0.0, 1.25], [0, 0, 0], 1, True], [[5.0, 0.0, 2.5], [0, 0, 0], 1, True], [[7.5, 0.0, 3.75], [0, 0, 0], 1, True], [[10.0, 0.0, 5.0], [0, 0, 0], 1, True], [[1.250000642043016, 1.2500006420430159, 1.093757840242485], [0, 0, 0], 1, False], [[3.749999167060534, 1.250002842879245, 2.0312572523033605], [0, 0, 0], 1, False], [[6.250000832939466, 1.250002842879245, 2.9687427476966395], [0, 0, 0], 1, False], [[8.749999357956984, 1.2500006420430159, 3.9062421597575145], [0, 0, 0], 1, False], [[0.0, 2.5, 1.25], [0, 0, 0], 1, True], [[2.5000018763119476, 2.5000018763119476, 1.8750155278412162], [0, 0, 0], 1, False], [[5.0, 2.500006767978834, 2.5], [0, 0, 0], 1, False], [[7.499998123688052, 2.5000018763119476, 3.1249844721587836], [0, 0, 0], 1, False], [[10.0, 2.5, 3.75], [0, 0, 0], 1, True], [[1.250002842879245, 3.7499991670605346, 2.0312572523033605], [0, 0, 0], 1, False], [[3.7500041495172023, 3.750004149517202, 2.343758247380392], [0, 0, 0], 1, False], [[6.249995850482797, 3.750004149517202, 2.656241752619608], [0, 0, 0], 1, False], [[8.749997157120756, 3.7499991670605346, 2.96874274769664], [0, 0, 0], 1, False], [[0.0, 5.0, 2.5], [0, 0, 0], 1, True], [[2.500006767978834, 5.0, 2.5], [0, 0, 0], 1, False], [[5.0, 5.0, 2.5], [0, 0, 0], 1, False], [[7.499993232021167, 5.0, 2.5], [0, 0, 0], 1, False], [[10.0, 5.0, 2.5], [0, 0, 0], 1, True], [[1.250002842879245, 6.250000832939466, 2.9687427476966395], [0, 0, 0], 1, False], [[3.750004149517202, 6.249995850482797, 2.656241752619608], [0, 0, 0], 1, False], [[6.249995850482797, 6.249995850482797, 2.343758247380392], [0, 0, 0], 1, False], [[8.749997157120756, 6.250000832939466, 2.0312572523033605], [0, 0, 0], 1, False], [[0.0, 7.5, 3.75], [0, 0, 0], 1, True], [[2.500001876311947, 7.499998123688053, 3.1249844721587836], [0, 0, 0], 1, False], [[5.0, 7.499993232021166, 2.5], [0, 0, 0], 1, False], [[7.499998123688052, 7.499998123688052, 1.8750155278412162], [0, 0, 0], 1, False], [[10.0, 7.5, 1.25], [0, 0, 0], 1, True], [[1.2500006420430159, 8.749999357956984, 3.9062421597575145], [0, 0, 0], 1, False], [[3.7499991670605346, 8.749997157120756, 2.9687427476966395], [0, 0, 0], 1, False], [[6.250000832939465, 8.749997157120756, 2.0312572523033605], [0, 0, 0], 1, False], [[8.749999357956984, 8.749999357956984, 1.093757840242485], [0, 0, 0], 1, False], [[0.0, 10.0, 5.0], [0, 0, 0], 1, True], [[2.5, 10.0, 3.75], [0, 0, 0], 1, True], [[5.0, 10.0, 2.5], [0, 0, 0], 1, True], [[7.5, 10.0, 1.25], [0, 0, 0], 1, True], [[10.0, 10.0, 0.0], [0, 0, 0], 1, True]]
    connectivity_matrix = [[0, 5, 9, 1, 'default'], [5, 1, 9, 1, 'default'], [5, 9, 9, 1, 'default'], [5, 10, 9, 1, 'default'], [6, 1, 9, 1, 'default'], [6, 2, 9, 1, 'default'], [6, 10, 9, 1, 'default'], [6, 11, 9, 1, 'default'], [7, 2, 9, 1, 'default'], [7, 3, 9, 1, 'default'], [7, 11, 9, 1, 'default'], [7, 12, 9, 1, 'default'], [8, 3, 9, 1, 'default'], [8, 4, 9, 1, 'default'], [8, 12, 9, 1, 'default'], [8, 13, 9, 1, 'default'], [10, 14, 9, 1, 'default'], [10, 15, 9, 1, 'default'], [11, 15, 9, 1, 'default'], [11, 16, 9, 1, 'default'], [12, 16, 9, 1, 'default'], [12, 17, 9, 1, 'default'], [14, 9, 9, 1, 'default'], [14, 18, 9, 1, 'default'], [14, 19, 9, 1, 'default'], [15, 19, 9, 1, 'default'], [15, 20, 9, 1, 'default'], [16, 20, 9, 1, 'default'], [16, 21, 9, 1, 'default'], [17, 13, 9, 1, 'default'], [17, 21, 9, 1, 'default'], [17, 22, 9, 1, 'default'], [19, 23, 9, 1, 'default'], [19, 24, 9, 1, 'default'], [20, 24, 9, 1, 'default'], [20, 25, 9, 1, 'default'], [21, 25, 9, 1, 'default'], [21, 26, 9, 1, 'default'], [23, 18, 9, 1, 'default'], [23, 27, 9, 1, 'default'], [23, 28, 9, 1, 'default'], [24, 28, 9, 1, 'default'], [24, 29, 9, 1, 'default'], [25, 29, 9, 1, 'default'], [25, 30, 9, 1, 'default'], [26, 22, 9, 1, 'default'], [26, 30, 9, 1, 'default'], [26, 31, 9, 1, 'default'], [28, 32, 9, 1, 'default'], [28, 33, 9, 1, 'default'], [29, 33, 9, 1, 'default'], [29, 34, 9, 1, 'default'], [30, 34, 9, 1, 'default'], [30, 35, 9, 1, 'default'], [32, 27, 9, 1, 'default'], [32, 36, 9, 1, 'default'], [32, 37, 9, 1, 'default'], [33, 37, 9, 1, 'default'], [33, 38, 9, 1, 'default'], [34, 38, 9, 1, 'default'], [34, 39, 9, 1, 'default'], [35, 31, 9, 1, 'default'], [35, 39, 9, 1, 'default'], [35, 40, 9, 1, 'default']]

        
    SaddleForm = FEM_structure(initial_conditions, connectivity_matrix)
    ax,fig = SaddleForm.plot(color='red')
    fe = np.zeros(SaddleForm.N, dtype=DOUBLE)
    SaddleForm.update_internal_forces()
    SaddleForm.update_stiffness_matrix()
    SaddleForm.solve( fe = fe, max_iterations=1, tolerance=.1,limit=.1,offset=0.001)
    ax,fig = SaddleForm.plot(color='blue')
    l = 20
    vect = fe - SaddleForm.fi
    
    for node in range(len(initial_conditions)):
        coords = SaddleForm.ncoords_current[DOF//2*node:DOF//2*node+3]
        x = np.array([coords[0], coords[0]+vect[DOF*node]/l])
        y = np.array([coords[1], coords[1]+vect[DOF*node+1]/l])
        z = np.array([coords[2], coords[2]+vect[DOF*node+2]/l])
        ax.plot(x,y,z, color='green', linewidth=2, label='Internal Forces')
    
        
    KC = SaddleForm.KC0
    residual = fe - SaddleForm.fi
    vect = lsqr(KC, residual)[0]

    l=5
    for node in range(len(initial_conditions)):
        coords = SaddleForm.ncoords_current[DOF//2*node:DOF//2*node+3]
        x = np.array([coords[0], coords[0]+vect[DOF*node]/l])
        y = np.array([coords[1], coords[1]+vect[DOF*node+1]/l])
        z = np.array([coords[2], coords[2]+vect[DOF*node+2]/l])
        if initial_conditions[node][3] == False:
            ax.plot(x,y,z, color='orange', linewidth=2, label='Displacement Response')
    
    plt.show()

    
    
    # initial_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True], [[1.0, 0.0, 0.0], [0, 0, 0], 1, False], [[2.0, 0.0, 0.0], [0, 0, 0], 1, True], [[1.0,0.0,10.0], [0, 0, 0], 1, True]]
    # connectivity_matrix = [[0, 1, 1.0, 1.0, 'default'], [1, 2, 1.0, 1.0, 'default'],[1,3, 1.0, 1.0, 'noncompressive']]
    # Canopy = FEM_structure(initial_conditions, connectivity_matrix)
    # ax,fig = Canopy.plot(color='red')
    # fext = np.zeros(Canopy.N, dtype=DOUBLE)
    
    # fext[DOF+2] = 10  # Apply force in y-direction at node 1
    
    # Canopy.solve(fe=fext, max_iterations=3000, tolerance=.1, limit=.2, offset=0.1)
    # ax,fig = Canopy.plot(color='blue')
    # plt.show()
    

    
    # initial_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True], [[0.5, 0.0, 0.0], [0, 0, 0], 1, False], [[2.0, 0.0, 0.0], [0, 0, 0], 1, True], [[0.0,1.0,0.0], [0, 0, 0], 1, True],[[1.0,1.0,0.0], [0, 0, 0], 1, True]]
    # connectivity_matrix = [[0, 1, 1.0, 1.0, 'default'], [1, 2, 1.0, 1.0, 'default'],[0,3, 1.0, 1.0, 'noncompressive'],[3,4, 1.0, 1.0, 'noncompressive']]
    
    # initial_conditions = [[[0.0, 0, 0.0], [0, 0, 0], 1, True], [[5.0, 0, 2.5], [0, 0, 0], 1, True], [[10.0, 0, 5.0], [0, 0, 0], 1, True], [[2.5, 2.5, 2.5], [0, 0, 0], 1, False], [[7.5, 2.5, 2.5], [0, 0, 0], 1, False], [[0.0, 5.0, 2.5], [0, 0, 0], 1, True], [[5.0, 5.0, 2.5], [0, 0, 0], 1, True], [[10.0, 5.0, 2.5], [0, 0, 0], 1, True], [[2.5, 7.5, 2.5], [0, 0, 0], 1, False], [[7.5, 7.5, 2.5], [0, 0, 0], 1, False], [[0.0, 10.0, 5.0], [0, 0, 0], 1, True], [[5.0, 10.0, 2.5], [0, 0, 0], 1, True], [[10.0, 10.0, 0.0], [0, 0, 0], 1, True]]
    # connectivity_matrix = [[3, 0, 9, 1, 'default'], [3, 1, 9, 1, 'default'], [3, 5, 9, 1, 'default'], [3, 6, 9, 1, 'default'], [4, 1, 9, 1, 'default'], [4, 2, 9, 1, 'default'], [4, 6, 9, 1, 'default'], [4, 7, 9, 1, 'default'], [6, 8, 9, 1, 'default'], [6, 9, 9, 1, 'default'], [8, 5, 9, 1, 'default'], [8, 10, 9, 1, 'default'], [8, 11, 9, 1, 'default'], [9, 7, 9, 1, 'default'], [9, 11, 9, 1, 'default'], [9, 12, 9, 1, 'default']]
    # SaddleForm = FEM_structure(initial_conditions, connectivity_matrix)
    # ax,fig = SaddleForm.plot(color='red')
    # fe = np.zeros(SaddleForm.N, dtype=DOUBLE)
    # SaddleForm.solve( fe = fe, max_iterations=5, tolerance=.01,limit=.5,offset=0.001)
    # ax,fig = SaddleForm.plot(color='blue')
    # plt.show()
    
    # print(SaddleForm.ncoords_init-SaddleForm.ncoords_current)
    
    # connectivity_matrix = [[19, 2, 50000.0, 0.9, "default"], [2, 18, 50000.0, 0.9, "default"], [18, 1, 50000.0, 0.9, "noncompressive"], [1, 19, 50000.0, 0.9, "default"], [19, 18, 1000.0, 0.9, "noncompressive"], [2, 1, 1000.0, 0.9, "noncompressive"], [2, 3, 50000.0, 0.9, "default"], [3, 17, 50000.0, 0.9, "default"], [17, 18, 50000.0, 0.9, "noncompressive"], [18, 2, 50000.0, 0.9, "default"], [2, 17, 1000.0, 0.9, "noncompressive"], [3, 18, 1000.0, 0.9, "noncompressive"], [3, 4, 50000.0, 0.9, "default"], [4, 16, 50000.0, 0.9, "default"], [16, 17, 50000.0, 0.9, "noncompressive"], [17, 3, 50000.0, 0.9, "default"], [3, 16, 1000.0, 0.9, "noncompressive"], [4, 17, 1000.0, 0.9, "noncompressive"], [4, 5, 50000.0, 0.9, "default"], [5, 15, 50000.0, 0.9, "default"], [15, 16, 50000.0, 0.9, "noncompressive"], [16, 4, 50000.0, 0.9, "default"], [4, 15, 1000.0, 0.9, "noncompressive"], [5, 16, 1000.0, 0.9, "noncompressive"], [5, 6, 50000.0, 0.9, "default"], [6, 14, 50000.0, 0.9, "default"], [14, 15, 50000.0, 0.9, "noncompressive"], [15, 5, 50000.0, 0.9, "default"], [5, 14, 1000.0, 0.9, "noncompressive"], [6, 15, 1000.0, 0.9, "noncompressive"], [6, 7, 50000.0, 0.9, "default"], [7, 13, 50000.0, 0.9, "default"], [13, 14, 50000.0, 0.9, "noncompressive"], [14, 6, 50000.0, 0.9, "default"], [6, 13, 1000.0, 0.9, "noncompressive"], [7, 14, 1000.0, 0.9, "noncompressive"], [7, 8, 50000.0, 0.9, "default"], [8, 12, 50000.0, 0.9, "default"], [12, 13, 50000.0, 0.9, "noncompressive"], [13, 7, 50000.0, 0.9, "default"], [7, 12, 1000.0, 0.9, "noncompressive"], [8, 13, 1000.0, 0.9, "noncompressive"], [8, 9, 50000.0, 0.9, "default"], [9, 11, 50000.0, 0.9, "default"], [11, 12, 50000.0, 0.9, "noncompressive"], [12, 8, 50000.0, 0.9, "default"], [8, 11, 1000.0, 0.9, "noncompressive"], [9, 12, 1000.0, 0.9, "noncompressive"], [9, 20, 50000.0, 0.9, "default"], [20, 10, 50000.0, 0.9, "default"], [10, 11, 50000.0, 0.9, "noncompressive"], [11, 9, 50000.0, 0.9, "default"], [9, 10, 1000.0, 0.9, "noncompressive"], [20, 11, 1000.0, 0.9, "noncompressive"], [0, 21, 50000.0, 0.9, "noncompressive"], [21, 22, 50000.0, 0.9, "noncompressive"], [21, 23, 50000.0, 0.9, "noncompressive"], [21, 27, 50000.0, 0.9, "noncompressive"], [22, 24, 50000.0, 0.9, "pulley"], [22, 28, 50000.0, 0.9, "pulley"], [23, 24, 50000.0, 0.9, "pulley"], [23, 1, 50000.0, 0.9, "noncompressive"], [27, 28, 50000.0, 0.9, "pulley"], [27, 10, 50000.0, 0.9, "noncompressive"], [24, 25, 50000.0, 0.9, "noncompressive"], [24, 26, 50000.0, 0.9, "noncompressive"], [28, 29, 50000.0, 0.9, "noncompressive"], [28, 30, 50000.0, 0.9, "noncompressive"], [25, 18, 50000.0, 0.9, "noncompressive"], [25, 17, 50000.0, 0.9, "noncompressive"], [29, 11, 50000.0, 0.9, "noncompressive"], [29, 12, 50000.0, 0.9, "noncompressive"], [26, 16, 50000.0, 0.9, "noncompressive"], [26, 15, 50000.0, 0.9, "noncompressive"], [30, 13, 50000.0, 0.9, "noncompressive"], [30, 14, 50000.0, 0.9, "noncompressive"], [0, 31, 50000.0, 0.9, "noncompressive"], [0, 34, 50000.0, 0.9, "noncompressive"], [31, 32, 50000.0, 0.9, "noncompressive"], [31, 33, 50000.0, 0.9, "noncompressive"], [34, 35, 50000.0, 0.9, "noncompressive"], [34, 36, 50000.0, 0.9, "noncompressive"], [32, 2, 50000.0, 0.9, "noncompressive"], [32, 3, 50000.0, 0.9, "noncompressive"], [35, 9, 50000.0, 0.9, "noncompressive"], [35, 8, 50000.0, 0.9, "noncompressive"], [33, 4, 50000.0, 0.9, "noncompressive"], [33, 5, 50000.0, 0.9, "noncompressive"], [36, 7, 50000.0, 0.9, "noncompressive"], [36, 6, 50000.0, 0.9, "noncompressive"], [24, 19, 50000.0, 0.9, "noncompressive"], [28, 20, 50000.0, 0.9, "noncompressive"], [31, 19, 50000.0, 0.9, "noncompressive"], [34, 20, 50000.0, 0.9, "noncompressive"]]
    # initial_conditions = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.661345363420986, True], [[1.5450326776836745, 4.130039796867375, 7.261364950271782], [0.0, 0.0, 0.0], 0.6054251074570364, False], [[-0.01769468978128639, 3.9841196033007544, 8.497033387639695], [0.0, 0.0, 0.0], 0.5707213947602356, False], [[-0.23864988658488862, 3.147085371141592, 9.816746227504566], [0.0, 0.0, 0.0], 0.5752129756233141, False], [[-0.3852940131556698, 1.9677011127725468, 10.6073565473802], [0.0, 0.0, 0.0], 0.5760255005323119, False], [[-0.4584192791195412, 0.6669541551721638, 10.948973784300064], [0.0, 0.0, 0.0], 0.5788983834737577, False], [[-0.4584192791195412, -0.6669541551721638, 10.948973784300064], [0.0, 0.0, 0.0], 0.5788983834737577, False], [[-0.3852940131556698, -1.9677011127725468, 10.6073565473802], [0.0, 0.0, 0.0], 0.5760255005323119, False], [[-0.23864988658488862, -3.147085371141592, 9.816746227504566], [0.0, 0.0, 0.0], 0.5752129756233141, False], [[-0.01769468978128639, -3.9841196033007544, 8.497033387639695], [0.0, 0.0, 0.0], 0.5707213947602356, False], [[1.5450326776836745, -4.130039796867375, 7.261364950271782], [0.0, 0.0, 0.0], 0.6054251074570364, False], [[1.7103966474299823, -3.9715968676171474, 8.492040169869373], [0.0, 0.0, 0.0], 0.5654576443671213, False], [[2.0106621688737727, -3.12943184814695, 9.787197879905024], [0.0, 0.0, 0.0], 0.5728514884728007, False], [[2.118729000086964, -1.9545056515510937, 10.55854695742184], [0.0, 0.0, 0.0], 0.575770732822115, False], [[2.16733994663813, -0.6634087911809183, 10.889841638961673], [0.0, 0.0, 0.0], 0.578854001974007, False], [[2.16733994663813, 0.6634087911809183, 10.889841638961673], [0.0, 0.0, 0.0], 0.578854001974007, False], [[2.118729000086964, 1.9545056515510937, 10.55854695742184], [0.0, 0.0, 0.0], 0.575770732822115, False], [[2.0106621688737727, 3.12943184814695, 9.787197879905024], [0.0, 0.0, 0.0], 0.5728514884728007, False], [[1.7103966474299823, 3.9715968676171474, 8.492040169869373], [0.0, 0.0, 0.0], 0.5654576443671213, False], [[0.8630767430175426, 4.1565, 7.423819809051551], [0.0, 0.0, 0.0], 0.6206859968952704, False], [[0.8630767430175426, -4.1565, 7.423819809051551], [0.0, 0.0, 0.0], 0.0706859968952703, False], [[0.17884119251811076, 0.0, 0.9330143770911036], [0.0, 0.0, 0.0], 8.447411093293947, False], [[0.4176659156066993, 0.0, 2.0047264427326446], [0.0, 0.0, 0.0], 0.06165064275796516, False], [[0.4765199236180079, 0.515701635045511, 2.4187272340933115], [0.0, 0.0, 0.0], 0.09102212303016835, False], [[1.0118690598081908, 0.85948465466864, 4.671178520435504], [0.0, 0.0, 0.0], 0.24485626871062102, False], [[1.447067774285811, 2.71993276851239, 7.354965134765166], [0.0, 0.0, 0.0], 0.0680648306878911, False], [[1.5342469795828642, 1.3437933594787979, 7.833401527870664], [0.0, 0.0, 0.0], 0.08390189207122503, False], [[0.4765199236180079, -0.515701635045511, 2.4187272340933115], [0.0, 0.0, 0.0], 0.09102212303016835, False], [[1.0118690598081908, -0.85948465466864, 4.671178520435504], [0.0, 0.0, 0.0], 0.24485626871062102, False], [[1.447067774285811, -2.71993276851239, 7.354965134765166], [0.0, 0.0, 0.0], 0.0680648306878911, False], [[1.5342469795828642, -1.3437933594787979, 7.833401527870664], [0.0, 0.0, 0.0], 0.08390189207122503, False], [[-0.06579255760134889, 1.3268467003328777, 5.531595097127407], [0.0, 0.0, 0.0], 0.12090058106424205, False], [[-0.042661843847079224, 2.0553030306836795, 7.2551007793587665], [0.0, 0.0, 0.0], 0.0628358384714946, False], [[-0.09211118715063556, 1.267408888894257, 7.827716375899725], [0.0, 0.0, 0.0], 0.07567094165901125, False], [[-0.06579255760134889, -1.3268467003328777, 5.531595097127407], [0.0, 0.0, 0.0], 0.12090058106424205, True], [[-0.042661843847079224, -2.0553030306836795, 7.2551007793587665], [0.0, 0.0, 0.0], 0.0628358384714946, False], [[-0.09211118715063556, -1.267408888894257, 7.827716375899725], [0.0, 0.0, 0.0], 0.07567094165901125, False]]
    # fext_raw = [5.907129052020721, 0.0, 1.9526610413851657, 2.015164300914973, 7.446125789778666, 2.17662384683654, -2.241614545265641, 87.04883707570737, 35.19190090029271, -5.088668571284915, 131.06491501351405, 124.35294182705051, -9.516065916794336, 91.82823795380547, 197.11645666147078, -11.14547630385018, 31.955725253992284, 231.2055991164709, -11.145476303849945, -31.95572525399233, 231.20559911647132, -9.51606591679433, -91.82823795380547, 197.11645666147078, -5.088668571284968, -131.06491501351383, 124.35294182705023, -2.241614545265664, -87.04883707570727, 35.19190090029261, 2.015164300914981, -7.446125789778677, 2.176623846836544, -1.0883146061334417, -45.65634960612242, 17.553379013651032, -2.0720126953780893, -62.41186429214944, 59.62901014852281, -4.0488258280559, -43.727732358954995, 94.35066813619626, -4.75204822735654, -15.217012025710634, 110.64644697251512, -4.75204822735665, 15.217012025710615, 110.64644697251491, -4.048825828055902, 43.72773235895499, 94.35066813619626, -2.0720126953780635, 62.41186429214956, 59.62901014852295, -1.0883146061334323, 45.65634960612251, 17.55337901365108, 0.3319544741706002, 39.85502122122765, 6.275311425915698, 0.3319544741706357, -39.85502122122762, 6.275311425915707, 6.943984425887798, 0.0, 1.9541420163112841, 2.7850539901986195, 0.0, 1.5314330527886186, 4.269981430384881, 0.0, 2.1700379712646236, 7.104255704461362, 0.0, 3.0302176450483893, 3.2656607105813302, 0.0, 1.5640348931113555, 3.8675557043145643, 0.0, 2.0449520669576273, 4.269981430384881, 0.0, 2.1700379712646236, 7.104255704461362, 0.0, 3.0302176450483893, 3.2656607105813302, 0.0, 1.5640348931113555, 3.8675557043145643, 0.0, 2.0449520669576273, 6.229349134775813, 0.0, 2.3075755414136094, 3.416395768582101, 0.0, 0.9437198382179653, 4.143191118551853, 0.0, 1.0205688432130426, 6.229349134775813, 0.0, 2.3075755414136094, 3.416395768582101, 0.0, 0.9437198382179653, 4.143191118551853, 0.0, 1.0205688432130426]
    # self.l0_lst = [1.3990213376738259, 1.7281439236583667, 1.2518030597706775, 0.7015380391093075, 1.3759487374332864, 1.9975700275639268, 1.5783185624280678, 2.249575398725647, 1.573796301066927, 1.7281439236583667, 2.551322622085479, 2.496687353016216, 1.4274159893458738, 2.5045334390096428, 1.4096486820629126, 2.249575398725647, 2.7440389803737433, 2.7861953570331908, 1.3468453098963156, 2.626427362762899, 1.3338104411792946, 2.5045334390096428, 2.8804368453103164, 2.907217779250614, 1.3339083103443277, 2.626427362762899, 1.3268175823618367, 2.626427362762899, 2.9441422675022175, 2.9441422675022175, 1.3468453098963156, 2.5045334390096428, 1.3338104411792946, 2.626427362762899, 2.907217779250614, 2.8804368453103164, 1.4274159893458738, 2.249575398725647, 1.4096486820629126, 2.5045334390096428, 2.7861953570331908, 2.7440389803737433, 1.5783185624280678, 1.7281439236583667, 1.573796301066927, 2.249575398725647, 2.496687353016216, 2.551322622085479, 1.3990213376738259, 0.7015380391093075, 1.2518030597706775, 1.7281439236583667, 1.9975700275639268, 1.3759487374332864, 0.95, 1.0979999999999996, 1.6005947322165974, 1.6005947322165974, 2.8638711442513793, 2.8638711442513793, 2.3405816071107988, 6.136472920477778, 2.3405816071107988, 6.136472920477778, 3.294446189671765, 3.2414638608775626, 3.294446189671765, 3.2414638608775626, 1.7114160066670918, 2.5300364156199593, 1.7114160066670918, 2.5300364156199593, 2.853244880514887, 3.194613594536709, 2.853244880514887, 3.194613594536709, 5.688883470891625, 5.688883470891625, 1.8712710898826828, 2.2970412383152796, 1.8712710898826828, 2.2970412383152796, 2.2941999331130436, 2.791491965569317, 2.2941999331130436, 2.791491965569317, 2.8814518651535197, 3.199527358061682, 2.8814518651535197, 3.199527358061682, 4.297613687231579, 4.297613687231579, 3.528491226341037, 3.528491226341037]
    # fext = []
    # i=0
    # j=0
    # for aa in range(len(initial_conditions*6)):
    #     if i < 3:
    #         fext.append(fext_raw[j])
    #         j+= 1
    #     else:
    #         fext.append(0.0)
    #     if i >= 5:
    #         i = 0
    #     i += 1
    # fext = np.array(fext, dtype=DOUBLE)
    
    
    # kite = FEM_structure(initial_conditions, connectivity_matrix)
    # ax, fig = kite.plot(color='red')
    # ax.set_xlim([-5, 5])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([0, 10])
    # kite.solve(fe= fext, max_iterations=1000, tolerance=.1, limit=.1, offset=0.001)
    # # print(fe=fext)
    # ax, fig = kite.plot(color='blue',ax=ax,fig=fig)
    # ax.set_xlim([-5, 5])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([0, 10])
    # plt.show()
    
