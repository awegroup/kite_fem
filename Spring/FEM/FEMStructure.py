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
            
    def solve(self, fe=None, max_iterations=1000, tolerance=1e-2,limit=1e-5,offset=0.001,relax=0.5):
        if fe is None:
            fe = np.zeros(self.N, dtype=DOUBLE)
        self.fe = fe
        u = np.zeros(self.N, dtype=DOUBLE)
        uu = u[self.bu]
        residual = 0
        for iteration in range(max_iterations):
            self.update_internal_forces()
            self.update_stiffness_matrix()
            
            residual_pure = self.fe - self.fi
            residual = (self.fe-self.fi)[self.bu]
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
            duu = self.__sparse_solve(self.KC0, residual_pure)
            
            # if any(np.isnan(duu)):
            #     duu = np.where(np.isnan(duu), self.rng.uniform(-offset,offset,size=np.size(duu)), duu)  

            uu += np.clip(duu[self.bu]*relax,-limit,limit) #TODO Add line search?
            

            u[self.bu] = uu
            self.ncoords_current = self.ncoords_init + u[self.xyz]
            
    def __sparse_solve(self, K, F):
        # try:
        #     with warnings.catch_warnings():
        #         warnings.filterwarnings("error", category=MatrixRankWarning)
        #         u = spsolve(K, F)
        # except MatrixRankWarning:
        #     # print("Matrix is singular, using LSQR solver instead.")
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
    SaddleForm.solve( fe = fe, max_iterations=2000, tolerance=10,limit=.01,offset=0.001)
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
    
