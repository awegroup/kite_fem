from SpringElement import SpringElement
from pyfe3d import DOF, INT, DOUBLE, SpringData
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve,lsmr
import numpy as np
import matplotlib.pyplot as plt
from saddle_form_input import initial_conditions, connectivity_matrix

class FEM_structure:
    def __init__(self, initial_conditions, connectivity_matrix):
        self.init_k_KC0 = 0
        self.num_nodes = len(initial_conditions)
        self.num_elements = len(connectivity_matrix)
        self.N = DOF * self.num_nodes
        self.springdata = SpringData()
        self.xyz = np.zeros(self.N, dtype=bool) 
        self.xyz[0::DOF] = self.xyz[1::DOF] = self.xyz[2::DOF] = True  

        
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
        self.ncoords_current = self.ncoords_init.copy()
        
    def __setup_spring_elements(self, connectivity_matrix):
        self.KC0r = np.zeros(self.springdata.KC0_SPARSE_SIZE * self.num_elements, dtype=INT)  
        self.KC0c = np.zeros(self.springdata.KC0_SPARSE_SIZE * self.num_elements, dtype=INT)
        self.KC0v = np.zeros(self.springdata.KC0_SPARSE_SIZE * self.num_elements, dtype=DOUBLE)
        
        self.spring_elements = []
        for (n1,n2,k,c, springtype) in connectivity_matrix:
            spring_element = SpringElement(n1, n2, self.init_k_KC0)
            unit_vector,l0 = spring_element.unit_vector(self.ncoords_init)
            spring_element.set_spring_properties(0, k, springtype) 
            self.spring_elements.append(spring_element)
            self.init_k_KC0 += self.springdata.KC0_SPARSE_SIZE

    def update_stiffness_matrix(self):
        for spring_element in self.spring_elements:
            spring_element.update_KC0(self.KC0r, self.KC0c, self.KC0v, self.ncoords_init)
        self.KC0 = coo_matrix((self.KC0v, (self.KC0r, self.KC0c)), shape=(self.N, self.N)).tocsc()

    def update_internal_forces(self):
        self.fi = np.zeros(self.N, dtype=DOUBLE)
        for spring_element in self.spring_elements:
            fi_element = spring_element.spring_internal_forces(self.ncoords_init)
            bu1 = self.bu[spring_element.spring.c1:spring_element.spring.c1+DOF]
            bu2 = self.bu[spring_element.spring.c2:spring_element.spring.c2+DOF]
            self.fi[spring_element.spring.n1*DOF:(spring_element.spring.n1+1)*DOF] -= fi_element*bu1
            self.fi[spring_element.spring.n2*DOF:(spring_element.spring.n2+1)*DOF] += fi_element*bu2
            
    def solve(self, fe=None):
        if fe is None:
            fe = np.zeros(self.N, dtype=DOUBLE)
        self.fe = fe
        
        
        self.update_internal_forces()
        self.update_stiffness_matrix()
        residual = self.fe - self.fi
        u = np.zeros(self.N, dtype=DOUBLE)
        uu = lsmr(self.KC0[self.bu, :][:, self.bu], residual[self.bu])[0]
        u[self.bu] = uu
        self.ncoords_current += u[self.xyz]
    
    def plot(self,color):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')        
        ax.scatter(self.ncoords_current[0::DOF//2], self.ncoords_current[1::DOF//2], self.ncoords_current[2::DOF//2], color= color)
        
        for spring_element in self.spring_elements:
            n1 = spring_element.spring.n1
            n2 = spring_element.spring.n2
            ax.plot([self.ncoords_current[n1*DOF//2], self.ncoords_current[n2*DOF//2]],
                    [self.ncoords_current[n1*DOF//2+1], self.ncoords_current[n2*DOF//2+1]],
                    [self.ncoords_current[n1*DOF//2+2], self.ncoords_current[n2*DOF//2+2]], color=color)
        
        return ax,fig
    
if __name__ == "__main__":
    initial_conditions = [[[0.0, 0, 0.0], [0, 0, 0], 1, True], [[2.5, 0, 1.25], [0, 0, 0], 1, True], [[5.0, 0, 2.5], [0, 0, 0], 1, True], [[7.5, 0, 3.75], [0, 0, 0], 1, True], [[10.0, 0, 5.0], [0, 0, 0], 1, True], [[1.25, 1.25, 2.5], [0, 0, 0], 1, False], [[3.75, 1.25, 2.5], [0, 0, 0], 1, False], [[6.25, 1.25, 2.5], [0, 0, 0], 1, False], [[8.75, 1.25, 2.5], [0, 0, 0], 1, False], [[0.0, 2.5, 1.25], [0, 0, 0], 1, True], [[2.5, 2.5, 2.5], [0, 0, 0], 1, False], [[5.0, 2.5, 2.5], [0, 0, 0], 1, False], [[7.5, 2.5, 2.5], [0, 0, 0], 1, False], [[10.0, 2.5, 3.75], [0, 0, 0], 1, True], [[1.25, 3.75, 2.5], [0, 0, 0], 1, False], [[3.75, 3.75, 2.5], [0, 0, 0], 1, False], [[6.25, 3.75, 2.5], [0, 0, 0], 1, False], [[8.75, 3.75, 2.5], [0, 0, 0], 1, False], [[0.0, 5.0, 2.5], [0, 0, 0], 1, True], [[2.5, 5.0, 2.5], [0, 0, 0], 1, False], [[5.0, 5.0, 2.5], [0, 0, 0], 1, False], [[7.5, 5.0, 2.5], [0, 0, 0], 1, False], [[10.0, 5.0, 2.5], [0, 0, 0], 1, True], [[1.25, 6.25, 2.5], [0, 0, 0], 1, False], [[3.75, 6.25, 2.5], [0, 0, 0], 1, False], [[6.25, 6.25, 2.5], [0, 0, 0], 1, False], [[8.75, 6.25, 2.5], [0, 0, 0], 1, False], [[0.0, 7.5, 3.75], [0, 0, 0], 1, True], [[2.5, 7.5, 2.5], [0, 0, 0], 1, False], [[5.0, 7.5, 2.5], [0, 0, 0], 1, False], [[7.5, 7.5, 2.5], [0, 0, 0], 1, False], [[10.0, 7.5, 1.25], [0, 0, 0], 1, True], [[1.25, 8.75, 2.5], [0, 0, 0], 1, False], [[3.75, 8.75, 2.5], [0, 0, 0], 1, False], [[6.25, 8.75, 2.5], [0, 0, 0], 1, False], [[8.75, 8.75, 2.5], [0, 0, 0], 1, False], [[0.0, 10.0, 5.0], [0, 0, 0], 1, True], [[2.5, 10.0, 3.75], [0, 0, 0], 1, True], [[5.0, 10.0, 2.5], [0, 0, 0], 1, True], [[7.5, 10.0, 1.25], [0, 0, 0], 1, True], [[10.0, 10.0, 0.0], [0, 0, 0], 1, True]]
    connectivity_matrix = [[5, 0, 9, 1, 'default'], [5, 1, 9, 1, 'default'], [5, 9, 9, 1, 'default'], [5, 10, 9, 1, 'default'], [6, 1, 9, 1, 'default'], [6, 2, 9, 1, 'default'], [6, 10, 9, 1, 'default'], [6, 11, 9, 1, 'default'], [7, 2, 9, 1, 'default'], [7, 3, 9, 1, 'default'], [7, 11, 9, 1, 'default'], [7, 12, 9, 1, 'default'], [8, 3, 9, 1, 'default'], [8, 4, 9, 1, 'default'], [8, 12, 9, 1, 'default'], [8, 13, 9, 1, 'default'], [10, 14, 9, 1, 'default'], [10, 15, 9, 1, 'default'], [11, 15, 9, 1, 'default'], [11, 16, 9, 1, 'default'], [12, 16, 9, 1, 'default'], [12, 17, 9, 1, 'default'], [14, 9, 9, 1, 'default'], [14, 18, 9, 1, 'default'], [14, 19, 9, 1, 'default'], [15, 19, 9, 1, 'default'], [15, 20, 9, 1, 'default'], [16, 20, 9, 1, 'default'], [16, 21, 9, 1, 'default'], [17, 13, 9, 1, 'default'], [17, 21, 9, 1, 'default'], [17, 22, 9, 1, 'default'], [19, 23, 9, 1, 'default'], [19, 24, 9, 1, 'default'], [20, 24, 9, 1, 'default'], [20, 25, 9, 1, 'default'], [21, 25, 9, 1, 'default'], [21, 26, 9, 1, 'default'], [23, 18, 9, 1, 'default'], [23, 27, 9, 1, 'default'], [23, 28, 9, 1, 'default'], [24, 28, 9, 1, 'default'], [24, 29, 9, 1, 'default'], [25, 29, 9, 1, 'default'], [25, 30, 9, 1, 'default'], [26, 22, 9, 1, 'default'], [26, 30, 9, 1, 'default'], [26, 31, 9, 1, 'default'], [28, 32, 9, 1, 'default'], [28, 33, 9, 1, 'default'], [29, 33, 9, 1, 'default'], [29, 34, 9, 1, 'default'], [30, 34, 9, 1, 'default'], [30, 35, 9, 1, 'default'], [32, 27, 9, 1, 'default'], [32, 36, 9, 1, 'default'], [32, 37, 9, 1, 'default'], [33, 37, 9, 1, 'default'], [33, 38, 9, 1, 'default'], [34, 38, 9, 1, 'default'], [34, 39, 9, 1, 'default'], [35, 31, 9, 1, 'default'], [35, 39, 9, 1, 'default'], [35, 40, 9, 1, 'default']]
    SaddleForm = FEM_structure(initial_conditions, connectivity_matrix)
    ax,fig = SaddleForm.plot(color='red')
    plt.show()