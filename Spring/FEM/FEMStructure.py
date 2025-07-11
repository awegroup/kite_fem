from SpringElement import SpringElement
from pyfe3d import DOF, INT, DOUBLE, SpringData
import numpy as np

class FEM_structure:
    def __init__(self, initial_conditions, connectivity_matrix):
        self.init_k_KC0 = 0
        self.num_nodes = len(initial_conditions)
        self.num_elements = len(connectivity_matrix)
        self.N = DOF * self.num_nodes
        self.springdata = SpringData()
        
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
        
    def __setup_spring_elements(self, connectivity_matrix):
        self.KC0r = np.zeros(self.springdata.KC0_SPARSE_SIZE * self.num_elements, dtype=INT)  
        self.KC0c = np.zeros(self.springdata.KC0_SPARSE_SIZE * self.num_elements, dtype=INT)
        self.KC0v = np.zeros(self.springdata.KC0_SPARSE_SIZE * self.num_elements, dtype=DOUBLE)
        
        self.spring_elements = []
        for (n1,n2,k,c, springtype) in connectivity_matrix:
            spring_element = SpringElement(n1, n2, self.init_k_KC0)
            unit_vector,l0 = spring_element.unit_vector(self.ncoords_init)
            spring_element.set_spring_properties(l0, k, springtype) 
            self.spring_elements.append(spring_element)
            self.init_k_KC0 += self.springdata.KC0_SPARSE_SIZE



if __name__ == "__main__":
    initial_conditions = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, True], [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, False]]
    connectivity_matrix = [[0, 1, 1.0 , 0.0 , "default"]]

    Example = FEM_structure(initial_conditions,connectivity_matrix)
