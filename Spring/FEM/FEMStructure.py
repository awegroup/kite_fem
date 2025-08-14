from SpringElement import SpringElement
from pyfe3d import DOF, INT, DOUBLE, SpringData
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import lsqr,lsmr
import numpy as np
import matplotlib.pyplot as plt
import time

class FEM_structure:
    def __init__(self, initial_conditions, spring_matrix = None, pulley_matrix = None):
        self.num_nodes = len(initial_conditions)
        self.num_elements = 0
        if spring_matrix is not None:
            self.num_elements += len(spring_matrix)
        if pulley_matrix is not None:
            self.num_elements += len(pulley_matrix)*2
        self.N = DOF * self.num_nodes
        self.__xyz = np.zeros(self.N, dtype=bool) 
        self.__xyz[0::DOF] = self.__xyz[1::DOF] = self.__xyz[2::DOF] = True  
        self.fe = np.zeros(self.N, dtype=DOUBLE)  
        self.fi = np.zeros(self.N, dtype=DOUBLE)  
        self.__springdata = SpringData()
        self.__identity_matrix = identity(self.N, format='csc')*25

        self.__KC0r = np.zeros(self.__springdata.KC0_SPARSE_SIZE * self.num_elements, dtype=INT)  
        self.__KC0c = np.zeros(self.__springdata.KC0_SPARSE_SIZE * self.num_elements, dtype=INT)
        self.__KC0v = np.zeros(self.__springdata.KC0_SPARSE_SIZE * self.num_elements, dtype=DOUBLE)
        self.init_KC0 = 0
        self.spring_elements = []
        self.__setup_initial_conditions(initial_conditions) 
        if spring_matrix is not None:       
            self.__setup_spring_elements(spring_matrix)
        if pulley_matrix is not None:
            self.__setup_pulley_elements(pulley_matrix)

    def __setup_initial_conditions(self, initial_conditions):
        self.__bu = np.zeros(self.N, dtype=bool)
        self.ncoords_init = np.zeros((self.num_nodes, 3), dtype=float)
        for id, (pos,vel,mass,fixed) in enumerate(initial_conditions):
            self.ncoords_init[id] = pos
            if fixed == False:
                self.__bu[DOF*id:DOF*id+3] = True
        
        self.ncoords_init = self.ncoords_init.flatten()
        self.ncoords_current = self.ncoords_init.flatten()
        
    def __setup_spring_elements(self, connectivity_matrix):
        for (n1,n2,k,c,l0, springtype) in connectivity_matrix:

            spring_element = SpringElement(n1, n2, self.init_KC0)
            spring_element.set_spring_properties(l0, k, springtype) 
            self.spring_elements.append(spring_element)
            self.init_KC0 += self.__springdata.KC0_SPARSE_SIZE

    def __setup_pulley_elements(self,connectivity_matrix):
        for (n1,n2,n3,k,c,l0) in connectivity_matrix:
            i_other_pulley = len(self.spring_elements)+1
            spring_element = SpringElement(n1, n2, self.init_KC0)
            spring_element.set_spring_properties(l0, k, "pulley", i_other_pulley) 
            self.spring_elements.append(spring_element)
            i_other_pulley -= 1
            self.init_KC0 += self.__springdata.KC0_SPARSE_SIZE
            spring_element = SpringElement(n2, n3, self.init_KC0)
            spring_element.set_spring_properties(l0, k, "pulley", i_other_pulley) 
            self.spring_elements.append(spring_element)
            self.init_KC0 += self.__springdata.KC0_SPARSE_SIZE

    def __update_stiffness_matrix(self):
        self.__KC0v *= 0  
        for spring_element in self.spring_elements:
            self.__KC0r, self.__KC0c, self.__KC0v = spring_element.update_KC0(self.__KC0r, self.__KC0c, self.__KC0v, self.ncoords_current)
        if np.count_nonzero(np.isnan(self.__KC0v)) > 0:
            raise ValueError(f"NaN detected in stiffness matrix, element: {spring_element.spring.n1}-{spring_element.spring.n2}")
        self.KC0 = coo_matrix((self.__KC0v, (self.__KC0r, self.__KC0c)), shape=(self.N, self.N)).tocsc()
        self.KC0 += self.__identity_matrix
        self.Kuu = self.KC0[self.__bu, :][:, self.__bu]

    def __update_internal_forces(self):
        self.fi = np.zeros(self.N, dtype=DOUBLE)
        for spring_element in self.spring_elements:
            if spring_element.springtype == "pulley":
                other_element = self.spring_elements[spring_element.i_other_pulley]
                l_other_pulley = other_element.unit_vector(self.ncoords_current)[1]
                fi_element = spring_element.spring_internal_forces(self.ncoords_current, l_other_pulley)
            else:
                fi_element = spring_element.spring_internal_forces(self.ncoords_current)
            bu1 = self.__bu[spring_element.spring.c1:spring_element.spring.c1+DOF]
            bu2 = self.__bu[spring_element.spring.c2:spring_element.spring.c2+DOF]
            self.fi[spring_element.spring.n1*DOF:(spring_element.spring.n1+1)*DOF] -= fi_element*bu1
            self.fi[spring_element.spring.n2*DOF:(spring_element.spring.n2+1)*DOF] += fi_element*bu2

    def solve(self, fe=None, max_iterations=100, tolerance=1e-2,limit_init=0.2,relax_init=0.5,relax_update=0.95, k_update=1):
        if fe is not None:
            self.fe = fe
        displacement = np.zeros(self.N, dtype=DOUBLE)
        self.iteration_history = []
        self.residual_norm_history = []
        start_time = time.perf_counter()
        timings = {
            "update_internal_forces": 0.0,
            "update_stiffness": 0.0,
            "linear_solve": 0.0,
        }
        relax = relax_init
        limit = limit_init
        for iteration in range(max_iterations+1):
            t0 = time.perf_counter()
            self.__update_internal_forces()
            timings["update_internal_forces"] += time.perf_counter() - t0

            if iteration % k_update == 0:
                t0 = time.perf_counter()
                self.__update_stiffness_matrix()
                timings["update_stiffness"] += time.perf_counter() - t0

            residual = self.fe - self.fi
            residual_norm = np.linalg.norm(residual[self.__bu])
            self.residual_norm_history.append(residual_norm)
            self.iteration_history.append(iteration)
            
            if residual_norm < tolerance:
                print(f"Converged after {iteration} iterations. Residual: {residual_norm}")
                break
                
            if iteration == max_iterations:
                print(f"Did not converge after {max_iterations} iterations. Residual: {residual_norm}")
                break
            
            if iteration > 10 and self.residual_norm_history[-1] >= min(self.residual_norm_history[-10:-1]):
                if iteration % k_update != 0:
                    t0 = time.perf_counter()
                    self.__update_stiffness_matrix()
                    timings["update_stiffness"] += time.perf_counter() - t0
                relax *= relax_update

            t0 = time.perf_counter()
            
            displacement_delta = lsqr(self.Kuu, residual[self.__bu])[0]

            timings["linear_solve"] += time.perf_counter() - t0

            displacement[self.__bu] += np.clip(displacement_delta*relax, -limit, limit)
            
            self.ncoords_current = self.ncoords_init + displacement[self.__xyz]

        end_time = time.perf_counter()
        total = end_time - start_time
        print(f"Solver time: {total:.4f} s")
        iters = max(1, len(self.iteration_history))
        print("Timing summary (total / per-iter) [s]:")
        for k, v in timings.items():
            print(f"  {k:22s}: {v:.4f} / {v/iters:.6f}")
        return
    
    def reinitialise(self):
        self.ncoords_init = self.ncoords_current
        #add rotational DOF's for beam?
    
    def plot_3D(self, color, ax=None, fig=None, plot_forces_displacements=False, fe = None):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')        
        if fe is not None:
            self.fe = fe

        node_types = {True: ("Free Node", color), False: ("Fixed Node", "black")}
        label_set = {"Free Node": False, "Fixed Node": False} 
        for n in range(self.num_nodes):
            label, c = node_types[self.__bu[n * DOF]]
            ax.scatter(self.ncoords_current[n * DOF // 2], self.ncoords_current[n * DOF // 2 + 1], self.ncoords_current[n * DOF // 2 + 2], color=c, label=label if not label_set[label] else None)
            label_set[label] = True  

        for i, spring_element in enumerate(self.spring_elements):
            c = color
            if spring_element.springtype == "pulley":
                c = 'orange'
            # if spring_element.springtype == "noncompressive":
            #     c = 'green'
            n1 = spring_element.spring.n1
            n2 = spring_element.spring.n2
            ax.plot([self.ncoords_current[n1 * DOF // 2], self.ncoords_current[n2 * DOF // 2]], [self.ncoords_current[n1 * DOF // 2 + 1], self.ncoords_current[n2 * DOF // 2 + 1]], [self.ncoords_current[n1 * DOF // 2 + 2], self.ncoords_current[n2 * DOF // 2 + 2]],
            color=c, label="Spring Element" if i == 0 else None)

        if plot_forces_displacements:
            self.__update_internal_forces()
            self.__update_stiffness_matrix()
            residual = self.fe - self.fi
            displacement = lsqr(self.KC0, residual)[0]
            scale = 150
            for node in range(self.num_nodes):
                coords = self.ncoords_current[node * DOF // 2:node * DOF // 2 + 3]
                residual_vector = coords + residual[DOF * node:DOF * node + 3] / scale
                external_force_vector = coords + self.fe[DOF * node:DOF * node + 3] / scale
                displacement_vector = (coords + displacement[DOF * node:DOF * node + 3]*self.__bu[DOF * node:DOF * node + 3])
                ax.plot([coords[0], residual_vector[0]], [coords[1], residual_vector[1]], [coords[2], residual_vector[2]], color='green', linewidth=2, label='Residual Force Vector' if node == 0 else None)
                ax.plot([coords[0], displacement_vector[0]], [coords[1], displacement_vector[1]], [coords[2], displacement_vector[2]], color='orange', linewidth=2, label='Displacement Response' if node == 0 else None)                    
                ax.plot([coords[0], external_force_vector[0]], [coords[1], external_force_vector[1]], [coords[2], external_force_vector[2]], color='red', linewidth=2, label='External Force Vector' if node == 0 else None)
        ax.set(xlabel='X', ylabel='Y', zlabel='Z')
        return ax, fig
    
    def plot_convergence(self, ax=None, fig=None):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        ax.plot(self.iteration_history, self.residual_norm_history)
        ax.set(xlabel='Iteration', ylabel='Residual')
        return ax, fig
    
if __name__ == "__main__":
    # Define initial conditions and connectivity matrix
    initial_conditions = [[[0.0, 0, 0.0], [0, 0, 0], 1, True], [[2.5, 0, 1.25], [0, 0, 0], 1, True], [[5.0, 0, 2.5], [0, 0, 0], 1, True], [[7.5, 0, 3.75], [0, 0, 0], 1, True], [[10.0, 0, 5.0], [0, 0, 0], 1, True], [[1.25, 1.25, 2.5], [0, 0, 0], 1, False], [[3.75, 1.25, 2.5], [0, 0, 0], 1, False], [[6.25, 1.25, 2.5], [0, 0, 0], 1, False], [[8.75, 1.25, 2.5], [0, 0, 0], 1, False], [[0.0, 2.5, 1.25], [0, 0, 0], 1, True], [[2.5, 2.5, 2.5], [0, 0, 0], 1, False], [[5.0, 2.5, 2.5], [0, 0, 0], 1, False], [[7.5, 2.5, 2.5], [0, 0, 0], 1, False], [[10.0, 2.5, 3.75], [0, 0, 0], 1, True], [[1.25, 3.75, 2.5], [0, 0, 0], 1, False], [[3.75, 3.75, 2.5], [0, 0, 0], 1, False], [[6.25, 3.75, 2.5], [0, 0, 0], 1, False], [[8.75, 3.75, 2.5], [0, 0, 0], 1, False], [[0.0, 5.0, 2.5], [0, 0, 0], 1, True], [[2.5, 5.0, 2.5], [0, 0, 0], 1, False], [[5.0, 5.0, 2.5], [0, 0, 0], 1, False], [[7.5, 5.0, 2.5], [0, 0, 0], 1, False], [[10.0, 5.0, 2.5], [0, 0, 0], 1, True], [[1.25, 6.25, 2.5], [0, 0, 0], 1, False], [[3.75, 6.25, 2.5], [0, 0, 0], 1, False], [[6.25, 6.25, 2.5], [0, 0, 0], 1, False], [[8.75, 6.25, 2.5], [0, 0, 0], 1, False], [[0.0, 7.5, 3.75], [0, 0, 0], 1, True], [[2.5, 7.5, 2.5], [0, 0, 0], 1, False], [[5.0, 7.5, 2.5], [0, 0, 0], 1, False], [[7.5, 7.5, 2.5], [0, 0, 0], 1, False], [[10.0, 7.5, 1.25], [0, 0, 0], 1, True], [[1.25, 8.75, 2.5], [0, 0, 0], 1, False], [[3.75, 8.75, 2.5], [0, 0, 0], 1, False], [[6.25, 8.75, 2.5], [0, 0, 0], 1, False], [[8.75, 8.75, 2.5], [0, 0, 0], 1, False], [[0.0, 10.0, 5.0], [0, 0, 0], 1, True], [[2.5, 10.0, 3.75], [0, 0, 0], 1, True], [[5.0, 10.0, 2.5], [0, 0, 0], 1, True], [[7.5, 10.0, 1.25], [0, 0, 0], 1, True], [[10.0, 10.0, 0.0], [0, 0, 0], 1, True]]
    connectivity_matrix = [[0, 5, 9, 1, 0, 'default'], [5, 1, 9, 1, 0, 'default'], [5, 9, 9, 1, 0, 'default'], [5, 10, 9, 1, 0, 'default'], [6, 1, 9, 1, 0, 'default'], [6, 2, 9, 1, 0, 'default'], [6, 10, 9, 1, 0, 'default'], [6, 11, 9, 1, 0, 'default'], [7, 2, 9, 1, 0, 'default'], [7, 3, 9, 1, 0, 'default'], [7, 11, 9, 1, 0, 'default'], [7, 12, 9, 1, 0, 'default'], [8, 3, 9, 1, 0, 'default'], [8, 4, 9, 1, 0, 'default'], [8, 12, 9, 1, 0, 'default'], [8, 13, 9, 1, 0, 'default'], [10, 14, 9, 1, 0, 'default'], [10, 15, 9, 1, 0, 'default'], [11, 15, 9, 1, 0, 'default'], [11, 16, 9, 1, 0, 'default'], [12, 16, 9, 1, 0, 'default'], [12, 17, 9, 1, 0, 'default'], [14, 9, 9, 1, 0, 'default'], [14, 18, 9, 1, 0, 'default'], [14, 19, 9, 1, 0, 'default'], [15, 19, 9, 1, 0, 'default'], [15, 20, 9, 1, 0, 'default'], [16, 20, 9, 1, 0, 'default'], [16, 21, 9, 1, 0, 'default'], [17, 13, 9, 1, 0, 'default'], [17, 21, 9, 1, 0, 'default'], [17, 22, 9, 1, 0, 'default'], [19, 23, 9, 1, 0, 'default'], [19, 24, 9, 1, 0, 'default'], [20, 24, 9, 1, 0, 'default'], [20, 25, 9, 1, 0, 'default'], [21, 25, 9, 1, 0, 'default'], [21, 26, 9, 1, 0, 'default'], [23, 18, 9, 1, 0, 'default'], [23, 27, 9, 1, 0, 'default'], [23, 28, 9, 1, 0, 'default'], [24, 28, 9, 1, 0, 'default'], [24, 29, 9, 1, 0, 'default'], [25, 29, 9, 1, 0, 'default'], [25, 30, 9, 1, 0, 'default'], [26, 22, 9, 1, 0, 'default'], [26, 30, 9, 1, 0, 'default'], [26, 31, 9, 1, 0, 'default'], [28, 32, 9, 1, 0, 'default'], [28, 33, 9, 1, 0, 'default'], [29, 33, 9, 1, 0, 'default'], [29, 34, 9, 1, 0, 'default'], [30, 34, 9, 1, 0, 'default'], [30, 35, 9, 1, 0, 'default'], [32, 27, 9, 1, 0, 'default'], [32, 36, 9, 1, 0, 'default'], [32, 37, 9, 1, 0, 'default'], [33, 37, 9, 1, 0, 'default'], [33, 38, 9, 1, 0, 'default'], [34, 38, 9, 1, 0, 'default'], [34, 39, 9, 1, 0, 'default'], [35, 31, 9, 1, 0, 'default'], [35, 39, 9, 1, 0, 'default'], [35, 40, 9, 1, 0, 'default']] 
    
    # Create FEM structure and solve
    SaddleForm = FEM_structure(initial_conditions, connectivity_matrix)
    ax1, fig1 = SaddleForm.plot_3D(color='red', plot_forces_displacements=True)
    SaddleForm.solve(fe=None, max_iterations=1000, tolerance=.1, limit_init=0.25, relax_init=0.25, relax_update=0.95, k_update=30)
    ax2, fig2 = SaddleForm.plot_3D(color='blue', plot_forces_displacements=True)
    ax3, fig3 = SaddleForm.plot_convergence()

    elements = SaddleForm.spring_elements
    ncoords = SaddleForm.ncoords_current
    u = ncoords
    for i in range(SaddleForm.num_nodes):
        u = np.insert(u, (SaddleForm.num_nodes-i)*3, [0,0,0])
        
    # print(u)
    # fint = np.zeros(SaddleForm.N)
    # L = 50
    # for element in elements:
    #     fint = np.zeros(SaddleForm.N)
    #     element.spring.update_probe_ue(u)
    #     element.spring.update_fint(fint)
    #     fint *= -1
    #     n1 = element.spring.n1
    #     n2 = element.spring.n2
    #     n1x1, n1y1, n1z1 = ncoords[n1*3], ncoords[n1*3+1], ncoords[n1*3+2]
    #     n1x2, n1y2, n1z2 = ncoords[n1*3] + fint[n1*6]/L, ncoords[n1*3+1] + fint[n1*6+1]/L, ncoords[n1*3+2] + fint[n1*6+2]/L
    #     n2x1, n2y1, n2z1 = ncoords[n2*3], ncoords[n2*3+1], ncoords[n2*3+2]
    #     n2x2, n2y2, n2z2 = ncoords[n2*3] + fint[n2*6]/L, ncoords[n2*3+1] + fint[n2*6+1]/L, ncoords[n2*3+2] + fint[n2*6+2]/L
    #     ax2.plot([n1x1,n1x2],[n1y1,n1y2],[n1z1,n1z2], color='orange')
    #     ax2.plot([n2x1,n2x2],[n2y1,n2y2],[n2z1,n2z2], color='orange')
        # print(fint)

    fint = np.zeros(SaddleForm.N)
    for element in elements:
        # element.spring.update_probe_ue(u)
        element.spring.update_probe_ue(u)
        element.spring.update_fint(fint)
            
    fint *= -1
    L = 25
    for node in range(len(initial_conditions)):
        x = ncoords[node * 3]
        x2 = x + fint[node * 6]/L
        y = ncoords[node * 3 + 1]
        y2 = y + fint[node * 6 + 1]/L
        z = ncoords[node * 3 + 2]
        z2 = z + fint[node * 6 + 2]/L
        if initial_conditions[node][3] == False:
            ax1.plot([x,x2], [y,y2], [z,z2], color='purple')
        # ax1.plot([x,x2], [y,y2], [z,z2], color='purple')
    
    
    # Add legends and show plots
    ax1.legend()
    ax1.set_xlim([-1,11])
    ax1.set_ylim([-1,11])
    ax1.set_zlim([-1,6])
    # ax2.legend()
    # ax3.grid()
    plt.show()



