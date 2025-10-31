from kite_fem.SpringElement import SpringElement
from kite_fem.BeamElement import BeamElement
from pyfe3d import DOF, INT, DOUBLE, SpringData, BeamCData
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import lsqr, spsolve
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings


class FEM_structure:
    def __init__(self, initial_conditions, spring_matrix=None, pulley_matrix=None, beam_matrix=None):
        #Determining number of nodes,DOF's and elements
        self.num_nodes = len(initial_conditions)
        self.N = DOF * self.num_nodes
        num_spring_elements = 0
        num_beam_elements = 0
        if spring_matrix is not None:
            num_spring_elements += len(spring_matrix)
        if pulley_matrix is not None:
            num_spring_elements += 2*len(pulley_matrix)
        if beam_matrix is not None:
            num_beam_elements += len(beam_matrix)

        #allocating spare arrays for stiffness matrix
        self.__springdata = SpringData()
        self.__beamdata = BeamCData()
        array_size = (self.__springdata.KC0_SPARSE_SIZE * num_spring_elements + self.__beamdata.KC0_SPARSE_SIZE * num_beam_elements)
        self.__KC0r = np.zeros(array_size, dtype=INT)
        self.__KC0c = np.zeros(array_size, dtype=INT)
        self.__KC0v = np.zeros(array_size, dtype=DOUBLE)
        self.__init_KC0 = 0

        #Setting up the initial conditions and elements
        self.__setup_initial_conditions(initial_conditions)
        self.spring_elements = []
        self.beam_elements = []
        if spring_matrix is not None:
            self.__setup_spring_elements(spring_matrix)
        if pulley_matrix is not None:
            self.__setup_pulley_elements(pulley_matrix)
        if beam_matrix is not None:
            self.__setup_beam_elements(beam_matrix)
        #Overwriting boundary conditions from elements with fixed nodes from initial_conditions
        self.bc = np.where(self.fixed == True, False, self.bc)

        #mask to extract coords array from coords_rotations array
        self.__coordmask = np.zeros(self.N, dtype=bool)
        self.__coordmask[0::DOF] = self.__coordmask[1::DOF] = self.__coordmask[2::DOF] = True
        #Allocating force arrays
        self.fe = np.zeros(self.N, dtype=DOUBLE)
        self.fi = np.zeros(self.N, dtype=DOUBLE)
        #Identity matrix for stiffness improvement
        self.__identity_matrix = identity(self.N, format="csc")
        self.__I_stiffness = 0

    def __setup_initial_conditions(self, initial_conditions):
        #sets up initial positions, velocities, masses and fixed nodes. Velocities and masses are not used, but were included to match PSS inputs (https://github.com/awegroup/Particle_System_Simulator)
        self.fixed = np.zeros(self.N, dtype=bool)
        self.coords_init = np.zeros((self.num_nodes, 3), dtype=np.float64)
        self.coords_rotations_init = np.zeros((self.num_nodes, 6), dtype=np.float64)
        #assigning all initial conditions, and setting fixed DOF's
        for id, (pos, vel, mass, fixed) in enumerate(initial_conditions):
            self.coords_init[id] = pos
            self.coords_rotations_init[id] = np.concatenate([pos, [0, 0, 0]])
            if fixed == True:
                self.fixed[DOF * id : DOF * id + 6] = True
        #Initalising coords (translational DOF's) and coords_rotation (translational+rotational DOF's) flat arrays
        self.coords_init = self.coords_init.flatten()
        self.coords_current = self.coords_init.flatten()
        self.coords_rotations_init = self.coords_rotations_init.flatten()
        self.coords_rotations_current = self.coords_rotations_init.flatten()
        #allocating displacement array for reinitialisation
        self.displacement_reinit = np.zeros(self.N, dtype=DOUBLE)
        #Initialising boundary conditions array (True = free DOF, False = fixed DOF)
        self.bc = np.ones(self.N, dtype=bool)

    def __setup_spring_elements(self, connectivity_matrix):
        for n1, n2, k, c, l0, springtype in connectivity_matrix:
            #initialise sprint element and assign properties
            spring_element = SpringElement(n1, n2, self.__init_KC0)
            spring_element.set_spring_properties(l0, k, springtype)
            self.spring_elements.append(spring_element)
            #update index for sparse stiffness matrix (required for pyfe3d)
            self.__init_KC0 += self.__springdata.KC0_SPARSE_SIZE
            #fixes rotational DOF's for the nodes connected by the spring
            for id in [n1, n2]:
                self.bc[DOF * id+3 : DOF * id + 6] = False

    def __setup_pulley_elements(self, connectivity_matrix):
        for n1, n2, n3, k, c, l0 in connectivity_matrix:
            #initialise pulley as two spring elements
            i_other_pulley = len(self.spring_elements) + 1
            spring_element = SpringElement(n1, n2, self.__init_KC0)
            #first spring gets the index of the second spring to later access its length
            spring_element.set_spring_properties(l0, k, "pulley", i_other_pulley)
            self.spring_elements.append(spring_element)
            i_other_pulley -= 1
            self.__init_KC0 += self.__springdata.KC0_SPARSE_SIZE
            spring_element = SpringElement(n2, n3, self.__init_KC0)
            #second spring gets the index of the second spring to later access its length
            spring_element.set_spring_properties(l0, k, "pulley", i_other_pulley)
            self.spring_elements.append(spring_element)
            #update index for sparse stiffness matrix (required for pyfe3d)
            self.__init_KC0 += self.__springdata.KC0_SPARSE_SIZE
            #fixes rotational DOF's for the nodes connected by the springs
            for id in [n1, n2, n3]:
                self.bc[DOF * id+3 : DOF * id + 6] = False
                
    def __setup_beam_elements(self, connectivity_matrix): #TODO move L to connectivity matrix
        for n1, n2, d, p in connectivity_matrix:
            #initialise beam element and assign properties
            beam_element = BeamElement(n1, n2, self.__init_KC0,self.N)
            L = beam_element.unit_vector(self.coords_init)[1] 
            beam_element.set_inflatable_beam_properties(d,p,L)
            self.beam_elements.append(beam_element)
            #update index for sparse stiffness matrix (required for pyfe3d)
            self.__init_KC0 += self.__beamdata.KC0_SPARSE_SIZE
            #frees all DOF's for the nodes connected by the beam (overwrites fixed DOF's from spring elements)
            for id in [n1, n2]:
                self.bc[DOF * id+3 : DOF * id + 6] = True
        
    def update_stiffness_matrix(self):
        self.__KC0v *= 0
        #Update stiffness matrix due to spring elements
        for spring_element in self.spring_elements:
            self.__KC0r, self.__KC0c, self.__KC0v = spring_element.update_KC0(self.__KC0r, self.__KC0c, self.__KC0v, self.coords_current)
        #Update stiffness matrix due to beam elements
        for beam_element in self.beam_elements:
            self.__KC0r, self.__KC0c, self.__KC0v = beam_element.update_KC0(self.__KC0r, self.__KC0c, self.__KC0v, self.coords_current)

        if np.count_nonzero(np.isnan(self.__KC0v)) > 0:
            raise ValueError("NaN detected in stiffness matrix")
        
        # Assemble global stiffness matrix        
        self.KC0 = coo_matrix((self.__KC0v, (self.__KC0r, self.__KC0c)), shape=(self.N, self.N)).tocsc()
        # Add identity matrix to improve convergence, this adds stiffness in each DOF
        self.KC0 += self.__identity_matrix*self.__I_stiffness
        # Extract matrix for free DOF's
        self.Kbc= self.KC0[self.bc, :][:, self.bc]
    
    def update_internal_forces(self):
        self.fi = np.zeros(self.N, dtype=DOUBLE)
        #Add spring and pulley internal forces
        for spring_element in self.spring_elements:
            #retrieve internal forces from spring element
            if spring_element.springtype == "pulley":
                #add length of matching spring for pulley systems
                other_element = self.spring_elements[spring_element.i_other_pulley]
                l_other_pulley = other_element.unit_vector(self.coords_current)[1]
                fi_element = spring_element.spring_internal_forces(self.coords_current, l_other_pulley)
            else:
                fi_element = spring_element.spring_internal_forces(self.coords_current)
            #allocation of spring forces to nodes
            self.fi[spring_element.spring.n1 * DOF : (spring_element.spring.n1 + 1) * DOF] -= fi_element
            self.fi[spring_element.spring.n2 * DOF : (spring_element.spring.n2 + 1) * DOF] += fi_element

        #Add beam internal forces
        displacement = self.coords_rotations_current - self.coords_rotations_init
        for beam_element in self.beam_elements:
            self.fi = beam_element.beam_internal_forces(displacement,self.coords_current,self.fi)

        
    def solve(
        self,
        fe=None,                    #external force vector for each DOF (length is self.N), if None then zero vector is used
        max_iterations=100,         #maximum number of iterations
        tolerance=1e-2,             #convergence tolerance in based on norm of residual forces (residual = fe - fi) [N]
        step_limit=0.2,             #maximum displacement or rotation step for each DOF per iteration (important for convergence)
        relax_init=0.5,             #initial relaxation factor to scale displacement updats
        relax_update=0.95,          #relaxation factor update if not converging
        k_update=1,                 #frequency of stiffness matrix updates k_update=1 means updating every iteration.     
        I_stiffness=25,             #identity matrix stiffness addition to improve convergence
        print_info = True,          #print solver timing and convergence info
    ):
        #set timing information
        start_time = time.perf_counter()
        timings = {
            "update_internal_forces": 0.0,
            "update_stiffness": 0.0,
            "linear_solve": 0.0,
        }
        #set external force vector to 0 if no input is given, else use input
        if fe is None:
            self.fe *= 0
        else:
            self.fe = fe

        #set solver parameters
        self.__I_stiffness = I_stiffness
        relax = relax_init

        #set displacements to zero and clear history
        displacement = np.zeros(self.N, dtype=DOUBLE)
        self.iteration_history = []
        self.residual_norm_history = []
        converged = False

        #start of newton-raphson solver
        for iteration in range(max_iterations + 1):
            #calculate internal forces
            t0 = time.perf_counter()
            self.update_internal_forces()
            timings["update_internal_forces"] += time.perf_counter() - t0
            
            #update stiffness matrix, initially and every k_update iterations
            if iteration % k_update == 0:
                t0 = time.perf_counter()
                self.update_stiffness_matrix()
                timings["update_stiffness"] += time.perf_counter() - t0
            
            #determine residual, add norm to history
            residual = self.fe - self.fi
            residual_norm = np.linalg.norm(residual[self.bc])
            self.residual_norm_history.append(residual_norm)
            self.iteration_history.append(iteration)

            #check for convergence
            #TODO: look into chrisfield convergence criteria            
            if residual_norm < tolerance:
                if print_info:
                    print(
                        f"Converged after {iteration} iterations. Residual: {residual_norm:.3g} N"
                    )
                converged = True
                break

            #check for max iterations reached
            if iteration == max_iterations:
                if print_info:
                    print(
                        f"Did not converge after {max_iterations} iterations. Residual: {residual_norm:.3g} N"
                    )
                break

            #update relaxation factor if not converging over the last 10 steps
            if iteration > 10 and self.residual_norm_history[-1] >= min(
                self.residual_norm_history[-10:-1]
            ):
                relax *= relax_update

            #TODO: add decisiom making between lsqr solver and spsolve
            #TODO: test pypardiso spsolve

            #solve the linear system Ku=r for u (displacement delta), use spsolve with fallback on lsqr
            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    displacement_delta = spsolve(self.Kbc, residual[self.bc])
                    #fall back on lsqr solver if spsolve generates warnings
                    if w:  
                        if print_info:
                            print(f"spsolve generated warnings: {[warning.message for warning in w]}. Falling back to lsqr solver.")
                        displacement_delta = lsqr(self.Kbc, residual[self.bc], atol=1e-7, btol=1e-7)[0]
                except Exception as e:
                    #fall back on lsqr solver if spsolve fails
                    if print_info:
                        print(f"spsolve failed with error: {e}. Falling back to lsqr solver.")
                    displacement_delta = lsqr(self.Kbc, residual[self.bc], atol=1e-7, btol=1e-7)[0]
            timings["linear_solve"] += time.perf_counter() - t0

            #relax displacement delta and apply step limits, then update displacement array
            displacement[self.bc] += np.clip(
                displacement_delta * relax, -step_limit, step_limit
            )
            
            #update current coordinates with new displacements
            self.coords_rotations_current = self.coords_rotations_init + self.displacement_reinit + displacement
            self.coords_current = self.coords_rotations_current[self.__coordmask]
            

        if print_info:
            #print timing information
            end_time = time.perf_counter()
            total = end_time - start_time
            print(f"Solver time: {total:.4f} s")
            iters = max(1, len(self.iteration_history))
            print("Timing summary (total / per-iter) [s]:")
            for k, v in timings.items():
                print(f"  {k:22s}: {v:.4f} / {v/iters:.6f}")

        return converged

    def reset(self):
        #Resets the structure to the initial conditions
        self.displacement_reinit *= 0

    def reinitialise(self):
        #Reinitialises the structure, such that the current displacement is the new starting point for the next solve
        self.displacement_reinit =  self.coords_rotations_current - self.coords_rotations_init
        
    def modify_get_spring_rest_length(self, spring_ids = [], new_l0s = []):
        #allows for modifying the rest length of a spring (usefull for power and steering lines), and returns all rest lengths
        for spring_id, new_l0 in zip(spring_ids, new_l0s):
            self.spring_elements[spring_id].l0 = new_l0
        rest_lengths = np.array([spring.l0 for spring in self.spring_elements])
        return rest_lengths   


