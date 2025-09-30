from kite_fem.SpringElement import SpringElement
from kite_fem.BeamElement import BeamElement
from pyfe3d import DOF, INT, DOUBLE, SpringData, BeamCData
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import lsqr, lsmr
import numpy as np
import matplotlib.pyplot as plt
import time


class FEM_structure:
    def __init__(self, initial_conditions, spring_matrix=None, pulley_matrix=None, beam_matrix=None):
        self.num_nodes = len(initial_conditions)
        self.num_spring_elements = 0
        self.num_beam_elements = 0
        if spring_matrix is not None:
            self.num_spring_elements += len(spring_matrix)
        if pulley_matrix is not None:
            self.num_spring_elements += 2*len(pulley_matrix)
        if beam_matrix is not None:
            self.num_beam_elements += len(beam_matrix)
        self.num_elements = self.num_spring_elements + self.num_beam_elements
        self.N = DOF * self.num_nodes
        self.__xyz = np.zeros(self.N, dtype=bool)
        self.__xyz[0::DOF] = self.__xyz[1::DOF] = self.__xyz[2::DOF] = True
        self.fe = np.zeros(self.N, dtype=DOUBLE)
        self.fi = np.zeros(self.N, dtype=DOUBLE)
        self.fi_beams = np.zeros(self.N, dtype=DOUBLE)
        self.__springdata = SpringData()
        self.__beamdata = BeamCData()
        self.__identity_matrix = identity(self.N, format="csc")
        array_size = (self.__springdata.KC0_SPARSE_SIZE * self.num_spring_elements + self.__beamdata.KC0_SPARSE_SIZE * self.num_beam_elements)
        self.__KC0r = np.zeros(array_size, dtype=INT)
        self.__KC0c = np.zeros(array_size, dtype=INT)
        self.__KC0v = np.zeros(array_size, dtype=DOUBLE)
        self.I_stiffness = 0
        self.init_KC0 = 0
        self.spring_elements = []
        self.beam_elements = []
        self.__bu = np.ones(self.N, dtype=bool)
        self.__setup_initial_conditions(initial_conditions)
        if spring_matrix is not None:
            self.__setup_spring_elements(spring_matrix)
        if pulley_matrix is not None:
            self.__setup_pulley_elements(pulley_matrix)
        if beam_matrix is not None:
            self.__setup_beam_elements(beam_matrix)
        self.__bu = np.where(self.__fixed == True, False, self.__bu)

    def __setup_initial_conditions(self, initial_conditions):
        self.__fixed = np.zeros(self.N, dtype=bool)
        self.coords_init = np.zeros((self.num_nodes, 3), dtype=float)
        self.coords_rotations_init = np.zeros((self.num_nodes, 6), dtype=float)
        for id, (pos, vel, mass, fixed) in enumerate(initial_conditions):
            self.coords_init[id] = pos
            self.coords_rotations_init[id] = np.concatenate([pos, [0, 0, 0]])
            if fixed == True:
                self.__fixed[DOF * id : DOF * id + 6] = True
        self.coords_init = self.coords_init.flatten()
        self.coords_current = self.coords_init.flatten()
        self.coords_rotations_init = self.coords_rotations_init.flatten()
        self.coords_rotations_current = self.coords_rotations_init.flatten()
        self.coords_rotations_previous = self.coords_rotations_init.flatten()
        
    def __setup_spring_elements(self, connectivity_matrix):
        for n1, n2, k, c, l0, springtype in connectivity_matrix:

            spring_element = SpringElement(n1, n2, self.init_KC0)
            spring_element.set_spring_properties(l0, k, springtype)
            self.spring_elements.append(spring_element)
            self.init_KC0 += self.__springdata.KC0_SPARSE_SIZE
            for id in [n1, n2]:
                self.__bu[DOF * id+3 : DOF * id + 6] = False

    def __setup_pulley_elements(self, connectivity_matrix):
        for n1, n2, n3, k, c, l0 in connectivity_matrix:
            i_other_pulley = len(self.spring_elements) + 1
            spring_element = SpringElement(n1, n2, self.init_KC0)
            spring_element.set_spring_properties(l0, k, "pulley", i_other_pulley)
            self.spring_elements.append(spring_element)
            i_other_pulley -= 1
            self.init_KC0 += self.__springdata.KC0_SPARSE_SIZE
            spring_element = SpringElement(n2, n3, self.init_KC0)
            spring_element.set_spring_properties(l0, k, "pulley", i_other_pulley)
            self.spring_elements.append(spring_element)
            self.init_KC0 += self.__springdata.KC0_SPARSE_SIZE
            for id in [n1, n2, n3]:
                self.__bu[DOF * id+3 : DOF * id + 6] = False
                
    def __setup_beam_elements(self, connectivity_matrix):
        for n1, n2, E, A, I in connectivity_matrix:
            beam_element = BeamElement(n1, n2, self.init_KC0,self.N)
            L = beam_element.unit_vector(self.coords_init)[1]
            beam_element.set_beam_properties(E, A, I,L)
            self.beam_elements.append(beam_element)
            self.init_KC0 += self.__beamdata.KC0_SPARSE_SIZE
            for id in [n1, n2]:
                self.__bu[DOF * id+3 : DOF * id + 6] = True
        
    def __update_stiffness_matrix(self):
        self.__KC0v *= 0
        for spring_element in self.spring_elements:
            self.__KC0r, self.__KC0c, self.__KC0v = spring_element.update_KC0(self.__KC0r, self.__KC0c, self.__KC0v, self.coords_current)

        for beam_element in self.beam_elements:
            self.__KC0r, self.__KC0c, self.__KC0v = beam_element.update_KC0(self.__KC0r, self.__KC0c, self.__KC0v, self.coords_current)
            
        if np.count_nonzero(np.isnan(self.__KC0v)) > 0:
            print(f"NaN detected in stiffness matrix")
                
        self.KC0 = coo_matrix((self.__KC0v, (self.__KC0r, self.__KC0c)), shape=(self.N, self.N)).tocsc()
        self.KC0 += self.__identity_matrix*self.I_stiffness
        self.Kuu = self.KC0[self.__bu, :][:, self.__bu]

    def __update_internal_forces(self):
        self.fi = np.zeros(self.N, dtype=DOUBLE)
        self.fi_beams = np.zeros(self.N, dtype=DOUBLE)
        for spring_element in self.spring_elements:
            if spring_element.springtype == "pulley":
                other_element = self.spring_elements[spring_element.i_other_pulley]
                l_other_pulley = other_element.unit_vector(self.coords_current)[1]
                fi_element = spring_element.spring_internal_forces(self.coords_current, l_other_pulley)
            else:
                fi_element = spring_element.spring_internal_forces(self.coords_current)
            bu1 = self.__bu[spring_element.spring.c1 : spring_element.spring.c1 + DOF]
            bu2 = self.__bu[spring_element.spring.c2 : spring_element.spring.c2 + DOF]
            self.fi[spring_element.spring.n1 * DOF : (spring_element.spring.n1 + 1) * DOF] -= (fi_element * bu1)
            self.fi[spring_element.spring.n2 * DOF : (spring_element.spring.n2 + 1) * DOF] += (fi_element * bu2)

        dif = self.coords_rotations_current - self.coords_rotations_previous
        for beam_element in self.beam_elements:
            self.fi_beams += beam_element.beam_internal_forces(dif,self.coords_current)

        self.fi += self.fi_beams
        
    def solve(
        self,
        fe=None,
        max_iterations=100,
        tolerance=1e-2,
        step_limit=0.2,
        relax_init=0.5,
        relax_update=0.95,
        k_update=1,
        I_stiffness=25
    ):
        if fe is not None:
            self.fe = fe
        displacement = np.zeros(self.N, dtype=DOUBLE)
        self.iteration_history = []
        self.residual_norm_history = []
        start_time = time.perf_counter()
        self.I_stiffness = I_stiffness
        timings = {
            "update_internal_forces": 0.0,
            "update_stiffness": 0.0,
            "linear_solve": 0.0,
        }
        relax = relax_init
        for iteration in range(max_iterations + 1):
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
                print(
                    f"Converged after {iteration} iterations. Residual: {residual_norm}"
                )
                break

            if iteration == max_iterations:
                print(
                    f"Did not converge after {max_iterations} iterations. Residual: {residual_norm}"
                )
                break

            if iteration > 10 and self.residual_norm_history[-1] >= min(
                self.residual_norm_history[-10:-1]
            ):
                if iteration % k_update != 0:
                    t0 = time.perf_counter()
                    self.__update_stiffness_matrix()
                    timings["update_stiffness"] += time.perf_counter() - t0
                relax *= relax_update

            t0 = time.perf_counter()

            displacement_delta = lsqr(
                self.Kuu, residual[self.__bu], atol=1e-7, btol=1e-7
            )[0]

            timings["linear_solve"] += time.perf_counter() - t0

            displacement[self.__bu] += np.clip(
                displacement_delta * relax, -step_limit, step_limit
            )
            self.coords_rotations_previous = self.coords_rotations_current
            self.coords_rotations_current = self.coords_rotations_init + displacement
            self.coords_current = self.coords_rotations_current[self.__xyz]

        end_time = time.perf_counter()
        total = end_time - start_time
        print(f"Solver time: {total:.4f} s")
        iters = max(1, len(self.iteration_history))
        print("Timing summary (total / per-iter) [s]:")
        for k, v in timings.items():
            print(f"  {k:22s}: {v:.4f} / {v/iters:.6f}")


        return

    def reset(self):
        for beam_element in self.beam_elements:
            beam_element.beam.probe.ue=np.array(beam_element.beam.probe.ue)*0
            beam_element.fi *=0
        self.coords_current = self.coords_init
        self.coords_rotations_current = self.coords_rotations_init
        self.coords_rotations_previous = self.coords_rotations_init

    def reinitialise(self):
        self.coords_init = self.coords_current
        self.coords_rotations_init = self.coords_rotations_current
        self.coords_rotations_previous = self.coords_rotations_current
        self.reinitialised = True

    def plot_3D(
        self, color="blue", ax=None, fig=None, plot_forces_displacements=False, fe=None, show_plot=True, show_legend=True
    ):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection="3d")
        if fe is not None:
            self.fe = fe

        node_types = {True: ("Free Node", color), False: ("Fixed Node", "black")}
        label_set = {"Free Node": False, "Fixed Node": False}
        for n in range(self.num_nodes):
            label, c = node_types[self.__bu[n * DOF]]
            ax.scatter(
                self.coords_current[n * DOF // 2],
                self.coords_current[n * DOF // 2 + 1],
                self.coords_current[n * DOF // 2 + 2],
                color=c,
                label=label if not label_set[label] else None,
            )
            label_set[label] = True

        for i, spring_element in enumerate(self.spring_elements):
            c = color
            if spring_element.springtype == "pulley":
                c = "orange"
            # if spring_element.springtype == "noncompressive":
            #     c = 'green'
            n1 = spring_element.spring.n1
            n2 = spring_element.spring.n2
            ax.plot(
                [
                    self.coords_current[n1 * DOF // 2],
                    self.coords_current[n2 * DOF // 2],
                ],
                [
                    self.coords_current[n1 * DOF // 2 + 1],
                    self.coords_current[n2 * DOF // 2 + 1],
                ],
                [
                    self.coords_current[n1 * DOF // 2 + 2],
                    self.coords_current[n2 * DOF // 2 + 2],
                ],
                color=c,
                label="Spring Element" if i == 0 else None,
            )

        for i, beam_element in enumerate(self.beam_elements):
            c = color
            n1 = beam_element.beam.n1
            n2 = beam_element.beam.n2
            ax.plot(
                [
                    self.coords_current[n1 * DOF // 2],
                    self.coords_current[n2 * DOF // 2],
                ],
                [
                    self.coords_current[n1 * DOF // 2 + 1],
                    self.coords_current[n2 * DOF // 2 + 1],
                ],
                [
                    self.coords_current[n1 * DOF // 2 + 2],
                    self.coords_current[n2 * DOF // 2 + 2],
                ],
                color=c,
                label="Beam Element" if i == 0 else None,
            )

        if plot_forces_displacements:
            self.__update_internal_forces()
            self.__update_stiffness_matrix()
            residual = self.fe - self.fi
            displacement = lsqr(self.KC0, residual)[0]
            scale = 250
            for node in range(self.num_nodes):
                coords = self.coords_current[node * DOF // 2 : node * DOF // 2 + 3]
                residual_vector = coords + residual[DOF * node : DOF * node + 3] / scale
                external_force_vector = (
                    coords + self.fe[DOF * node : DOF * node + 3] / scale
                )
                displacement_vector = (
                    coords
                    + displacement[DOF * node : DOF * node + 3]
                    * self.__bu[DOF * node : DOF * node + 3]
                )
                ax.plot(
                    [coords[0], residual_vector[0]],
                    [coords[1], residual_vector[1]],
                    [coords[2], residual_vector[2]],
                    color="green",
                    linewidth=2,
                    label="Residual Force Vector" if node == 0 else None,
                )
                ax.plot(
                    [coords[0], displacement_vector[0]],
                    [coords[1], displacement_vector[1]],
                    [coords[2], displacement_vector[2]],
                    color="orange",
                    linewidth=2,
                    label="Displacement Response" if node == 0 else None,
                )
                ax.plot(
                    [coords[0], external_force_vector[0]],
                    [coords[1], external_force_vector[1]],
                    [coords[2], external_force_vector[2]],
                    color="red",
                    linewidth=2,
                    label="External Force Vector" if node == 0 else None,
                )
        ax.set(xlabel="X", ylabel="Y", zlabel="Z")
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        xmid = (xlim[0] + xlim[1]) / 2
        ymid = (ylim[0] + ylim[1]) / 2
        zmid = (zlim[0] + zlim[1]) / 2
        maximum = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
        ax.set_xlim([xmid - maximum / 2, xmid + maximum / 2])
        ax.set_ylim([ymid - maximum / 2, ymid + maximum / 2])
        ax.set_zlim([zmid - maximum / 2, zmid + maximum / 2])
        ax.set_box_aspect([1, 1, 1])
        if show_legend:
            ax.legend()
        if show_plot:
            plt.show()
        return ax, fig

    def plot_convergence(self, ax=None, fig=None, show_plot=True):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        ax.plot(self.iteration_history, self.residual_norm_history)
        ax.set(xlabel="Iteration", ylabel="Residual")
        if show_plot:
            plt.show()
        return ax, fig
