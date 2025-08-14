import numpy as np
import matplotlib.pyplot as plt

from PSS.particleSystem.ParticleSystem import ParticleSystem

# dictionary of required parameters
params = {
    # model parameters
    "c": 100,  # [N s/m] damping coefficient
    "m_segment": 4,  # [kg] mass of each node
    # simulation settings
    "dt": 0.05,  # [s]       simulation timestep
    "t_steps": 1000,  # [-]      number of simulated time steps
    "abs_tol": 1e-50,  # [m/s]     absolute error tolerance iterative solver
    "rel_tol": 1e-5,  # [-]       relative error tolerance iterative solver
    "max_iter": 1e5,  # [-]       maximum number of iterations
}

#Inputs
params["k"] = 1  # [N/m] spring stiffness
P = 1 # [N] force applied 
L = 1 # [m] length of the spring

# Set up coordinates
params["n"] = 2  # Number of nodes
xyz_coordinates = np.empty((params["n"], 3))
xyz_coordinates[0] = [0, 0, 0] #coordinates of the first node
xyz_coordinates[1] = [L, 0, 0] #coordinates of the second node


# Create the initial conditions for the nodes
initial_conditions = []
initial_conditions.append([xyz_coordinates[0], np.zeros(3), params["m_segment"], True])  # Fixed node 1
initial_conditions.append([xyz_coordinates[1], np.zeros(3), params["m_segment"], False]) # Free node 2

# Create the connections between nodes
connections = []
connections.append([0,1,params["k"], params["c"]])

# Set up the external forces
f_ext = np.empty((params["n"], 3))
f_ext[1] = [P ,0 , 0]  # Force applied to the free node
f_ext = f_ext.flatten()

# Setup the particle system and simulation
PS = ParticleSystem(connections, initial_conditions, params,init_surface=False)

t_vector = np.linspace(
    params["dt"], params["t_steps"] * params["dt"], params["t_steps"]
)
final_step = 0
E_kin = []
f_int = []

# Run the simulation
for step in t_vector:
    PS.kin_damp_sim(f_ext)

    final_step = step
    (
        x,
        v,
    ) = PS.x_v_current
    E_kin.append(np.linalg.norm(v * v))
    f_int.append(np.linalg.norm(PS.f_int))

    converged = False
    if step > 10:
        if np.max(E_kin[-10:-1]) <= 1e-29:
            converged = True
    if converged and step > 1:
        print("Kinetic damping PS converged", step)
        break

particles = PS.particles

final_positions = [
    [particle.x, particle.v, particle.m, particle.fixed, particle.constraint_type] for particle in PS.particles
]

# Plot the results
fig, ax = plt.subplots(figsize=(6,6))

# Plot the nodes
for i, node in enumerate(initial_conditions):
    if node[3]:  # Fixed node
        ax.scatter(node[0][0], node[0][1], color="red", marker="o")
    else:  # Free node
        ax.scatter(node[0][0], node[0][1], color="blue", marker="o")
        
for i, node in enumerate(final_positions):
    if node[3]:  # Fixed node
        ax.scatter(node[0][0], node[0][1], color="red", marker="o")
    else:  # Free node
        ax.scatter(node[0][0], node[0][1], color="blue", marker="o")
        

# Plot the connections between nodes
for i, connection in enumerate(connections):
    line = np.column_stack(
        [final_positions[connection[0]][0][:2], final_positions[connection[1]][0][:2]]
    )
    ax.plot(line[0], line[1], color="red",label="Final state" if i == 0 else "")
    line = np.column_stack(
        [initial_conditions[connection[0]][0][:2], initial_conditions[connection[1]][0][:2]]
    )
    ax.plot(line[0], line[1], color="black",label="Initial state" if i == 0 else "")

print("Spring stiffness:", params["k"], "[N/m]")
print("Spring rest length:", L, "[m]")
print("Force applied:", P, "[N]")
print("Elongation of the spring:", final_positions[1][0][0] - initial_conditions[1][0][0], "[m]")

plt.legend()
plt.show()