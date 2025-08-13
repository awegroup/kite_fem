import numpy as np
import matplotlib.pyplot as plt

# --- Element Properties and Problem Definition ---
E = 1 # Young's modulus 
A = 1 # Cross-sectional area 
L0 = 1.0       # Unstressed length of the spring (m)
Lc = L0 # Initial length of the spring (m)
k_spring = E * A / L0 # Spring constant (N/m)

# Nodal Coordinates
# Node 1 is fixed at the origin
p1_initial = np.array([0.0, 0.0, 0.0])
# Node 2 initial position
p2_initial = np.array([2, .1,-.3])

# Applied external force at Node 2
F_ext_node2 = np.array([1.0, 1.0, 0.0]) # (Fx, Fy, Fz)

# --- Helper Functions ---
def get_current_geometry(p1, p2):
    """Calculates current length and direction cosines."""
    vec_12 = p2 - p1
    Lc = np.linalg.norm(vec_12)
    
    cx = vec_12[0] / Lc
    cy = vec_12[1] / Lc
    cz = vec_12[2] / Lc 
    return Lc, np.array([cx, cy, cz])

def get_elastic_stiffness_Ke22(k_spring_const, cx, cy, cz):
    """Forms the 3x3 elastic stiffness sub-matrix for Node 2."""
    S0_mat = np.array([
        [cx**2, cx*cy, cx*cz],
        [cy*cx, cy**2, cy*cz],
        [cz*cx, cz*cy, cz**2]
    ])
    return k_spring_const * S0_mat

def get_geometric_stiffness_Kg22(P_force, Lc, cx, cy, cz):
    """Forms the 3x3 geometric stiffness sub-matrix for Node 2."""
    if Lc == 0:
        return np.zeros((3,3)) # No geometric stiffness if length is zero
        
    G0_sub_mat = np.array([
        [1 , 0,  0 ],
        [0,  1 , 0 ],
        [0,  0 , 1 ]
    ])
    return (P_force / Lc) * G0_sub_mat

# --- Newton-Raphson Solver ---
p1_current = np.copy(p1_initial)
p2_current = np.copy(p2_initial) # Current position of Node 2, starts at initial

# Solver parameters
num_iterations = 50
tolerance = 1e-8
under_relaxation =1 # Can help with convergence for some problems

print("Starting Newton-Raphson iterations...")
print("------------------------------------")
print(f"Initial P2: {p2_current}")

history_p2 = [np.copy(p2_current)]

for i in range(num_iterations):
    # 1. Calculate current geometry
    Lc, dir_cosines = get_current_geometry(p1_current, p2_current)
    cx, cy, cz = dir_cosines[0], dir_cosines[1], dir_cosines[2]
    k_spring_iterate = E * A *Lc**2/ L0**3
    # 2. Calculate internal force in the spring
    P_axial_force = k_spring * (Lc - L0) # Positive for tension

    # 3. Internal force vector exerted by the spring on Node 2
    # (This force opposes elongation if P_axial_force is positive)
    F_int_node2 = P_axial_force * dir_cosines

    # 4. Calculate residual force vector at Node 2
    Residual_node2 = F_ext_node2 - F_int_node2
    residual_norm = np.linalg.norm(Residual_node2)

    print(f"Iter {i+1:2d}: P2=({p2_current[0]:.6f}, {p2_current[1]:.6f}, {p2_current[2]:.6f}), Lc={Lc:.6f}, P_axial={P_axial_force:.6f}, Res_norm={residual_norm:.4e}")

    # 5. Check for convergence
    if residual_norm < tolerance:
        print(f"\nConverged after {i+1} iterations.")
        break

    # 6. Form tangent stiffness matrix for Node 2's DOFs
    Ke_22 = get_elastic_stiffness_Ke22(k_spring_iterate, cx, cy, cz)
    Kg_22 = get_geometric_stiffness_Kg22(P_axial_force, Lc-L0, cx, cy, cz)
    KT_22 = Ke_22 + Kg_22

    # 7. Solve for incremental displacement of Node 2
    # delta_u_node2 = np.linalg.solve(KT_22, Residual_node2) # Can be unstable if KT_22 is singular
    try:
        # Using pseudo-inverse for stability, though for a well-posed problem direct solve is fine.
        # For this simple problem, KT_22 should be invertible unless Lc becomes zero or P_axial_force is pathologically large.
        delta_u_node2 = np.linalg.solve(KT_22, Residual_node2)
    except np.linalg.LinAlgError:
        print("Singular matrix encountered. Using pseudo-inverse.")
        delta_u_node2 = np.linalg.pinv(KT_22) @ Residual_node2
        if np.linalg.norm(delta_u_node2) > L0 * 10: # Heuristic for divergence
            print("Warning: Large displacement increment, solution might be diverging.")


    # 8. Update current position of Node 2
    p2_current = p2_current + under_relaxation * delta_u_node2
    history_p2.append(np.copy(p2_current))

else: # If loop finishes without breaking
    print(f"\nDid not converge after {num_iterations} iterations. Final residual norm: {residual_norm:.4e}")

print("------------------------------------")
print(f"Final Node 2 position: ({p2_current[0]:.6f}, {p2_current[1]:.6f}, {p2_current[2]:.6f})")

# Analytical solution for comparison
F_mag = np.linalg.norm(F_ext_node2)
L_final_analytical = L0 + F_mag / k_spring
dir_force = F_ext_node2 / F_mag
p2_final_analytical = p1_initial + L_final_analytical * dir_force
print(f"Analytical Node 2 pos: ({p2_final_analytical[0]:.6f}, {p2_final_analytical[1]:.6f}, {p2_final_analytical[2]:.6f})")


# --- Plotting the result (2D projection on XY plane) ---
plt.figure(figsize=(8, 8))

# Initial state
plt.plot([p1_initial[0], p2_initial[0]], [p1_initial[1], p2_initial[1]], 'ko-', label='Initial State', linewidth=2, markerfacecolor='blue', markersize=8)
plt.plot(p1_initial[0], p1_initial[1], 'ko', markerfacecolor='black', markersize=8) # Node 1 initial
plt.plot(p2_initial[0], p2_initial[1], 'ko', markerfacecolor='blue', markersize=8)  # Node 2 initial


# Final state
plt.plot([p1_current[0], p2_current[0]], [p1_current[1], p2_current[1]], 'ro-', label='Final State (FEM)', linewidth=2, markerfacecolor='red', markersize=8)
plt.plot(p1_current[0], p1_current[1], 'ko', markerfacecolor='black', markersize=8) # Node 1 final (same as initial)
plt.plot(p2_current[0], p2_current[1], 'ro', markerfacecolor='red', markersize=8)    # Node 2 final

# Plot analytical solution for reference
plt.plot([p1_initial[0], p2_final_analytical[0]], [p1_initial[1], p2_final_analytical[1]], 'g--', label='Final State (Analytical)', linewidth=1.5)


# Plot path of Node 2 during iteration (optional)
# path_p2 = np.array(history_p2)
# plt.plot(path_p2[:,0], path_p2[:,1], 'c:', label='Node 2 Path (Iterative)')


plt.xlabel('X-coordinate (m)')
plt.ylabel('Y-coordinate (m)')
plt.title(f'3D Axial Spring Deformation (k={k_spring} N/m, L0={L0}m)')
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.grid(True, linestyle=':')
plt.legend()
plt.axis('equal') # Important for correct visual representation of angles

# Determine plot limits dynamically
all_x = [p1_initial[0], p2_initial[0], p2_current[0], p2_final_analytical[0]]
all_y = [p1_initial[1], p2_initial[1], p2_current[1], p2_final_analytical[1]]
x_min, x_max = min(all_x)-0.5, max(all_x)+0.5
y_min, y_max = min(all_y)-0.5, max(all_y)+0.5
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.show()