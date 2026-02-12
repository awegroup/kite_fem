import matplotlib.pyplot as plt
import numpy as np
from kite_fem.FEMStructure import FEM_structure
from kite_fem.Plotting import plot_structure, plot_convergence


def angle_between(v1, v2):
    """Angle in degrees between two vectors."""
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-30)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def verify_pulley_angles(structure, fe):
    """
    Print pulley angle checks for the 7-node pulley topology.
    """
    x = structure.coords_current.reshape(structure.num_nodes, 3)
    force_vectors = fe.reshape(structure.num_nodes, 6)[:, :3]

    node_checks = [
        ("Node 4", 4, 0, 1, 6),
        ("Node 5", 5, 2, 3, 6),
    ]

    print("\nPulley angle verification (frictionless => equal angles):")
    all_ok = True

    for label, P, A, B, R in node_checks:
        vec_AP = x[P] - x[A]
        vec_BP = x[P] - x[B]
        vec_PR = x[R] - x[P]

        angle_left = angle_between(vec_AP, vec_PR)
        angle_right = angle_between(vec_BP, vec_PR)
        diff = abs(angle_left - angle_right)
        ok = diff < 1.0
        status = "OK" if ok else "MISMATCH"
        if not ok:
            all_ok = False
        print(
            f"  {label}:  angle({A}->{P}->{R}) = {angle_left:6.2f} deg,  "
            f"angle({B}->{P}->{R}) = {angle_right:6.2f} deg,  "
            f"delta = {diff:.2f} deg  [{status}]"
        )

    P, A, B = 6, 4, 5
    vec_AP = x[P] - x[A]
    vec_BP = x[P] - x[B]
    f_dir = force_vectors[P]

    angle_left = angle_between(vec_AP, f_dir)
    angle_right = angle_between(vec_BP, f_dir)
    diff = abs(angle_left - angle_right)
    ok = diff < 1.0
    status = "OK" if ok else "MISMATCH"
    if not ok:
        all_ok = False
    print(
        f"  Node 6:  angle({A}->{P}->f) = {angle_left:6.2f} deg,  "
        f"angle({B}->{P}->f) = {angle_right:6.2f} deg,  "
        f"delta = {diff:.2f} deg  [{status}]"
    )

    if all_ok:
        print("\nPulley angle check PASSED.")
    else:
        print("\nPulley angle check FAILED.")
# Define initial conditions and connectivity matrix
initial_conditions = [[[0.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, 0.0, 0.0], [0, 0, 0], 1, True],[[2.0, 0.0, 0.0], [0, 0, 0], 1, True],[[3.0, 0.0, 0.0], [0, 0, 0], 1, True],[[1.0, -1.0, 0.0], [0, 0, 0], 1, False],[[2.0, -1.0, 0.0], [0, 0, 0], 1, False],[[1.5, -2.0, 0.0], [0, 0, 0], 1, False]]
l0 = 3
pulley_matrix = [[0,4,1,1000,0,l0],[2,5,3,1000,0,l0],[4,6,5,1000,0,l0]]

Pulleys = FEM_structure(initial_conditions, pulley_matrix = pulley_matrix)
fe = np.zeros(Pulleys.N)
fe[(Pulleys.num_nodes-1)*6+1] = -100
fe[(Pulleys.num_nodes-1)*6] = 50

ax1,fig1 = plot_structure(Pulleys,fe=fe, plot_displacements=True,fe_magnitude=0.35,plot_2d=True)
Pulleys.solve(fe = fe, tolerance=1e-8, max_iterations=5000, step_limit=0.3, relax_init=0.5,relax_update=0.95, k_update=1,I_stiffness=1)
verify_pulley_angles(Pulleys, fe)
fi = Pulleys.fi
res = fe-fi
ax2,fig2 = plot_structure(Pulleys, fe=fe, fe_magnitude=0.35,plot_2d=True,plot_internal_forces=True,plot_external_forces=True)
ax3,fig3 = plot_convergence(Pulleys)

ax1.set_title("Initial Configuration")
ax2.set_title("Deformed Configuration")
ax3.set_title("Convergence History")
ax1.legend()
ax2.legend()
plt.show()
