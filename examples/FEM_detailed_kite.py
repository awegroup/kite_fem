from pathlib import Path

from kitesim import (
    structural_kite_fem_level_2,
    read_struc_geometry_level_2_yaml,
)
from kitesim.utils import (
    load_yaml,
)
from kite_fem.FEMStructure import FEM_structure
from kite_fem.Plotting import (
    plot_structure,
    plot_structure_with_strain,
    plot_convergence
)
from kite_fem.Functions import tensionbridles, relaxbridles,fix_nodes,adapt_stiffnesses

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
kite_name = "TUDELFT_V3_KITE"  
struc_geometry_path = (
    Path(PROJECT_DIR)
    / "data"
    / f"{kite_name}"
    / "struc_geometry_hanging_test.yaml"
)
struc_geometry = load_yaml(struc_geometry_path)

(
    # node level
    struc_nodes,
    m_arr,
    struc_node_le_indices,
    struc_node_te_indices,
    power_tape_index,
    steering_tape_indices,
    pulley_node_indices,
    canopy_sections,
    strut_sections,
    simplified_bridle_points,
    # element level
    kite_connectivity_arr,
    bridle_connectivity_arr,
    bridle_diameter_arr,
    l0_arr,
    k_arr,
    c_arr,
    linktype_arr,
    pulley_line_indices,
    pulley_line_to_other_node_pair_dict,
) = read_struc_geometry_level_2_yaml.main(struc_geometry)

config = {"is_with_initial_point_velocity": False}
kite = structural_kite_fem_level_2.instantiate(
    config,
    struc_geometry,
    struc_nodes,
    kite_connectivity_arr,
    l0_arr,
    k_arr,
    c_arr,
    m_arr,
    linktype_arr,
    pulley_line_to_other_node_pair_dict,
)[0]

canopy_nodes = list(set([node for section in canopy_sections + strut_sections for node in section]))



kite = fix_nodes(kite,[0,129,128,127,72,81,82,115,116,106])
origin =  [82,116]
kite = relaxbridles(kite,canopy_nodes,origin)
kite = fix_nodes(kite,[0,129,128,127,72,81,82,115,116,106])




ax10,fig10 = plot_structure_with_strain(kite)



gravity = m_arr*10
fe = np.zeros(kite.N)
fe[2::6] = gravity 

# fe[2*6+1] = 50
# fe[56*6+1] = -50

fe[29*6+2] = 250

ax1,fig1 = plot_structure(kite,plot_nodes=False,fe=fe,plot_external_forces=True,linewidth = [1,0.75,1,3.5],plot_node_numbers=True)
ax2,fig2 = plot_structure(kite, plot_nodes=False,plot_displacements=False,solver="spsolve",e_colors = ['black', 'black', 'black', 'black'],linewidth = [1,0.75,1,3.5],plot_2d=True,plot_2d_plane="yz")

# ax1.legend()
# plt.show()
# breakpoint()

kite.solve(fe=fe, max_iterations=10, tolerance=0.01, step_limit=.005, relax_init=.25, relax_min=0.00, relax_update=0.998, k_update=1,I_stiffness=15)
# adapt_stiffnesses(kite)
# kite.solve(fe=fe, max_iterations=10000, tolerance=0.001, step_limit=.005, relax_init=.25, relax_min=0.00, relax_update=0.998, k_update=1,I_stiffness=15)
# adapt_stiffnesses(kite)
# kite.solve(fe=fe, max_iterations=10000, tolerance=0.001, step_limit=.005, relax_init=.25, relax_min=0.00, relax_update=0.998, k_update=1,I_stiffness=15)
# adapt_stiffnesses(kite)


def extract_lengths_validation(kite,strut_sections):
    phi = []
    strut_sections = np.array(strut_sections)
    #extract lengths billowing segments
    coords = kite.coords_current.reshape(-1,3)
    te_ids = strut_sections[:,-1]
    le_ids = strut_sections[:,0]
    te_ids = te_ids[::-1]
    te_id1 = te_ids[0:-1]
    te_id2 = te_ids[1:]
    for id1,id2 in zip(te_id1,te_id2):
        print(id1,id2)
        coord1 = coords[id1]
        coord2 = coords[id2]
        length = np.linalg.norm(coord1-coord2)
        phi.append(length)
    #span
    spanid1 = te_ids[0]
    spanid2 = te_ids[-1]
    print("spanid",spanid1,spanid2)
    coord1 = coords[spanid1]
    coord2 = coords[spanid2]
    span = np.linalg.norm(coord1-coord2)
    phi.append(span)
    #tip_leading_edge distance
    middle_idx = len(le_ids) // 2
    # For uneven length, middle_idx is the exact center
    # Get the points on either side of the center
    left_le_id = le_ids[middle_idx]
    right_le_id = le_ids[middle_idx-1]
    rightids = [right_le_id, te_ids[-1]]
    leftids = [left_le_id, te_ids[0]]
    right = np.linalg.norm(coords[rightids[0]]-coords[rightids[1]])
    left = np.linalg.norm(coords[leftids[0]]-coords[leftids[1]])
    print("tip-front ids")
    print(rightids[0],rightids[1])
    print(leftids[0],leftids[1])
    phi.append(right)
    phi.append(left)
    return phi

phi = extract_lengths_validation(kite,strut_sections)



fi = kite.fi
residual = fe-fi
print(np.max(residual[kite.bc]))

# kite.reinitialise()
# kite.solve(fe=fe, max_iterations=3000, tolerance=5, step_limit=.005, relax_init=.25, relax_min=0.025, relax_update=0.995, k_update=1,I_stiffness=15)

ax3,fig3 = plot_structure(kite,fe=fe,fe_magnitude=1.5, plot_residual_forces=False,plot_external_forces=True,plot_nodes=False,plot_displacements=True,solver="spsolve",linewidth = [1,0.75,1,3.5])
ax2,fig2 = plot_structure(kite, fe=fe,plot_nodes=False,plot_external_forces=True,plot_displacements=False,solver="spsolve",e_colors = ['red', 'red', 'red', 'red'], linewidth = [1,0.75,1,3.5],ax=ax2,fig=fig2,plot_2d=True,plot_2d_plane="yz")
ax4,fig4 = plot_structure_with_strain(kite)

ax5,fig5 = plot_convergence(kite,convergence_criteria = "crisfield")
# ax2.legend()
ax1.legend()
ax3.legend()
ax1.set_title("Initial")
ax2.set_title("Initial (black) vs Final (red)")
ax3.set_title("Final")
plt.show()