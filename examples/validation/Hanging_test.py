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
from kite_fem.Functions import tensionbridles, fix_nodes

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
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

kite = tensionbridles(kite,canopy_nodes,offset=1,scale=0.9)
kite = fix_nodes(kite,[0,127,126,125,70,79,80,113,114,104])

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
        coord1 = coords[id1]
        coord2 = coords[id2]
        length = np.linalg.norm(coord1-coord2)
        phi.append(length)
    #span
    spanid1 = te_ids[0]
    spanid2 = te_ids[-1]
    coord1 = coords[spanid1]
    coord2 = coords[spanid2]
    span = np.linalg.norm(coord1-coord2)
    phi.append(span)
    #tip_leading_edge distance
    middle_idx = len(le_ids) // 2
    middle_le_ids = [le_ids[middle_idx - 1], le_ids[middle_idx]]
    rightids = [middle_le_ids[0],te_ids[-1]]
    leftids = [middle_le_ids[1],te_ids[0]]
    right = np.linalg.norm(coords[rightids[0]]-coords[rightids[1]])
    left = np.linalg.norm(coords[leftids[0]]-coords[leftids[1]])
    phi.append(right)
    phi.append(left)
    return phi

# gravity = m_arr*10
# fe = np.zeros(kite.N)
# fe[2::6] = gravity 

# fe[2*6+1] = 50
# fe[56*6+1] = -50

# # fe[27*6+2] = 125
# # fe[29*6+2] = 125


plot_structure(kite,plot_node_numbers=True,plot_nodes=False)


extract_lengths_validation(kite,strut_sections)
# kite.solve(fe=fe, max_iterations=10000, tolerance=0.01, step_limit=.005, relax_init=.25, relax_min=0.00, relax_update=0.998, k_update=1,I_stiffness=15)
plt.show()
