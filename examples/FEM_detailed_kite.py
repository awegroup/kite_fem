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

kite = tensionbridles(kite,canopy_nodes,offset=1,scale=0.9)
kite = fix_nodes(kite,[0,127,126,125,70,79,80,113,114,104])


ax10,fig10 = plot_structure_with_strain(kite)



gravity = m_arr*10
fe = np.zeros(kite.N)
fe[2::6] = gravity 

# fe[2*6+1] = 50
# fe[56*6+1] = -50

fe[27*6+2] = 125
fe[29*6+2] = 125

ax1,fig1 = plot_structure(kite,plot_nodes=True,fe=fe,plot_external_forces=True,linewidth = [1,0.75,1,3.5],plot_node_numbers=False)
ax2,fig2 = plot_structure(kite, plot_nodes=False,plot_displacements=False,solver="spsolve",e_colors = ['black', 'black', 'black', 'black'],linewidth = [1,0.75,1,3.5],plot_2d=True,plot_2d_plane="yz")

# ax1.legend()
# plt.show()
# breakpoint()

kite.solve(fe=fe, max_iterations=10000, tolerance=0.01, step_limit=.005, relax_init=.25, relax_min=0.00, relax_update=0.998, k_update=1,I_stiffness=15)
fi = kite.fi
residual = fe-fi
print(np.max(residual[kite.bc]))

# kite.reinitialise()
# kite.solve(fe=fe, max_iterations=3000, tolerance=5, step_limit=.005, relax_init=.25, relax_min=0.025, relax_update=0.995, k_update=1,I_stiffness=15)

ax3,fig3 = plot_structure(kite,fe=fe,fe_magnitude=1.5, plot_residual_forces=True,plot_external_forces=False,plot_nodes=False,plot_displacements=True,solver="spsolve",linewidth = [1,0.75,1,3.5])
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