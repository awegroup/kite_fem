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
    plot_convergence,
    plot_structure_with_collapsed_beams,
    plot_cross_sections
)
from kite_fem.Functions import relaxbridles,fix_nodes,adapt_stiffnesses
from kite_fem.saveload import save_fem_structure,load_fem_structure

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
kite_name = "TUDELFT_V3_KITE"  
struc_geometry_path = (
    Path(PROJECT_DIR)
    / "data"
    / f"{kite_name}"
    / "struc_geometry_all_in_surfplan.yaml"
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
    canopy_sections,
    strut_sections,
)[0]

canopy_nodes = list(set([node for section in canopy_sections + strut_sections for node in section]))
all_sections = canopy_sections + strut_sections
all_sections.sort(key=lambda section: section[0])

fe = np.array([3.81E+00,-5.15E-02,-6.59E-01,-1.56E+00,
2.53E+00,-3.90E+00,-3.38E-03,5.01E-03,
-9.11E-02,-2.37E+00,5.35E+00,-5.14E+00,
-8.46E-01,2.13E+00,-9.04E-01,-5.67E+00,
2.15E+01,-7.46E-01,-9.81E-01,2.96E+00,
-6.69E-01,-6.18E+00,3.91E+01,7.86E+00,
-6.05E-01,3.41E+00,4.19E-01,-6.90E+00,
5.14E+01,2.49E+01,-7.61E-01,5.13E+00,
2.56E+00,-1.33E+01,6.31E+01,6.28E+01,
-9.58E-01,5.03E+00,4.37E+00,-1.43E+01,
6.11E+01,7.27E+01,-9.69E-01,4.08E+00,
4.72E+00,-1.06E+01,5.58E+01,8.27E+01,
-6.87E-01,3.22E+00,4.03E+00,-9.49E+00,
3.74E+01,9.31E+01,-4.52E-01,1.81E+00,
3.33E+00,-9.84E+00,3.63E+01,1.01E+02,
-4.58E-01,1.42E+00,3.29E+00,-7.77E+00,
3.50E+01,1.13E+02,-2.90E-01,1.07E+00,
3.01E+00,-8.45E+00,1.11E+01,1.10E+02,
-3.15E-01,3.05E-01,3.22E+00,-8.01E+00,
8.75E+00,1.15E+02,-2.92E-01,1.46E-01,
2.71E+00,-7.60E+00,9.94E+00,1.11E+02,
-2.73E-01,1.91E-01,2.69E+00,-7.60E+00,
-9.94E+00,1.11E+02,-2.73E-01,-1.91E-01,
2.69E+00,-8.01E+00,-8.75E+00,1.15E+02,
-2.92E-01,-1.46E-01,2.71E+00,-8.45E+00,
-1.11E+01,1.10E+02,-3.15E-01,-3.05E-01,
3.22E+00,-7.77E+00,-3.50E+01,1.13E+02,
-2.90E-01,-1.07E+00,3.01E+00,-9.84E+00,
-3.63E+01,1.01E+02,-4.58E-01,-1.42E+00,
3.29E+00,-9.49E+00,-3.74E+01,9.31E+01,
-4.52E-01,-1.81E+00,3.33E+00,-1.06E+01,
-5.58E+01,8.27E+01,-6.87E-01,-3.22E+00,
4.03E+00,-1.43E+01,-6.11E+01,7.27E+01,
-9.69E-01,-4.08E+00,4.72E+00,-1.33E+01,
-6.31E+01,6.28E+01,-9.58E-01,-5.03E+00,
4.37E+00,-6.90E+00,-5.14E+01,2.49E+01,
-7.61E-01,-5.13E+00,2.55E+00,-6.18E+00,
-3.91E+01,7.86E+00,-6.05E-01,-3.41E+00,
4.19E-01,-5.67E+00,-2.15E+01,-7.87E-01,
-9.81E-01,-2.96E+00,-6.88E-01,-2.37E+00,
-5.35E+00,-5.15E+00,-8.46E-01,-2.13E+00,
-9.11E-01,-1.56E+00,-2.53E+00,-3.90E+00,
-3.38E-03,-5.01E-03,-9.11E-02,2.70E-02,
1.60E-04,-1.88E-01,2.57E-02,-7.10E-04,
-1.78E-01,2.24E-01,-4.31E-03,-9.67E-01,
2.00E-02,-7.10E-04,-4.13E-01,2.13E-01,
8.56E-03,-9.68E-01,2.46E-02,-1.30E-03,
-1.58E-01,1.99E-02,2.35E-03,-3.79E-01,
2.12E-01,9.70E-03,-9.68E-01,2.13E-02,
2.77E-03,-3.78E-01,4.95E-01,-1.83E-02,
-1.01E+00,4.71E-01,3.61E-02,-1.01E+00,
9.14E-01,1.33E-02,-9.58E-02,2.09E-02,
-1.85E-03,-1.22E-01,2.88E+00,3.50E-01,
-5.49E-01,2.42E+00,4.72E-01,-1.64E+00,
9.56E-01,1.11E-01,-7.02E-02,3.26E-01,
4.58E-02,-9.71E-01,4.64E-01,4.14E-02,
-1.01E+00,1.84E-02,4.68E-03,-6.24E-01,
2.52E-02,-3.40E-04,-2.11E+00,2.35E-02,
-1.13E-03,-2.03E+00,3.15E-02,-1.15E-03,
-1.78E+00,9.94E-01,4.87E-02,-1.14E+00,
1.78E+00,3.24E-01,-3.95E-01,4.61E-01,
4.12E-02,-9.97E-02,8.75E-01,1.46E-01,
-1.32E-01,9.37E-01,1.40E-02,-2.15E-01,
0.00E+00,0.00E+00,-1.51E-01,1.19E+00,
2.28E-01,-1.10E+00,8.18E-02,1.38E-02,
-1.09E-01,3.35E-01,6.99E-02,-5.57E-01,
3.59E-01,4.57E-02,-1.76E+00,3.64E-01,
3.04E-02,-1.96E+00,3.80E-01,-1.91E-02,
-2.06E+00,2.70E-02,-1.60E-04,-1.88E-01,
2.57E-02,7.10E-04,-1.78E-01,2.24E-01,
4.31E-03,-9.67E-01,2.00E-02,7.10E-04,
-4.13E-01,2.13E-01,-8.56E-03,-9.68E-01,
2.46E-02,1.30E-03,-1.58E-01,1.99E-02,
-2.35E-03,-3.79E-01,2.12E-01,-9.70E-03,
-9.68E-01,2.13E-02,-2.77E-03,-3.78E-01,
4.95E-01,1.83E-02,-1.01E+00,4.71E-01,
-3.61E-02,-1.01E+00,9.14E-01,-1.33E-02,
-9.58E-02,2.09E-02,1.85E-03,-1.22E-01,
4.35E+00,-4.01E-01,-6.03E-01,2.42E+00,
-4.72E-01,-1.64E+00,9.56E-01,-1.11E-01,
-7.02E-02,3.26E-01,-4.58E-02,-9.71E-01,
4.64E-01,-4.14E-02,-1.01E+00,1.84E-02,
-4.68E-03,-6.24E-01,2.52E-02,3.40E-04,
-2.11E+00,2.35E-02,1.13E-03,-2.03E+00,
3.15E-02,1.15E-03,-1.78E+00,9.94E-01,
-4.87E-02,-1.14E+00,1.78E+00,-3.24E-01,
-3.95E-01,4.61E-01,-4.12E-02,-9.97E-02,
8.75E-01,-1.46E-01,-1.32E-01,9.37E-01,
-1.40E-02,-2.15E-01,0.00E+00,0.00E+00,
-1.51E-01,1.19E+00,-2.28E-01,-1.10E+00,
8.18E-02,-1.38E-02,-1.09E-01,3.35E-01,
-6.99E-02,-5.57E-01,3.59E-01,-4.57E-02,
-1.76E+00,3.64E-01,-3.04E-02,-1.96E+00,
3.80E-01,1.91E-02,-2.06E+00,1.20E+00,
0.00E+00,-3.72E-01,1.47E+00,-1.75E-01,
-2.91E-01,1.47E+00,1.75E-01,-2.91E-01,
4.76E-02,8.00E-05,-6.33E-01,4.53E-02,
2.80E-04,-6.03E-01,5.04E-02,-3.00E-05,
-5.44E-01,4.76E-02,-8.00E-05,-6.33E-01,
4.53E-02,-2.80E-04,-6.03E-01,5.04E-02,
3.00E-05,-5.44E-01,2.01E-01,-8.46E-03,
-2.45E-02,1.93E-01,1.62E-02,-2.08E-02,
2.04E-01,1.78E-02,-2.25E-02,2.01E-01,
8.46E-03,-2.45E-02,1.93E-01,-1.62E-02,
-2.08E-02,2.04E-01,-1.78E-02,-2.25E-02,
0.00E+00,0.00E+00,-1.17E-01,0.00E+00,
0.00E+00,-1.23E-01,0.00E+00,0.00E+00,
-7.01E-02,0.00E+00,0.00E+00,-1.20E+00,
0.00E+00,0.00E+00,-1.03E+00,0.00E+00,
0.00E+00,-1.20E+00,0.00E+00,0.00E+00,
-1.03E+00,0.00E+00,0.00E+00,-1.17E-01,
0.00E+00,0.00E+00,-1.23E-01,0.00E+00,
0.00E+00,-7.01E-02,0.00E+00,0.00E+00,
-1.74E-02,0.00E+00,0.00E+00,-5.03E-02,
0.00E+00,0.00E+00,-9.09E-02,0.00E+00,
0.00E+00,-7.59E-02,0.00E+00,0.00E+00,
-4.26E-02,0.00E+00,0.00E+00,-5.65E-02,
0.00E+00,0.00E+00,-2.27E-01,0.00E+00,
0.00E+00,-4.22E-01,0.00E+00,0.00E+00,
-4.20E-01,0.00E+00,0.00E+00,-2.54E-01,
0.00E+00,0.00E+00,-6.59E-02,0.00E+00,
0.00E+00,-2.71E-01,0.00E+00,0.00E+00,
-4.99E-01,0.00E+00,0.00E+00,-5.89E-01,
0.00E+00,0.00E+00,-3.99E-01,0.00E+00,
0.00E+00,-8.51E-02,0.00E+00,0.00E+00,
-2.47E-01,0.00E+00,0.00E+00,-4.16E-01,
0.00E+00,0.00E+00,-8.32E-01,0.00E+00,
0.00E+00,-7.11E-01,0.00E+00,0.00E+00,
-8.95E-02,0.00E+00,0.00E+00,-2.18E-01,
0.00E+00,0.00E+00,-3.50E-01,0.00E+00,
0.00E+00,-9.15E-01,0.00E+00,0.00E+00,
-8.33E-01,0.00E+00,0.00E+00,-9.59E-02,
0.00E+00,0.00E+00,-1.98E-01,0.00E+00,
0.00E+00,-3.15E-01,0.00E+00,0.00E+00,
-1.02E+00,0.00E+00,0.00E+00,-9.55E-01,
0.00E+00,0.00E+00,-9.84E-02,0.00E+00,
0.00E+00,-2.01E-01,0.00E+00,0.00E+00,
-3.23E-01,0.00E+00,0.00E+00,-1.05E+00,
0.00E+00,0.00E+00,-9.79E-01,0.00E+00,
0.00E+00,-1.01E-01,0.00E+00,0.00E+00,
-2.08E-01,0.00E+00,0.00E+00,-3.33E-01,
0.00E+00,0.00E+00,-1.08E+00,0.00E+00,
0.00E+00,-1.01E+00,0.00E+00,0.00E+00,
-1.02E-01,0.00E+00,0.00E+00,-2.11E-01,
0.00E+00,0.00E+00,-3.35E-01,0.00E+00,
0.00E+00,-1.09E+00,0.00E+00,0.00E+00,
-1.02E+00,0.00E+00,0.00E+00,-1.02E-01,
0.00E+00,0.00E+00,-2.11E-01,0.00E+00,
0.00E+00,-3.35E-01,0.00E+00,0.00E+00,
-1.09E+00,0.00E+00,0.00E+00,-1.02E+00,
0.00E+00,0.00E+00,-1.01E-01,0.00E+00,
0.00E+00,-2.08E-01,0.00E+00,0.00E+00,
-3.33E-01,0.00E+00,0.00E+00,-1.08E+00,
0.00E+00,0.00E+00,-1.01E+00,0.00E+00,
0.00E+00,-9.84E-02,0.00E+00,0.00E+00,
-2.01E-01,0.00E+00,0.00E+00,-3.23E-01,
0.00E+00,0.00E+00,-1.05E+00,0.00E+00,
0.00E+00,-9.79E-01,0.00E+00,0.00E+00,
-9.59E-02,0.00E+00,0.00E+00,-1.98E-01,
0.00E+00,0.00E+00,-3.15E-01,0.00E+00,
0.00E+00,-1.02E+00,0.00E+00,0.00E+00,
-9.55E-01,0.00E+00,0.00E+00,-8.95E-02,
0.00E+00,0.00E+00,-2.17E-01,0.00E+00,
0.00E+00,-3.48E-01,0.00E+00,0.00E+00,
-9.17E-01,0.00E+00,0.00E+00,-8.36E-01,
0.00E+00,0.00E+00,-8.51E-02,0.00E+00,
0.00E+00,-2.45E-01,0.00E+00,0.00E+00,
-4.11E-01,0.00E+00,0.00E+00,-8.34E-01,
0.00E+00,0.00E+00,-7.15E-01,0.00E+00,
0.00E+00,-9.72E-02,0.00E+00,0.00E+00,
-2.67E-01,0.00E+00,0.00E+00,-4.66E-01,
0.00E+00,0.00E+00,-5.56E-01,0.00E+00,
0.00E+00,-4.01E-01,0.00E+00,0.00E+00,
-1.07E-01,0.00E+00,0.00E+00,-2.22E-01,
0.00E+00,0.00E+00,-3.73E-01,0.00E+00,
0.00E+00,-3.65E-01,0.00E+00,0.00E+00,
-2.54E-01,0.00E+00,0.00E+00,-3.65E-02,
0.00E+00,0.00E+00,-4.88E-02,0.00E+00,
0.00E+00,-7.26E-02,0.00E+00,0.00E+00,
-5.49E-02,0.00E+00,0.00E+00,-4.19E-02])

fe = fe.reshape(-1,3)
fe = np.hstack([fe, np.zeros((fe.shape[0], 3))])
fe = fe.flatten()
print(fe)

print("DOF",np.sum(kite.bc))
print("Beam",np.size(kite.beam_elements))
print("Springs",np.size(kite.spring_elements))

ax1,fig1 = plot_structure(kite,plot_nodes=False,fe=fe,plot_external_forces=False,linewidth = [1,0.75,1,3.5],plot_node_numbers=False)
ax2,fig2 = plot_structure(kite, plot_nodes=False,plot_displacements=False,solver="spsolve",e_colors = ['black', 'black', 'black', 'black'],linewidth = [1,0.75,1,3.5],plot_2d=True,plot_2d_plane="yz")
kite.solve(fe=fe, max_iterations=1, tolerance=0.01, step_limit=.005, relax_init=.25, relax_min=0.00, relax_update=0.998, k_update=1,I_stiffness=15)
ax1.legend(fontsize=14)
ax1.tick_params(axis='both', labelsize=14)
ax1.set_xlabel(ax1.get_xlabel(), fontsize=14)
ax1.set_ylabel(ax1.get_ylabel(), fontsize=14)
ax1.set_zlabel(ax1.get_zlabel(), fontsize=14)
# adapt_stiffnesses(kite)
# kite.solve(fe=fe, max_iterations=15000, tolerance=0.01, step_limit=.005, relax_init=.25, relax_min=0.00, relax_update=0.998, k_update=1,I_stiffness=15)

# plot_cross_sections(kite,all_sections)
ax3,fig3 = plot_structure(kite,fe_magnitude=1.5, plot_residual_forces=False,plot_external_forces=True,plot_nodes=False,plot_displacements=True,solver="spsolve",linewidth = [1,0.75,1,3.5])
ax2,fig2 = plot_structure(kite,plot_nodes=False,plot_external_forces=True,plot_displacements=False,solver="spsolve",e_colors = ['red', 'red', 'red', 'red'], linewidth = [1,0.75,1,3.5],plot_2d=True,plot_2d_plane="yz",ax=ax2,fig=fig2)
ax4,fig4 = plot_structure_with_strain(kite)
ax5,fig5 = plot_structure_with_collapsed_beams(kite,plot_nodes=False)
ax6,fig6 = plot_convergence(kite,"crisfield")
# ax3.legend()
# ax2.set_title("Initial (black) vs Final (red)")
# ax3.set_title("Final")
plt.show()