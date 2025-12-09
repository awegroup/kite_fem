from kite_fem.saveload import load_fem_structure
from kite_fem.Plotting import plot_structure,plot_structure_with_strain,plot_convergence
from kite_fem.Functions import fix_nodes
import matplotlib.pyplot as plt
from pathlib import Path


load_case = 6 #select load case (1-10)

result_dir = Path(__file__).resolve().parent / "results"
result_dir.mkdir(exist_ok=True)
init_path = result_dir / f"initial.npz"
result_path = result_dir / f"load_case_{load_case}.npz"
init = load_fem_structure(init_path)
result = load_fem_structure(result_path)
fe = result.fe
result = fix_nodes(result,[0,129,128,127,72,81,82,115,116,106])
init = fix_nodes(init,[0,129,128,127,72,81,82,115,116,106])


ax,fig = plot_structure(init, plot_nodes=False,plot_displacements=False,solver="spsolve",e_colors = ['black', 'black', 'black', 'black'],linewidth = [1,0.75,1,3.5],plot_2d=True,plot_2d_plane="yz")
ax,fig = plot_structure(result, plot_nodes=False,plot_displacements=False,solver="spsolve",e_colors = ['red', 'red', 'red', 'red'],linewidth = [1,0.75,1,3.5],plot_2d=True,plot_2d_plane="yz",ax=ax,fig=fig)
ax2,fig2 = plot_structure(init,fe=fe,fe_magnitude=1.5,plot_external_forces=True,plot_nodes=False,linewidth = [1,0.75,1,3.5])
ax3,fig3 = plot_structure(result,fe=fe,fe_magnitude=1.5,plot_external_forces=True,plot_nodes=False,linewidth = [1,0.75,1,3.5])
ax4,fig4 = plot_structure_with_strain(result)
ax5,fig5 = plot_convergence(result,convergence_criteria="crisfield")
plt.show()