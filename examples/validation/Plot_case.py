from kite_fem.saveload import load_fem_structure
from kite_fem.Plotting import plot_structure,plot_structure_with_strain,plot_convergence
from Hanging_test import get_load_cases
from kite_fem.Functions import fix_nodes
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def create_composite_view(load_cases=range(1, 11), elev=10, azim=180, invert_zaxis=True):
    """
    Create a composite view of all load cases in a single figure.
    
    Parameters
    ----------
    load_cases : iterable
        Load case numbers to plot (default: 1-10)
    elev : float
        Elevation angle in degrees (default: 20)
    azim : float
        Azimuth angle in degrees (default: 45)
    invert_zaxis : bool
        Whether to invert the z-axis (default: False)
    """
    result_dir = Path(__file__).resolve().parent / "results"
    
    # Create figure with subplots (2 rows x 5 columns for 10 load cases)
    fig = plt.figure(figsize=(20, 8))
    
    for idx, load_case in enumerate(load_cases, start=1):
        # Load the result for this load case
        result_path = result_dir / f"load_case_{load_case}.npz"
        
        if not result_path.exists():
            print(f"Warning: {result_path} not found, skipping load case {load_case}")
            continue
        
        result = load_fem_structure(result_path)
        fe = result.fe
        result = fix_nodes(result, [0, 129, 128, 127, 72, 81, 82, 115, 116, 106])
        
        # Determine fe_magnitude based on load case
        if load_case in [1, 6]:
            fe_magnitude = 0.5
        else:
            fe_magnitude = 1
        
        # Create subplot
        ax = fig.add_subplot(2, 5, idx, projection='3d')
        
        # Plot structure with external forces
        ax, _ = plot_structure(
            result, 
            fe=fe, 
            fe_magnitude=fe_magnitude,
            plot_external_forces=True, 
            plot_nodes=False, 
            linewidth=[1, 0.75, 1, 3.5],
            ax=ax,
            fig=fig
        )
        
        # Set title for each subplot
        case_info = get_load_cases()[load_case-1] 
        pressure    = case_info[0]
        center_load = case_info[2]
        tip_load    = case_info[1]
        
        # Build title based on what loads are present
        title_parts = [f'Load Case {load_case}']
        title_parts.append(f'P={pressure} bar')
        if center_load != 0:
            title_parts.append(f'Center={center_load} kg')
        if tip_load != 0:
            title_parts.append(f'Tip={tip_load} kg')
        
        ax.set_title(', '.join(title_parts), fontsize=10, fontweight='bold')
        
        # Adjust viewing angle for better visualization
        ax.view_init(elev=elev, azim=azim)
        
        # Invert z-axis if requested
        if invert_zaxis:
            ax.invert_zaxis()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the composite figure
    plt.show()

    return ax,fig


def plot_single_case(load_case=1, elev=10, azim=180, invert_zaxis=True):
    """
    Plot a single load case.
    
    Parameters
    ----------
    load_case : int
        Load case number (1-10)
    elev : float
        Elevation angle in degrees (default: 20)
    azim : float
        Azimuth angle in degrees (default: 45)
    invert_zaxis : bool
        Whether to invert the z-axis (default: False)
    """
    result_dir = Path(__file__).resolve().parent / "results"
    result_dir.mkdir(exist_ok=True)
    init_path = result_dir / f"initial.npz"
    result_path = result_dir / f"load_case_{load_case}.npz"
    init = load_fem_structure(init_path)
    result = load_fem_structure(result_path)
    fe = result.fe
    result = fix_nodes(result, [0, 129, 128, 127, 72, 81, 82, 115, 116, 106])
    init = fix_nodes(init, [0, 129, 128, 127, 72, 81, 82, 115, 116, 106])
    
    # Determine fe_magnitude based on load case
    if load_case in [1, 6]:
        fe_magnitude = 0.5
    else:
        fe_magnitude = 1
    
    ax1,fig1 = plot_structure(init, plot_nodes=False,plot_displacements=False,solver="spsolve",e_colors = ['black', 'black', 'black', 'black'],linewidth = [1,0.75,1,3.5],plot_2d=True,plot_2d_plane="yz")
    ax1,fig1 = plot_structure(result, plot_nodes=False,plot_displacements=False,solver="spsolve",e_colors = ['red', 'red', 'red', 'red'],linewidth = [1,0.75,1,3.5],plot_2d=True,plot_2d_plane="yz",ax=ax1,fig=fig1)
    ax2,fig2 = plot_structure(init,fe=fe,fe_magnitude=fe_magnitude,plot_external_forces=True,plot_nodes=False,linewidth = [1,0.75,1,3.5])
    ax3,fig3 = plot_structure(result,fe=fe,fe_magnitude=fe_magnitude,plot_external_forces=True,plot_nodes=False,linewidth = [1,0.75,1,3.5])
    ax4,fig4 = plot_structure_with_strain(result)
    ax5,fig5 = plot_convergence(result,convergence_criteria="crisfield")



    # Adjust viewing angle
    ax2.view_init(elev=elev, azim=azim)
    ax3.view_init(elev=elev, azim=azim)
    ax4.view_init(elev=elev, azim=azim)

    # Invert z-axis if requested
    if invert_zaxis:
        ax1.invert_yaxis()
        ax2.invert_zaxis()
        ax3.invert_zaxis()
        ax4.invert_zaxis()

    plt.show()



if __name__ == "__main__":
    # Create composite view of all load cases
    # You can change the viewing angle by adjusting elev and azim:
    # elev: elevation angle (up/down), azim: azimuth angle (rotation around)
    # Examples:
    #   - Top view: elev=90, azim=0
    #   - Front view: elev=0, azim=0
    #   - Side view: elev=0, azim=90
    #   - Default: elev=20, azim=45
    # Set invert_zaxis=True to flip the z-axis
    # fig_composite = create_composite_view(elev=10, azim=180, invert_zaxis=True)
    
    # Optionally, plot a single case as well
    load_case = 10
    plot_single_case(load_case)

