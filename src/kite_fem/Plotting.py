import matplotlib.pyplot as plt
from pyfe3d import DOF
from scipy.sparse.linalg import lsqr, spsolve
import numpy as np

def plot_structure(
        structure,
        ax=None,
        fig=None,
        fe=None,
        fe_magnitude=1,
        plot_displacements=False,
        displacement_magnitude=1,
        I_stiffness = 25,
        plot_nodes=True,
        plot_node_numbers=False,
        set_labels=True,
        plot_2d=False,
        plot_2d_plane='xy',
        e_colors = ['red', 'blue', 'orange', 'green'], #"Spring", "Non-compressive Spring", "Pulley spring", "Beam"
        linewidth = [1,1,1,1],  #"Spring", "Non-compressive Spring", "Pulley spring", "Beam"
        n_colors = ['black', 'grey','red'],
        n_scale = [15, 15,15],
        v_colors = ['magenta', 'cyan', 'purple', 'darkgreen'], # External, Internal, Residual, Displacement forces
        plot_external_forces=False,  # Plot external forces (fe)
        plot_internal_forces=False, # Plot internal forces (-fi)
        plot_residual_forces=False, # Plot residual forces (fe - fi)
        solver="spsolve"
        ):
    # Validate 2D plane option
    valid_planes = ['xy', 'xz', 'yz']
    if plot_2d and plot_2d_plane not in valid_planes:
        raise ValueError(f"plot_2d_plane must be one of {valid_planes}, got '{plot_2d_plane}'")
    
    # Helper function to get coordinates for the specified plane
    def get_plane_coords(x, y, z, plane):
        if plane == 'xy':
            return x, y
        elif plane == 'xz':
            return x, z
        elif plane == 'yz':
            return y, z
        else:
            raise ValueError(f"Invalid plane: {plane}")
    
    # Helper function to get axis labels for the specified plane
    def get_plane_labels(plane):
        if plane == 'xy':
            return "X", "Y"
        elif plane == 'xz':
            return "X", "Z"
        elif plane == 'yz':
            return "Y", "Z"
        else:
            raise ValueError(f"Invalid plane: {plane}")
    
    if ax is None or fig is None:
        fig = plt.figure()
        if plot_2d:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')

    elements = ["Spring", "Non-compressive Spring", "Pulley spring", "Beam"]
    linetypes = ['-', '-', '-','-']
    element_dict = {element: {'color': color, 'linetype': linetype, 'linewidth': linewidth} 
                 for element, color, linetype, linewidth in zip(elements, e_colors, linetypes, linewidth)}
    
    nodes = ["Fixed Node", "Free Node", "Pulley Node"]
    
    node_dict = {node: {'color': color, 'size': s} 
                 for node, color, s in zip(nodes, n_colors, n_scale)}
    
    vectors = ["External Force Vector", "Internal Force Vector", "Residual Force Vector", "Displacement Vector"]
    magnitudes = [fe_magnitude, fe_magnitude, fe_magnitude, displacement_magnitude]
    linewidths = [1, 1, 1, 1]
    vector_dict = {vector: {'color': color, 'magnitude': magnitude, 'linewidth': linewidth}
                   for vector, color, magnitude, linewidth in zip(vectors, v_colors, magnitudes, linewidths)}
    
    if set_labels:
        label_set = {"Free Node": False, "Fixed Node": False, "Pulley Node": False, "Spring": False, "Non-compressive Spring": False, "Pulley spring" : False, "Beam": False, "External Force Vector": False, "Internal Force Vector": False, "Residual Force Vector": False, "Displacement Vector": False}
    else:
        label_set = {"Free Node": True, "Fixed Node": True, "Pulley Node": True, "Spring": True, "Non-compressive Spring": True, "Pulley spring" : True, "Beam": True, "External Force Vector": True, "Internal Force Vector": True, "Residual Force Vector": True, "Displacement Vector": True}
    
    if plot_nodes:
        pulley_nodes = structure.pulley_ids
        for n in range(structure.num_nodes):
            x = structure.coords_current[n * DOF // 2]
            y = structure.coords_current[n * DOF // 2 + 1]
            z = structure.coords_current[n * DOF // 2 + 2]
            fixed = structure.fixed[n*DOF]
            if n in pulley_nodes:
                node_type = "Pulley Node"
            elif fixed:
                node_type = "Fixed Node"
            else:
                node_type = "Free Node"

            if plot_2d:
                coord1, coord2 = get_plane_coords(x, y, z, plot_2d_plane)
                ax.scatter(coord1, coord2, 
                    color=node_dict[node_type]['color'], 
                    s=node_dict[node_type]['size'],zorder=20,
                    edgecolors="black",
                    linewidths=0.5,
                    label=node_type if not label_set[node_type] else None
                )
            else:
                ax.scatter(x, y, z, 
                    color=node_dict[node_type]['color'], 
                    s=node_dict[node_type]['size'],zorder=20,
                    edgecolors="black",
                    linewidths=0.5,
                    label=node_type if not label_set[node_type] else None
                )
            label_set[node_type] = True

    # Plot node numbers independently of node plotting
    if plot_node_numbers:
        for n in range(structure.num_nodes):
            x = structure.coords_current[n * DOF // 2]
            y = structure.coords_current[n * DOF // 2 + 1]
            z = structure.coords_current[n * DOF // 2 + 2]
            
            if plot_2d:
                coord1, coord2 = get_plane_coords(x, y, z, plot_2d_plane)
                ax.text(coord1, coord2, str(n), fontsize=8, ha='center', va='center', 
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7),zorder=80)
            else:
                ax.text(x, y, z, str(n), fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7),zorder=80)

    # springs
    for spring_element in structure.spring_elements:
        n1 = spring_element.spring.n1
        n2 = spring_element.spring.n2
        x_coords = [structure.coords_current[n1 * DOF // 2], structure.coords_current[n2 * DOF // 2]]
        y_coords = [structure.coords_current[n1 * DOF // 2 + 1], structure.coords_current[n2 * DOF // 2 + 1]]
        z_coords = [structure.coords_current[n1 * DOF // 2 + 2], structure.coords_current[n2 * DOF // 2 + 2]]
        if spring_element.springtype == "default":
            element_type = "Spring"
        elif spring_element.springtype == "noncompressive":
            element_type = "Non-compressive Spring"
        elif spring_element.springtype == "pulley":
            element_type = "Pulley spring"
        
        if plot_2d:
            coord1_list = [get_plane_coords(x_coords[0], y_coords[0], z_coords[0], plot_2d_plane)[0],
                          get_plane_coords(x_coords[1], y_coords[1], z_coords[1], plot_2d_plane)[0]]
            coord2_list = [get_plane_coords(x_coords[0], y_coords[0], z_coords[0], plot_2d_plane)[1],
                          get_plane_coords(x_coords[1], y_coords[1], z_coords[1], plot_2d_plane)[1]]
            ax.plot(coord1_list, coord2_list, 
                    color=element_dict[element_type]['color'], 
                    linestyle=element_dict[element_type]['linetype'],
                    linewidth=element_dict[element_type]['linewidth'],
                    label=element_type if not label_set[element_type] else None
                   )
        else:
            ax.plot(x_coords, y_coords, z_coords, 
                    color=element_dict[element_type]['color'], 
                    linestyle=element_dict[element_type]['linetype'],
                    linewidth=element_dict[element_type]['linewidth'],
                    label=element_type if not label_set[element_type] else None
                   )
        label_set[element_type] = True

    # beams
    for beam_element in structure.beam_elements:
        n1 = beam_element.beam.n1
        n2 = beam_element.beam.n2
        x_coords = [structure.coords_current[n1 * DOF // 2], structure.coords_current[n2 * DOF // 2]]
        y_coords = [structure.coords_current[n1 * DOF // 2 + 1], structure.coords_current[n2 * DOF // 2 + 1]]
        z_coords = [structure.coords_current[n1 * DOF // 2 + 2], structure.coords_current[n2 * DOF // 2 + 2]]
        element_type = "Beam"
        
        if plot_2d:
            coord1_list = [get_plane_coords(x_coords[0], y_coords[0], z_coords[0], plot_2d_plane)[0],
                          get_plane_coords(x_coords[1], y_coords[1], z_coords[1], plot_2d_plane)[0]]
            coord2_list = [get_plane_coords(x_coords[0], y_coords[0], z_coords[0], plot_2d_plane)[1],
                          get_plane_coords(x_coords[1], y_coords[1], z_coords[1], plot_2d_plane)[1]]
            ax.plot(coord1_list, coord2_list, 
                    color=element_dict[element_type]['color'], 
                    linestyle=element_dict[element_type]['linetype'],
                    linewidth=element_dict[element_type]['linewidth'],
                    label="Inflatable beam" if not label_set[element_type] else None,
                    zorder = 12
                   )
        else:
            ax.plot(x_coords, y_coords, z_coords, 
                    color=element_dict[element_type]['color'], 
                    linestyle=element_dict[element_type]['linetype'],
                    linewidth=element_dict[element_type]['linewidth'],
                    label="Inflatable beam" if not label_set[element_type] else None,
                    zorder = 12
                   )
        label_set[element_type] = True
    
    # Function to plot force vectors
    def plot_force_vectors(force_vector, vector_type, scale_factor=None):
        if force_vector is None:
            return
            
        force_array = np.array(force_vector)
        force_array = np.where(structure.bc==True, force_array, 0)
        
        # Calculate magnitudes for scaling
        magnitudes = []
        for n in range(structure.num_nodes):
            fx = force_array[n * DOF]
            fy = force_array[n * DOF + 1]
            fz = force_array[n * DOF + 2]        
            magnitudes.append(np.linalg.norm([fx, fy, fz]))
        
        if not magnitudes or max(magnitudes) == 0:
            return
            
        max_magnitude = max(magnitudes)
        if scale_factor is None:
            scale = vector_dict[vector_type]['magnitude'] / max_magnitude
        else:
            scale = scale_factor / max_magnitude
            
        for n in range(structure.num_nodes):
            fx = force_array[n * DOF]
            fy = force_array[n * DOF + 1]
            fz = force_array[n * DOF + 2]
            length = np.linalg.norm([fx, fy, fz]) * scale
            
            if fx != 0 or fy != 0 or fz != 0:
                x = structure.coords_current[n * DOF // 2]
                y = structure.coords_current[n * DOF // 2 + 1]
                z = structure.coords_current[n * DOF // 2 + 2]
                
                if plot_2d:
                    pos1, pos2 = get_plane_coords(x, y, z, plot_2d_plane)
                    force1, force2 = get_plane_coords(fx, fy, fz, plot_2d_plane)
                    ax.quiver(pos1, pos2, force1, force2, 
                              color=vector_dict[vector_type]['color'], 
                              scale=1/scale if scale != 0 else 1, 
                              scale_units='xy', 
                              angles='xy',
                              linewidth=vector_dict[vector_type]['linewidth'],
                              zorder=15)
                    if not label_set[vector_type]:
                        ax.scatter([], [], marker=r'$\longrightarrow$', c=vector_dict[vector_type]['color'], s=150, label=vector_type)
                                        
                else:
                    ax.quiver(x, y, z, fx, fy, fz, 
                              color=vector_dict[vector_type]['color'], 
                              length=length, 
                              normalize=True,
                              linewidth=vector_dict[vector_type]['linewidth'],
                              zorder=15)
                    if not label_set[vector_type]:
                        ax.scatter([], [], [], marker=r'$\longrightarrow$', c=vector_dict[vector_type]['color'], s=150, label=vector_type)
                label_set[vector_type] = True

    # Plot external forces
    if plot_external_forces and fe is not None:
        plot_force_vectors(fe, "External Force Vector")
    
    # Plot internal forces and residual forces (requires internal force calculation)
    if plot_internal_forces or plot_residual_forces:
        structure.I_stiffness = I_stiffness
        structure.update_internal_forces()
        
        if plot_internal_forces:
            # Plot negative internal forces (-fi)
            plot_force_vectors(-structure.fi, "Internal Force Vector")
            
        if plot_residual_forces and fe is not None:
            # Plot residual forces (fe - fi)
            residual = np.array(fe) - structure.fi
            plot_force_vectors(residual, "Residual Force Vector")


    if plot_displacements:
        structure.I_stiffness = I_stiffness
        structure.update_internal_forces()
        structure.update_stiffness_matrix()
        if fe is None:
            fe = np.zeros(structure.N)
        residual = fe - structure.fi
        displacements = np.zeros(structure.N)
        if solver == "spsolve":
            displacements[structure.bc]= spsolve(structure.Kbc, residual[structure.bc])
        elif solver == "lsqr":
            displacements[structure.bc]= lsqr(structure.Kbc, residual[structure.bc])[0]

        vector_type = "Displacement Vector"
        magnitudes = []
        for n in range(structure.num_nodes):
            fx = displacements[n * DOF]
            fy = displacements[n * DOF + 1]
            fz = displacements[n * DOF + 2]        
            magnitudes.append(np.linalg.norm([fx, fy, fz]))

        max_magnitude = max(magnitudes)
        scale = vector_dict[vector_type]['magnitude'] / max_magnitude 
        for n in range(structure.num_nodes):
            dx = displacements[n * DOF]
            dy = displacements[n * DOF + 1]
            dz = displacements[n * DOF + 2]
            length = np.linalg.norm([dx, dy, dz])*scale
            if dx != 0 or dy != 0 or dz != 0:
                x = structure.coords_current[n * DOF // 2]
                y = structure.coords_current[n * DOF // 2 + 1]
                z = structure.coords_current[n * DOF // 2 + 2]
                
                if plot_2d:
                    pos1, pos2 = get_plane_coords(x, y, z, plot_2d_plane)
                    disp1, disp2 = get_plane_coords(dx, dy, dz, plot_2d_plane)
                    ax.quiver(pos1, pos2, disp1, disp2, 
                              color=vector_dict[vector_type]['color'], 
                              scale=1/scale if scale != 0 else 1, 
                              scale_units='xy', 
                              angles='xy',
                              linewidth=vector_dict[vector_type]['linewidth'],
                              label=vector_type if not label_set[vector_type] else None)
                else:
                    ax.quiver(x, y, z, dx, dy, dz, 
                              color=vector_dict[vector_type]['color'], 
                              length=length, 
                              normalize=True,
                              linewidth=vector_dict[vector_type]['linewidth'],
                              label=vector_type if not label_set[vector_type] else None)
                label_set[vector_type] = True

    #resize axes to be equal
    if plot_2d:
        # For 2D plots, only handle the two selected axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmid = (xlim[0] + xlim[1]) / 2
        ymid = (ylim[0] + ylim[1]) / 2
        maximum = max(xlim[1]-xlim[0], ylim[1]-ylim[0])
        ax.set_xlim(xmid - maximum / 2, xmid + maximum / 2)
        ax.set_ylim(ymid - maximum / 2, ymid + maximum / 2)
        ax.set_aspect('equal')
        
        # Set appropriate labels based on the selected plane
        xlabel, ylabel = get_plane_labels(plot_2d_plane)
        ax.set(xlabel=f"{xlabel.lower()} [m]", ylabel=f"{ylabel.lower()} [m]")
    else:
        # For 3D plots, handle x, y, and z axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()  # type: ignore
        xmid = (xlim[0] + xlim[1]) / 2
        ymid = (ylim[0] + ylim[1]) / 2
        zmid = (zlim[0] + zlim[1]) / 2
        maximum = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
        ax.set_xlim(xmid - maximum / 2, xmid + maximum / 2)
        ax.set_ylim(ymid - maximum / 2, ymid + maximum / 2)
        ax.set_zlim(zmid - maximum / 2, zmid + maximum / 2)  # type: ignore
        ax.set_box_aspect([1, 1, 1])  # type: ignore
        ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    return ax, fig


def plot_structure_with_strain(
    structure,
    ax=None,
    fig=None,
    plot_nodes=True,
    strain_range=(-5, 5),  # Strain range for colormap in percentage
    colormap='RdYlBu_r',     # Red-Blue colormap with clear negative/positive distinction
    show_colorbar=True,
    linewidth_base=2,      # Normal linewidth for elements
    linewidth_scale=False, # Don't scale linewidth by strain magnitude
    node_colors=['black', 'grey', 'red'],  # Fixed, free, pulley nodes
    node_sizes=[15, 15, 15]
):
    """
    Plot 3D structure with elements colored according to their strain levels.
    
    Parameters:
    -----------
    structure : FEM_structure
        The FEM structure object
    ax : matplotlib axis, optional
        Existing axis to plot on
    fig : matplotlib figure, optional
        Existing figure to plot on
    plot_nodes : bool, default=True
        Whether to plot nodes
    strain_range : tuple, default=(-5, 5)
        Min and max strain values for colormap scaling (percentage)
    colormap : str, default='RdBu_r'
        Colormap name (RdBu_r: red=compression, blue=tension)
    show_colorbar : bool, default=True
        Whether to show the strain colorbar
    linewidth_base : float, default=2
        Base linewidth for elements
    linewidth_scale : bool, default=True
        Whether to scale linewidth by strain magnitude
    node_colors : list, default=['black', 'grey', 'red']
        Colors for [fixed, free, pulley] nodes
    node_sizes : list, default=[15, 15, 15]
        Sizes for [fixed, free, pulley] nodes
    
    Returns:
    --------
    ax, fig : matplotlib objects
    """
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    
    # Create figure and axis if not provided
    if ax is None or fig is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Calculate strains for all elements
    element_strains = []
    element_info = []
    
    # Calculate spring element strains
    for i, spring_element in enumerate(structure.spring_elements):
        n1 = spring_element.spring.n1
        n2 = spring_element.spring.n2
        
        # Get node coordinates
        x1 = structure.coords_current[n1 * DOF // 2]
        y1 = structure.coords_current[n1 * DOF // 2 + 1]
        z1 = structure.coords_current[n1 * DOF // 2 + 2]
        x2 = structure.coords_current[n2 * DOF // 2]
        y2 = structure.coords_current[n2 * DOF // 2 + 1]
        z2 = structure.coords_current[n2 * DOF // 2 + 2]
        
        # Calculate current length of this element
        current_length = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        
        # Handle pulley elements differently
        if spring_element.springtype == "pulley":
            # For pulley elements, add the length of the matching element
            other_element = structure.spring_elements[spring_element.i_other_pulley]
            other_length = other_element.unit_vector(structure.coords_current)[1]
            total_current_length = current_length + other_length
            rest_length = spring_element.l0  # This is the total rest length for both elements
            strain_percent = (total_current_length - rest_length) / rest_length * 100
        else:
            # For regular springs, use individual length
            rest_length = spring_element.l0
            strain_percent = (current_length - rest_length) / rest_length * 100
        
        element_strains.append(strain_percent)
        element_info.append({
            'type': 'spring',
            'coords': [[x1, x2], [y1, y2], [z1, z2]],
            'strain': strain_percent,
            'spring_type': spring_element.springtype,
            'element_id': i
        })
    
    # Calculate beam element strains
    for i, beam_element in enumerate(structure.beam_elements):
        n1 = beam_element.beam.n1
        n2 = beam_element.beam.n2
        
        # Get node coordinates
        x1 = structure.coords_current[n1 * DOF // 2]
        y1 = structure.coords_current[n1 * DOF // 2 + 1]
        z1 = structure.coords_current[n1 * DOF // 2 + 2]
        x2 = structure.coords_current[n2 * DOF // 2]
        y2 = structure.coords_current[n2 * DOF // 2 + 1]
        z2 = structure.coords_current[n2 * DOF // 2 + 2]
        
        # Calculate current length and strain
        current_length = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        rest_length = beam_element.L
        strain_percent = (current_length - rest_length) / rest_length * 100
        
        element_strains.append(strain_percent)
        element_info.append({
            'type': 'beam',
            'coords': [[x1, x2], [y1, y2], [z1, z2]],
            'strain': strain_percent,
            'element_id': i
        })
    
    # Set up colormap normalization with clear zero crossing
    # Ensure the strain range is symmetric around zero for clear negative/positive distinction
    max_abs_strain = max(abs(strain_range[0]), abs(strain_range[1]))
    symmetric_range = (-max_abs_strain, max_abs_strain)
    norm = colors.TwoSlopeNorm(vmin=symmetric_range[0], vcenter=0, vmax=symmetric_range[1])
    cmap = cm.get_cmap(colormap)
    
    # Plot elements with strain-based coloring
    for info in element_info:
        strain = info['strain']
        coords = info['coords']
        
        # Determine color based on strain
        color = cmap(norm(strain))
        
        # Determine linewidth - use consistent thickness
        linewidth = linewidth_base
        
        # Plot the element - all with solid lines and consistent thickness
        if info['type'] == 'spring':
            ax.plot(coords[0], coords[1], coords[2], 
                   color=color, linewidth=linewidth, linestyle='-', alpha=1.0)
        
        elif info['type'] == 'beam':
            # Beams with slightly thicker lines but still normal
            ax.plot(coords[0], coords[1], coords[2], 
                   color=color, linewidth=linewidth*1.5, linestyle='-', alpha=1.0)
    
    # Plot nodes if requested
    if plot_nodes:
        pulley_nodes = structure.pulley_ids
        for n in range(structure.num_nodes):
            x = structure.coords_current[n * DOF // 2]
            y = structure.coords_current[n * DOF // 2 + 1]
            z = structure.coords_current[n * DOF // 2 + 2]
            fixed = structure.fixed[n * DOF]
            
            if n in pulley_nodes:
                color, size = node_colors[2], node_sizes[2]  # Pulley node
            elif fixed:
                color, size = node_colors[0], node_sizes[0]  # Fixed node
            else:
                color, size = node_colors[1], node_sizes[1]  # Free node
            
            ax.scatter(x, y, z, color=color, s=size, zorder=20, 
                      edgecolors="black", linewidths=0.5)
    
    # Set axis properties for equal scaling
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    xmid = (xlim[0] + xlim[1]) / 2
    ymid = (ylim[0] + ylim[1]) / 2
    zmid = (zlim[0] + zlim[1]) / 2
    maximum = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
    ax.set_xlim(xmid - maximum / 2, xmid + maximum / 2)
    ax.set_ylim(ymid - maximum / 2, ymid + maximum / 2)
    ax.set_zlim(zmid - maximum / 2, zmid + maximum / 2)
    ax.set_box_aspect([1, 1, 1])
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    
    # Add colorbar if requested
    if show_colorbar:
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Strain [%]', rotation=270, labelpad=15)
        
        # Add horizontal line at zero strain for clarity
        cbar.ax.axhline(y=0, color='black', linewidth=1.5, alpha=0.8)
    
    # Add title
    if element_strains:
        max_strain = max(element_strains)
        min_strain = min(element_strains)
        avg_strain = np.mean(element_strains)
        ax.set_title(f'Structure with Strain Visualization\n'
                    f'Strain range: {min_strain:.1f}% to {max_strain:.1f}% (avg: {avg_strain:.1f}%)')
    
    return ax, fig


def check_element_strain(structure, print_results=True, return_data=False):
    """
    Check current element lengths against their rest lengths (l0) with percentage strain.
    
    Parameters:
    -----------
    structure : FEM_structure
        The FEM structure object
    print_results : bool, default=True
        Whether to print the strain results to console
    return_data : bool, default=False
        Whether to return the strain data as a dictionary
    
    Returns:
    --------
    strain_data : dict (optional, if return_data=True)
        Dictionary containing strain information for springs and beams
    """
    
    strain_data = {
        'spring_strains': [],
        'beam_strains': [],
        'spring_info': [],
        'beam_info': []
    }
    
    if print_results:
        print("\n" + "="*60)
        print("ELEMENT STRAIN ANALYSIS")
        print("="*60)
    
    # Check spring elements
    if structure.spring_elements:
        if print_results:
            print(f"\nSPRING ELEMENTS ({len(structure.spring_elements)} total):")
            print("-" * 60)
            print(f"{'ID':<4} {'Type':<15} {'Current [m]':<12} {'Rest [m]':<10} {'Strain [%]':<12}")
            print("-" * 60)
        
        for i, spring_element in enumerate(structure.spring_elements):
            n1 = spring_element.spring.n1
            n2 = spring_element.spring.n2
            
            # Calculate current length
            x1 = structure.coords_current[n1 * DOF // 2]
            y1 = structure.coords_current[n1 * DOF // 2 + 1]
            z1 = structure.coords_current[n1 * DOF // 2 + 2]
            x2 = structure.coords_current[n2 * DOF // 2]
            y2 = structure.coords_current[n2 * DOF // 2 + 1]
            z2 = structure.coords_current[n2 * DOF // 2 + 2]
            
            current_length = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            
            # Handle pulley elements differently
            if spring_element.springtype == "pulley":
                # For pulley elements, add the length of the matching element
                other_element = structure.spring_elements[spring_element.i_other_pulley]
                other_length = other_element.unit_vector(structure.coords_current)[1]
                total_current_length = current_length + other_length
                rest_length = spring_element.l0  # This is the total rest length for both elements
                strain_percent = (total_current_length - rest_length) / rest_length * 100
                
                # Store information about combined length for pulley elements
                spring_info_entry = {
                    'id': i,
                    'type': spring_element.springtype.capitalize(),
                    'current_length': total_current_length,
                    'individual_length': current_length,
                    'other_element_length': other_length,
                    'rest_length': rest_length,
                    'strain_percent': strain_percent,
                    'nodes': (n1, n2),
                    'other_element_id': spring_element.i_other_pulley
                }
            else:
                # For regular springs, use individual length
                rest_length = spring_element.l0
                strain_percent = (current_length - rest_length) / rest_length * 100
                
                spring_info_entry = {
                    'id': i,
                    'type': spring_element.springtype.capitalize(),
                    'current_length': current_length,
                    'rest_length': rest_length,
                    'strain_percent': strain_percent,
                    'nodes': (n1, n2)
                }
            
            strain_data['spring_strains'].append(strain_percent)
            strain_data['spring_info'].append(spring_info_entry)
            
            if print_results:
                if spring_element.springtype == "pulley":
                    print(f"{i:<4} {spring_element.springtype.capitalize():<15} {total_current_length:<12.6f} {rest_length:<10.6f} {strain_percent:<12.2f}")
                else:
                    print(f"{i:<4} {spring_element.springtype.capitalize():<15} {current_length:<12.6f} {rest_length:<10.6f} {strain_percent:<12.2f}")
    
    # Check beam elements
    if structure.beam_elements:
        if print_results:
            print(f"\nBEAM ELEMENTS ({len(structure.beam_elements)} total):")
            print("-" * 60)
            print(f"{'ID':<4} {'Type':<15} {'Current [m]':<12} {'Rest [m]':<10} {'Strain [%]':<12}")
            print("-" * 60)
        
        for i, beam_element in enumerate(structure.beam_elements):
            n1 = beam_element.beam.n1
            n2 = beam_element.beam.n2
            
            # Calculate current length
            x1 = structure.coords_current[n1 * DOF // 2]
            y1 = structure.coords_current[n1 * DOF // 2 + 1]
            z1 = structure.coords_current[n1 * DOF // 2 + 2]
            x2 = structure.coords_current[n2 * DOF // 2]
            y2 = structure.coords_current[n2 * DOF // 2 + 1]
            z2 = structure.coords_current[n2 * DOF // 2 + 2]
            
            current_length = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            rest_length = beam_element.L  # Beam rest length
            strain_percent = (current_length - rest_length) / rest_length * 100
            
            strain_data['beam_strains'].append(strain_percent)
            strain_data['beam_info'].append({
                'id': i,
                'type': 'Beam',
                'current_length': current_length,
                'rest_length': rest_length,
                'strain_percent': strain_percent,
                'nodes': (n1, n2)
            })
            
            if print_results:
                print(f"{i:<4} {'Beam':<15} {current_length:<12.6f} {rest_length:<10.6f} {strain_percent:<12.2f}")
    
    # Summary statistics
    all_strains = strain_data['spring_strains'] + strain_data['beam_strains']
    
    if all_strains and print_results:
        print("\n" + "="*60)
        print("STRAIN SUMMARY:")
        print("-" * 60)
        print(f"Total elements: {len(all_strains)}")
        print(f"Maximum strain: {max(all_strains):.2f}%")
        print(f"Minimum strain: {min(all_strains):.2f}%")
        print(f"Average strain: {np.mean(all_strains):.2f}%")
        print(f"Standard deviation: {np.std(all_strains):.2f}%")
        
        # Count elements by strain level
        compression = [s for s in all_strains if s < -1]
        tension = [s for s in all_strains if s > 1]
        neutral = [s for s in all_strains if -1 <= s <= 1]
        
        print(f"\nStrain distribution:")
        print(f"  High compression (< -1%): {len(compression)} elements")
        print(f"  Neutral (-1% to 1%):      {len(neutral)} elements")
        print(f"  High tension (> 1%):      {len(tension)} elements")
        
        if compression:
            print(f"  Max compression: {min(compression):.2f}%")
        if tension:
            print(f"  Max tension: {max(tension):.2f}%")
    
    if return_data:
        return strain_data


def plot_convergence(structure):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(structure.iteration_history, structure.residual_norm_history, 'b-', label='Residual')
    ax.set(xlabel="Iteration", ylabel="Residual [N]")
    ax.set_ylim(0,structure.residual_norm_history[0])
    ax.tick_params(axis='y', labelcolor='b')
    
    # Create second y-axis for relaxation factor
    ax2 = ax.twinx()
    ax2.plot(structure.iteration_history, structure.relax_history, 'r-', label='Relaxation Factor')
    ax2.set_ylabel('Relaxation Factor', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    return ax, fig

