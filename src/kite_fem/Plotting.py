import matplotlib.pyplot as plt
from pyfe3d import DOF
from scipy.sparse.linalg import lsqr
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
        set_labels=True,
        plot_2d=False,
        plot_2d_plane='xy',
        e_colors = ['red', 'blue', 'orange', 'green'] ,
        n_colors = ['black', 'grey','red'],
        n_scale = [15, 15,15],
        v_colors = ['magenta', 'darkgreen']
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
    linewidth = [1,1,1,1]
    element_dict = {element: {'color': color, 'linetype': linetype, 'linewidth': linewidth} 
                 for element, color, linetype, linewidth in zip(elements, e_colors, linetypes, linewidth)}
    
    nodes = ["Fixed Node", "Free Node", "Pulley Node"]
    
    node_dict = {node: {'color': color, 'size': s} 
                 for node, color, s in zip(nodes, n_colors, n_scale)}
    
    vectors = ["External Force Vector", "Displacement Vector"]
    magnitudes = [fe_magnitude, displacement_magnitude]
    linewidth = [1, 1]
    vector_dict = {vector: {'color': color, 'magnitude': magnitudes, 'linewidth': linewidth}
                   for vector, color, magnitudes, linewidth in zip(vectors, v_colors, magnitudes, linewidth)}
    
    if set_labels:
        label_set = {"Free Node": False, "Fixed Node": False, "Pulley Node": False, "Spring": False, "Non-compressive Spring": False, "Pulley spring" : False, "Beam": False, "External Force Vector": False, "Displacement Vector": False}
    else:
        label_set = {"Free Node": True, "Fixed Node": True, "Pulley Node": True, "Spring": True, "Non-compressive Spring": True, "Pulley spring" : True, "Beam": True, "External Force Vector": True, "Displacement Vector": True}
    
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
    
    if fe is not None:
        # plot forces as arrows
        vector_type = "External Force Vector"
        magnitudes = []

        for n in range(structure.num_nodes):
            fx = fe[n * DOF]
            fy = fe[n * DOF + 1]
            fz = fe[n * DOF + 2]        
            magnitudes.append(np.linalg.norm([fx, fy, fz]))
        max_magnitude = max(magnitudes)
        scale = vector_dict[vector_type]['magnitude'] / max_magnitude 
        for n in range(structure.num_nodes):
            fx = fe[n * DOF]
            fy = fe[n * DOF + 1]
            fz = fe[n * DOF + 2]
            length = np.linalg.norm([fx, fy, fz])*scale
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


    if plot_displacements:
        structure.I_stiffness = I_stiffness
        structure.update_internal_forces()
        structure.update_stiffness_matrix()
        if fe is None:
            fe = np.zeros(structure.N)
        residual = fe - structure.fi
        displacements = np.zeros(structure.N)
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


def plot_convergence(structure):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(structure.iteration_history, structure.residual_norm_history)
    ax.set(xlabel="Iteration", ylabel="Residual [N]")
    return ax, fig
