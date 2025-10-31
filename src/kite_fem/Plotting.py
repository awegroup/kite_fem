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
        ):
    if ax is None or fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    elements = ["Spring", "Non-compressive Spring", "Pulley", "Beam"]
    e_colors = ['red', 'blue', 'orange', 'green'] 
    linetypes = ['-', '-', '-','-']
    linewidth = [1,1,1,1]
    element_dict = {element: {'color': color, 'linetype': linetype, 'linewidth': linewidth} 
                 for element, color, linetype, linewidth in zip(elements, e_colors, linetypes, linewidth)}
    
    nodes = ["Fixed Node", "Free Node"]
    n_colors = ['black', 'grey']
    scale = [10, 10]
    node_dict = {node: {'color': color, 'size': s} 
                 for node, color, s in zip(nodes, n_colors, scale)}
    
    vectors = ["External Force Vector", "Displacement Vector"]
    v_colors = ['magenta', 'darkgreen']
    magnitudes = [fe_magnitude, displacement_magnitude]
    linewidth = [1, 1]
    vector_dict = {vector: {'color': color, 'magnitude': magnitudes, 'linewidth': linewidth}
                   for vector, color, magnitudes, linewidth in zip(vectors, v_colors, magnitudes, linewidth)}
    
    if set_labels:
        label_set = {"Free Node": False, "Fixed Node": False, "Spring": False, "Non-compressive Spring": False, "Pulley" : False, "Beam": False, "External Force Vector": False, "Displacement Vector": False}
    else:
        label_set = {"Free Node": True, "Fixed Node": True, "Spring": True, "Non-compressive Spring": True, "Pulley" : True, "Beam": True, "External Force Vector": True, "Displacement Vector": True}
    
    if plot_nodes:
        for n in range(structure.num_nodes):
            x = structure.coords_current[n * DOF // 2]
            y = structure.coords_current[n * DOF // 2 + 1]
            z = structure.coords_current[n * DOF // 2 + 2]
            fixed = structure.fixed[n*DOF]
            if fixed:
                node_type = "Fixed Node"
            else:
                node_type = "Free Node"
            ax.scatter(x, y, z, 
                color=node_dict[node_type]['color'], 
                s=node_dict[node_type]['size'],
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
            element_type = "Pulley"
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
                ax.quiver(x, y, z, fx, fy, fz, 
                          color=vector_dict[vector_type]['color'], 
                          length=length, 
                          normalize=True,
                          linewidth=vector_dict[vector_type]['linewidth'],
                          label=vector_type if not label_set[vector_type] else None)
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
                ax.quiver(x, y, z, dx, dy, dz, 
                          color=vector_dict[vector_type]['color'], 
                          length=length, 
                          normalize=True,
                          linewidth=vector_dict[vector_type]['linewidth'],
                          label=vector_type if not label_set[vector_type] else None)
                label_set[vector_type] = True

    #resize axes to be equal
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    xmid = (xlim[0] + xlim[1]) / 2
    ymid = (ylim[0] + ylim[1]) / 2
    zmid = (zlim[0] + zlim[1]) / 2
    maximum = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
    ax.set_xlim([xmid - maximum / 2, xmid + maximum / 2])
    ax.set_ylim([ymid - maximum / 2, ymid + maximum / 2])
    ax.set_zlim([zmid - maximum / 2, zmid + maximum / 2])
    ax.set_box_aspect([1, 1, 1])
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    return ax, fig


def plot_convergence(structure):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(structure.iteration_history, structure.residual_norm_history)
    ax.set(xlabel="Iteration", ylabel="Residual [N]")
    return ax, fig
