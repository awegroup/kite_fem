import numpy as np
from pathlib import Path


def save_fem_structure(fem_structure, filepath):
    """
    Save a FEM_structure object to a single compressed npz file.
    
    Alternative simpler format using numpy's compressed format.
    Useful for quick saves but less human-readable than the directory format.
    
    Parameters
    ----------
    fem_structure : FEM_structure
        The FEM structure object to save
    filepath : str or Path
        Path to save the structure (will add .npz extension if not present)
    
    Returns
    -------
    str
        Path to the saved file
    """
    filepath = Path(filepath)
    if filepath.suffix != '.npz':
        filepath = filepath.with_suffix('.npz')
    
    # Prepare all data for saving
    save_dict = {
        'coords_current': fem_structure.coords_current,
        'coords_rotations_current': fem_structure.coords_rotations_current,
        'fe': fem_structure.fe,
        'fi': fem_structure.fi,
    }
    
    # Save connectivity matrices
    if fem_structure.spring_matrix is not None:
        save_dict['spring_matrix'] = np.array(fem_structure.spring_matrix)
    if fem_structure.pulley_matrix is not None:
        save_dict['pulley_matrix'] = np.array(fem_structure.pulley_matrix)
    if fem_structure.beam_matrix is not None:
        save_dict['beam_matrix'] = np.array(fem_structure.beam_matrix)
    
    # Save solver history
    if hasattr(fem_structure, 'iteration_history') and fem_structure.iteration_history:
        save_dict['iteration_history'] = np.array(fem_structure.iteration_history)
    if hasattr(fem_structure, 'crisfield_history') and fem_structure.crisfield_history:
        save_dict['crisfield_history'] = np.array(fem_structure.crisfield_history)
    if hasattr(fem_structure, 'residual_norm_history') and fem_structure.residual_norm_history:
        save_dict['residual_norm_history'] = np.array(fem_structure.residual_norm_history)
    if hasattr(fem_structure, 'relax_history') and fem_structure.relax_history:
        save_dict['relax_history'] = np.array(fem_structure.relax_history)
    
    # Save spring properties as arrays
    if fem_structure.spring_elements:
        spring_k = np.array([s.k for s in fem_structure.spring_elements])
        spring_l0 = np.array([s.l0 for s in fem_structure.spring_elements])
        save_dict['spring_k'] = spring_k
        save_dict['spring_l0'] = spring_l0
    
    # Save beam properties as arrays
    if fem_structure.beam_elements:
        beam_E = np.array([b.E if hasattr(b, 'E') else 0.0 for b in fem_structure.beam_elements])
        beam_G = np.array([b.G if hasattr(b, 'G') else 0.0 for b in fem_structure.beam_elements])
        beam_k = np.array([b.k if hasattr(b, 'k') else 0.0 for b in fem_structure.beam_elements])
        beam_collapsed = np.array([b.collapsed if hasattr(b, 'collapsed') else False for b in fem_structure.beam_elements])
        save_dict['beam_E'] = beam_E
        save_dict['beam_G'] = beam_G
        save_dict['beam_k'] = beam_k
        save_dict['beam_collapsed'] = beam_collapsed
    
    # Convert initial conditions to array for storage
    # Store as a structured format that can be reconstructed
    initial_conds_array = []
    for pos, vel, mass, fixed in fem_structure.initial_conditions:
        initial_conds_array.append([*pos, *vel, mass, int(fixed)])
    save_dict['initial_conditions_array'] = np.array(initial_conds_array)
    
    np.savez_compressed(filepath, **save_dict)
    
    print(f"FEM structure saved to: {filepath}")
    return str(filepath)


def load_fem_structure(filepath):
    """
    Load a FEM_structure object from a compressed npz file.
    
    Loads from the simple format created by save_fem_structure_simple.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the saved .npz file
    
    Returns
    -------
    FEM_structure
        Reconstructed FEM structure object
    """
    from kite_fem.FEMStructure import FEM_structure
    
    filepath = Path(filepath)
    if filepath.suffix != '.npz':
        filepath = filepath.with_suffix('.npz')
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load all data
    data = np.load(filepath, allow_pickle=True)
    
    # Reconstruct initial conditions
    initial_conds_array = data['initial_conditions_array']
    initial_conditions = []
    for row in initial_conds_array:
        pos = row[0:3]
        vel = row[3:6]
        mass = row[6]
        fixed = bool(row[7])
        initial_conditions.append([pos.tolist(), vel.tolist(), mass, fixed])
    
    # Get connectivity matrices and convert to lists with proper types
    spring_matrix = None
    if 'spring_matrix' in data:
        spring_array = data['spring_matrix']
        spring_matrix = []
        for row in spring_array:
            # Convert: [n1(int), n2(int), k(float), c(float), l0(float), springtype(str)]
            spring_matrix.append([int(row[0]), int(row[1]), float(row[2]), 
                                 float(row[3]), float(row[4]), str(row[5])])
    
    pulley_matrix = None
    if 'pulley_matrix' in data:
        pulley_array = data['pulley_matrix']
        pulley_matrix = []
        for row in pulley_array:
            # Convert: [n1(int), n2(int), n3(int), k(float), c(float), l0(float)]
            pulley_matrix.append([int(row[0]), int(row[1]), int(row[2]), 
                                 float(row[3]), float(row[4]), float(row[5])])
    
    beam_matrix = None
    if 'beam_matrix' in data:
        beam_array = data['beam_matrix']
        beam_matrix = []
        for row in beam_array:
            # Convert: [n1(int), n2(int), d(float), p(float), l0(float)]
            beam_matrix.append([int(row[0]), int(row[1]), float(row[2]), 
                               float(row[3]), float(row[4])])
    
    # Create FEM structure
    fem_structure = FEM_structure(
        initial_conditions=initial_conditions,
        spring_matrix=spring_matrix,
        pulley_matrix=pulley_matrix,
        beam_matrix=beam_matrix
    )
    
    # Load deformed state
    fem_structure.coords_current = data['coords_current']
    fem_structure.coords_rotations_current = data['coords_rotations_current']
    
    # Load force vectors
    if 'fe' in data:
        fem_structure.fe = data['fe']
    if 'fi' in data:
        fem_structure.fi = data['fi']
    
    # Update spring properties
    if 'spring_k' in data:
        spring_k = data['spring_k']
        spring_l0 = data['spring_l0']
        for i, spring_elem in enumerate(fem_structure.spring_elements):
            if i < len(spring_k):
                spring_elem.k = spring_k[i]
                spring_elem.l0 = spring_l0[i]
                spring_elem.spring.kxe = spring_k[i]
    
    # Update beam properties
    if 'beam_E' in data:
        beam_E = data['beam_E']
        beam_G = data['beam_G']
        beam_k = data['beam_k']
        for i, beam_elem in enumerate(fem_structure.beam_elements):
            if i < len(beam_E):
                beam_elem.E = beam_E[i]
                beam_elem.prop.E = beam_E[i]
                beam_elem.G = beam_G[i]
                beam_elem.prop.G = beam_G[i]
                beam_elem.k = beam_k[i]
    
    # Load beam collapsed states
    if 'beam_collapsed' in data:
        beam_collapsed = data['beam_collapsed']
        for i, beam_elem in enumerate(fem_structure.beam_elements):
            if i < len(beam_collapsed):
                beam_elem.collapsed = bool(beam_collapsed[i])
    
    # Load solver history
    if 'iteration_history' in data:
        fem_structure.iteration_history = data['iteration_history'].tolist()
    if 'crisfield_history' in data:
        fem_structure.crisfield_history = data['crisfield_history'].tolist()
    if 'residual_norm_history' in data:
        fem_structure.residual_norm_history = data['residual_norm_history'].tolist()
    if 'relax_history' in data:
        fem_structure.relax_history = data['relax_history'].tolist()
    
    print(f"FEM structure loaded from: {filepath}")
    return fem_structure
