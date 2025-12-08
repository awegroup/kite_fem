import numpy as np
import json
import os
from pathlib import Path


def save_fem_structure(fem_structure, filepath):
    """
    Save a FEM_structure object to files for later reconstruction.
    
    This function saves:
    1. Initial conditions and connectivity matrices (structure setup)
    2. Current deformed state (coords_current, coords_rotations_current)
    3. Dynamic properties (spring stiffnesses, beam E and G values)
    
    Parameters
    ----------
    fem_structure : FEM_structure
        The FEM structure object to save
    filepath : str or Path
        Path to save the structure. Extension will be added automatically.
        Creates a directory with the structure data.
    
    Returns
    -------
    str
        Path to the saved directory
    
    Examples
    --------
    >>> save_fem_structure(fem_struct, "results/my_structure")
    'results/my_structure'
    """
    # Convert to Path object and create directory
    save_dir = Path(filepath)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store metadata and list data
    data = {}
    
    # ===== 1. Save initial conditions and connectivity matrices =====
    data['initial_conditions'] = fem_structure.initial_conditions
    
    # Save connectivity matrices (check if they exist)
    if fem_structure.spring_matrix is not None:
        data['spring_matrix'] = fem_structure.spring_matrix.tolist() if isinstance(
            fem_structure.spring_matrix, np.ndarray) else fem_structure.spring_matrix
    else:
        data['spring_matrix'] = None
    
    if fem_structure.pulley_matrix is not None:
        data['pulley_matrix'] = fem_structure.pulley_matrix.tolist() if isinstance(
            fem_structure.pulley_matrix, np.ndarray) else fem_structure.pulley_matrix
    else:
        data['pulley_matrix'] = None
    
    if fem_structure.beam_matrix is not None:
        data['beam_matrix'] = fem_structure.beam_matrix.tolist() if isinstance(
            fem_structure.beam_matrix, np.ndarray) else fem_structure.beam_matrix
    else:
        data['beam_matrix'] = None
    
    # ===== 2. Save current deformed state =====
    # Save as numpy arrays for precision
    np.save(save_dir / 'coords_current.npy', fem_structure.coords_current)
    np.save(save_dir / 'coords_rotations_current.npy', fem_structure.coords_rotations_current)
    
    # ===== 3. Save dynamic properties =====
    # Spring stiffnesses and rest lengths
    spring_properties = []
    for i, spring_elem in enumerate(fem_structure.spring_elements):
        spring_props = {
            'index': i,
            'k': float(spring_elem.k),
            'l0': float(spring_elem.l0),
            'springtype': spring_elem.springtype,
            'n1': int(spring_elem.spring.n1),
            'n2': int(spring_elem.spring.n2)
        }
        if spring_elem.springtype == "pulley":
            spring_props['i_other_pulley'] = int(spring_elem.i_other_pulley)
        spring_properties.append(spring_props)
    
    data['spring_properties'] = spring_properties
    
    # Beam properties
    beam_properties = []
    for i, beam_elem in enumerate(fem_structure.beam_elements):
        beam_props = {
            'index': i,
            'E': float(beam_elem.E) if hasattr(beam_elem, 'E') else None,
            'G': float(beam_elem.G) if hasattr(beam_elem, 'G') else None,
            'k': float(beam_elem.k) if hasattr(beam_elem, 'k') else None,
            'n1': int(beam_elem.beam.n1),
            'n2': int(beam_elem.beam.n2),
            'length': float(beam_elem.beam.length)
        }
        # Save additional beam properties if available
        if hasattr(beam_elem, 'r'):
            beam_props['r'] = float(beam_elem.r)
        if hasattr(beam_elem, 'p'):
            beam_props['p'] = float(beam_elem.p)
        if hasattr(beam_elem, 'A'):
            beam_props['A'] = float(beam_elem.A)
        if hasattr(beam_elem, 'I'):
            beam_props['I'] = float(beam_elem.I)
        if hasattr(beam_elem, 'J'):
            beam_props['J'] = float(beam_elem.J)
        
        beam_properties.append(beam_props)
    
    data['beam_properties'] = beam_properties
    
    # Save number of nodes for validation
    data['num_nodes'] = int(fem_structure.num_nodes)
    data['N'] = int(fem_structure.N)
    
    # Write metadata to JSON file
    with open(save_dir / 'structure_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"FEM structure saved to: {save_dir}")
    return str(save_dir)


def load_fem_structure(filepath):
    """
    Load a FEM_structure object from saved files.
    
    This function loads and reconstructs a complete FEM_structure with:
    1. Structure setup from initial conditions and connectivity matrices
    2. Deformed state from saved coordinates
    3. Dynamic properties (updated spring k, beam E and G values)
    
    Parameters
    ----------
    filepath : str or Path
        Path to the saved structure directory
    
    Returns
    -------
    FEM_structure
        Reconstructed FEM structure object with deformed state and dynamic properties
    
    Examples
    --------
    >>> fem_struct = load_fem_structure("results/my_structure")
    >>> # Structure is ready to use with saved deformation and properties
    """
    from kite_fem.FEMStructure import FEM_structure
    
    # Convert to Path object
    load_dir = Path(filepath)
    
    if not load_dir.exists():
        raise FileNotFoundError(f"Directory not found: {load_dir}")
    
    # Load metadata from JSON
    with open(load_dir / 'structure_data.json', 'r') as f:
        data = json.load(f)
    
    # ===== 1. Reconstruct FEM_structure with initial setup =====
    initial_conditions = data['initial_conditions']
    
    # Convert matrices back to proper format if they exist
    # Keep as list format since FEM_structure expects lists, not numpy arrays
    spring_matrix = data['spring_matrix'] if data['spring_matrix'] is not None else None
    pulley_matrix = data['pulley_matrix'] if data['pulley_matrix'] is not None else None
    beam_matrix = data['beam_matrix'] if data['beam_matrix'] is not None else None
    
    # Create the FEM structure
    fem_structure = FEM_structure(
        initial_conditions=initial_conditions,
        spring_matrix=spring_matrix,
        pulley_matrix=pulley_matrix,
        beam_matrix=beam_matrix
    )
    
    # ===== 2. Load and apply deformed state =====
    coords_current = np.load(load_dir / 'coords_current.npy')
    coords_rotations_current = np.load(load_dir / 'coords_rotations_current.npy')
    
    fem_structure.coords_current = coords_current
    fem_structure.coords_rotations_current = coords_rotations_current
    
    # ===== 3. Apply dynamic properties =====
    # Update spring stiffnesses
    for spring_data in data['spring_properties']:
        idx = spring_data['index']
        if idx < len(fem_structure.spring_elements):
            spring_elem = fem_structure.spring_elements[idx]
            spring_elem.k = spring_data['k']
            spring_elem.l0 = spring_data['l0']
            spring_elem.spring.kxe = spring_data['k']
    
    # Update beam properties
    for beam_data in data['beam_properties']:
        idx = beam_data['index']
        if idx < len(fem_structure.beam_elements):
            beam_elem = fem_structure.beam_elements[idx]
            
            # Update stiffness properties
            if beam_data['E'] is not None:
                beam_elem.E = beam_data['E']
                beam_elem.prop.E = beam_data['E']
            
            if beam_data['G'] is not None:
                beam_elem.G = beam_data['G']
                beam_elem.prop.G = beam_data['G']
            
            if beam_data['k'] is not None:
                beam_elem.k = beam_data['k']
            
            # Update geometric properties if available
            if 'r' in beam_data and beam_data['r'] is not None:
                beam_elem.r = beam_data['r']
            if 'p' in beam_data and beam_data['p'] is not None:
                beam_elem.p = beam_data['p']
            if 'A' in beam_data and beam_data['A'] is not None:
                beam_elem.A = beam_data['A']
                beam_elem.prop.A = beam_data['A']
            if 'I' in beam_data and beam_data['I'] is not None:
                beam_elem.I = beam_data['I']
                beam_elem.prop.Iyy = beam_data['I']
                beam_elem.prop.Izz = beam_data['I']
            if 'J' in beam_data and beam_data['J'] is not None:
                beam_elem.J = beam_data['J']
                beam_elem.prop.J = beam_data['J']
    
    # Validate loaded structure
    if fem_structure.num_nodes != data['num_nodes']:
        raise ValueError(
            f"Mismatch in number of nodes: expected {data['num_nodes']}, got {fem_structure.num_nodes}"
        )
    
    print(f"FEM structure loaded from: {load_dir}")
    print(f"  Nodes: {fem_structure.num_nodes}")
    print(f"  Spring elements: {len(fem_structure.spring_elements)}")
    print(f"  Beam elements: {len(fem_structure.beam_elements)}")
    
    return fem_structure


def save_fem_structure_simple(fem_structure, filepath):
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
    }
    
    # Save connectivity matrices
    if fem_structure.spring_matrix is not None:
        save_dict['spring_matrix'] = np.array(fem_structure.spring_matrix)
    if fem_structure.pulley_matrix is not None:
        save_dict['pulley_matrix'] = np.array(fem_structure.pulley_matrix)
    if fem_structure.beam_matrix is not None:
        save_dict['beam_matrix'] = np.array(fem_structure.beam_matrix)
    
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
        save_dict['beam_E'] = beam_E
        save_dict['beam_G'] = beam_G
        save_dict['beam_k'] = beam_k
    
    # Convert initial conditions to array for storage
    # Store as a structured format that can be reconstructed
    initial_conds_array = []
    for pos, vel, mass, fixed in fem_structure.initial_conditions:
        initial_conds_array.append([*pos, *vel, mass, int(fixed)])
    save_dict['initial_conditions_array'] = np.array(initial_conds_array)
    
    np.savez_compressed(filepath, **save_dict)
    
    print(f"FEM structure saved to: {filepath}")
    return str(filepath)


def load_fem_structure_simple(filepath):
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
    
    print(f"FEM structure loaded from: {filepath}")
    return fem_structure
