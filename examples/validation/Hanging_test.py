from pathlib import Path
from multiprocessing import Pool, cpu_count
import copy

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
from kite_fem.Functions import relaxbridles, fix_nodes,set_pressure, check_element_strain, adapt_stiffnesses

import matplotlib.pyplot as plt
import numpy as np
import csv

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

def create_kite():
    """Factory function to create a fresh kite instance"""
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
    kite = fix_nodes(kite,[0,129,128,127,72,81,82,115,116,106])
    kite = fix_nodes(kite,[0,129,128,127,72,81,82,115,116,106])
    origin =  [82,116]
    kite = relaxbridles(kite,canopy_nodes,origin)
    kite = fix_nodes(kite,[0,129,128,127,72,81,82,115,116,106])
    return kite

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
    # For uneven length, middle_idx is the exact center
    # Get the points on either side of the center
    left_le_id = le_ids[middle_idx]
    right_le_id = le_ids[middle_idx-1]
    rightids = [right_le_id, te_ids[-1]]
    leftids = [left_le_id, te_ids[0]]
    right = np.linalg.norm(coords[rightids[0]]-coords[rightids[1]])
    left = np.linalg.norm(coords[leftids[0]]-coords[leftids[1]])
    phi.append(right)
    phi.append(left)
    return phi

def loading(N,m_arr,tip_load,point_load):
    fe = np.zeros(N)
    gravity = m_arr*9.81
    fe[2::6] = gravity 
    fe[2*6+1] += tip_load*9.81
    fe[58*6+1] += -tip_load*9.81
    fe[29*6+2] += point_load*9.81
    return fe

def solve_single_case(args):
    """Worker function to solve a single load case"""
    pressure, tip_load, point_load, load_case = args
    
    # Create fresh kite instance for this process
    kite = create_kite()
    kite = set_pressure(kite, pressure)
    fe = loading(kite.N, m_arr, tip_load, point_load)
    max_strain = 100
    iteration = 1
    print("set up kite for case",load_case)
    while max_strain >1 and iteration <5:
        print("load case",load_case, "iteration",iteration)
        kite.solve(fe=fe, max_iterations=15000, tolerance=0.01, step_limit=.005, 
                relax_init=.25, relax_min=0.00, relax_update=0.998, k_update=1, I_stiffness=15)
        strain_data = check_element_strain(kite, False)
        all_strains = strain_data['spring_strains'] + strain_data['beam_strains']
        max_strain = max(all_strains)
        adapt_stiffnesses(kite)
        iteration += 1
        
    phi = extract_lengths_validation(kite, strut_sections)
    tolerance = kite.crisfield_history[-1]
    
    output = phi
    output.append(tolerance)
    output.insert(0, load_case)
    output.append(max_strain)
    
    return output

def write_results(results):
    """Write all results to CSV file"""
    csv_path = Path(__file__).parent / "model_results.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Load case",'La','Lb','Lc','Ld','Le','Lf','Lg','Lh','Li','b','LcsTL','LcsTR',"Tolerance","max strain"])
        writer.writerows(results)

if __name__ == '__main__':
    pressures = [0.15, 0.25]
    tip_loads = [2, 5]           #kg
    point_loads = [9.7, 25.2]     #kg
    
    # Build list of all load cases
    load_cases = []
    load_case = 0
    
    for pressure in pressures:
        # Base case (no additional loads)
        load_case += 1
        load_cases.append((pressure, 0, 0, load_case))
        # Point load cases
        for point_load in point_loads:
            load_case += 1
            load_cases.append((pressure, 0, point_load, load_case))
        # Tip load cases
        for tip_load in tip_loads:
            load_case += 1
            load_cases.append((pressure, tip_load, 0, load_case))
    
    # Run simulations in parallel
    n_cores = cpu_count()
    print(f"Running {len(load_cases)} load cases on {n_cores} CPU cores...")
    
    with Pool(processes=n_cores) as pool:
        results = pool.map(solve_single_case, load_cases)
    
    # Write all results at once
    write_results(results)
    print("All simulations complete!")






