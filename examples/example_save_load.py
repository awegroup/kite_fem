"""
Example demonstrating how to save and load FEM_structure objects.

This example shows:
1. Creating a simple FEM structure
2. Solving to get a deformed state
3. Saving the structure with all properties
4. Loading the structure back
5. Verifying the loaded structure matches the original
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from kite_fem.FEMStructure import FEM_structure
from kite_fem.saveload import save_fem_structure, load_fem_structure
from kite_fem.saveload import save_fem_structure_simple, load_fem_structure_simple


def create_simple_structure():
    """Create a simple cantilever beam structure for testing."""
    # Initial conditions: [position, velocity, mass, fixed]
    initial_conditions = [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, True],   # Fixed base
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, False],  # Middle node
        [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, False],  # Tip node
    ]
    
    # Spring connectivity: [n1, n2, k, c, l0, springtype]
    spring_matrix = [
        [0, 1, 1000.0, 0.0, 1.0, "default"],
        [1, 2, 1000.0, 0.0, 1.0, "default"],
    ]
    
    # Create structure
    fem_struct = FEM_structure(
        initial_conditions=initial_conditions,
        spring_matrix=spring_matrix,
        pulley_matrix=None,
        beam_matrix=None
    )
    
    return fem_struct


def example_directory_format():
    """Example using the directory-based save/load format."""
    print("=" * 60)
    print("Example 1: Directory-based format")
    print("=" * 60)
    
    # Create and solve structure
    print("\n1. Creating FEM structure...")
    fem_struct = create_simple_structure()
    
    # Apply a load to deform the structure
    print("2. Solving with applied load...")
    fe = np.zeros(fem_struct.N)
    fe[2*6 + 2] = -10.0  # Apply downward force at tip (z-direction)
    
    converged, runtime = fem_struct.solve(
        fe=fe,
        max_iterations=50,
        tolerance=1e-3,
        print_info=True
    )
    
    if converged:
        print(f"   Converged in {runtime:.4f} seconds")
    
    # Modify spring stiffness to simulate dynamic behavior
    print("3. Modifying spring stiffnesses...")
    fem_struct.spring_elements[0].k = 1200.0
    fem_struct.spring_elements[1].k = 800.0
    
    # Save structure
    print("\n4. Saving structure...")
    save_path = save_fem_structure(fem_struct, "test_save_dir")
    
    # Load structure
    print("\n5. Loading structure...")
    loaded_struct = load_fem_structure(save_path)
    
    # Verify
    print("\n6. Verification:")
    print(f"   Original tip position: {fem_struct.coords_current[-3:]}")
    print(f"   Loaded tip position:   {loaded_struct.coords_current[-3:]}")
    print(f"   Position match: {np.allclose(fem_struct.coords_current, loaded_struct.coords_current)}")
    
    print(f"   Original spring k's: {[s.k for s in fem_struct.spring_elements]}")
    print(f"   Loaded spring k's:   {[s.k for s in loaded_struct.spring_elements]}")
    print(f"   Stiffness match: {[s1.k == s2.k for s1, s2 in zip(fem_struct.spring_elements, loaded_struct.spring_elements)]}")
    
    return fem_struct, loaded_struct


def example_simple_format():
    """Example using the simple npz file format."""
    print("\n" + "=" * 60)
    print("Example 2: Simple NPZ format")
    print("=" * 60)
    
    # Create and solve structure
    print("\n1. Creating FEM structure...")
    fem_struct = create_simple_structure()
    
    # Apply a load
    print("2. Solving with applied load...")
    fe = np.zeros(fem_struct.N)
    fe[2*6 + 2] = -15.0  # Apply larger downward force at tip
    
    converged, runtime = fem_struct.solve(
        fe=fe,
        max_iterations=50,
        tolerance=1e-3,
        print_info=False
    )
    
    if converged:
        print(f"   Converged in {runtime:.4f} seconds")
    
    # Save structure
    print("\n3. Saving structure (simple format)...")
    save_path = save_fem_structure_simple(fem_struct, "test_save_simple.npz")
    
    # Load structure
    print("\n4. Loading structure (simple format)...")
    loaded_struct = load_fem_structure_simple(save_path)
    
    # Verify
    print("\n5. Verification:")
    print(f"   Original tip position: {fem_struct.coords_current[-3:]}")
    print(f"   Loaded tip position:   {loaded_struct.coords_current[-3:]}")
    print(f"   Position match: {np.allclose(fem_struct.coords_current, loaded_struct.coords_current)}")
    
    return fem_struct, loaded_struct


def example_with_beams():
    """Example with beam elements (if available)."""
    print("\n" + "=" * 60)
    print("Example 3: Structure with beam elements")
    print("=" * 60)
    
    # Initial conditions
    initial_conditions = [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, True],   # Fixed base
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, False],  # Tip node
    ]
    
    # Beam connectivity: [n1, n2, diameter, pressure, length]
    beam_matrix = [
        [0, 1, 0.1, 1000.0, 1.0],  # Inflatable beam
    ]
    
    print("\n1. Creating FEM structure with beam...")
    fem_struct = FEM_structure(
        initial_conditions=initial_conditions,
        spring_matrix=None,
        pulley_matrix=None,
        beam_matrix=beam_matrix
    )
    
    # Apply a load
    print("2. Solving with applied load...")
    fe = np.zeros(fem_struct.N)
    fe[1*6 + 2] = -5.0  # Apply downward force at tip
    
    converged, runtime = fem_struct.solve(
        fe=fe,
        max_iterations=100,
        tolerance=1e-3,
        print_info=False
    )
    
    if converged:
        print(f"   Converged in {runtime:.4f} seconds")
    
    # Save structure
    print("\n3. Saving structure with beam...")
    save_path = save_fem_structure(fem_struct, "test_save_beam")
    
    # Load structure
    print("\n4. Loading structure with beam...")
    loaded_struct = load_fem_structure(save_path)
    
    # Verify beam properties
    print("\n5. Verification:")
    print(f"   Original beam E: {fem_struct.beam_elements[0].E:.2f}")
    print(f"   Loaded beam E:   {loaded_struct.beam_elements[0].E:.2f}")
    print(f"   Original beam G: {fem_struct.beam_elements[0].G:.2f}")
    print(f"   Loaded beam G:   {loaded_struct.beam_elements[0].G:.2f}")
    print(f"   Beam properties match: {np.isclose(fem_struct.beam_elements[0].E, loaded_struct.beam_elements[0].E)}")
    
    return fem_struct, loaded_struct


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FEM Structure Save/Load Examples")
    print("="*60)
    
    # Run examples
    try:
        original1, loaded1 = example_directory_format()
        original2, loaded2 = example_simple_format()
        
        # Try beam example if beam matrix is supported
        try:
            original3, loaded3 = example_with_beams()
        except Exception as e:
            print(f"\nBeam example skipped: {e}")
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during examples: {e}")
        import traceback
        traceback.print_exc()
