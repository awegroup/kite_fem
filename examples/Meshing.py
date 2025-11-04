from kite_fem.MeshBuilder import InteractiveMeshBuilder
# Start the interactive mesh builder
# You can add points by pressing a and clicking on the window
# You can add connections of springs,beams and pulleys by pressing 1,2,3 respectively and clicking on nodes. A pulley requires three nodes

builder = InteractiveMeshBuilder()
builder.run()

#the output can be used to initialize a kite_fem simulation as follows:
#Model = FEM_structure(initial_conditions, spring_matrix=spring_matrix, beam_matrix=beam_matrix, pulley_matrix=pulley_matrix)

#You can also provide an initial point cloud as follows:
# Format: [[x,y,z], [vx,vy,vz], mass, fixed.      [vx,vy,vz] and mass are not used but kept for compatibility]

initial_conditions = [
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1, True], 
    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1, True], 
    [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], 1, True], 
    [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], 1, True],
    [[0.5, 0.5, 0.0], [0.0, 0.0, 0.0], 1, False]
]

print("Starting Interactive Mesh Builder with kite point cloud...")
print(f"Loaded {len(initial_conditions)} points")

builder = InteractiveMeshBuilder(initial_conditions)
builder.run()