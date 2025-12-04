

import numpy as np

def triangle_area(p1, p2, p3):
    # Vectors for two sides of the triangle
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product magnitude gives 2 * triangle area
    return 0.5 * np.linalg.norm(np.cross(v1, v2))

def quad_area(A, B, C, D):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)

    # Split quad into triangles ABC and ACD
    area1 = triangle_area(A, B, C)
    area2 = triangle_area(A, C, D)
    return area1 + area2


def coords(idx):
    if idx <= 4:
        x = idx
        y = 0
        z = 0
    if 8 >= idx > 4:
        x = idx-4
        y = 1
        z = 0
    else:
        x = idx-8
        y = 2
        z = 0
    return np.array([x,y,z])


section1 =[1,2,3,4]
section2 =[5,6,7,8]
section3 =[9,10,11,12]
canopy_sections = [section1,section2,section3]
density = 120 #g/m^2

# Generate all quadrilateral connections between adjacent sections
quads = []
quad_areas = []

# Dictionary to store mass at each node
node_masses = {}

for i in range(len(canopy_sections) - 1):
    section_a = canopy_sections[i]
    section_b = canopy_sections[i + 1]
    # Create quads by connecting adjacent nodes in consecutive sections
    for j in range(len(section_a) - 1):
        quad = [section_a[j], section_a[j+1], section_b[j+1], section_b[j]]
        # Get coordinates and calculate area
        corners = [coords(node) for node in quad]
        area = quad_area(corners[0], corners[1], corners[2], corners[3])
        # Distribute quad mass to its 4 nodes (each gets 1/4 of quad mass)
        quad_mass = area * density / 1000  # Convert g/m^2 to kg/m^2
        for node in quad:
            node_masses[node] = node_masses.get(node, 0) + quad_mass / 4

print("Node masses (kg):", node_masses)

print("Quads:", quads)
print("Areas:", quad_areas)
print("total Area:", sum(quad_areas))
