import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamC, BeamCData, BeamCProbe, DOF, INT, DOUBLE
from pyfe3d import BeamLR, BeamLRData, BeamLRProbe, DOF, INT, DOUBLE

# def calculate_vxy_vector(node1_coords, node2_coords):
#     # 1. Calculate the vector along the beam's length (local x-axis direction)
#     beam_vector = node2_coords - node1_coords
#     # Check for zero-length beam element (nodes are coincident)
#     length = np.linalg.norm(beam_vector)
#     if length < 1e-9: # Use a small tolerance for floating point comparisons
#         raise ValueError("Nodes are coincident, cannot define beam element direction.")
#     e_x = beam_vector / length # Normalized vector for local x-axis
#     # 2. Define a primary global reference vector for orienting the element's cross-section.
#     global_z_ref = np.array([0.0, 0.0, 1.0])
#     # 3. Calculate a candidate vector for the local y-axis.
#     vxy_candidate = np.cross(global_z_ref, e_x)
#     # 4. Handle the edge case where the beam's x-axis is parallel to global_z_ref.
#     if np.linalg.norm(vxy_candidate) < 1e-9:
#         global_y_ref = np.array([0.0, 1.0, 0.0])
#         vxy_candidate = np.cross(global_y_ref, e_x)
#         if np.linalg.norm(vxy_candidate) < 1e-9:
#             global_x_ref = np.array([1.0, 0.0, 0.0])
#             vxy_candidate = np.cross(global_x_ref, e_x)
#             if np.linalg.norm(vxy_candidate) < 1e-9:
#                 raise RuntimeError("Could not robustly determine vxy vector for beam. This implies a highly unusual or degenerate input configuration.")
#     # 5. Normalize the vxy_candidate to get the final vxy vector.
#     vxy = vxy_candidate / np.linalg.norm(vxy_candidate)
#     return vxy


n = 15
L = 1
P = -5

nu = 0.3
rho = 7.83e3 # kg/m3

x = np.linspace(0, L, n)
y = np.zeros_like(x)
hy = 1 # m
hz = 1 # m
A = hy*hz
Izz = hz*hy**3/12
Iyy = hz**3*hy/12

Load_param = 1

EI = -P*L**2/Load_param
E = EI/Izz

print('E', E)
# print('Iyy', Iyy)
# print('Izz', Izz)

ncoords = np.vstack((x, y, np.zeros_like(x))).T
nids = 1 + np.arange(ncoords.shape[0])
nid_pos = dict(zip(nids, np.arange(len(nids))))

n1s = nids[0:-1]
n2s = nids[1:]

num_elements = len(n1s)
# print('num_elements', num_elements)

p = BeamCProbe()
data = BeamCData()

KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
KGr = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
KGc = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
KGv = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
N = DOF*n
# print('num_DOF', N)

prop = BeamProp()
prop.A = A
prop.E = E
scf = 5/6.
prop.G = scf*E/2/(1+nu)
prop.Izz = Izz
prop.Iyy = Iyy
prop.J = Izz + Iyy

ncoords_flatten = ncoords.flatten()
beams = []
init_k_KC0 = 0
init_k_KG = 0
for n1, n2 in zip(n1s, n2s):
    pos1 = nid_pos[n1]
    pos2 = nid_pos[n2]
    beam = BeamC(p)
    beam.init_k_KC0 = init_k_KC0
    beam.init_k_KG = init_k_KG
    beam.n1 = n1
    beam.n2 = n2
    beam.c1 = DOF*pos1
    beam.c2 = DOF*pos2
    beam.update_rotation_matrix(1,1,0,ncoords_flatten)
    beam.update_probe_xe(ncoords_flatten)
    beam.update_KC0(KC0r, KC0c, KC0v, prop)
    beam.update_KG(KC0r, KC0c, KC0v, prop)
    beams.append(beam)
    init_k_KC0 += data.KC0_SPARSE_SIZE
    init_k_KG += data.KG_SPARSE_SIZE

# print('elements created')

KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    


# print('sparse KC0 created')

bk = np.zeros(N, dtype=bool)

left_end = np.isclose(x, 0.)
right_end = np.isclose(x, L)

bk[0::DOF][left_end] = True
bk[1::DOF][left_end] = True
bk[2::DOF][left_end] = True
bk[3::DOF][left_end] = True
bk[4::DOF][left_end] = True
bk[5::DOF][left_end] = True

bu = ~bk

fext = np.zeros(N)
fext[1::DOF][-1] = P

Kuu = KC0[bu, :][:, bu]
KGuu = KG[bu, :][:, bu]

Kt = Kuu + KGuu
relax = 0.5
u = np.zeros(N)
u[bu] = relax*spsolve(Kt, fext[bu])


x = u[0::DOF] + ncoords_flatten[0::3]
y = u[1::DOF] + ncoords_flatten[1::3]

plt.plot(x, y, label="linear beam deflection")

xyz = np.zeros_like(u, dtype=bool)
xyz[0::DOF] = xyz[1::DOF] = xyz[2::DOF] = True  


for i in range(1000):
    ncoords_flatten += u[xyz]
    for beam in beams:
        beam.update_probe_ue(u)
        beam.update_probe_xe(ncoords_flatten)   
        # i,j,k = calculate_vxy_vector(np.array(beam.probe.xe)[0:3], np.array(beam.probe.xe)[3:6])
        beam.update_rotation_matrix(1,1,0,ncoords_flatten)
        beam.update_KC0(KC0r, KC0c, KC0v, prop)
        beam.update_KG(KGr, KGc, KGv, prop)

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()

    Kuu = KC0[bu, :][:, bu]
    KGuu = KG[bu, :][:, bu]

    Kt = Kuu + KGuu
    u = np.zeros(N)
    u[bu] = spsolve(Kt, fext[bu])*relax
    x = u[0::DOF] + ncoords_flatten[0::3]
    y = u[1::DOF] + ncoords_flatten[1::3]
    z = u[2::DOF] + ncoords_flatten[2::3]
    plt.plot(x, y, label="nonlinear beam deflection i="+str(i+1))


plt.plot(0.95,-0.3, 'ro', label="Target deflection")
plt.xlabel('X-coordinate (m)')
plt.ylabel('Y-coordinate (m)')
plt.title("cantilever beam deflection under load")
# plt.legend()
plt.grid()
plt.show()


