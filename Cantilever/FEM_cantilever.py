import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamLR, BeamLRData, BeamLRProbe, DOF, INT, DOUBLE

n = 50
L = 10
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

Load_param = 10

EI = -P*L**2/Load_param
E = EI/Izz

# print('E', E)
# print('Iyy', Iyy)
# print('Izz', Izz)

ncoords = np.vstack((x, y, np.zeros_like(x))).T
nids = 1 + np.arange(ncoords.shape[0])
nid_pos = dict(zip(nids, np.arange(len(nids))))

n1s = nids[0:-1]
n2s = nids[1:]

num_elements = len(n1s)
# print('num_elements', num_elements)

p = BeamLRProbe()
data = BeamLRData()

KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
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
for n1, n2 in zip(n1s, n2s):
    pos1 = nid_pos[n1]
    pos2 = nid_pos[n2]
    beam = BeamLR(p)
    beam.init_k_KC0 = init_k_KC0
    beam.n1 = n1
    beam.n2 = n2
    beam.c1 = DOF*pos1
    beam.c2 = DOF*pos2
    beam.update_rotation_matrix(1., 1., 0, ncoords_flatten)
    beam.update_probe_xe(ncoords_flatten)
    beam.update_KC0(KC0r, KC0c, KC0v, prop)
    beams.append(beam)
    init_k_KC0 += data.KC0_SPARSE_SIZE

# print('elements created')

KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

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

u = np.zeros(N)
u[bu] = spsolve(Kuu, fext[bu])

dx = u[0::DOF]
dy = u[1::DOF]

plt.plot(x, y, 'b-', label='Initial')
plt.plot(x, y+dy, 'r-', label='Displacement')

plt.show()


# Print maximum deflections and angle at the tip
max_deflection_y = -np.min(dy)/L  # Most negative value (downwards)
max_deflection_x = dx[np.argmin(dy)]/L  # x-deflection at max y-deflection
angle_tip = -np.arctan2(dy[-1] - dy[-2], x[-1] - x[-2])

print(Load_param)
print(max_deflection_y)
print(max_deflection_x)
print(angle_tip)


