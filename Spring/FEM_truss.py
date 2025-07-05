import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg, spsolve
from pyfe3d.beamprop import BeamProp
from pyfe3d import Truss, TrussData, TrussProbe, DOF, INT, DOUBLE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 5
L = 10

E = 1 # Pa
rho = 1 # kg/m3

x = np.linspace(0, L, n)
y = np.zeros_like(x)
b = 1 # m
h = 1 # m
A = h*b

ncoords = np.vstack((x, y, np.zeros_like(x))).T
nids = 1 + np.arange(ncoords.shape[0])
nid_pos = dict(zip(nids, np.arange(len(nids))))

n1s = nids[0:-1]
n2s = nids[1:]

num_elements = len(n1s)

p = TrussProbe()
data = TrussData()

KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
Mr = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
Mc = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
Mv = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=DOUBLE)
N = DOF*n

prop = BeamProp()
prop.A = A
prop.E = E
scf = 5/6.
prop.G = scf*E/2/(1+0.3)
prop.intrho = rho*A
prop.intrhoy2 = 0 # used to calculate torsional constant
prop.intrhoz2 = 0 # used to calculate torsional constant

ncoords_flatten = ncoords.flatten()
print('ncoords_flatten', ncoords_flatten)
trusses = []
init_k_KC0 = 0
init_k_M = 0
for n1, n2 in zip(n1s, n2s):
    pos1 = nid_pos[n1]
    pos2 = nid_pos[n2]
    truss = Truss(p)
    truss.init_k_KC0 = init_k_KC0
    truss.init_k_M = init_k_M
    truss.n1 = n1
    truss.n2 = n2
    truss.c1 = DOF*pos1
    truss.c2 = DOF*pos2
    truss.update_rotation_matrix(ncoords_flatten)
    truss.update_probe_xe(ncoords_flatten)
    truss.update_KC0(KC0r, KC0c, KC0v, prop)
    truss.update_M(Mr, Mc, Mv, prop)
    trusses.append(truss)
    init_k_KC0 += data.KC0_SPARSE_SIZE
    init_k_M += data.M_SPARSE_SIZE

for truss in trusses:

    print(np.array(truss.probe.xe))
# print('elements created')

KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()

# print('sparse KC0 and M created')
# applying boundary conditions
bk = np.zeros(N, dtype=bool)
# Constraining 
bk[0:N*DOF] = True  # all nodes are fully constrained
bk[0::DOF] = False # free x coords
bk[0] = True  # constrain node 1 in x direction


bu = ~bk
# force applied on node 1
fext = np.zeros(N)
fext[DOF*num_elements] = 10  # force in x-direction at node 1

Kuu = KC0[bu, :][:, bu]
# print('Kuu', Kuu)
u = np.zeros(N)
print(fext[bu])

u[bu] = spsolve(Kuu, fext[bu])

positions = np.zeros_like(u, dtype=bool)
positions[0::DOF] = positions[1::DOF] = positions[2::DOF] = True  

for truss in trusses:
    truss.update_probe_ue(u)
    truss.update_probe_xe(ncoords_flatten+ u[positions])
    # print("ue",np.array(truss.probe.ue))
    # print("xe",np.array(np.round(truss.probe.xe,1)))




