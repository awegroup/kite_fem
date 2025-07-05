import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamC, BeamCData, BeamCProbe, DOF, INT, DOUBLE
from pyfe3d import BeamLR, BeamLRData, BeamLRProbe, DOF, INT, DOUBLE

n = 15
L = 10
P = -50

nu = 0.3
rho = 7.83e3 # kg/m3

x = np.linspace(0, L, n)
y = np.zeros_like(x)
hy = 1.862 # m
hz = 1.862 # m
A = hy*hz
Izz = hz*hy**3/12
Iyy = hz**3*hy/12
print(Izz, Iyy)
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
scf = 5/6
prop.G = scf*E/2/(1+nu)
prop.Izz = Izz
prop.Iyy = Iyy
prop.J = Izz + Iyy

ncoords_flatten = ncoords.flatten()
beams = []
init_k_KC0 = 0
init_k_KG = 0
for n1, n2 in zip(n1s, n2s):
    p = BeamCProbe()
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
relax = 1
Kt = Kuu + KGuu
u = np.zeros(N)
u[bu] = spsolve(Kt, fext[bu])


x = u[0::DOF] + ncoords_flatten[0::3]
y = u[1::DOF] + ncoords_flatten[1::3]

plt.plot(x, y, label="linear beam deflection")

xyz = np.zeros_like(u, dtype=bool)
xyz[0::DOF] = xyz[1::DOF] = xyz[2::DOF] = True  

ncoords_init = ncoords_flatten.copy()
ncoords_current = np.zeros_like(ncoords_flatten)
ncoords_current = ncoords_init + u[xyz]

# KC0v *= 0
# KGv *= 0
# for beam in beams:
#     beam.update_probe_ue(u)
#     beam.update_probe_xe(ncoords_init + u[xyz])   
#     beam.update_rotation_matrix(1,1,0,ncoords_init + u[xyz])
#     beam.update_KC0(KC0r, KC0c, KC0v, prop)
#     beam.update_KG(KGr, KGc, KGv, prop)
# Calculate residual:
       
for i in range(100):
    ncoords_current = ncoords_init + u[xyz]
    KC0v *= 0
    KGv *= 0
    for beam in beams:
        beam.update_probe_ue(u)
        beam.update_probe_xe(ncoords_current)   
        beam.update_rotation_matrix(1,1,0,ncoords_current)
        beam.update_KC0(KC0r, KC0c, KC0v, prop)
        beam.update_KG(KGr, KGc, KGv, prop)

       
    beam.update_probe_ue(u)
    beam.update_probe_xe(ncoords_current)   
    beam.update_rotation_matrix(1,1,0,ncoords_current)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()

    Kuu = KC0[bu, :][:, bu]
    KGuu = KG[bu, :][:, bu]

    Kt = Kuu + KGuu
    u = np.zeros(N)
    u[bu] = spsolve(Kt, fext[bu])
    x = ncoords_current[0::3]
    y = ncoords_current[1::3]
    z = ncoords_current[2::3]
    
plt.plot(x, y, label="nonlinear beam deflection")
    

    
# plt.plot(x, y, label="nonlinear beam deflection i="+str(i+1))


# plt.plot(9.5,-3, 'ro', label="Target deflection")
plt.xlabel('X-coordinate (m)')
plt.ylabel('Y-coordinate (m)')
plt.title("cantilever beam deflection under load")
plt.legend()
plt.grid()
plt.show()


