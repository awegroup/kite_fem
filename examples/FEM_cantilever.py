import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve, lsqr
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamC, BeamCData, BeamCProbe, DOF, INT, DOUBLE
from pyfe3d import BeamLR, BeamLRData, BeamLRProbe, DOF, INT, DOUBLE
from pyfe3d import Spring, SpringData, SpringProbe

def unit_vector(beam, ncoords):
    xi = ncoords[beam.c2//2 + 0] - ncoords[beam.c1//2 + 0]
    xj = ncoords[beam.c2//2 + 1] - ncoords[beam.c1//2 + 1]
    xk = ncoords[beam.c2//2 + 2] - ncoords[beam.c1//2 + 2]
    l = (xi**2 + xj**2 + xk**2)**0.5
    unit_vect = np.array([xi, xj, xk])/l
    return unit_vect,l
    
def plot(fig,ax, c='blue'):
    for beam in beams:
        n1 = beam.n1-1
        n2 = beam.n2-1
        x1 = ncoords_current[n1*3]
        x2 = ncoords_current[n2*3]
        y1 = ncoords_current[n1*3 + 1]
        y2 = ncoords_current[n2*3 + 1]
        ax.plot([x1, x2], [y1, y2], color=c)
    return fig, ax

n = 2
L = 10
P = -5

nu = 0
rho = 0

x = np.linspace(0, L, n)
y = np.zeros_like(x)

# R = 0.05 # m
dy = 0.1 # m
dx = 0.1 # m
A = dx * dy # m^2
Izz = dy * dx**3 / 12
Iyy = dx * dy**3 / 12


Load_params = [0.5,1, 2, 3, 4, 5,7.5,10]
wL_lst = [0.161665,0.30172,0.49346,0.60325,0.66996,0.71379,0.7767,0.81061]
uL_lst = [0.01642,0.05643,0.16064,0.25442,0.32894,0.38763,0.48957,0.555]
theta_lst = [0.24395,0.46135,0.78175,0.98602,1.12124,1.21537,1.35593,1.43029]
Load_param = 10
integer = np.where(np.isclose(Load_params, Load_param))[0][0]
wL = wL_lst[integer]
uL = uL_lst[integer]
theta = theta_lst[integer]


EI = -P*L**2/Load_param




E = EI/Izz
ncoords = np.vstack((x, y, np.zeros_like(x))).T
nids = 1 + np.arange(ncoords.shape[0])
nid_pos = dict(zip(nids, np.arange(len(nids))))

n1s = nids[0:-1]
n2s = nids[1:]

num_elements = len(n1s)

print('num_elements', num_elements)

data = BeamCData()
data_spring = SpringData()
KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
KC0r_s = np.zeros(data_spring.KC0_SPARSE_SIZE*num_elements, dtype=INT)
KC0c_s = np.zeros(data_spring.KC0_SPARSE_SIZE*num_elements, dtype=INT)
KC0v_s = np.zeros(data_spring.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
KGc = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
KGr = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
KGv = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)

N = DOF*n
print('num_DOF', N)

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
fint_beam = np.zeros(N)
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
    unit_vect, l = unit_vector(beam, ncoords_flatten)
    xi, xj ,xk = unit_vect[0], unit_vect[1], unit_vect[2]      
    vxyi, vxyj, vxyk =  unit_vect[1], unit_vect[2], unit_vect[0] 
    if xi == xj  and xj == xk: # Edge case, if all are the same then KC0 returns NaN's
        vxyi *= -1
    beam.update_rotation_matrix(vxyi, vxyj, vxyk, ncoords_flatten)
    beam.update_probe_xe(ncoords_flatten)
    beam.update_KC0(KC0r, KC0c, KC0v, prop)
    beam.update_KG(KGr, KGc, KGv, prop)
    beam.update_fint(fint_beam, prop)
    beams.append(beam)
    init_k_KC0 += data.KC0_SPARSE_SIZE
    init_k_KG += data.KG_SPARSE_SIZE
    

init_k_KC0 = 0
init_k_KG = 0
fint_spring = np.zeros(N)
springs = []
for n1, n2 in zip(n1s, n2s):
    p = SpringProbe()
    pos1 = nid_pos[n1]
    pos2 = nid_pos[n2]
    spring = Spring(p)
    spring.init_k_KC0 = init_k_KC0
    spring.n1 = n1
    spring.n2 = n2
    spring.c1 = DOF*pos1
    spring.c2 = DOF*pos2
    unit_vect, l = unit_vector(spring, ncoords_flatten)
    xi, xj ,xk = unit_vect[0], unit_vect[1], unit_vect[2]
    vxyi, vxyj, vxyk =  unit_vect[1], unit_vect[2], unit_vect[0]
    if xi == xj  and xj == xk: # Edge case, if
        vxyi *= -1
    spring.update_rotation_matrix(xi,xj,xk, vxyi, vxyj, vxyk)
    spring.kxe = 100
    spring.update_KC0(KC0r_s, KC0c_s, KC0v_s)
    springs.append(spring)
    init_k_KC0 += data_spring.KC0_SPARSE_SIZE
    spring.update_fint(fint_spring)
    
print('elements created')


fint = fint_beam - fint_spring
print('sparse KC0 created')

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


KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
KC0_s = coo_matrix((KC0v_s, (KC0r_s, KC0c_s)), shape=(N, N)).tocsc()
KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()

KC0uu = KC0[bu, :][:, bu]
KGuu = KG[bu, :][:, bu]
KC0_suu = KC0_s[bu, :][:, bu]
Ktuu = KC0uu
residual = fext - fint
max_iterations = 1
u = np.zeros(N)
u[bu] = np.linalg.solve(Ktuu.toarray(), fext[bu])


print(u)
# du = np.zeros(N)
# xyz = np.zeros(N, dtype=bool)
# xyz[0::DOF] = xyz[1::DOF] = xyz[2::DOF] = True
# fig, ax = plt.subplots()
# for iteration in range(max_iterations+1):
#     print("residual norm", np.linalg.norm(residual[bu]))
#     if np.linalg.norm(residual[bu]) < 1 and iteration > 0:
#         print(f'Converged after {iteration} iterations')
#         break
    
#     du = lsqr(Ktuu, residual[bu])[0]
#     u[bu] += du
#     ncoords_current = ncoords_flatten + u[xyz]

#     fint_beam = np.zeros(N)
#     fint_spring = np.zeros(N)
#     # KC0v *= 0
#     # KGv *= 0
#     for beam,spring in zip(beams, springs):
#         unit_vect, l = unit_vector(beam, ncoords_current)
#         xi, xj ,xk = unit_vect[0], unit_vect[1], unit_vect[2]      
#         vxyi, vxyj, vxyk =  unit_vect[1], unit_vect[2], unit_vect[0] 
#         if xi == xj  and xj == xk: # Edge case, if all are the same then KC0 returns NaN's
#             vxyi *= -1
#         beam.update_rotation_matrix(vxyi, vxyj, vxyk, ncoords_current)
#         beam.update_probe_ue(u)
#         # beam.update_probe_xe(ncoords_current)
#         # beam.update_KC0(KC0r, KC0c, KC0v, prop)
#         # beam.update_KG(KGr, KGc, KGv, prop)
#         beam.update_fint(fint_beam, prop)
#         spring.update_rotation_matrix(xi, xj, xk, vxyi, vxyj, vxyk)
#         spring.update_probe_ue(u)
#         spring.update_KC0(KC0r_s, KC0c_s, KC0v_s)
#         spring.update_fint(fint_spring)
#     fint = fint_beam
#     KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
#     KC0_s = coo_matrix((KC0v_s, (KC0r_s, KC0c_s)), shape=(N, N)).tocsc()
#     KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
#     KC0uu = KC0[bu, :][:, bu]   
#     KGuu = KG[bu, :][:, bu]
#     KC0_suu = KC0_s[bu, :][:, bu]
#     Ktuu = KC0uu 
#     residual = fext - fint
    
#     fig,ax = plot(fig, ax)
    
# fig,ax = plot(fig, ax, c='red')

# ax.scatter(L*(1-uL),-L*(wL), color='red', label='Target Deflection')


# plt.show()
    







# Kuu = KC0[bu, :][:, bu]

# u = np.zeros(N)
# du = np.zeros(N)
# du = lsqr(Kuu, fext[bu])[0]
# u[bu] += du

# xyz = np.zeros(N, dtype=bool)
# xyz[0::DOF] = xyz[1::DOF] = xyz[2::DOF] = True

# ncoords_current = ncoords_flatten + u[xyz]


# fint = np.zeros(N)

# KC0v *= 0





# for beam in beams:
#     beam.update_rotation_matrix(1., 1., 0, ncoords_current)
#     beam.update_KC0(KC0r, KC0c, KC0v, prop)
#     beam.update_probe_ue(u)
#     beam.update_fint(fint, prop)

# fint *= -1
# KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
# Kuu = KC0[bu, :][:, bu]
# residual = fext - fint
# du = np.zeros(N)
# du = lsqr(Kuu, residual[bu])[0]
# u[bu] += du
# print('residual norm', np.linalg.norm(residual[bu]))

# fig,ax = plot(fig, ax)
# ncoords_current = ncoords_flatten + u[xyz]
# print(u)

# fint = np.zeros(N)

# KC0v *= 0
# for beam in beams:
#     beam.update_rotation_matrix(1., 1., 0, ncoords_current)
#     beam.update_KC0(KC0r, KC0c, KC0v, prop)

# for beam in beams:
#     beam.update_probe_ue(u)
#     beam.update_fint(fint, prop)

# fint *= -1

    
# KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
# Kuu = KC0[bu, :][:, bu]
# residual = fext - fint
# du = np.zeros(N)
# du = lsqr(Kuu, residual[bu])[0]
# u[bu] += du
# print('residual norm', np.linalg.norm(residual[bu]))
# print(u)

# ncoords_current = ncoords_flatten + u[xyz]



# fig, ax = plot(fig, ax)

# fint = np.zeros(N)


# KC0v *= 0
# for beam in beams:
#     beam.update_rotation_matrix(1., 1., 0, ncoords_current)
#     beam.update_KC0(KC0r, KC0c, KC0v, prop)


# for beam in beams:
#     beam.update_probe_ue(u)
#     beam.update_fint(fint, prop)

# fint *= -1

    
# KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
# Kuu = KC0[bu, :][:, bu]
# residual = fext - fint
# du = np.zeros(N)
# du = lsqr(Kuu, residual[bu])[0]
# u[bu] += du
# print('residual norm', np.linalg.norm(residual[bu]))
# print(u)

# ncoords_current = ncoords_flatten + u[xyz]



# fig, ax = plot(fig, ax)

# ax.scatter(L*(1-uL),-L*(wL), color='red', label='Target Deflection')

# plt.show()
# # update ue
# # update rotation matrix
# # update fint