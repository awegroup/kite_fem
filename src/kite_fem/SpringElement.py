import numpy as np
from pyfe3d import Spring, SpringProbe, DOF


class SpringElement:
    def __init__(self, n1: int, n2: int, init_k_KC0: int):
        # initialising pyfe3d spring element
        springprobe = SpringProbe()
        self.spring = Spring(springprobe)
        self.spring.init_k_KC0 = init_k_KC0
        self.spring.n1 = n1
        self.spring.n2 = n2
        self.spring.c1 = DOF * n1
        self.spring.c2 = DOF * n2
        self.update_KC0v_only = 0
        self.slack = False

    def set_spring_properties(
        self, l0: float, k: float, springtype: str, i_other_pulley: int = 0
    ):
        # setting spring properties
        self.l0 = l0
        self.k = k
        self.spring.kxe = k
        self.springtype = springtype.lower()

        # track indice of matching spring element in pulley system
        if self.springtype == "pulley":
            self.i_other_pulley = i_other_pulley

        # error if invalid spring type
        if self.springtype not in ("noncompressive", "default", "pulley"):
            raise ValueError(
                "Invalid spring type. Choose from 'noncompressive', 'default', or 'pulley'."
            )

    def unit_vector(self, coords: np.ndarray):
        # calculate unit vector and length of the element
        xi = coords[self.spring.c2 // 2 + 0] - coords[self.spring.c1 // 2 + 0]
        xj = coords[self.spring.c2 // 2 + 1] - coords[self.spring.c1 // 2 + 1]
        xk = coords[self.spring.c2 // 2 + 2] - coords[self.spring.c1 // 2 + 2]
        l = np.linalg.norm([xi, xj, xk])
        unit_vect = np.array([xi, xj, xk]) / l
        return unit_vect, l

    def __update_rotation_matrix(self, coords: np.ndarray):
        # determine arbitrary vector on plane xy to describe coordinate system along with vector x
        unit_vect = self.unit_vector(coords)[0]
        xi, xj, xk = unit_vect[0], unit_vect[1], unit_vect[2]
        vxyi, vxyj, vxyk = unit_vect[1], unit_vect[2], unit_vect[0]
        if (
            xi == xj and xj == xk
        ):  # Edge case, if all are the same then KC0 returns NaN's
            vxyi *= -1
        # update element rotation matrix in pyfe3d
        self.spring.update_rotation_matrix(xi, xj, xk, vxyi, vxyj, vxyk)

    def update_KC0(
        self, KC0r: np.ndarray, KC0c: np.ndarray, KC0v: np.ndarray, coords: np.ndarray
    ):
        # update element rotation matrix and adds contribution to global stiffness matrix
        self.__update_rotation_matrix(coords)
        if not self.slack:
            self.spring.update_KC0(KC0r, KC0c, KC0v, self.update_KC0v_only)
        # set flag to only update KC0v from now on
        self.update_KC0v_only = 1
        return KC0r, KC0c, KC0v

    def spring_internal_forces(self, coords: np.ndarray, l_other_pulley: float = 0.0):
        # Set spring stiffness
        k_fi = self.k
        self.spring.kxe = self.k
        self.slack = False
        # Set noncompressive and pulley spring stiffness to zero if compressed
        unit_vector, l = self.unit_vector(coords)
        if self.springtype == "noncompressive" or self.springtype == "pulley":
            if l + l_other_pulley < (self.l0):
                self.spring.kxe = 0 * self.k
                k_fi = 0 * self.k
                self.slack = True
        # calculate spring force and allign with unit vector
        f_s = k_fi * (l + l_other_pulley - self.l0)
        fi = f_s * unit_vector

        # append with zeros for rotational DOF's
        fi = np.append(fi, [0, 0, 0])
        return fi
