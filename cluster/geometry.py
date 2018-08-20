"""Geometry classes for representing vectors and performing geometric
transformations on sets of vectors"""

import numpy as np
import numpy.linalg as la
import logging

LOGGER = logging.getLogger(__name__)

def tol_eq(a, b):
    return np.allclose(a, b)

def tol_gt(a, b):
    return a > b and not tol_eq(a, b)

def tol_lt(a, b):
    return a < b and not tol_eq(a, b)

def tol_ge(a, b):
    return a > b or tol_eq(a, b)

def tol_le(a, b):
    return a < b or tol_eq(a, b)

def tol_zero(a):
    return tol_eq(a, np.zeros_like(a))

class Vector(np.ndarray):
    """Two-dimensional column vector in Cartesian coordinates"""

    def __new__(cls, *args, **kwargs):
        obj = np.array(*args, **kwargs)

        # return view as Vector
        return obj.view(cls)

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, x):
        self[0] = float(x)

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, y):
        self[1] = float(y)

    def __str__(self):
        return f"({self.x:.3f}, {self.y:.3f})"

    @classmethod
    def origin(cls):
        """The coordinates of the origin"""

        return cls([0, 0])

    def unit(self):
        """Unit vector"""
        return self / self.length

    @property
    def length(self):
        """Length between point defined by coordinates and the origin"""
        return np.sqrt(self.x * self.x + self.y * self.y)

    def distance_to(self, other):
        return (other - self).length

    def angle_between(self, a, c):
        """Angle of the triangle formed by ABC where B is this vector

        The angle is signed and in the range [-pi, pi] corresponding to a clockwise rotation.
        If a - b - c is clockwise, then angle > 0.

        Raises
        ------
        DegenerateError
            If the angle is degenerate, i.e. the triangle has zero area.
        """
        # distances between points
        ab = self - a
        cb = self - c

        if ab.tol_zero() or cb.tol_zero():
            # degenerate angle
            raise DegenerateError("triangle is degenerate")

        # vectors between points
        vab = ab.unit()
        vcb = cb.unit()

        # calculate vector rotating vab to vcb
        rotation = np.dot(vab, vcb)

        # clip to +/-1.0 to fix floating point errors
        if rotation > 1.0:
            rotation = 1.0
        elif rotation < -1.0:
            rotation = -1.0

        # calculate angle from rotation vector
        angle = np.arccos(rotation)

        if self.is_counterclockwise(a, self, c):
            # flip angle
            angle = -angle

        return angle

    @classmethod
    def is_clockwise(cls, a, b, c):
        """Calculates whether or not triangle ABC is orientated clockwise

        Returns
        -------
        :class:`bool`
            True if clockwise, False otherwise
        """
        u = b - a
        v = c - b

        # vector perpendicular to u
        perp_u = cls([-u.y, u.x])

        # check that a < 0 within tolerance
        return tol_lt(np.dot(perp_u, v), 0)

    @classmethod
    def is_counterclockwise(cls, a, b, c):
        """Calculates whether or not triangle ABC is orientated counter-clockwise

        Returns
        -------
        :class:`bool`
            True if counter-clockwise, False otherwise
        """
        u = b - a
        v = c - b

        # vector perpendicular to u
        perp_u = cls([-u.y, u.x])

        # check that a > 0 within tolerance
        return tol_gt(np.dot(perp_u, v), 0)

    def tol_eq(self, other):
        return tol_eq(self.x, other.x) and tol_eq(self.y, other.y)

    def tol_gt(self, other):
        return self > other and not self.tol_eq(other)

    def tol_lt(self, other):
        return self < other and not self.tol_eq(other)

    def tol_ge(self, other):
        return self > other or self.tol_eq(other)

    def tol_le(self, other):
        return self < other or self.tol_eq(other)

    def tol_zero(self):
        return tol_zero(self)

def cc_int(p1, r1, p2, r2):
    """Intersect circle (p1, r1) with circle (p2, r2)

    :param p1: vector representing centre of first circle
    :type p1: :class:`Vector`
    :param r1: scalar representing the radius of first circle
    :type r1: float
    :param p2: vector representing centre of second circle
    :type p2: :class:`Vector`
    :param r2: scalar representing the radius of first circle
    :type r2: float
    :returns: list of zero, one or two solution points
    :rtype: list
    """
    LOGGER.debug(f"intersecting circles ({p1}, r={r1}) and ({p2}, r={r2})")

    # distance between circle centres
    d = (p2-p1).length

    # check if d <= 0 within tolerance
    if tol_le(d, 0):
        # no solutions
        return

    u = ((r1*r1 - r2*r2) / d + d) / 2

    a = r1*r1
    b = u*u

    # check that a < b within tolerance
    # FIXME: what's going on here?  elif block seems to repeat earlier check
    if tol_lt(a, b):
        return
    elif a < b:
        v = 0.0
    else:
        v = np.sqrt(a-b)

    s = (p2 - p1) * u / d

    if tol_zero(s.length):
        yield p1 + Vector([p2.y - p1.y, p1.x - p2.x]) * r1 / d

        if not tol_zero(r1 / d):
            # second solution
            yield p1 + Vector([p1.y - p2.y, p2.x - p1.x]) * r1/d
    else:
        yield p1 + s + Vector([s.y, -s.x]) * v / s.length

        if not tol_zero(v / s.length):
            # second solution
            yield p1 + s + Vector([-s.y, s.x]) * v / s.length

def cl_int(p1, r, p2, v):
    """Intersect circle (p1, r) with line (p2, v)

    :param p1: vector representing centre of circle
    :type p1: :class:`Vector`
    :param r: scalar representing the radius of circle
    :type r1: float
    :param p2: vector representing a point on the line
    :type p2: :class:`Vector`
    :param v: vector representing the direction of the line
    :type v: :class:`Vector`
    :returns: list of zero, one or two solution points
    :rtype: list
    """
    LOGGER.debug(f"intersecting circle ({p1}, r={r}) with line ({p2}, dir={v})")

    # vector between centre of circle and start of line
    p = p2 - p1

    # squared length of line
    d2 = v.x * v.x + v.y * v.y

    D = p.x * v.y - v.x * p.y
    E = r * r * d2 - D * D

    # check that d2 and E are both > 0 within tolerance
    if tol_gt(d2, 0) and tol_gt(E, 0):
        sE = np.sqrt(E)

        x1 = p1.x + (D * v.y + np.sign(v.y) * v.x * sE) / d2
        y1 = p1.y + (-D * v.x + np.abs(v.y) * sE) / d2

        yield Vector([x1, y1])

        x2 = p1.x + (D * v.y - np.sign(v.y) * v.x * sE) / d2
        y2 = p1.y + (-D * v.x - np.abs(v.y) * sE) / d2

        yield Vector([x2, y2])
    elif tol_zero(E):
        x1 = p1.x + D * v.y / d2
        y1 = p1.y + -D * v.x / d2

        yield Vector([x1, y1])

def cr_int(p1, r, p2, v):
    """Intersect a circle (p1, r) with ray (p2, v) (a half-line)

    :param p1: vector representing centre of circle
    :type p1: :class:`Vector`
    :param r: scalar representing the radius of circle
    :type r1: float
    :param p2: vector representing a point on the ray
    :type p2: :class:`Vector`
    :param v: vector representing the direction of the ray
    :type v: :class:`Vector`
    :returns: list of zero, one or two solution points
    :rtype: list
    """
    LOGGER.debug(f"intersecting circle ({p1}, r={r}) with ray ({p2}, dir={v})")

    # loop over solutions of the circle and line intercept
    for s in cl_int(p1, r, p2, v):
        # check if a is >= 0 within tolerance
        if tol_ge(np.dot(s - p2, v), 0):
            yield s

def ll_int(p1, v1, p2, v2):
    """Intersect two lines

    :param p1: vector representing a point on the first line
    :type p1: :class:`Vector`
    :param v1: vector represting the direction of the first line
    :type v1: :class:`Vector`
    :param p2: vector representing a point on the second line
    :type p2: :class:`Vector`
    :param v2: vector represting the direction of the second line
    :type v2: :class:`Vector`
    :returns: list of zero or one solution points
    :rtype: list
    """
    LOGGER.debug(f"intersecting lines ({p1}, dir={v1}) and ({p2}, dir={v2})")

    if tol_zero(v1.x * v2.y - v1.y * v2.x):
        # lines don't intersect
        return

    d = p2 - p1

    if tol_zero(v2.y):
        t1 = d.y / v1.y
    else:
        r2 = -v2.x / v2.y
        f = v1.x + v1.y * r2
        t1 = (d.x + d.y * r2) / f

    yield p1 + v1 * t1

def lr_int(p1, v1, p2, v2):
    """Intersect line with ray

    :param p1: vector representing a point on the line
    :type p1: :class:`Vector`
    :param v1: vector represting the direction of the line
    :type v1: :class:`Vector`
    :param p2: vector representing a point on the ray
    :type p2: :class:`Vector`
    :param v2: vector represting the direction of the ray
    :type v2: :class:`Vector`
    :returns: list of zero or one solution points
    :rtype: list
    """
    LOGGER.debug(f"intersecting line ({p1}, dir={v1}) with ray ({p2}, dir={v2})")

    # assume ray is a line and get intersection with line
    for s in ll_int(p1, v1, p2, v2):
        # check if s > 0 and a >= 0 within tolerance
        if tol_ge(np.dot(s - p2, v2), 0):
            yield s

def rr_int(p1, v1, p2, v2):
    """Intersect ray with ray

    :param p1: vector representing a point on the first ray
    :type p1: :class:`Vector`
    :param v1: vector represting the direction of the first ray
    :type v1: :class:`Vector`
    :param p2: vector representing a point on the second ray
    :type p2: :class:`Vector`
    :param v2: vector represting the direction of the second ray
    :type v2: :class:`Vector`
    :returns: list of zero or one solution points
    :rtype: list
    """
    LOGGER.debug(f"intersecting rays ({p1}, dir={v1}) and ({p2}, dir={v2})")

    # assume rays are lines and get intersection
    for s in ll_int(p1, v1, p2, v2):
        a1 = np.dot(s - p1, v1)
        a2 = np.dot(s - p2, v2)

        # check len(s) > 0 and a1 >= 0 and a2 >= 0 within tolerance
        if tol_ge(a1, 0) and tol_ge(a2, 0):
            yield s

def is_clockwise(p1, p2, p3):
    """Calculates whether or not triangle p1, p2, p3 is orientated clockwise

    :param p1: first point
    :type p1: :class:`Vector`
    :param p2: second point
    :type p2: :class:`Vector`
    :param p3: third point
    :type p3: :class:`Vector`
    :returns: True if clockwise, otherwise False
    :rtype: boolean
    """

    u = p2 - p1
    v = p3 - p2
    perp_u = Vector([-u.y, u.x])

    # check a < 0 within tolerance
    return tol_lt(np.dot(perp_u, v), 0)

def is_counterclockwise(p1, p2, p3):
    """Calculates whether or not triangle p1, p2, p3 is orientated \
    counter-clockwise

    :param p1: first point
    :type p1: :class:`Vector`
    :param p2: second point
    :type p2: :class:`Vector`
    :param p3: third point
    :type p3: :class:`Vector`
    :returns: True if counter-clockwise, otherwise False
    :rtype: boolean
    """

    u = p2 - p1
    v = p3 - p2
    perp_u = Vector([-u.y, u.x])

    # check that a > 0 within tolerance
    return tol_gt(np.dot(perp_u, v), 0)

def is_flat(p1, p2, p3):
    """Calculates wheter or not triangle p1, p2, p3 is flat (neither \
    clockwise nor counter-clockwise)

    :param p1: first point
    :type p1: :class:`Vector`
    :param p2: second point
    :type p2: :class:`Vector`
    :param p3: third point
    :type p3: :class:`Vector`
    :returns: True if flat, otherwise False
    :rtype: boolean
    """

    u = p2 - p1
    v = p3 - p2
    perp_u = Vector([-u.y, u.x])

    return tol_zero(np.dot(perp_u, v))

def is_acute(p1, p2, p3):
    """Calculates whether or not angle p1, p2, p3 is acute, i.e. less than \
    pi / 2

    :param p1: first point
    :type p1: :class:`Vector`
    :param p2: second point
    :type p2: :class:`Vector`
    :param p3: third point
    :type p3: :class:`Vector`
    :returns: True if acute, otherwise False
    :rtype: boolean
    """

    # calculate angle between points
    angle = p2.angle_between(p1, p3)

    if angle is None:
        return False

    return tol_lt(np.abs(angle), np.pi / 2)

def is_obtuse(p1,p2,p3):
    """Calculates whether or not angle p1, p2, p3 is obtuse, i.e. greater than \
    pi / 2

    :param p1: first point
    :type p1: :class:`Vector`
    :param p2: second point
    :type p2: :class:`Vector`
    :param p3: third point
    :type p3: :class:`Vector`
    :returns: True if obtuse, otherwise False
    :rtype: boolean
    """

    # calculate angle between points
    angle = p2.angle_between(p1, p3)

    if angle is None:
        return False

    return tol_gt(np.abs(angle), np.pi / 2)

def make_hcs(a, b, scale=False):
    """Build a homogeneous coordiate system from two vectors, normalised

    :param a: first vector
    :type a: :class:`Vector`
    :param b: second vector
    :type b: :class:`Vector`
    :returns: 3x3 homogeneous coordinate matrix
    :rtype: :class:`Vector`
    """

    # resultant vector
    u = b - a

    if tol_zero(u.length):
        # vectors are on top of each other (within tolerance)
        return None

    if not scale:
        # normalise resultant
        u = u / u.length

    # mirror of u
    v = Vector([-u.y, u.x])

    # return new coordinate system
    return np.array([[u.x, v.x, a.x],
                     [u.y, v.y, a.y],
                     [0.0, 0.0, 1.0]])

def make_hcs_scaled(*args, **kwargs):
    """Build a homogeneous coordiate system from two vectors

    :param a: first vector
    :type a: :class:`Vector`
    :param b: second vector
    :type b: :class:`Vector`
    :returns: 3x3 homogeneous coordinate matrix
    :rtype: :class:`Vector`
    """

    return make_hcs(scale=True, *args, **kwargs)

def cs_transform_matrix(from_cs, to_cs):
    """Calculate the transformation matrix from one coordinate system to another

    :param from_cs: initial coordinate system
    :type from_cs: :class:`Vector`
    :param to_cs: target coordinate system
    :type to_cs: :class:`Vector`
    :returns: transformation matrix to convert from_cs to to_cs
    :rtype: :class:`Vector`
    """

    return np.dot(to_cs, la.inv(from_cs))


class DegenerateError(ValueError):
    pass




# -------------------------test code -----------------

def test_ll_int():
    """test random line-line intersection. returns True iff succesful"""
    # generate three points A,B,C an two lines AC, BC.
    # then calculate the intersection of the two lines
    # and check that it equals C
    p_a = vector.randvec(2, 0.0, 10.0, 1.0)
    p_b = vector.randvec(2, 0.0, 10.0, 1.0)
    p_c = vector.randvec(2, 0.0, 10.0, 1.0)
    # print p_a, p_b, p_c
    if np.allclose(la.norm(p_c - p_a),0) or np.allclose(la.norm(p_c - p_b),0):
        return True # ignore this case
    v_ac = (p_c - p_a) / la.norm(p_c - p_a)
    v_bc = (p_c - p_b) / la.norm(p_c - p_b)
    s = ll_int(p_a, v_ac, p_b, v_bc)
    if np.allclose(np.absolute(np.dot(v_ac, v_bc)),1.0):
        return len(s) == 0
    else:
        if len(s) > 0:
            p_s = s[0]
            return np.allclose(p_s[0],p_c[0]) and np.allclose(p_s[1],p_c[1])
        else:
            return False

def test_rr_int():
    """test random ray-ray intersection. returns True iff succesful"""
    # generate tree points A,B,C an two rays AC, BC.
    # then calculate the intersection of the two rays
    # and check that it equals C
    p_a = vector.randvec(2, 0.0, 10.0,1.0)
    p_b = vector.randvec(2, 0.0, 10.0,1.0)
    p_c = vector.randvec(2, 0.0, 10.0,1.0)
    # print p_a, p_b, p_c
    if np.allclose(la.norm(p_c - p_a),0) or np.allclose(la.norm(p_c - p_b),0):
        return True # ignore this case
    v_ac = (p_c - p_a) / la.norm(p_c - p_a)
    v_bc = (p_c - p_b) / la.norm(p_c - p_b)
    s = rr_int(p_a, v_ac, p_b, v_bc)
    if np.allclose(np.absolute(np.dot(v_ac, v_bc)),1.0):
        return len(s) == 0
    else:
        if len(s) > 0:
            p_s = s[0]
            return np.allclose(p_s[0],p_c[0]) and np.allclose(p_s[1],p_c[1])
        else:
            return False

def test1():
    sat = True
    for i in range(0,100):
        sat = sat and test_ll_int()
        if not sat:
            print("ll_int() failed")
            return
    if sat:
        print("ll_int() passed")
    else:
        print("ll_int() failed")

    sat = True
    for i in range(0,100):
        sat = sat and test_rr_int()
        if not sat:
            print("rr_int() failed")
            return

    if sat:
        print("rr_int() passed")
    else:
        print("rr_int() failed")

    print("2D angles")
    for i in range(9):
        a = i * 45 * np.pi / 180
        p1 = np.array([1.0,0.0])
        p2 = np.array([0.0,0.0])
        p3 = np.array([np.cos(a),np.sin(a)])
        print((p3, angle_3p(p1,p2,p3) * 180 / np.pi, "flip", angle_3p(p3,p2,p1) * 180 / np.pi))

if __name__ == '__main__': test1()
