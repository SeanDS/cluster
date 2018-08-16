"""Geometric objects"""

import numpy as np
from numpy.linalg import norm

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

def degrees(angle):
    return np.degrees(angle)

def radians(angle):
    return np.radians(angle)

class Vector(np.ndarray):
    """2D vector"""

    def __new__(cls, *args, **kwargs):
        obj = np.array(*args, **kwargs)

        # return view as Vector
        return obj.view(cls)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @classmethod
    def origin(cls):
        """Origin vector"""
        return cls((0, 0))

    @property
    def length(self):
        return norm(self)

    def unit(self):
        """Unit vector"""
        return self / self.length

    def tol_eq(self, other):
        return tol_eq(self, other)

    def tol_gt(self, other):
        return tol_gt(self, other)

    def tol_lt(self, other):
        return tol_lt(self, other)

    def tol_ge(self, other):
        return tol_ge(self, other)

    def tol_le(self, other):
        return tol_le(self, other)

    def tol_zero(self):
        return tol_zero(self)

    def distance_to(self, other):
        return norm(other - self)

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
        rotation = vab.dot(vcb)

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


class DegenerateError(ValueError):
    pass
