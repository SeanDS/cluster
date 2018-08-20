"""Provides a Configuration class representing a set of named points with
coordinates"""

import logging
import numpy as np

from .geometry import Vector, tol_gt, tol_zero, make_hcs, make_hcs_scaled, cs_transform_matrix

LOGGER = logging.getLogger(__name__)

class Configuration:
    """A set of named points with coordinates.

    Immutable. Defines equality and a hash function.
    """

    def __init__(self, mapping):
        """Instantiate a configuration

        :param mapping: dictionary mapping between variables and points, e.g. \
        {v0: p0, v1: p1, v2: p2}, note that points are objects of class \
        :class:`Vector`
        """
        # map of variables to points
        self.mapping = mapping

        # flag indicating an underconstrained merge (i.e. not a unique solution)
        self.underconstrained = False

    def __copy__(self):
        obj = self.__class__(self.mapping)

        # copy constrainedness
        obj.underconstrained = self.underconstrained

        return obj

    @property
    def variables(self):
        """return list of variables"""
        return self.mapping.keys()

    def position(self, var):
        """return position of point var"""
        return self.mapping[var]

    def transform(self, t):
        """returns a new configuration, which is this one transformed by matrix t"""
        new_map = {}

        for v in self.mapping:
            # transform in projective space
            new_map[v] = Vector.from_projective(np.dot(t, self.mapping[v].to_projective()))

        return Configuration(new_map)

    def add(self, c):
        """return a new configuration which is this configuration extended with all points in c not in this configuration"""
        new_map = {}

        for v in self.mapping:
            new_map[v] = self.mapping[v]

        for v in c.mapping:
            if v not in new_map:
                new_map[v] = c.mapping[v]

        return Configuration(new_map)

    def select(self, vars):
        """return a new configuration that is a subconfiguration of this configuration, containing only the selected variables"""
        new_map = {}

        for v in vars:
            new_map[v] = self.mapping[v]

        return Configuration(new_map)

    def merge(self, other):
        """returns a new configurations which is this one plus the given other configuration transformed, such that common points will overlap (if possible)."""
        LOGGER.debug(f"merging '{self}' with '{other}'")

        t, underconstrained = self.merge_transform(other)
        result = self.add(other.transform(t))

        return result, underconstrained

    def merge_transform(self, other):
        """returns a new configurations which is this one plus the given other configuration transformed, such that common points will overlap (if possible)."""
        shared = set(self.variables) & other.variables

        underconstrained = self.underconstrained or other.underconstrained

        if len(shared) == 0:
            underconstrained = True
            cs1 = make_hcs(Vector.origin(), Vector([1.0, 0.0]))
            cs2 = make_hcs(Vector.origin(), Vector([1.0, 0.0]))
        elif len(shared) == 1:
            if len(self.variables) > 1 and len(other.variables) > 1:
                underconstrained = True

            v1 = list(shared)[0]
            p11 = self.mapping[v1]
            p21 = other.mapping[v1]
            cs1 = make_hcs(p11, p11 + Vector([1.0, 0.0]))
            cs2 = make_hcs(p21, p21 + Vector([1.0, 0.0]))
        else:   # len(shared) >= 2:
            v1 = list(shared)[0]
            v2 = list(shared)[1]
            p11 = self.mapping[v1]
            p12 = self.mapping[v2]

            if tol_zero(p11.distance_to(p12)):
                underconstrained = True
                cs1 = make_hcs(p11, p11 + Vector([1.0, 0.0]))
            else:
                cs1 = make_hcs(p11, p12)

            p21 = other.mapping[v1]
            p22 = other.mapping[v2]

            if tol_zero(p21.distance_to(p22)):
                underconstrained = True
                cs2 = make_hcs(p21, p21 + Vector([1.0, 0.0]))
            else:
                cs2 = make_hcs(p21, p22)

        t = cs_transform_matrix(cs2, cs1)

        return t, underconstrained

    def merge_scale(self, other, shared=None):
        """returns a new configurations which is this one plus the given other configuration transformed, such that common points will overlap (if possible)."""
        if shared is None:
            shared = set(self.variables) & other.variables

        underconstrained = self.underconstrained or other.underconstrained

        if len(shared) < 2:
            raise ValueError("must have >=2 shared point vars")

        v1 = list(shared)[0]
        v2 = list(shared)[1]
        p11 = self.mapping[v1]
        p12 = self.mapping[v2]

        if tol_zero(p11.distance_to(p12)):
            underconstrained = True
            cs1 = make_hcs_scaled(p11, p11 + Vector([1.0, 0.0]))
        else:
            cs1 = make_hcs_scaled(p11, p12)

        p21 = other.mapping[v1]
        p22 = other.mapping[v2]

        if tol_zero(p21.distance_to(p22)):
            underconstrained = True
            cs2 = make_hcs_scaled(p21, p21 + Vector([1.0, 0.0]))
        else:
            cs2 = make_hcs_scaled(p21, p22)

        t = cs_transform_matrix(cs2, cs1)
        othert = other.transform(t)
        result = self.add(othert)

        return result, underconstrained

    def __eq__(self, other):
        """two configurations are equal if they map onto eachother modulo rotation and translation"""
        if hash(self) != hash(other):
            return False

        if len(self.mapping) != len(other.mapping):
            return False

        if not isinstance(other, Configuration):
            return False

        for var in self.mapping:
            if var not in other.mapping:
                return False

        # determine a rotation-translation transformation
        # to transform other onto self
        t, _ = self.merge_transform(other)
        othertransformed = other.transform(t)

        # test if point map onto eachother (distance metric tolerance)
        for var in self.mapping:
            d = othertransformed.position(var).distance_to(self.position(var))
            # check that d is greater than 0 within tolerance
            if not tol_gt(d, 0):
                return False

        return True

    def __hash__(self):
        # hash the configuration's points
        return hash(frozenset(self.mapping))

    def __str__(self):
        return "Configuration({0})".format(self.mapping)


def test():
    p1 = np.array([0.0,0.0,0.0])
    p2 = np.array([1.0,0.0,0.0])
    p3 = np.array([0.0,1.0,0.0])
    c1 = Configuration({1:p1,2:p2})
    q1 = np.array([0.0,0.0,0.0])
    q2 = np.array([1.0,0.0,0.0])
    q3 = np.array([0.0,-1.0,0.0])
    c2 = Configuration({1:q1,2:q2})
    print((c1 == c2))

if __name__ == "__main__": test()
