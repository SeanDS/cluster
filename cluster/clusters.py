"""Clusters are generalised constraints on sets of points in
:math:`\mathbb{R}^2`. Cluster types are :class:`Rigid`, :class:`Hedgehog` and
:class:`Balloon`."""

import abc
import logging
import itertools

from .methods import Variable

LOGGER = logging.getLogger(__name__)

class PointRelation(metaclass=abc.ABCMeta):
    """Represents a relation between a set of points"""

    def __init__(self, name, points):
        """Creates a new point relation

        :param name: name of relation
        :param points: list of points
        :type name: unicode
        :type points: list
        """
        self.name = name
        self.points = points

    def __str__(self):
        # comma separated points
        points = ", ".join([str(point) for point in self.points])

        return "{0}({1})".format(self.name, points)


class Distance(PointRelation):
    """Represents a known distance between two points"""

    def __init__(self, a, b):
        """Creates a new known distance

        The distance is defined between points *a* and *b*

        :param a: first point
        :param b: second point
        :type a: :class:`~.np.ndarray`
        :type b: :class:`~.np.ndarray`
        """

        # call parent constructor
        super().__init__("dist", [a, b])

    def __eq__(self, other):
        # check other object is a Distance
        if isinstance(other, Distance):
            # check points are the same
            return frozenset(self.points) == frozenset(other.points)

        return False

    def __hash__(self):
        return hash(frozenset(self.points))


class Angle(PointRelation):
    """Represents a known angle between three points"""

    def __init__(self, a, b, c):
        """Creates a new known angle

        The angle is defined at the *b* edge between *a* and *c*.

        :param a: first point
        :param b: second point
        :param c: third point
        :type a: :class:`~.np.ndarray`
        :type b: :class:`~.np.ndarray`
        :type c: :class:`~.np.ndarray`
        """

        # call parent constructor
        super().__init__("ang", [a, b, c])

    def __eq__(self, other):
        # check other object is an Angle
        if isinstance(other, Angle):
            # check the middle point is identical, and that the other points are
            # the included
            return self.points[2] == other.points[2] and frozenset(self.points) == frozenset(other.points)

        return False

    def __hash__(self):
        return hash(frozenset(self.points))


class Cluster(Variable, metaclass=abc.ABCMeta):
    """A set of points, satisfying some constaint"""

    NAME = "Cluster"

    def __init__(self, variables, *args, **kwargs):
        """Create a new cluster

        Specified variables should be hashable.

        :param variables: cluster variables
        :type variables: list
        """
        # call parent constructor
        super().__init__(*args, **kwargs)

        # set variables
        self.variables = set(variables)

        # set default overconstrained value
        self.overconstrained = False

    def __str__(self):
        # create character string to represent whether the cluster is
        # overconstrained
        if self.overconstrained:
            overconstrained = "!"
        else:
            overconstrained = ""

        # get parent string
        parent_str = super().__str__()

        return f"{overconstrained}{parent_str}({self._variable_str()})"

    def _variable_str(self):
        return ", ".join(self.variables)

    def intersection(self, other):
        """Get the intersection between this cluster and the specified cluster

        :param other: other cluster
        :type other: :class:`Cluster`
        :returns: new cluster with intersection of input clusters' variables
        :rtype: :class:`Cluster`
        :raises TypeError: if cluster types are unknown
        """
        # get shared points between this cluster and the other cluster
        shared_points = self.variables & other.variables

        # note, a one point cluster is never returned
        # because it is not a constraint
        if len(shared_points) < 2:
            return None

        return self._intersect_with(other, shared_points)

    @abc.abstractmethod
    def _intersect_with(self, other, shared_points):
        raise NotImplementedError

    def over_constraints(self, other):
        """returns the over-constraints (duplicate distances and angles) for
        a pair of clusters (rigid, angle or scalable)."""
        return self._over_distances(other) | self._over_angles(other)

    def _over_distances(self, other):
        """Default cluster overdistances"""
        return set()

    @abc.abstractmethod
    def _over_angles(self, other):
        raise NotImplementedError

    def __copy__(self):
        return self.__class__(self.variables)


class Rigid(Cluster):
    """Represents a set of points that form a rigid body"""

    NAME = "Rigid"

    def _intersect_with(self, other, shared_points):
        if isinstance(other, self.__class__):
            if len(shared_points) >= 2:
                return self.__class__(shared_points)
        elif isinstance(other, Balloon):
            if len(shared_points) >= 3:
                return Balloon(shared_points)
        elif isinstance(other, Hedgehog):
            xvars = shared_points - set([other.cvar])

            if other.cvar in self.variables and len(xvars) >= 2:
                return Hedgehog(other.cvar, xvars)
        else:
            raise ValueError(f"unrecognised cluster '{other}'")

        # default
        return None

    def _over_distances(self, other):
        if not isinstance(other, self.__class__):
            return set()

        # other is also rigid
        shared = self.variables & other.variables

        return set([Distance(v1, v2) for v1, v2
                    in itertools.combinations(shared, 2)])

    def _over_angles(self, other):
        overangles = set()

        if isinstance(other, (self.__class__, Balloon)):
            # determine duplicate angles
            shared = self.variables & other.variables

            # generate combinations of three from shared variables
            for v1, v2, v3 in itertools.combinations(shared, 3):
                # add clockwise angles
                overangles.add(Angle(v1, v2, v3))
                overangles.add(Angle(v2, v3, v1))
                overangles.add(Angle(v3, v1, v2))
        elif isinstance(other, Hedgehog):
            # determine duplicate angles
            shared = self.variables & other.xvars

            if other.cvar in self.variables:
                for v1, v2 in itertools.combinations(shared, 2):
                    overangles.add(Angle(v1, other.cvar, v2))

        return overangles

class Hedgehog(Cluster):
    """Represents a set of points (C, X1...XN) where all angles a(Xi, C, Xj) are
    known"""

    NAME = "Hedgehog"

    def __init__(self, cvar, xvars):
        """Creates a new hedgehog cluster

        :param cvar: center variable
        :param xvars: other variables
        :type cvar: object
        :type xvars: list
        :raises ValueError: if less than three variables are specified between \
        *cvar* and *xvars*
        """
        # set central variable
        self.cvar = cvar

        # check there are enough other variables
        if len(xvars) < 2:
            raise ValueError("Hedgehog must have at least three variables")

        # set other variables
        self.xvars = set(xvars)

        # call parent constructor with all variables
        super().__init__(self.xvars.union([self.cvar]))

    def _variable_str(self):
        extra = ", ".join(self.xvars)
        return f"{self.cvar}, [{extra}]"

    def _intersect_with(self, other, shared_points):
        if isinstance(other, self.__class__):
            xvars = self.xvars & other.xvars

            if self.cvar == other.cvar and len(xvars) >= 2:
                return self.__class__(self.cvar,xvars)
        elif isinstance(other, Rigid):
            return other._intersect_with(self)
        elif isinstance(other, Balloon):
            xvars = shared_points - set([self.cvar])

            if self.cvar in other.variables and len(xvars) >= 2:
                return Hedgehog(self.cvar,xvars)
        else:
            raise ValueError(f"unrecognised cluster '{other}'")

        # default
        return None

    def __copy__(self):
        return self.__class__(self.cvar, self.xvars)

    def _over_angles(self, other):
        overangles = set()

        if isinstance(other, self.__class__):
            # determine duplicate angles
            shared = self.xvars & other.xvars

            if self.cvar == other.cvar:
                for v1 in shared:
                    for v2 in shared:
                        overangles.add(Angle(v1, self.cvar, v2))
        elif isinstance(other, Rigid):
            return other._over_angles(self)
        elif isinstance(other, Balloon):
            # determine duplicate angles
            shared = self.xvars & other.variables

            if self.cvar in other.variables:
                for v1, v2 in itertools.combinations(shared, 2):
                    overangles.add(Angle(v1, self.cvar, v2))
        else:
            raise ValueError(f"unrecognised cluster '{other}'")

        return overangles

class Balloon(Cluster):
    """Represents a set of points that is invariant to rotation, translation and
    scaling"""

    NAME = "Balloon"

    def __init__(self, *args, **kwargs):
        """Create a new balloon

        :raises ValueError: if less than three variables are specified
        """

        # call parent
        super().__init__(*args, **kwargs)

        # check there are enough variables for a balloon
        if len(self.variables) < 3:
            raise ValueError("Balloon must have at least three variables")

    def _intersect_with(self, other, shared_points):
        if isinstance(other, self.__class__):
            if len(shared_points) >= 3:
                return self.__class__(shared_points)
        elif isinstance(other, (Rigid, Hedgehog)):
            return other._intersect_with(self)
        else:
            raise ValueError(f"unrecognised cluster '{other}'")

        # default
        return None

    def _over_angles(self, other):
        overangles = set()

        if isinstance(other, self.__class__):
            # determine duplicate angles
            shared = self.variables & other.variables

            # generate combinations of three from shared variables
            for v1, v2, v3 in itertools.combinations(shared, 3):
                # add clockwise angles
                overangles.add(Angle(v1, v2, v3))
                overangles.add(Angle(v2, v3, v1))
                overangles.add(Angle(v3, v1, v2))
        elif isinstance(other, (Rigid, Hedgehog)):
            return other._over_angles(self)
        else:
            raise ValueError(f"unrecognised cluster '{other}'")

        return overangles
