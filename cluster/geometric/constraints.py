import logging
import numpy as np

from ..constraint import ParametricConstraint
from ..geometry import tol_eq

LOGGER = logging.getLogger(__name__)

class FixConstraint(ParametricConstraint):
    """A constraint to fix a point relative to the coordinate system"""

    def __init__(self, var, pos):
        """Create a new DistanceConstraint instance

           keyword args:
            var    - a point variable name
            pos    - the position parameter
        """
        super().__init__()

        self._variables = [var]
        self.set_parameter(pos)

    def satisfied(self, mapping):
        """return True iff mapping from variable names to points satisfies constraint"""

        a = mapping[self._variables[0]]

        result = a.tol_eq(self._value)

        return result

    def __str__(self):
        return "FixConstraint({0}, {1})".format(self._variables[0], self._value)

class DistanceConstraint(ParametricConstraint):
    """A constraint on the Euclidean distance between two points"""

    def __init__(self, a, b, dist):
        """Create a new DistanceConstraint instance

           keyword args:
            a    - a point variable name
            b    - a point variable name
            dist - the distance parameter value
        """
        super().__init__()

        self._variables = [a, b]
        self.set_parameter(dist)

    def satisfied(self, mapping):
        """return True iff mapping from variable names to points satisfies constraint"""

        a = mapping[self._variables[0]]
        b = mapping[self._variables[1]]

        result = tol_eq(a.distance_to(b), self._value)

        return result

    def __str__(self):
        return "DistanceConstraint({0}, {1}, {2})".format(self._variables[0], \
        self._variables[1], self._value)

class AngleConstraint(ParametricConstraint):
    """A constraint on the angle in point B of a triangle ABC"""

    def __init__(self, a, b, c, ang):
        """Create a new AngleConstraint instance.

           keyword args:
            a    - a point variable name
            b    - a point variable name
            c    - a point variable name
            ang  - the angle parameter value
        """

        super(AngleConstraint, self).__init__()

        self._variables = [a,b,c]
        self.set_parameter(ang)

    def satisfied(self, mapping):
        """return True iff mapping from variable names to points satisfies constraint"""

        a = mapping[self._variables[0]]
        b = mapping[self._variables[1]]
        c = mapping[self._variables[2]]

        ang = b.angle_between(a, c)

        if ang is None:
            result = False
        else:
            result = tol_eq(ang, self._value)

        if not result:
            LOGGER.debug("measured angle = %s, parameter value = %s, geometric", ang, self._value)

        return result

    def angle_degrees(self):
        return np.degrees(self._value)

    def __str__(self):
        return "AngleConstraint({0}, {1}, {2}, {3})".format(\
        self._variables[0], self._variables[1], self._variables[2], \
        self.angle_degrees())
