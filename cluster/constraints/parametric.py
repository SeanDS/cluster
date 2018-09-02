import abc
import logging
import numpy as np

from ..notify import Notifier
from ..geometry import Vector, tol_eq, angle_3p
from .base import Constraint

LOGGER = logging.getLogger(__name__)


class ParametricConstraint(Constraint, Notifier, metaclass=abc.ABCMeta):
    """A constraint with a parameter and notification when parameter changes"""

    NAME = "ParametricConstraint"

    def __init__(self, value=None, *args, **kwargs):
        """initialize ParametricConstraint"""
        super().__init__(*args, **kwargs)

        self._value = None

        # set properties
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

        # fire event
        self.send_notify(("set_parameter", value))

    @abc.abstractmethod
    def satisfied(self, mapping):
        raise NotImplementedError

    def __str__(self):
        return f"{self.NAME}(({self._variable_str}) = {self.value})"


class FixConstraint(ParametricConstraint):
    """A constraint to fix a point relative to the coordinate system"""

    NAME = "FixConstraint"

    def __init__(self, variable, position):
        """Create a new FixConstraint instance

           keyword args:
            var    - a point variable name
            pos    - the position parameter
        """
        super().__init__(variables=[variable], value=position)

    @property
    def point(self):
        return self.variables[0]

    @property
    def position(self):
        return self.value

    @position.setter
    def position(self, position):
        self.value = position

    def satisfied(self, mapping):
        """return True iff mapping from variable names to points satisfies constraint"""
        point = mapping[self.point]

        if len(point) != len(self.value):
            raise ValueError("fix constraint vectors of unequal length")

        result = True

        for i in range(len(self.value)):
            result &= tol_eq(point[i], self.value[i])

        return result

    def __str__(self):
        return f"{self.NAME}({self.point} = {self.position})"


class DistanceConstraint(ParametricConstraint):
    """A constraint on the Euclidean distance between two points"""

    NAME = "DistanceConstraint"

    def __init__(self, point_a, point_b, distance):
        """Create a new DistanceConstraint instance
           keyword args:
            a    - a point variable name
            b    - a point variable name
            dist - the distance parameter value
        """
        super().__init__(variables=[point_a, point_b], value=distance)

    @property
    def point_a(self):
        return self.variables[0]

    @property
    def point_b(self):
        return self.variables[1]

    @property
    def distance(self):
        return self.value

    @distance.setter
    def distance(self, distance):
        self.value = distance

    def satisfied(self, mapping):
        """return True iff mapping from variable names to points satisfies constraint"""
        a = mapping[self.point_a]
        b = mapping[self.point_b]

        return tol_eq(a.distance_to(b), np.abs(self._value))

    def __str__(self):
        return f"{self.NAME}(|{self._variable_str}| = {self.distance:.3f})"


class AngleConstraint(ParametricConstraint):
    """A constraint on the angle in point B of a triangle ABC"""

    NAME = "AngleConstraint"

    def __init__(self, point_a, point_b, point_c, angle):
        """Create a new AngleConstraint instance.
           keyword args:
            a    - a point variable name
            b    - a point variable name
            c    - a point variable name
            ang  - the angle parameter value
        """
        super().__init__(variables=[point_a, point_b, point_c], value=angle)

    @property
    def point_a(self):
        return self.variables[0]

    @property
    def point_b(self):
        return self.variables[1]

    @property
    def point_c(self):
        return self.variables[2]

    @property
    def angle(self):
        return self.value

    @property
    def angle_degrees(self):
        return np.degrees(self.angle)

    def satisfied(self, mapping):
        """return True iff mapping from variable names to points satisfies constraint"""
        a = mapping[self.point_a]
        b = mapping[self.point_b]
        c = mapping[self.point_c]

        ang = angle_3p(a, b, c)

        if ang == None:
            # if the angle is indeterminate, its probably ok.
            result = True
        else:
            # in 3d, ignore the sign of the angle
            if len(a) >= 3:
                cmp = abs(self._value)
            else:
                cmp = self._value

            result = tol_eq(ang, cmp)

        if result == False:
            LOGGER.debug("measured angle: {ang}, parameter value: {cmp}")

        return result

    def __str__(self):
        return f"{self.NAME}(∠({self._variable_str}) = {self.angle_degrees:.3f}°)"


class RigidConstraint(ParametricConstraint):
    """A constraint to set the relative position of a set of points"""

    NAME = "RigidConstraint"

    def __init__(self, configuration):
        """Create a new AngleConstraint instance.
           keyword args:
            a    - a point variable name
            b    - a point variable name
            c    - a point variable name
            ang  - the angle parameter value
        """
        super().__init__(variables=[configuration.vars()], value=configuration.copy())

    @property
    def configuration(self):
        return self.value

    def satisfied(self, mapping):
        """return True iff mapping from variable names to points satisfies constraint"""
        result = True

        for index in range(1, len(self.variables) - 1):
            p1 = mapping[self.variables[index - 1]]
            p2 = mapping[self.variables[index]]
            p3 = mapping[self.variables[index + 1]]

            c1 = self.configuration.map[self.variables[index - 1]]
            c2 = self.configuration.map[self.variables[index]]
            c3 = self.configuration.map[self.variables[index + 1]]

            result &= tol_eq(p1.distance_to(p2), c1.distance_to(c2))
            result &= tol_eq(p1.distance_to(p3), c1.distance_to(c3))
            result &= tol_eq(p2.distance_to(p3), c2.distance_to(c3))

        return result

    def __str__(self):
        return f"{self.NAME}({self._variable_str})"
