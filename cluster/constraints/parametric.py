import abc
import numpy as np

from ..event import Observable, Event
from ..geometry import Vector, tol_eq
from ..configuration import Configuration
from ..clusters import Rigid, Hedgehog
from .base import Constraint

class ParametricConstraint(Constraint, Observable, metaclass=abc.ABCMeta):
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
        self.fire(Event("set_parameter", constraint=self, value=value))

    @abc.abstractmethod
    def mapped_value(self, mapping):
        raise NotImplementedError

    @abc.abstractmethod
    def default_config(self, problem):
        raise NotImplementedError

    @abc.abstractmethod
    def default_cluster(self):
        raise NotImplementedError

    def satisfied(self, mapping):
        """return True iff mapping from variable names to points satisfies constraint"""
        return tol_eq(self.mapped_value(mapping), self.value)

    def __str__(self):
        return f"{self.NAME}(({self._variable_str}) = {self.value})"


class FixConstraint(ParametricConstraint):
    """A constraint to fix a point relative to the coordinate system"""

    NAME = "FixConstraint"

    def __init__(self, variable, position=None):
        """Create a new DistanceConstraint instance

           keyword args:
            var    - a point variable name
            pos    - the position parameter
        """
        if position is None:
            position = Vector.origin()

        super().__init__(variables=[variable], value=position)

    def default_config(self, problem):
        raise NotImplementedError

    def default_cluster(self):
        raise NotImplementedError

    def mapped_value(self, mapping):
        return mapping[self.point]

    @property
    def point(self):
        return self.variables[0]

    @property
    def position(self):
        return self.value

    @position.setter
    def position(self, position):
        self.value = position

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

    def default_config(self, problem):
        return Configuration({self.point_a: Vector.origin(),
                              self.point_b: Vector([self.distance, 0])})

    def default_cluster(self):
        return Rigid([self.point_a, self.point_b])

    def mapped_value(self, mapping):
        point_a = mapping[self.point_a]
        point_b = mapping[self.point_b]
        return point_a.distance_to(point_b)

    @property
    def point_a(self):
        return self.variables[0]

    @property
    def point_b(self):
        return self.variables[1]

    @property
    def distance(self):
        return self.value

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

    def default_config(self, problem):
        return Configuration({self.point_a: Vector([1.0, 0.0]),
                              self.point_b: Vector.origin(),
                              self.point_c: Vector([np.cos(self.angle), np.sin(self.angle)])})

    def default_cluster(self):
        # hedgehog with second point of constraint as the central point and the
        # other points specified with respect to it
        return Hedgehog(self.point_b, [self.point_a, self.point_c])

    def mapped_value(self, mapping):
        point_a = mapping[self.point_a]
        point_b = mapping[self.point_b]
        point_c = mapping[self.point_c]
        return point_b.angle_between(point_a, point_c)

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

    def __str__(self):
        return f"{self.NAME}(∠({self._variable_str}) = {self.angle_degrees:.3f}°)"
