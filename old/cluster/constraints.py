"""Constraints between points"""

import abc

from .geometry import tol_eq, degrees, DegenerateError

class Constraint(metaclass=abc.ABCMeta):
    """Abstract constraint

    A constraint defines a relation between variables that should be satisfied.

    Constraints are immutable, hashable objects.
    """
    def __init__(self, variables):
        self.variables = variables

    @abc.abstractmethod
    def satisfied(self, mapping):
        """Returns true if this constraint is satisfied by the specified mapping

        Parameters
        ----------
        mapping : :class:`dict`
            Mapping of variables to values.

        Returns
        -------
        :class:`bool`
            Whether the constraint is satisfied.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abc.abstractmethod
    def __hash__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class ParametricConstraint(Constraint, metaclass=abc.ABCMeta):
    """Constraint with a parameter"""
    def __init__(self, parameter=None, *args, **kwargs):
        """initialize ParametricConstraint"""
        super().__init__(*args, **kwargs)

        # set value
        self.parameter = parameter

    def __eq__(self, other):
        return self.variables == other.variables and self.parameter == other.parameter

    def __hash__(self):
        return hash((tuple(self.variables), self.parameter))


class FixConstraint(ParametricConstraint):
    """Constraint to fix a point relative to the coordinate system"""
    def __init__(self, variable, position):
        super().__init__(parameter=position, variables=[variable])

    @property
    def variable(self):
        return self.variables[0]

    def satisfied(self, mapping):
        """Returns true if this constraint is satisfied by the specified mapping

        Parameters
        ----------
        mapping : :class:`dict`
            Mapping of variables to values.

        Returns
        -------
        :class:`bool`
            Whether the constraint is satisfied.
        """
        return self.variable.tol_eq(mapping[self.variable])

    def __str__(self):
        return f"FixConstraint({self.variable}, {self.parameter})"


class DistanceConstraint(ParametricConstraint):
    """Constraint on the Euclidean distance between two points"""
    def __init__(self, a, b, distance):
        super().__init__(parameter=distance, variables=[a, b])

    @property
    def point_a(self):
        return self.variables[0]

    @property
    def point_b(self):
        return self.variables[1]

    def satisfied(self, mapping):
        """Returns true if this constraint is satisfied by the specified mapping

        Parameters
        ----------
        mapping : :class:`dict`
            Mapping of variables to values.

        Returns
        -------
        :class:`bool`
            Whether the constraint is satisfied.
        """
        point_a = mapping[self.point_a]
        point_b = mapping[self.point_b]

        return self.parameter.tol_eq(point_a.distance_to(point_b))

    def __str__(self):
        return f"DistanceConstraint({self.point_a}, {self.point_b}, {self.parameter})"


class AngleConstraint(ParametricConstraint):
    """Constraint on the angle of triangle ABC"""
    def __init__(self, a, b, c, angle):
        super().__init__(parameter=angle, variables=[a, b, c])

    @property
    def point_a(self):
        return self.variables[0]

    @property
    def point_b(self):
        return self.variables[1]

    @property
    def point_c(self):
        return self.variables[2]

    def satisfied(self, mapping):
        """Returns true if this constraint is satisfied by the specified mapping

        Parameters
        ----------
        mapping : :class:`dict`
            Mapping of variables to values.

        Returns
        -------
        :class:`bool`
            Whether the constraint is satisfied.
        """
        a = mapping[self.point_a]
        b = mapping[self.point_b]
        c = mapping[self.point_c]

        # mapping angle
        try:
            return tol_eq(b.angle_between(a, c), self.parameter)
        except DegenerateError:
            return False

    def __str__(self):
        angle = degrees(self.parameter)

        return f"AngleConstraint({self.point_a}, {self.point_b}, {self.point_c}, {angle})"
