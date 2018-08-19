import abc
import logging
import numpy as np

from ..constraint import Constraint, ParametricConstraint
from ..geometry import tol_eq, is_clockwise, is_counterclockwise, is_obtuse, is_acute

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


class SelectionConstraint(Constraint, metaclass=abc.ABCMeta):
    """Constraints for solution selection"""

    def __init__(self, name, variables):
        self.name = str(name)
        self.variables = list(variables)

    @abc.abstractmethod
    def satisfied(self, mapping):
        raise NotImplementedError

    def __str__(self):
        return "{0}({1})".format(self.name, ", ".join(self.variables))


class FunctionConstraint(SelectionConstraint, metaclass=abc.ABCMeta):
    """Selects solutions where function returns True when applied to specified \
    variables"""

    def __init__(self, function, variables, name="FunctionConstraint"):
        """Instantiate a FunctionConstraint

        :param function: callable to call to check the constraint
        :param variables: list of variables as arguments for the function
        """

        # call parent constructor
        super(FunctionConstraint, self).__init__(name, variables)

        # set the function
        self.function = function

    def satisfied(self, mapping):
        """Check if mapping from variable names to points satisfies this \
        constraint

        :param mapping: map from variables to their points
        """

        # return whether the result of the function with the variables as
        # arguments is True or not
        return self.function(*[mapping[variable] for variable in \
        self.variables]) is True

    def __str__(self):
        return "{0}({1}, {2})".format(self.name, self.function.__name__, \
        ", ".join(self.variables))


class NotClockwiseConstraint(FunctionConstraint):
    """Selects triplets that are not clockwise (i.e. counter clockwise or \
    degenerate)"""

    def __init__(self, v1, v2, v3):
        """Instantiate a NotClockwiseConstraint

        :param v1: first variable
        :param v2: second variable
        :param v3: third variable
        """

        # create function to evaluate
        # checks if the points do not form a clockwise set
        function = lambda x, y, z: not is_clockwise(x, y, z)

        # call parent constructor
        super(NotClockwiseConstraint, self).__init__(function, [v1, v2, v3], \
        name="NotClockwiseConstraint")


class NotCounterClockwiseConstraint(FunctionConstraint):
    """Selects triplets that are not counter clockwise (i.e. clockwise or \
    degenerate)"""

    def __init__(self, v1, v2, v3):
        """Instantiate a NotCounterClockwiseConstraint

        :param v1: first variable
        :param v2: second variable
        :param v3: third variable
        """

        # create function to evaluate
        # checks if the points do not form a counter clockwise set
        function = lambda x, y, z: not is_counterclockwise(x, y, z)

        # call parent constructor
        super(NotCounterClockwiseConstraint, self).__init__(function, \
        [v1, v2, v3], name="NotCounterClockwiseConstraint")


class NotObtuseConstraint(FunctionConstraint):
    """Selects triplets that are not obtuse (i.e. acute or degenerate)"""

    def __init__(self, v1, v2, v3):
        """Instantiate a NotObtuseConstraint

        :param v1: first variable
        :param v2: second variable
        :param v3: third variable
        """

        # create function to evaluate
        # checks if the points do not form an obtuse angle
        function = lambda x, y, z: not is_obtuse(x, y, z)

        # call parent constructor
        super(NotObtuseConstraint, self).__init__(function, [v1, v2, v3], \
        name="NotObtuseConstraint")


class NotAcuteConstraint(FunctionConstraint):
    """Selects triplets that are not acute (i.e. obtuse or degenerate)"""

    def __init__(self,v1, v2, v3):
        """Instantiate a NotAcuteConstraint

        :param v1: first variable
        :param v2: second variable
        :param v3: third variable
        """

        # create function to evaluate
        # checks if the points do not form an acute angle
        function = lambda x, y, z: not is_acute(x, y, z)

        # call parent constructor
        super(NotAcuteConstraint, self).__init__(function, [v1, v2, v3], \
        name="NotAcuteConstraint")
