import abc
import logging
import numpy as np

from ..constraint import Constraint, ParametricConstraint
from ..geometry import tol_eq, is_clockwise, is_counterclockwise, is_obtuse, is_acute

LOGGER = logging.getLogger(__name__)

class FixConstraint(ParametricConstraint):
    """A constraint to fix a point relative to the coordinate system"""

    NAME = "FixConstraint"

    def __init__(self, variable, position):
        """Create a new DistanceConstraint instance

           keyword args:
            var    - a point variable name
            pos    - the position parameter
        """
        super().__init__(variables=[variable], value=position)

    def mapped_value(self, mapping):
        return mapping[self.point]

    @property
    def point(self):
        return self.variables[0]

    @property
    def position(self):
        return self.value

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


class SelectionConstraint(Constraint, metaclass=abc.ABCMeta):
    """Constraints for solution selection"""

    NAME = "SelectionConstraint"


class FunctionConstraint(SelectionConstraint, metaclass=abc.ABCMeta):
    """Selects solutions where function returns True when applied to specified \
    variables"""

    NAME = "FunctionConstraint"

    def __init__(self, function, variables):
        """Instantiate a FunctionConstraint

        :param function: callable to call to check the constraint
        :param variables: list of variables as arguments for the function
        """

        # call parent constructor
        super().__init__(variables=variables)

        # set the function
        self.function = function

    def satisfied(self, mapping):
        """Check if mapping from variable names to points satisfies this \
        constraint

        :param mapping: map from variables to their points
        """
        # return whether the result of the function with the variables as
        # arguments is True or not
        return self.function(*[mapping[variable] for variable in self.variables]) is True

    def __str__(self):
        variables = ", ".join(self.variables)
        return f"{self.NAME}({self.function.__name__}({variables}))"


class NotClockwiseConstraint(FunctionConstraint):
    """Selects triplets that are not clockwise (i.e. counter clockwise or \
    degenerate)"""

    NAME = "NotClockwiseConstraint"

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
        super().__init__(function, [v1, v2, v3])


class NotCounterClockwiseConstraint(FunctionConstraint):
    """Selects triplets that are not counter clockwise (i.e. clockwise or \
    degenerate)"""

    NAME = "NotCounterClockwiseConstraint"

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
        super().__init__(function, [v1, v2, v3])


class NotObtuseConstraint(FunctionConstraint):
    """Selects triplets that are not obtuse (i.e. acute or degenerate)"""

    NAME = "NotObtuseConstraint"

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
        super().__init__(function, [v1, v2, v3])


class NotAcuteConstraint(FunctionConstraint):
    """Selects triplets that are not acute (i.e. obtuse or degenerate)"""

    NAME = "NotAcuteConstraint"

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
        super().__init__(function, [v1, v2, v3])
