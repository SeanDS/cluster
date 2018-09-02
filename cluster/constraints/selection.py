import abc

from ..geometry import is_clockwise, is_counterclockwise, is_obtuse, is_acute
from .base import Constraint


class SelectionConstraint(Constraint, metaclass=abc.ABCMeta):
    """Selects solutions where function returns True when applied to specified variables"""

    NAME = "SelectionConstraint"

    def __init__(self, function, variables):
        """Instantiate a SelectionConstraint
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


class ClockwiseConstraint(SelectionConstraint):
    """Selects triplets that are clockwise"""

    NAME = "ClockwiseConstraint"

    def __init__(self, v1, v2, v3):
        """Instantiate a ClockwiseConstraint
        :param v1: first variable
        :param v2: second variable
        :param v3: third variable
        """

        # create function to evaluate
        # checks if the points do not form a clockwise set
        function = lambda x, y, z: is_clockwise(x, y, z)

        # call parent constructor
        super().__init__(function, [v1, v2, v3])


class NotClockwiseConstraint(SelectionConstraint):
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


class CounterClockwiseConstraint(SelectionConstraint):
    """Selects triplets that are counter clockwise"""

    NAME = "CounterClockwiseConstraint"

    def __init__(self, v1, v2, v3):
        """Instantiate a CounterClockwiseConstraint
        :param v1: first variable
        :param v2: second variable
        :param v3: third variable
        """

        # create function to evaluate
        # checks if the points do not form a counter clockwise set
        function = lambda x, y, z: is_counterclockwise(x, y, z)

        # call parent constructor
        super().__init__(function, [v1, v2, v3])


class NotCounterClockwiseConstraint(SelectionConstraint):
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


class ObtuseConstraint(SelectionConstraint):
    """Selects triplets that are obtuse"""

    NAME = "ObtuseConstraint"

    def __init__(self, v1, v2, v3):
        """Instantiate a ObtuseConstraint
        :param v1: first variable
        :param v2: second variable
        :param v3: third variable
        """

        # create function to evaluate
        # checks if the points do not form an obtuse angle
        function = lambda x, y, z: is_obtuse(x, y, z)

        # call parent constructor
        super().__init__(function, [v1, v2, v3])


class NotObtuseConstraint(SelectionConstraint):
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


class AcuteConstraint(SelectionConstraint):
    """Selects triplets that are acute"""

    NAME = "AcuteConstraint"

    def __init__(self,v1, v2, v3):
        """Instantiate a AcuteConstraint
        :param v1: first variable
        :param v2: second variable
        :param v3: third variable
        """

        # create function to evaluate
        # checks if the points do not form an acute angle
        function = lambda x, y, z: is_acute(x, y, z)

        # call parent constructor
        super().__init__(function, [v1, v2, v3])


class NotAcuteConstraint(SelectionConstraint):
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
