import abc
import logging

from ..geometry import tol_zero, distance_point_line
from ..primitives import GeometricVariable, Point, Line

LOGGER = logging.getLogger(__name__)


class Constraint(metaclass=abc.ABCMeta):
    """Abstract constraint
    A constraint defines a relation between variables that should be satisfied.
    Subclasses must define proper __init__(), variables() and satisfied()
    methods.
    Constraints must be immutable, hashable objects.
    """

    NAME = "Constraint"

    def __init__(self, variables=None):
        super().__init__()

        self.variables = variables

    @abc.abstractmethod
    def satisfied(self, mapping):
        """Returns true if this constraint is satisfied by the specified \
        mapping from variables to values
        :param mapping: dict containing mapping
        """
        raise NotImplementedError

    @property
    def _variable_str(self):
        return ", ".join(self.variables)

    def __str__(self):
        return f"{self.NAME}({self._variable_str})"

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash((self.NAME, tuple(self.variables)))


class CoincidenceConstraint(Constraint):
    """defines a coincidence between a point and another geometricvariable (i.e. point, line, plane)"""
    def __init__(self, point, geometry):
        assert isinstance(point, Point)
        assert isinstance(geometry, GeometricVariable)

        super().__init__(variables=[point, geometry])

    @property
    def point(self):
        return self.variables[0]

    @property
    def geometry(self):
        return self.variables[1]

    def satisfied(self, mapping):
        """return True iff mapping from variable names to points satisfies constraint"""
        if isinstance(self.geometry, Point):
            p1 = mapping[self.point]
            p2 = mapping[self.geometry]
            return tol_zero(p1.distance_to(p2))
        elif isinstance(self.geometry, Line):
            p = mapping[self.point]
            l = mapping[self.geometry]
            if len(l)==4:   #2D
                p1 = l[0:2]
                p2 = l[2:4]
            else:
                raise Exception("line has invalid number of values")

            d = distance_point_line(p, p1, p2)

            if not tol_zero(d):
                LOGGER.debug(f"coincidence constraint '{self}' not satisfied, distance: {d}")

            return tol_zero(d)
        else:
            raise Exception("unknown geometry type""")

    def __str__(self):
        return f"{self.NAME}({self.point}, {self.geometry})"
