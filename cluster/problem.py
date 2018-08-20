import logging

from .geometry import Vector
from .event import Observable, Observer, Event, UnknownEventException
from .constraints import (DistanceConstraint, AngleConstraint, FixConstraint, ParametricConstraint,
                          SelectionConstraint, ConstraintGraph)

LOGGER = logging.getLogger(__name__)

class GeometricProblem(Observable, Observer):
    """A geometric constraint problem with a prototype.

    A problem consists of point variables (just variables for short), prototype
    points for each variable and constraints.

    Variables are just names and can be identified by any hashable object
    (recommend strings).
    Supported constraints are instances of DistanceConstraint,
    AngleConstraint, FixConstraint or SelectionConstraint.

    Prototype points are instances of the :class:`Vector` class.

    GeometricProblem listens for changes in constraint parameters and passes
    these changes, and changes in the system of constraints and the prototype,
    to any other listeners (e.g. GeometricSolver)

    instance attributes:
        cg         - a ConstraintGraph instance
        prototype  - a dictionary mapping variables to points
    """

    def __init__(self):
        """initialize a new problem"""
        super().__init__()

        self.prototype = {}
        self.constraint_graph = ConstraintGraph()

    def add_point(self, point, position=None):
        """add a point variable with a prototype position"""
        if position is None:
            # default to origin
            position = Vector.origin()

        if point in self.prototype:
            raise ValueError(f"point '{point}' already in problem")

        # add point to prototype
        self.prototype[point] = position

        # add to constraint graph
        self.constraint_graph.add_variable(point)

    def remove_point(self, point):
        """remove a point variable from the constraint system"""
        if point not in self.prototype:
            raise ValueError(f"point {point} not in problem")

        self.constraint_graph.remove_variable(point)
        del(self.prototype[point])

    def set_point(self, point, position):
        """set prototype position of point variable"""
        if point not in self.prototype:
            raise ValueError(f"point '{point}' not in problem")

        self.prototype[point] = position

        # fire event
        self.fire(Event("set_point", point=point, position=position))

    def get_point(self, point):
        """get prototype position of point variable"""
        if point not in self.prototype:
            raise ValueError(f"point '{point}' not in problem")

        return self.prototype[point]

    def has_point(self, point):
        return point in self.prototype

    def add_constraint(self, constraint):
        """add a constraint"""
        if constraint in self.constraint_graph.constraints:
            raise ValueError(f"constraint '{constraint}' already in problem'")

        for variable in constraint.variables:
            if variable not in self.prototype:
                raise ValueError(f"constraint '{constraint}' point '{variable}' not in problem")

        # observe object for updates
        constraint.add_observer(self)

        self.constraint_graph.add_constraint(constraint)

    def remove_constraint(self, constraint):
        """remove a constraint from the constraint system"""
        if constraint not in self.constraint_graph.constraints:
            raise ValueError(f"constraint {constraint} not in problem")

        self.constraint_graph.remove_constraint(constraint)

    def get_fix(self, point):
        """return the fix constraint on given point, or None"""
        constraints = self.constraint_graph.constraints_on(point)
        constraints = [constraint for constraint in constraints if isinstance(constraint, FixConstraint)]

        if len(constraints) > 1:
            raise Exception("multiple constraints found")
        elif len(constraints) == 0:
            return None

        return constraints[0]

    def verify(self, solution):
        """returns true iff all constraints satisfied by given solution.
           solution is a dictionary mapping variables (names) to values (points)"""
        if solution is None:
            return False

        for constraint in self.constraint_graph.constraints:
            if not constraint.satisfied(solution):
                LOGGER.debug(f"{constraint} not satisfied")
                return False

            for variable in constraint.variables:
                if variable not in solution:
                    LOGGER.debug(f"{constraint} not solved")
                    return False

        return True

    def _handle_event(self, event):
        """When notified of changed constraint parameters, pass on to listeners"""
        if event.message == "set_parameter":
            # pass through
            self.fire(event)
        else:
            raise UnknownEventException(event)

    def __str__(self):
        points = ", ".join(self.prototype)
        return f"Problem({points})"

    def __repr__(self):
        # variable list on separate lines
        variables = "\n".join(["{0} = {1}".format(variable, \
        self.prototype[variable]) for variable in self.prototype])

        # constraints on separate lines
        constraints = "\n".join([str(constraint) for constraint
                                 in self.constraint_graph.constraints])

        return "{0}\n{1}".format(variables, constraints)
