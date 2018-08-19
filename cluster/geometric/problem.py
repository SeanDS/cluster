import logging

from ..notify import Notifier, Listener
from ..constraint import ConstraintGraph
from .constraints import (DistanceConstraint, AngleConstraint, FixConstraint, ParametricConstraint,
                          SelectionConstraint)

LOGGER = logging.getLogger(__name__)

class GeometricProblem(Notifier, Listener):
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

        Notifier.__init__(self)
        Listener.__init__(self)

        self.prototype = {}
        self.constraint_graph = ConstraintGraph()

    def add_point(self, variable, position):
        """add a point variable with a prototype position"""

        if variable not in self.prototype:
            # add point to prototype
            self.prototype[variable] = position

            # add to constraint graph
            self.constraint_graph.add_variable(variable)
        else:
            raise Exception("point already in problem")

    def set_point(self, variable, position):
        """set prototype position of point variable"""

        if variable not in self.prototype:
            raise Exception("unknown point variable")

        self.prototype[variable] = position
        self.send_notify(("set_point", (variable, position)))

    def get_point(self, variable):
        """get prototype position of point variable"""

        if variable not in self.prototype:
            raise Exception("unknown point variable")

        return self.prototype[variable]

    def has_point(self, variable):
        return variable in self.prototype

    def add_constraint(self, con):
        """add a constraint"""

        if isinstance(con, DistanceConstraint):
            for var in con.variables:
                if var not in self.prototype:
                    raise Exception("point variable not in problem")

            if self.get_distance(con.variables[0], con.variables[1]):
                raise Exception("distance already in problem")
            else:
                con.add_listener(self)

                self.constraint_graph.add_constraint(con)
        elif isinstance(con, AngleConstraint):
            for var in con.variables:
                if var not in self.prototype:
                    raise Exception("point variable not in problem")
            if self.get_angle(con.variables[0], con.variables[1], con.variables[2]):
                raise Exception("angle already in problem")
            else:
                con.add_listener(self)

                self.constraint_graph.add_constraint(con)
        elif isinstance(con, SelectionConstraint):
            for var in con.variables:
                if var not in self.prototype:
                    raise Exception("point variable not in problem")

            self.constraint_graph.add_constraint(con)
            self.send_notify(("add_selection_constraint", con))
        elif isinstance(con, FixConstraint):
            for var in con.variables:
                if var not in self.prototype:
                    raise Exception("point variable not in problem")

            if self.get_fix(con.variables[0]):
                raise Exception("fix already in problem")

            self.constraint_graph.add_constraint(con)
        else:
            raise Exception("unsupported constraint type")

    def get_distance(self, a, b):
        """return the distance constraint on given points, or None"""

        on_a = self.constraint_graph.get_constraints_on(a)
        on_b = self.constraint_graph.get_constraints_on(b)

        on_ab = [c for c in on_a if c in on_a and c in on_b]
        distances = list([c for c in on_ab if isinstance(c, DistanceConstraint)])

        if len(distances) > 1:
            raise Exception("multiple constraints found")
        elif len(distances) == 1:
            return distances[0]

        return None

    def get_angle(self, a, b, c):
        """return the angle constraint on given points, or None"""

        on_a = self.constraint_graph.get_constraints_on(a)
        on_b = self.constraint_graph.get_constraints_on(b)
        on_c = self.constraint_graph.get_constraints_on(c)

        on_abc = [x for x in on_a if x in on_a and x in on_b and x in on_c]
        angles = [x for x in on_abc if isinstance(x, AngleConstraint)]
        candidates = list([x for x in angles if x.variables[1] == b])

        if len(candidates) > 1:
            raise Exception("multiple constraints found")
        elif len(candidates) == 1:
            return candidates[0]

        return None

    def get_fix(self, p):
        """return the fix constraint on given point, or None"""

        on_p = self.constraint_graph.get_constraints_on(p)

        fixes = [x for x in on_p if isinstance(x, FixConstraint)]

        if len(fixes) > 1:
            raise Exception("multiple constraints found")
        elif len(fixes) == 1:
            return fixes[0]

        return None

    def verify(self, solution):
        """returns true iff all constraints satisfied by given solution.
           solution is a dictionary mapping variables (names) to values (points)"""

        if solution is None:
            sat = False
        else:
            sat = True

            for con in self.constraint_graph.constraints():
                solved = True

                for v in con.variables:
                    if v not in solution:
                        solved = False

                        break

                if not solved:
                    LOGGER.debug(f"{con} not solved", con)

                    sat = False
                elif not con.satisfied(solution):
                    LOGGER.debug(f"{con} not satisfied")

                    sat = False

        return sat

    def rem_point(self, var):
        """remove a point variable from the constraint system"""

        if var in self.prototype:
            self.constraint_graph.rem_variable(var)

            del( self.prototype[var])
        else:
            raise Exception("variable {0} not in problem.".format(var))

    def rem_constraint(self, con):
        """remove a constraint from the constraint system"""

        if con in self.constraint_graph.constraints():
            if isinstance(con, SelectionConstraint):
                self.send_notify(("rem_selection_constraint", con))

            self.constraint_graph.rem_constraint(con)
        else:
            raise Exception("no constraint {0} in problem.".format(con))

    def receive_notify(self, obj, notify):
        """When notified of changed constraint parameters, pass on to listeners"""

        if isinstance(object, ParametricConstraint):
            (message, data) = notify

            if message == "set_parameter":
                self.send_notify(("set_parameter", (obj, data)))

    def __str__(self):
        # variable list on separate lines
        variables = "\n".join(["{0} = {1}".format(variable, \
        self.prototype[variable]) for variable in self.prototype])

        # constraints on separate lines
        constraints = "\n".join([str(constraint) for constraint \
        in self.constraint_graph.constraints()])

        return "{0}\n{1}".format(variables, constraints)
