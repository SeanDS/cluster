import logging

from .geometry import Vector
from .clsolver import PrototypeMethod, SelectionMethod
from .clsolver2D import ClusterSolver2D
from .cluster import *
from .selconstr import SelectionConstraint, fnot
from .configuration import Configuration
from .constraint import Constraint, ConstraintGraph
from .notify import Notifier, Listener
from .geometry import (angle_3p, distance_2p, distance_point_line, is_clockwise,
                       is_counterclockwise, perp_2d, tol_eq)
from .geometric import (Point, Line, DistanceConstraint, AngleConstraint, RigidConstraint,
                        FixConstraint, ParametricConstraint)

LOGGER = logging.getLogger(__name__)


class GeometricProblem(Notifier, Listener):
    """A geometric constraint problem with a prototpe.

       A problem consists of geometric variables (just variables for short), a prototype
       for each variable and constraints.

       Variables are of type Point, Line, etc. Alternatively, variables of any other hashable type
        (e.g. strings) are assumed to be points. (Depricated - used for backwards compatibility)

       Prototypes are of type vector
       A point prototype must have length equal to the dimensionality as the problem (D).
       A line prototype must have length 2*D: it represents two points though which the line passes

       Supported constraints are instances of ParametricConstraint, FixConstraint, SelectionConstraint, etc.

       GeometricProblem listens for changes in constraint parameters and passes
       these changes, and changes in the system of constraints and the prototype,
       to any other listerers (e.g. GeometricSolver)
    """

    def __init__(self, use_prototype=True):
        """Initialize a new problem. Must specify dimensionality of the problem (i.e. dimension of points) and
            wheter to use the prototype for solution selection."""
        Notifier.__init__(self)
        Listener.__init__(self)
        self.prototype = {}             # mapping from variables to prototypes
        self.cg = ConstraintGraph()     # constraint graph
        self.use_prototype = use_prototype     # whether to use prototype for solution selection

    # ----------- prototype --------

    def set_prototype_selection(self, enabled):
        """Enable (True, the default) or disable (False) use of prototype for solution selection"""
        self.use_prototype = enabled
        self.dr.set_prototype_selection(self.use_prototype)
        self.send_notify(("set_prototype_selection", self.use_prototype))

    def get_prototype_selection(self):
        """Return True if prototype selection has been enabled (the default) or False otherwise"""
        return self.use_prototype

    # ---------------- variables -----------

    def add_variable(self, variable, prototype=None):
        """add a variable with a prototype"""

        if prototype is None:
            # assume at origin
            prototype = Vector.origin()

        prototypevector = Vector(prototype)
        # check dimension of prototype
        if isinstance(variable, Point):
            assert len(prototypevector) == 2
        elif isinstance(variable, Line):
            assert len(prototypevector) == 4
        else:
            # assume point
            assert len(prototypevector) == 2
        if variable not in self.prototype:
            self.prototype[variable] = prototypevector
            self.cg.add_variable(variable)
        else:
            raise Exception("variable already in problem")

    def has_variable(self, variable):
        """returns True if variable in problem"""
        return variable in self.prototype

    def rem_variable(self, variable):
        """remove a variable (and all constraints incident imposed on it)"""
        if variable in self.prototype:
            del self.prototype[variable]
            self.cg.rem_variable(variable)

    def set_prototype(self, variable, prototype):
        """set prototype of variable"""
        prototypevector = Vector(prototype)
        if variable in self.prototype:
            self.prototype[variable] = prototypevector
            self.send_notify(("set_prototype", (variable,prototypevector)))
        else:
            raise Exception("unknown variable variable")

    def get_prototype(self, variable):
        """get prototype of variable"""
        if variable in self.prototype:
            return self.prototype[variable]
        else:
            raise Exception("unknown variable "+str(variable))

    # ----------- constraints --------

    def add_constraint(self, con):
        """add a constraint"""
        # check that variables in problem
        for var in con.variables():
                if var not in self.prototype:
                    raise Exception("variable %s not in problem"%(var))
        # check that constraint not already in problem
        if isinstance(con, DistanceConstraint):
            if self.get_distance(con.variables()[0],con.variables()[1]):
                raise Exception("distance already in problem")
        elif isinstance(con, AngleConstraint):
            if self.get_angle(con.variables()[0],con.variables()[1], con.variables()[2]):
                raise Exception("angle already in problem")
        elif isinstance(con, RigidConstraint):
            if self.get_rigid(con.variables()):
                raise Exception("rigid already in problem")
        elif isinstance(con, SelectionConstraint):
            pass
        elif isinstance(con, FixConstraint):
            if self.get_fix(con.variables()[0]):
                raise Exception("fix already in problem")
        elif isinstance(con, CoincidenceConstraint):
            if self.get_coincidence(con.variables()[0], con.variables()[1]):
                raise Exception("coincidence already in problem")
        else:
            raise Exception("unsupported constraint type")
        # passed tests, add to poblem
        if isinstance(self, ParametricConstraint):
            con.add_listener(self)
        self.cg.add_constraint(con)

    def get_distance(self, a, b):
        """return the distance constraint on given points, or None"""
        on_a = self.cg.get_constraints_on(a)
        on_b = self.cg.get_constraints_on(b)
        on_ab = [c for c in on_a if c in on_a and c in on_b]
        distances = [c for c in on_ab if isinstance(c, DistanceConstraint)]
        if len(distances) > 1:
            raise Exception("multiple constraints found")
        elif len(distances) == 1:
            return distances[0]
        else:
            return None

    def get_angle(self, a, b, c):
        """return the angle constraint on given points, or None"""
        on_a = self.cg.get_constraints_on(a)
        on_b = self.cg.get_constraints_on(b)
        on_c = self.cg.get_constraints_on(c)
        on_abc = [x for x in on_a if x in on_a and x in on_b and x in on_c]
        angles = [x for x in on_abc if isinstance(x, AngleConstraint)]
        candidates = [x for x in angles if x.variables()[1] == b]
        if len(candidates) > 1:
            raise Exception("multiple constraints found")
        elif len(candidates) == 1:
            return candidates[0]
        else:
            return None

    def get_fix(self, p):
        """return the fix constraint on given point, or None"""
        on_p = self.cg.get_constraints_on(p)
        fixes = [x for x in on_p if isinstance(x, FixConstraint)]
        if len(fixes) > 1:
            raise Exception("multiple constraints found")
        elif len(fixes) == 1:
            return fixes[0]
        else:
            return None

    def get_constraints_with_type_on_variables(self, constrainttype, variables):
        candidates = None
        for var in variables:
            if candidates == None:
                candidates = set([c for c in self.cg.get_constraints_on(var) if isinstance(c,constrainttype)])
            else:
                candidates.intersection_update([c for c in self.cg.get_constraints_on(var) if isinstance(c,constrainttype)])
        return candidates

    def get_unique_constraint(self, constrainttype, variables):
        candidates = self.get_constraints_with_type_on_variables(constrainttype, variables)
        if len(candidates) == 0:
            return None
        elif len(candidates) == 1:
            return list(candidates)[0]
        else: # >= 1
            raise Exception("multiple constraints found")

    def get_coincidence(self, p, g):
        return self.get_unique_constraint(CoincidenceConstraint, [p,g])

    def get_rigid(self, variables):
        return self.get_unique_constraint(RigidConstraint, variables)

    def get_coincident_points(self, geometry):
        coincidences = self.get_constraints_with_type_on_variables(CoincidenceConstraint, [geometry])
        points = set()
        for constraint in coincidences:
            points.update([var for var in constraint.variables() if isinstance(var, Point) and var != geometry])
        return points

    def verify(self, solution):
        """returns true iff all constraints satisfied by given solution.
           solution is a dictionary mapping variables (names) to values (points)"""
        if solution == None:
            sat = False
        else:
            sat = True
            for con in self.cg.constraints():
                solved = True
                for v in con.variables():
                    if v not in solution:
                        solved = False
                        break
                if not solved:
                    LOGGER.debug(f"constraint '{con}' not solved")
                    sat = False
                elif not con.satisfied(solution):
                    LOGGER.debug(f"constraint '{con}' not satisfied")
                    sat = False
        return sat

    def rem_point(self, var):
        """remove a point variable from the constraint system"""
        if var in self.prototype:
            self.cg.rem_variable(var)
            del self.prototype[var]
        else:
            raise Exception("variable "+str(var)+" not in problem.")

    def rem_constraint(self, con):
        """remove a constraint from the constraint system"""
        if con in self.cg.constraints():
            if isinstance(con, SelectionConstraint):
                self.send_notify(("rem_selection_constraint", con))
            self.cg.rem_constraint(con)
        else:
            raise Exception("no constraint "+str(con)+" in problem.")

    def receive_notify(self, object, notify):
        """When notified of changed constraint parameters, pass on to listeners"""
        if isinstance(object, ParametricConstraint):
            (message, data) = notify
            if message == "set_parameter":
                self.send_notify(("set_parameter",(object,data)))
        #elif object == self.cg:
        #    self.send_notify(notify)

    def __str__(self):
        s = ""
        for v in self.prototype:
            s += str(v) + " = " + str(self.prototype[v]) + "\n"
        for con in self.cg.constraints():
            s += str(con) + "\n"
        s+= "prototype-based selection = " + str(self.use_prototype)
        return s
