"""Geometric constraint problem and solver. Uses ClusterSolver for solving
problems incrementally."""

import logging
import abc
import numpy as np

from .solve import PrototypeMethod, ClusterSolver, is_information_increasing
from .cluster import Rigid, Hedgehog
from .configuration import Configuration
from .constraint import Constraint, ConstraintGraph
from .notify import Notifier, Listener
from .selconstr import SelectionConstraint
from .geometry import Vector, tol_eq

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
            for var in con.variables():
                if var not in self.prototype:
                    raise Exception("point variable not in problem")

            if self.get_distance(con.variables()[0],con.variables()[1]):
                raise Exception("distance already in problem")
            else:
                con.add_listener(self)

                self.constraint_graph.add_constraint(con)
        elif isinstance(con, AngleConstraint):
            for var in con.variables():
                if var not in self.prototype:
                    raise Exception("point variable not in problem")
            if self.get_angle(con.variables()[0], con.variables()[1], \
            con.variables()[2]):
                raise Exception("angle already in problem")
            else:
                con.add_listener(self)

                self.constraint_graph.add_constraint(con)
        elif isinstance(con, SelectionConstraint):
            for var in con.variables():
                if var not in self.prototype:
                    raise Exception("point variable not in problem")

            self.constraint_graph.add_constraint(con)
            self.send_notify(("add_selection_constraint", con))
        elif isinstance(con, FixConstraint):
            for var in con.variables():
                if var not in self.prototype:
                    raise Exception("point variable not in problem")

            if self.get_fix(con.variables()[0]):
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
        candidates = list([x for x in angles if x.variables()[1] == b])

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

                for v in con.variables():
                    if v not in solution:
                        solved = False

                        break

                if not solved:
                    logging.getLogger("geometric").debug("%s not solved", con)

                    sat = False
                elif not con.satisfied(solution):
                    logging.getLogger("geometric").debug("%s not satisfied", \
                    con)

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


class GeometricSolver(Listener):
    """The GeometricSolver monitors changes in a GeometricProblem and
       maps any changes to corresponding changes in a GeometricCluster
    """

    def __init__(self, problem):
        """Create a new GeometricSolver instance

           keyword args
            problem        - the GeometricProblem instance to be monitored for changes
        """

        # call parent constructor
        super(GeometricSolver, self).__init__()

        # init variables
        self.problem = problem

        # constraint graph object
        self.constraint_graph = problem.constraint_graph

        # solver
        self.solver = ClusterSolver()

        # map
        self.mapping = {}

        # register listeners
        self.constraint_graph.add_listener(self)
        self.solver.add_listener(self)

        # create an initial fix cluster
        self.fixvars = []
        self.fixcluster = None

        # map current constraint graph variables
        for var in self.constraint_graph.variables():
            self._add_variable(var)

        # list of constraints from graph
        constraints = self.constraint_graph.constraints()

        # add fixed constraints first (avoids problems with invalid solutions)
        list(map(self._add_constraint, [x for x in constraints \
        if isinstance(x, FixConstraint)]))

        # add distances next (doing this before angles results in nicer
        # decompositions)
        list(map(self._add_constraint, [x for x in constraints \
        if isinstance(x, DistanceConstraint)]))

        # add everything else
        list(map(self._add_constraint, [x for x in constraints \
        if not isinstance(x, FixConstraint) \
        and not isinstance(x, DistanceConstraint)]))

    def get_constrainedness(self):
        # get cluster solver's top level solution(s)
        toplevel = self.solver.top_level()

        if len(toplevel) > 1:
            return "under-constrained"

        elif len(toplevel) == 1:
            cluster = toplevel[0]

            if isinstance(cluster, Rigid):
                configurations = self.solver.get(cluster)

                if configurations is None:
                    return "unsolved"
                elif len(configurations) > 0:
                    return "well-constrained"
                else:
                    return "over-constrained"
            else:
                return "under-constrained"
        elif len(toplevel) == 0:
            return "error"

    def decomposition(self):
        """Computes the solution(s) to the geometric problem"""

        # empty dict
        mapping = {}

        # get cluster solver object's rigid clusters
        # at this point, the solver may already have a solution, if the solver
        # has been run before
        for drcluster in self.solver.rigids():
            # create empty geometric cluster
            geocluster = GeometricCluster()

            # create map from geometric cluster to drcluster (and vice versa)
            mapping[drcluster] = geocluster
            mapping[geocluster] = drcluster

            # determine variables
            for var in drcluster.vars:
                geocluster.variables.append(var)

            # determine solutions
            solutions = self.solver.get(drcluster)

            underconstrained = False

            if solutions != None:
                for solution in solutions:
                    geocluster.solutions.append(solution.mapping)

                    if solution.underconstrained:
                        underconstrained = True

            # determine flag
            if drcluster.overconstrained:
                geocluster.flag = GeometricCluster.S_OVER
            elif len(geocluster.solutions) == 0:
                geocluster.flag = GeometricCluster.I_OVER
            elif underconstrained:
                geocluster.flag = GeometricCluster.I_UNDER
            else:
                geocluster.flag = GeometricCluster.OK

        # determine subclusters
        for method in self.solver.methods():
            for out in method.outputs:
                if isinstance(out, Rigid):
                    parent = mapping[out]

                    for inp in method.inputs:
                        if isinstance(inp, Rigid):
                            parent.subs.append(mapping[inp])

        # combine clusters due to selection
        for method in self.solver.methods():
            if isinstance(method, PrototypeMethod):
                incluster = method.inputs[0]
                outcluster = method.outputs[0]
                geoin = mapping[incluster]
                geoout = mapping[outcluster]
                geoout.subs = list(geoin.subs)

        # determine top-level result
        rigids = list([c for c in self.solver.top_level() if isinstance(c, Rigid)])

        if len(rigids) == 0:
            # no variables in problem?
            result = GeometricCluster()

            result.variables = []
            result.subs = []
            result.solutions = []
            result.flags = GeometricCluster.UNSOLVED
        elif len(rigids) == 1:
            # structurally well constrained
            result = mapping[rigids[0]]
        else:
            # structurally underconstrained cluster
            result = GeometricCluster()
            result.flag = GeometricCluster.S_UNDER

            for rigid in rigids:
                result.subs.append(mapping[rigid])

        return result

    def receive_notify(self, obj, message):
        """Take notice of changes in constraint graph"""

        if obj == self.constraint_graph:
            (dtype, data) = message
            if dtype == "add_constraint":
                self._add_constraint(data)
            elif dtype == "rem_constraint":
                self._rem_constraint(data)
            elif dtype == "add_variable":
                self._add_variable(data)
            elif dtype == "rem_variable":
                self._rem_variable(data)
            else:
                raise Exception("unknown message type {0}".format(dtype))
        elif obj == self.problem:
            (dtype, data) = message

            if dtype == "set_point":
                (variable, point) = data

                self._update_variable(variable)
            elif dtype == "set_parameter":
                (constraint, value) = data

                self._update_constraint(constraint)
            else:
                raise Exception("unknown message type {0}".format(dtype))
        elif obj == self.solver:
            pass
        else:
            raise Exception("message from unknown source {0} {1}".format(obj, message))

    def _add_variable(self, variable):
        if variable not in self.mapping:
            logging.getLogger("geometric").debug("Adding variable %s", variable)

            rigid = Rigid([variable])

            self.mapping[variable] = rigid
            self.mapping[rigid] = variable

            self.solver.add(rigid)

            self._update_variable(variable)

    def _rem_variable(self, var):
        logging.getLogger("geometric").debug("GeometricSolver._rem_variable")

        if var in self.mapping:
            self.solver.remove(self.mapping[var])

            del(self.mapping[var])

    def _add_constraint(self, con):
        logging.getLogger("geometric").debug("Adding constraint %s", con)

        if isinstance(con, AngleConstraint):
            # map to hedgehog
            vars = list(con.variables())

            # hedgehog with 2nd point of constraint as the main point, and the
            # other points specified w.r.t. it
            hog = Hedgehog(vars[1], [vars[0], vars[2]])

            self.mapping[con] = hog
            self.mapping[hog] = con

            self.solver.add(hog)

            # set configuration
            self._update_constraint(con)
        elif isinstance(con, DistanceConstraint):
            # map to rigid
            vars = list(con.variables())

            rig = Rigid([vars[0], vars[1]])

            self.mapping[con] = rig
            self.mapping[rig] = con

            self.solver.add(rig)

            # set configuration
            self._update_constraint(con)
        elif isinstance(con, FixConstraint):
            if self.fixcluster != None:
                self.solver.remove(self.fixcluster)
                self.fixcluster = None

            self.fixvars.append(con.variables()[0])

            # check if there are more fixed variables than dimensions
            if len(self.fixvars) >= 2:
                # TODO: check that a Rigid() is always correct to use here
                self.fixcluster = Rigid(self.fixvars)
                self.solver.add(self.fixcluster)
                self.solver.set_root(self.fixcluster)

            self._update_fix()
        else:
            pass

    def _rem_constraint(self, con):
        logging.getLogger("geometric").debug("GeometricSolver._rem_constraint")

        if isinstance(con,FixConstraint):
            if self.fixcluster != None:
                self.solver.remove(self.fixcluster)

            var = self.get(con.variables()[0])

            if var in self.fixvars:
                self.fixvars.remove(var)

            # check if there are less fixed variables than dimensions
            if len(self.fixvars) < 2:
                self.fixcluster = None
            else:
                self.fixcluster = Rigid(self.fixvars)
                self.solver.add(self.fixcluster)
                self.solver.set_root(self.fixcluster)
        elif con in self.mapping:
            self.solver.remove(self.mapping[con])
            del(self.mapping[con])

    def _update_constraint(self, con):
        if isinstance(con, AngleConstraint):
            # set configuration

            # the hog was added to the map for this constraint by _add_constraint,
            # which calls this method
            hog = self.mapping[con]

            # get variables associated with constraint
            variables = list(con.variables())

            v0 = variables[0]
            v1 = variables[1]
            v2 = variables[2]

            # get the constraint's specified angle
            angle = con.get_parameter()

            # create points representing the constraint
            p0 = Vector([1.0, 0.0])
            p1 = Vector.origin()
            p2 = Vector([np.cos(angle), np.sin(angle)])

            # create configuration
            conf = Configuration({v0: p0, v1: p1, v2: p2})

            # set the hedgehog's configuration in the solver
            self.solver.set(hog, [conf])

            assert con.satisfied(conf.mapping)
        elif isinstance(con, DistanceConstraint):
            # set configuration
            rig = self.mapping[con]

            variables = list(con.variables())

            v0 = variables[0]
            v1 = variables[1]

            dist = con.get_parameter()

            p0 = Vector.origin()
            p1 = Vector([dist, 0.0])

            conf = Configuration({v0: p0, v1: p1})

            self.solver.set(rig, [conf])

            assert con.satisfied(conf.mapping)
        elif isinstance(con, FixConstraint):
            self._update_fix()
        else:
            raise Exception("unknown constraint type")

    def _update_variable(self, variable):
        logging.getLogger("geometric").debug("Updating variable %s", variable)

        cluster = self.mapping[variable]
        proto = self.problem.get_point(variable)

        conf = Configuration({variable: proto})

        self.solver.set(cluster, [conf])

    def _update_fix(self):
        if not self.fixcluster:
            logging.getLogger("geometric").warning("No fix cluster to update")

            return

        variables = self.fixcluster.vars

        mapping = {}

        for var in variables:
            mapping[var] = self.problem.get_fix(var).get_parameter()

        conf = Configuration(mapping)

        self.solver.set(self.fixcluster, [conf])

class GeometricCluster(object):
    """Represents the result of solving a GeometricProblem. A cluster is a list of
       point variable names and a list of solutions for
       those variables. A solution is a dictionary mapping variable names to
       points. The cluster also keeps a list of sub-clusters (GeometricCluster)
       and a set of flags, indicating incidental/structural
       under/overconstrained

       instance attributes:
            variables       - a list of point variable names
            solutions       - a list of solutions. Each solution is a dictionary
                              mapping variable names to :class:`Vector`
                              objects.
            subs            - a list of sub-clusters
            flag            - value                 meaning
                              OK                    well constrained
                              I_OVER                incicental over-constrained
                              I_UNDER               incidental under-constrained
                              S_OVER                structural overconstrained
                              S_UNDER               structural underconstrained
                              UNSOLVED              unsolved
       """

    OK = "well constrained"
    I_OVER = "incidental over-constrained"
    I_UNDER = "incidental under-constrained"
    S_OVER = "structral over-constrained"
    S_UNDER = "structural under-constrained"
    UNSOLVED = "unsolved"

    def __init__(self):
        """initialise an empty new cluster"""

        self.variables = []
        self.solutions = []
        self.subs = []
        self.flag = GeometricCluster.OK

    def __str__(self):
        return self._str_recursive()

    def _str_recursive(result, depth=0, done=None):
        # create indent
        spaces = ""

        for i in range(depth):
            spaces = spaces + "|"

        # make done
        if done is None:
            done = set()

        # recurse
        s = ""

        if result not in done:
            # this one is done...
            done.add(result)

            # recurse
            for sub in result.subs:
                s = s + sub._str_recursive(depth+1, done)

        elif len(result.subs) > 0:
            s = s + spaces + "|...\n"

        # print cluster
        solutions = "solution"
        if len(result.solutions) != 1:
            solutions += "s"
        return f"{spaces}cluster {result.variables} {result.flag} {len(result.solutions)} {solutions}\n{s}"

class ParametricConstraint(Constraint, Notifier, metaclass=abc.ABCMeta):
    """A constraint with a parameter and notification when parameter changes"""

    def __init__(self):
        """initialize ParametricConstraint"""

        Constraint.__init__(self)
        Notifier.__init__(self)

        self._value = None

    def get_parameter(self):
        """get parameter value"""

        return self._value

    def set_parameter(self, value):
        """set parameter value and notify any listeners"""

        self._value = value
        self.send_notify(("set_parameter", value))

class FixConstraint(ParametricConstraint):
    """A constraint to fix a point relative to the coordinate system"""

    def __init__(self, var, pos):
        """Create a new DistanceConstraint instance

           keyword args:
            var    - a point variable name
            pos    - the position parameter
        """

        super(FixConstraint, self).__init__()

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
        super(DistanceConstraint, self).__init__()

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
            logging.getLogger("geometric").debug("measured angle = %s, parameter value = %s, geometric", ang, self._value)

        return result

    def angle_degrees(self):
        return np.degrees(self._value)

    def __str__(self):
        return "AngleConstraint({0}, {1}, {2}, {3})".format(\
        self._variables[0], self._variables[1], self._variables[2], \
        self.angle_degrees())
