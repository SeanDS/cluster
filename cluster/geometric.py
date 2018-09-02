"""Geometric constraint problem and solver. Uses ClusterSolver for solving
problems incrementally."""

import numpy as np

from .geometry import Vector
from .clsolver import PrototypeMethod, SelectionMethod
from .clsolver2D import ClusterSolver2D
from .cluster import *
from .selconstr import SelectionConstraint, fnot
from .configuration import Configuration
from .diagnostic import diag_print
from .constraint import Constraint, ConstraintGraph
from .notify import Notifier, Listener
from .geometry import (angle_3p, distance_2p, distance_point_line, is_clockwise,
                       is_counterclockwise, perp_2d, tol_eq)

# ----------- GeometricProblem -------------

class GeometricProblem (Notifier, Listener):
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

    def add_variable(self, variable, prototype):
        """add a variable with a prototype"""
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

    # --------------- points - depricated  -------------
    def add_point(self, variable, prototype):
        """depricated - use add_variable"""
        return self.add_variable(variable, prototype)

    def has_point(self, variable):
        """depricated - use has_variable"""
        return self.has_variable(variable)

    def set_point(self, variable, prototype):
        """depricated - use set_prototype"""
        return self.set_prototype(variable, prototype)

    def get_point(self, variable):
        """depricated - use get_prototype"""
        return self.get_prototype(variable)

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
                    diag_print(str(con)+" not solved", "GeometricProblem.verify")
                    sat = False
                elif not con.satisfied(solution):
                    diag_print(str(con)+" not satisfied", "GeometricProblem.verify")
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

#class GeometricProblem


# ---------- GeometricSolver --------------

class GeometricSolver (Listener):
    """The GeometricSolver monitors changes in a GeometricProblem and
       maps any changes to corresponding changes in a GeometricDecomposition
    """

    # public methods
    def __init__(self, problem):
        """Create a new GeometricSolver instance

           keyword args
            problem        - the GeometricProblem instance to be monitored for changes
        """
        # init superclasses
        Listener.__init__(self)

        # init variables
        # the problem on which this solver works
        self.problem = problem
        # shortcut to the constraint graph of the problem
        self.cg = problem.cg
        # the cluster-solver (or dr for decomposition-recombination)
        self.dr = ClusterSolver2D()

        # a map from problem variables and constraints to clusters, and vice versa
        self._map = {}

        # enable prototype based selection by default
        self._set_prototype_selection(problem.get_prototype_selection())

        # register
        self.cg.add_listener(self)
        self.dr.add_listener(self)

        # create an initial fix cluster
        self.fixvars = []
        self.fixcluster = None

        # add variables
        for var in self.cg.variables():
            self._add_variable(var)

        # add constraints
        toadd = set(self.cg.constraints())

        # add coincidences first. Prevents re-mapping of primitves and re-solving of problem
        for con in list(toadd):
            if isinstance(con, CoincidenceConstraint):
                self._add_constraint(con)
                toadd.remove(con)

        # add selection constraints first. Prevents re-evaluation
        for con in list(toadd):
            if isinstance(con, SelectionConstraint):
                self._add_constraint(con)
                toadd.remove(con)

        # add distances first. Nicer decomposition in Rigids
        for con in list(toadd):
            if isinstance(con, DistanceConstraint):
                self._add_constraint(con)
                toadd.remove(con)

        # add other constraints.
        for con in toadd:
            self._add_constraint(con)

    def get_decomposition(self):
        """Returns a GeometricDecomposition (the root of a tree of clusters),
         describing the solutions and the decomposition of the problem."""
        # several drcluster can maps to a single geoclusters
        map = {}
        geoclusters = []
        # map dr clusters
        for drcluster in [c for c in self.dr.clusters() if isinstance(c, Rigid)]:
            # create geocluster and map to drcluster (and vice versa)
            geocluster = GeometricDecomposition(drcluster.vars)
            if geocluster not in map:
                map[drcluster] = geocluster
                map[geocluster] = [drcluster]
                geoclusters.append(geocluster)
            else:
                geocluster = map[map[geocluster][0]]
                map[drcluster] = geocluster
                map[geocluster].append(drcluster)

        for geocluster in geoclusters:
            # pick newest drcluster
            drclusters = map[geocluster]
            drcluster = max(drclusters, key=lambda c: c.creationtime)
            # determine solutions
            geocluster.solutions = self._map_cluster_solutions(drcluster)
            # determine incidental underconstrainedness
            underconstrained = False
            configurations = self.dr.get(drcluster)
            if configurations != None:
                for config in configurations:
                    if config.underconstrained:
                        underconstrained = True
            # determine flag
            if drcluster.overconstrained:
                geocluster.flag = GeometricDecomposition.OVERCONSTRAINED
            elif geocluster.solutions == None:
                geocluster.flag = GeometricDecomposition.UNSOLVED
            elif len(geocluster.solutions) == 0:
                geocluster.flag = GeometricDecomposition.INCONSISTENT
            elif underconstrained:
                geocluster.flag = GeometricDecomposition.DEGENERATE
            else:
                geocluster.flag = GeometricDecomposition.OK


        # determine subclusters
        for method in self.dr.methods():
            if not isinstance(method, PrototypeMethod) and not isinstance(method, SelectionMethod):
                for out in method.outputs():
                    if isinstance(out, Rigid):
                        parent = map[out]
                        for inp in method.inputs():
                            if isinstance(inp, Rigid):
                                sub = map[inp]
                                if sub != parent and sub not in parent.subs:
                                    parent.subs.append(sub)

        # determine result from top-level clusters
        top = self.dr.top_level()
        rigids = [c for c in top if isinstance(c, Rigid)]
        if len(top) > 1:
            # structurally underconstrained cluster
            result = GeometricDecomposition(self.problem.cg.variables())
            result.flag = GeometricDecomposition.UNDERCONSTRAINED
            for cluster in rigids:
                result.subs.append(map[cluster])
        else:
            if len(rigids) == 1:
                # structurally well constrained, or structurally overconstrained
                result = map[rigids[0]]
            else:
                # no variables in problem?
                result = GeometricDecomposition(self.problem.cg.variables())
                result.variables = []
                result.subs = []
                result.solutions = []
                result.flags = GeometricDecomposition.UNSOLVED
        return result


    def get_solutions(self):
        """Returns a list of Configurations, which will be empty if the
           problem has no solutions. Note: this method is
           cheaper but less informative than get_decomposition.
        """
        #"""The list and the configurations should not be changed (since they are
        #references to objects in the solver)."""
        # find top level rigid and all its configurations
        rigids = [c for c in self.dr.top_level() if isinstance(c, Rigid)]
        if len(rigids) != 0:
            solutions = self._map_cluster_solutions(drcluster)
        else:
            solutions = []
        return solutions

    def _map_cluster_solutions(self, drcluster):
        # map dr-cluster configurations to solutions, i.e. a map from problem variables to values
        configurations = self.dr.get(drcluster)
        solutions = []
        diag_print("mapping cluster "+str(drcluster)+" #configurations="+str(len(configurations)),"GeometricSolver")
        for configuration in configurations:
            solution = {}
            for var in self.problem.cg.variables():
                if isinstance(var, Point):
                    assert len(self._map[var].vars) == 1
                    point = next(iter(self._map[var].vars))
                    if point in configuration:
                        solution[var] = configuration[point]
                elif isinstance(var, Line):
                    line_rigid = self._map[var]
                    line_vertex = line_rigid.vertex
                    line_normal = line_rigid.normal
                    if line_vertex in configuration and line_normal in configuration:
                        p1 = configuration[line_vertex]
                        n = configuration[line_normal]
                        p2 = p1 + perp_2d(n-p1)
                        solution[var] = p1.concatonated(p2)
                else:
                    # assume point - depricated
                    assert len(self._map[var].vars) == 1
                    point = next(iter(self._map[var].vars))
                    if point in configuration:
                        solution[var] = configuration[point]
            #for
            solutions.append(solution)
        #for
        return solutions

    def get_status(self):
        """Returns a symbolic flag, one of:
            GeometricDecomposition.UNDERCONSTRAINED,
            GeometricDecomposition.OVERCONSTRAINED,
            GeometricDecomposition.OK,
            GeometricDecomposition.UNSOLVED,
            GeometricDecomposition.EMPTY,
            GeometricDecomposition.INCONSISTENT,
            GeometricDecomposition.DEGENERATE.
           Note: this method is cheaper but less informative than get_decomposition.
        """
        rigids = [c for c in self.dr.top_level() if isinstance(c, Rigid)]
        if len(rigids) == 0:
            return GeometricDecomposition.EMPTY
        elif len(rigids) == 1:
            drcluster = rigids[0]
            solutions = self.dr.get(drcluster)
            underconstrained = False
            if solutions == None:
                return GeometricDecomposition.UNSOLVED
            else:
                for solution in solutions:
                    if solution.underconstrained:
                        underconstrained = True
            if drcluster.overconstrained:
                return GeometricDecomposition.OVERCONSTRAINED
            elif len(solutions) == 0:
                return GeometricDecomposition.INCONSISTENT
            elif underconstrained:
                return GeometricDecomposition.DEGENERATE
            else:
                return GeometricDecomposition.OK
        else:
            return GeometricDecomposition.UNDERCONSTRAINED


    def receive_notify(self, object, message):
        """Take notice of changes in constraint problem"""
        if object == self.cg:
            (type, data) = message
            if type == "add_constraint":
                self._add_constraint(data)
            elif type == "rem_constraint":
                self._rem_constraint(data)
            elif type == "add_variable":
                self._add_variable(data)
            elif type == "rem_variable":
                self._rem_variable(data)
            elif type == "set_prototype_selection":
                self.set_prototype_selection(data)
            else:
                raise Exception("unknown message type"+str(type))
        elif object == self.problem:
            (type, data) = message
            if type == "set_point":
                (variable, point) = data
                self._update_variable(variable)
            elif type == "set_parameter":
                (constraint, value) = data
                self._update_constraint(constraint)
            else:
                raise Exception("unknown message type"+str(type))
        elif object == self.dr:
            pass
        else:
            raise Exception("message from unknown source"+str((object, message)))

    # --------------- internal methods ------------------

    def _set_prototype_selection(self, enabled):
        """Enable (True) or disable (False) use of prototype for solution selection"""
        self.dr.set_prototype_selection(enabled)

    def _add_variable(self, var):
        if isinstance(var, Point):
            self._add_point(var)
        elif isinstance(var, Line):
            self._add_line(var)
        else:
            # assume point - depricated
            self._add_point(var)

    def _add_point(self, var):
        if var not in self._map:
            rigid = Rigid([var])
            self._map[var] = rigid
            self._map[rigid] = var
            self.dr.add(rigid)
            self._update_variable(var)

    def _add_line(self, var):
        diag_print("add line "+str(var),"GeometricSolver")
        # find coincident points
        points = list(self.problem.get_coincident_points(var))
        diag_print("on "+str(points),"GeometricSolver")

        if len(points) == 0:
            self._map_line_distance(var)
        elif len(points) >= 1:
            self._map_line_point_distance(var, points[0])

    def _map_line_distance(self,line):
        # map a line (coincident with no points) to a distance cluster (on two new point variables)
        v = str(line)+"_vertex"
        n = str(line)+"_normal"
        dist = Rigid([v,n])
        # add add-hoc attributes to rigid, so we can distinguish vertex and normal!
        dist.vertex = v
        dist.normal = n
        # add rigids for created points, needed for prototypes
        # NOTE: adding non-problem variables to mapping!
        # TODO: clean up after removal of line
        vertex_rigid = Rigid([dist.vertex])
        self.dr.add(vertex_rigid)
        self._map[dist.vertex] = vertex_rigid
        normal_rigid = Rigid([dist.normal])
        self.dr.add(normal_rigid)
        self._map[dist.normal] = normal_rigid
        # add line to mapping
        self._map[line] = dist
        self._map[dist] = line
        self.dr.add(dist)
        diag_print("mapped "+str(line)+" to "+str(dist),"GeometricSolver")
        # update configurations
        self._update_variable(line)

    def _map_line_point_distance(self,line, point):
        # map a line coincident with one point to a distance clusters (and one new point variable)
        v = list(self._map[point].vars)[0]
        n = str(line)+"_normal"
        dist = Rigid([v,n])
        # add add-hoc attributes to rigid, so we can distinguish vertex and normal!
        dist.vertex = v
        dist.normal = n
        # add rigids for created points, needed for prototypes
        # NOTE: adding non-problem variables to mapping!
        # TODO: clean up after removal of line
        normal_rigid = Rigid([dist.normal])
        self.dr.add(normal_rigid)
        self._map[dist.normal] = normal_rigid
        # add to mapping
        self._map[line] = dist
        self._map[dist] = line
        self.dr.add(dist)
        diag_print("mapped "+str(line)+" to "+str(dist),"GeometricSolver")
        self._update_variable(line)

    def _map_line_3d_distance(self,line):
        # map a line (coincident with no points) to a distance cluster (on two new point variables)
        v = str(line)+"_vertex"
        n1 = str(line)+"_normal1"
        n2 = str(line)+"_normal2"
        dist = Rigid([v,n1, n2])
        # add add-hoc attributes to rigid, so we can distinguish vertex and normal!
        dist.vertex = v
        dist.normal1 = n1
        dist.normal2 = n2
        # add rigids for created points, needed for prototypes
        # NOTE: adding non-problem variables to mapping!
        # TODO: clean up after removal of line
        vertex_rigid = Rigid([dist.vertex])
        self.dr.add(vertex_rigid)
        self._map[dist.vertex] = vertex_rigid
        normal1_rigid = Rigid([dist.normal1])
        self.dr.add(normal1_rigid)
        self._map[dist.normal1] = normal1_rigid
        normal2_rigid = Rigid([dist.normal2])
        self.dr.add(normal2_rigid)
        self._map[dist.normal2] = normal2_rigid
        # add line to mapping
        self._map[line] = dist
        self._map[dist] = line
        self.dr.add(dist)
        diag_print("mapped "+str(line)+" to "+str(dist),"GeometricSolver")
        # update configurations
        self._update_variable(line)

    def _map_line_3d_point_distance(self,line, point):
        # map a line coincident with one point to a distance clusters (and one new point variable)
        v = list(self._map[point].vars)[0]
        n1 = str(line)+"_normal1"
        n2 = str(line)+"_normal2"
        dist = Rigid([v,n1, n2])
        # add add-hoc attributes to rigid, so we can distinguish vertex and normal!
        dist.vertex = v
        dist.normal1 = n1
        dist.normal2 = n2
        # add rigids for created points, needed for prototypes
        # NOTE: adding non-problem variables to mapping!
        # TODO: clean up after removal of line
        normal1_rigid = Rigid([dist.normal1])
        self.dr.add(normal1_rigid)
        self._map[dist.normal1] = normal1_rigid
        normal2_rigid = Rigid([dist.normal2])
        self.dr.add(normal2_rigid)
        self._map[dist.normal2] = normal2_rigid
        # add to mapping
        self._map[line] = dist
        self._map[dist] = line
        self.dr.add(dist)
        diag_print("mapped "+str(line)+" to "+str(dist),"GeometricSolver")
        self._update_variable(line)

    def _rem_variable(self, var):
        diag_print("GeometricSolver._rem_variable","GeometricSolver")
        if var in self._map:
            self.dr.remove(self._map[var])
            # Note: CLSolver automatically removes variables with no dependent clusters
            del self._map[var]

    def _add_constraint(self, con):
        if isinstance(con, AngleConstraint):
            # map to hedgdehog
            vars = list(con.variables());
            hog = Hedgehog(vars[1],[vars[0],vars[2]])
            self._map[con] = hog
            self._map[hog] = con
            self.dr.add(hog)
            # set configuration
            self._update_constraint(con)
        elif isinstance(con, DistanceConstraint):
            # map to rigid
            vars = list(con.variables());
            rig = Rigid([vars[0],vars[1]])
            self._map[con] = rig
            self._map[rig] = con
            self.dr.add(rig)
            # set configuration
            self._update_constraint(con)
        elif isinstance(con, RigidConstraint):
            # map to rigid
            vars = list(con.variables());
            rig = Rigid(vars)
            self._map[con] = rig
            self._map[rig] = con
            self.dr.add(rig)
            # set configuration
            self._update_constraint(con)
        elif isinstance(con, FixConstraint):
            if self.fixcluster != None:
                self.dr.remove(self.fixcluster)
            self.fixvars.append(con.variables()[0])
            if len(self.fixvars) >= 1:
                self.fixcluster = Rigid(self.fixvars)
                self.dr.add(self.fixcluster)
                self.dr.set_root(self.fixcluster)
                self._update_fix()
        elif isinstance(con, SelectionConstraint):
            # add directly to clustersolver
            self.dr.add_selection_constraint(con)
        elif isinstance(con, CoincidenceConstraint):
            # re-map lines, etc
            lines = [var for var in con.variables() if isinstance(var,Line)]
            points = [var for var in con.variables() if isinstance(var,Point)]
            if len(lines)==1 and len(points)==1:
                line = next(iter(lines))
                point = next(iter(points))
                # re-map line if needed
                #self._rem_variable(line)
                #self._add_line(line)
                # map coincience constraint of a point with a line
                line_rigid  = self._map[line]
                point_rigid = self._map[point]
                point_vertex = next(iter(point_rigid.vars))
                if point_vertex not in line_rigid.vars:
                    line_vertex = line_rigid.vertex
                    line_normal = line_rigid.normal
                    angle_hog = Hedgehog(line_vertex,[line_normal, point_vertex])
                    self._map[con] = angle_hog
                    self._map[angle_hog] = con
                    self.dr.add(angle_hog)
                    diag_print("mapped "+str(con)+" to "+str(angle_hog),"GeometricSolver")
                    self._update_constraint(con)
        else:
            raise Exception("unknown constraint type")
            pass

    def _rem_constraint(self, con):
        diag_print("GeometricSolver._rem_constraint","GeometricSolver")
        if isinstance(con,FixConstraint):
            if self.fixcluster != None:
                self.dr.remove(self.fixcluster)
            var = con.variables()[0]
            if var in self.fixvars:
                self.fixvars.remove(var)
            if len(self.fixvars) == 0:
                self.fixcluster = None
            else:
                self.fixcluster = Rigid(self.fixvars)
                self.dr.add(self.fixcluster)
                self.dr.set_root(self.fixcluster)
        elif isinstance(con, SelectionConstraint):
            # remove directly from clustersolver
            self.dr.rem_selection_constraint(con)
        elif con in self._map:
            self.dr.remove(self._map[con])
            del self._map[con]

    # update methods: set the value of the variables in the constraint graph

    def _update_constraint(self, con):
        if isinstance(con, AngleConstraint):
            # set configuration
            hog = self._map[con]
            vars = list(con.variables())
            v0 = vars[0]
            v1 = vars[1]
            v2 = vars[2]
            angle = con.get_parameter()
            p0 = Vector([1.0,0.0])
            p1 = Vector.origin()
            p2 = Vector([np.cos(angle), np.sin(angle)])
            conf = Configuration({v0:p0,v1:p1,v2:p2})
            self.dr.set(hog, [conf])
            assert con.satisfied(conf.map)
        elif isinstance(con, DistanceConstraint):
            # set configuration
            rig = self._map[con]
            vars = list(con.variables())
            v0 = vars[0]
            v1 = vars[1]
            dist = con.get_parameter()
            #p0 = Vector.origin()
            #p1 = Vector([dist,0.0])
            # use prototype to orient rigid - minimize difference solution and prototype
            p0 = self.problem.get_prototype(v0)
            v = self.problem.get_prototype(v1) - p0
            if v.length != 0:
                v = v / v.length
            else:
                v = Vector.origin()
                v[0] = 1.0
            p1 = p0+v*dist
            conf = Configuration({v0:p0,v1:p1})
            self.dr.set(rig, [conf])
            assert con.satisfied(conf.map)
        elif isinstance(con, RigidConstraint):
            # set configuration
            rig = self._map[con]
            vars = list(con.variables())
            conf = con.get_parameter()
            self.dr.set(rig, [conf])
            assert con.satisfied(conf.map)
        elif isinstance(con, FixConstraint):
            self._update_fix()
        elif isinstance(con, CoincidenceConstraint):
            lines = [var for var in con.variables() if isinstance(var,Line)]
            points = [var for var in con.variables() if isinstance(var,Point)]
            if len(lines)==1 and len(points)==1:
                line = next(iter(lines))
                point = next(iter(points))
                line_rigid = self._map[line]
                point_rigid = self._map[point]
                point_vertex = next(iter(point_rigid.vars))
                print("point_vertex", point_vertex)
                line_vertex = line_rigid.vertex
                line_normal = line_rigid.normal
                angle_hog = self._map[con]
                pv = Vector([1.0,0.0])
                lv = Vector.origin()
                ln = Vector([0.0,1.0])
                conf1 = Configuration({line_vertex:lv, line_normal:ln, point_vertex: 1.0*pv})
                conf2 = Configuration({line_vertex:lv, line_normal:ln, point_vertex:-1.0*pv})
                self.dr.set(angle_hog, [conf1,conf2])
                diag_print("set "+str(angle_hog)+" to "+str(conf1),"GeometricSolver")
                diag_print("set "+str(angle_hog)+" to "+str(conf2),"GeometricSolver")
        else:
            raise Exception("unknown constraint type")

    def _update_variable(self, var):
        if isinstance(var, Point):
            self._update_point(var)
        elif isinstance(var, Line):
            self._update_line(var)
        else:
            # assume point - depricated
            self._update_point(var)

    def _update_point(self, variable):
        cluster = self._map[variable]
        proto = self.problem.get_prototype(variable)
        conf = Configuration({variable:proto})
        self.dr.set(cluster, [conf])

    def _update_line(self, variable):
        cluster = self._map[variable]
        proto = self.problem.get_prototype(variable)
        line_vertex = cluster.vertex
        line_normal = cluster.normal
        # determine vertex and normal prototype coordinates
        p1 = proto[0:2]
        p2 = proto[2:4]
        v = p1
        n = perp_2d(p2-p1)
        # update prototypes of created point variables
        if line_vertex in self._map:
            vertex_rigid = self._map[line_vertex]
            conf = Configuration({line_vertex: v})
            self.dr.set(vertex_rigid, [conf])
            diag_print("set "+str(vertex_rigid)+" to "+str(conf),"GeometricSolver")
        if line_normal in self._map:
            normal_rigid = self._map[line_normal]
            conf = Configuration({line_normal: n})
            self.dr.set(normal_rigid, [conf])
            diag_print("set "+str(normal_rigid)+" to "+str(conf),"GeometricSolver")
        # update line configuration
        conf = Configuration({line_vertex:v, line_normal:n})
        self.dr.set(cluster, [conf])
        diag_print("set "+str(cluster)+" to "+str(conf),"GeometricSolver")

    def _update_fix(self):
        if self.fixcluster:
            vars = self.fixcluster.vars
            map = {}
            for var in vars:
                map[var] = self.problem.get_fix(var).get_parameter()
            conf = Configuration(map)
            self.dr.set(self.fixcluster, [conf])
        else:
            diag_print("no fixcluster to update","geometric")
            pass

#class GeometricSolver


# ------------ GeometricDecomposition -------------

class GeometricDecomposition:
    """Represents the result of solving a GeometricProblem. A cluster is a list of
       point variable names and a list of solutions for
       those variables. A solution is a dictionary mapping variable names to
       points. The cluster also keeps a list of sub-clusters (GeometricDecomposition)
       and a set of flags, indicating incidental/structural
       under/overconstrained

       instance attributes:
            variables       - a list of int variable names
            solutions       - a list of solutions. Each solution is a dictionary
                              mapping variable names to vectors.
            subs            - a list of sub-clusters
            flag            - value                 meaning
                              OK                    well constrained
                              INCONSISTENT                incicental over-constrained
                              DEGENERATE               incidental under-constrained
                              OVERCONSTRAINED                structural overconstrained
                              UNDERCONSTRAINED               structural underconstrained
                              UNSOLVED              unsolved (no input values)
                              EMPTY                 empty (no variables)
       """

    OK = "well-constrained"
    UNSOLVED = "unsolved"
    EMPTY = "empty"
    OVERCONSTRAINED = "over-constrained"
    UNDERCONSTRAINED = "under-constrained"
    INCONSISTENT = "inconsistent"
    DEGENERATE = "degenerate"

    def __init__(self, variables):
        """initialise an empty new cluster"""
        self.variables = frozenset(variables)
        self.solutions = []
        self.subs = []
        self.flag = GeometricDecomposition.OK

    def __eq__(self, other):
        if isinstance(other, GeometricDecomposition):
            return self.variables == other.variables
        else:
            return False

    def __hash__(self):
        return hash(self.variables)

    def __str__(self):
        return self._str_recursive()

    def _str_recursive(result, depth=0, done=None):
        # create indent
        spaces = ""
        for i in range(depth):
            spaces = spaces + "|"

        # make done
        if done == None:
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

        # pritn cluster
        s = spaces + "cluster " + str(list(result.variables)) + " " + str(result.flag) + " " + str(len(result.solutions)) + " solutions\n" + s

        return s
    # def


# ------------------- variable type -------------

class GeometricVariable:
    """Abstract base class for geometric variabes (Point, Line, etc)
        A geometric variable is identified by its name attibute and its type.
        It is hasable so it can be used in sets etc.
    """

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.__class__.__name__+"("+repr(self.name)+")"


class Point(GeometricVariable):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "Point("+str(self.name)+")"

class Line(GeometricVariable):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "Line("+str(self.name)+")"

# --------------------- constraint types --------------------


class ParametricConstraint(Constraint, Notifier):
    """A constraint with a parameter and notification when parameter changes"""

    def __init__(self):
        """initialize ParametricConstraint"""
        Notifier.__init__(self)
        self._value = None

    def get_parameter(self):
        """get parameter value"""
        return self._value

    def set_parameter(self,value):
        """set parameter value and notify any listeners"""
        self._value = value
        self.send_notify(("set_parameter", value))

class FixConstraint(ParametricConstraint):
    """A constraint to fix a point relative to the coordinate system"""

    def __init__(self, var, pos):
        """Create a new FixConstraint instance

           keyword args:
            var    - a point variable name
            pos    - the position parameter
        """
        ParametricConstraint.__init__(self)
        self._variables = [var]
        self.set_parameter(Vector(pos))

    def satisfied(self, mapping):
        """return True iff mapping from variable names to points satisfies constraint"""
        point = mapping[self._variables[0]]
        if len(point) != len(self._value):
            diag_print("warning: FixConstraint.satisfied: vectors of unequal length", "geometric.FixConstraint.satisfied")
            return False
        result = True;
        for i in range(len(self._value)):
            result &= tol_eq(point[i], self._value[i])
        return result

    def __str__(self):
        return "FixConstraint("\
            +str(self._variables[0])+"="\
            +str(self._value)+")"

class DistanceConstraint(ParametricConstraint):
    """A constraint on the Euclidean distance between two points"""

    def __init__(self, a, b, dist):
        """Create a new DistanceConstraint instance

           keyword args:
            a    - a point variable name
            b    - a point variable name
            dist - the distance parameter value
        """
        ParametricConstraint.__init__(self)
        self._variables = [a,b]
        self.set_parameter(dist)

    def satisfied(self, mapping):
        """return True iff mapping from variable names to points satisfies constraint"""
        a = mapping[self._variables[0]]
        b = mapping[self._variables[1]]
        result = tol_eq(distance_2p(a,b), abs(self._value))
        return result

    def __str__(self):
        return "DistanceConstraint("\
            +str(self._variables[0])+","\
            +str(self._variables[1])+"="\
            +str(self._value)+")"

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
        ParametricConstraint.__init__(self)
        self._variables = [a,b,c]
        self.set_parameter(ang)

    def satisfied(self, mapping):
        """return True iff mapping from variable names to points satisfies constraint"""
        a = mapping[self._variables[0]]
        b = mapping[self._variables[1]]
        c = mapping[self._variables[2]]
        ang = angle_3p(a,b,c)
        if ang == None:
            # if the angle is indeterminate, its probably ok.
            result = True
        else:
            # in 3d, ignore the sign of the angle
            if len(a) >= 3:
                cmp = abs(self._value)
            else:
                cmp = self._value
            result = tol_eq(ang, cmp)
        if result == False:
            diag_print("measured angle = "+str(ang)+", parameter value = "+str(cmp), "satisfied")
        return result

    def __str__(self):
        return "AngleConstraint("\
            +str(self._variables[0])+","\
            +str(self._variables[1])+","\
            +str(self._variables[2])+"="\
            +str(self._value)+")"

class RigidConstraint(ParametricConstraint):
    """A constraint to set the relative position of a set of points"""

    def __init__(self, conf):
        """Create a new DistanceConstraint instance

           keyword args:
            conf    - a Configuration
        """
        ParametricConstraint.__init__(self)
        self._variables = list(conf.vars())
        self.set_parameter(conf.copy())

    def satisfied(self, mapping):
        """return True iff mapping from variable names to points satisfies constraint"""
        result = True
        conf = self._value
        for index in range(1,len(self._variables)-1):
            p1 = mapping[self._variables[index-1]]
            p2 = mapping[self._variables[index]]
            p3 = mapping[self._variables[index+1]]
            c1 = conf.map[self._variables[index-1]]
            c2 = conf.map[self._variables[index]]
            c3 = conf.map[self._variables[index+1]]
            result &= tol_eq(distance_2p(p1,p2), distance_2p(c1,c2))
            result &= tol_eq(distance_2p(p1,p3), distance_2p(c1,c3))
            result &= tol_eq(distance_2p(p2,p3), distance_2p(c2,c3))
        return result

    def __str__(self):
        return "RigidConstraint("+str(self._variables)+")"

class ClockwiseConstraint (SelectionConstraint):
    """A selection constraint for 3 points to have a clockwise orientation (not co-linear!)"""
    def __init__(self, v1, v2, v3):
        SelectionConstraint.__init__(self, is_clockwise, [v1,v2,v3])

class CounterClockwiseConstraint (SelectionConstraint):
    """A selection constraint for 3 points to have a counter-clockwise orientation (not co-linear!)"""
    def __init__(self, v1, v2, v3):
        SelectionConstraint.__init__(self, is_counterclockwise, [v1,v2,v3])

class NotClockwiseConstraint (SelectionConstraint):
    """A selection constraint for 3 points to not have a clockwise orientation (i.e. counter-clockwise or co-linear!)"""
    def __init__(self, v1, v2, v3):
        SelectionConstraint.__init__(self, fnot(is_clockwise), [v1,v2,v3])

class NotCounterClockwiseConstraint (SelectionConstraint):
    """A selection constraint for 3 points to not have a counter-clockwise orientation (i.e. clockwise or co-linear!)"""
    def __init__(self, v1, v2, v3):
        SelectionConstraint.__init__(self, fnot(is_counterclockwise), [v1,v2,v3])

class RightHandedConstraint (SelectionConstraint):
    """A selection constraint for 4 points to have a right-handed orientation (not co-planar!)"""
    def __init__(self, v1, v2, v3, v4):
        SelectionConstraint.__init__(self, is_right_handed, [v1,v2,v3,v4])

class LeftHandedConstraint (SelectionConstraint):
    """A selection constraint for 4 points to have a left-handed orientation (not co-planar!)"""
    def __init__(self, v1, v2, v3, v4):
        SelectionConstraint.__init__(self, is_left_handed, [v1,v2,v3,v4])

class NotRightHandedConstraint (SelectionConstraint):
    """A selection constraint for 4 points to not have a right-handed orientation, i.e. left-handed or co-planar"""
    def __init__(self, v1, v2, v3, v4):
        SelectionConstraint.__init__(self, fnot(is_right_handed), [v1,v2,v3,v4])

class NotLeftHandedConstraint (SelectionConstraint):
    """A selection constraint for 4 points to not have a left-handed orientation, i.e. right-handed or co-planar"""
    def __init__(self, v1, v2, v3, v4):
        SelectionConstraint.__init__(self, fnot(is_left_handed), [v1,v2,v3,v4])

class CoincidenceConstraint(Constraint):
    """defines a coincidence between a point and another geometricvariable (i.e. point, line, plane)"""
    def __init__(self, point, geometry):
        assert isinstance(point, Point)
        assert isinstance(geometry, GeometricVariable)
        self._point = point
        self._geometry = geometry
        self._variables = [point, geometry]

    def satisfied(self, mapping):
        """return True iff mapping from variable names to points satisfies constraint"""
        if isinstance(self._geometry, Point):
            p1 = mapping[self._point]
            p2 = mapping[self._geometry]
            return tol_eq(distance_2p(p1,p2),0)
        elif isinstance(self._geometry, Line):
            p = mapping[self._point]
            l = mapping[self._geometry]
            if len(l)==4:   #2D
                p1 = l[0:2]
                p2 = l[2:4]
            elif len(l)==6:   # 3D
                p1 = l[0:3]
                p2 = l[3:6]
            else:
                raise Exception("line has invalid number of values")
            d =  distance_point_line(p, p1, p2)
            if not tol_eq(d,0):
                diag_print("not satisfied "+ str(self)+" distance="+str(d),"CoincidenceConstraint")
                print("distance="+str(d),"CoincidenceConstraint")
            return tol_eq(d,0)
        else:
            raise Exception("unknown geometry type""")

    def __str__(self):
        return "CoincidenceConstraint("+str(self._point)+","+str(self._geometry)+")"


