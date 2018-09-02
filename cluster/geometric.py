"""Geometric constraint problem and solver. Uses ClusterSolver for solving
problems incrementally."""

import logging
import numpy as np

from .clsolver import PrototypeMethod, SelectionMethod
from .clsolver2D import ClusterSolver2D
from .cluster import *
from .configuration import Configuration
from .notify import Notifier, Listener
from .geometry import (Vector, angle_3p, distance_2p, distance_point_line, is_clockwise,
                       is_counterclockwise, perp_2d, tol_eq)
from .constraints import (DistanceConstraint, AngleConstraint, FixConstraint, RigidConstraint,
                          CoincidenceConstraint, SelectionConstraint)
from .primitives import Point, Line

LOGGER = logging.getLogger(__name__)


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

    def decomposition(self):
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
           cheaper but less informative than decomposition.
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
        n_configurations = len(configurations)
        solutions = []
        LOGGER.debug(f"mapping cluster '{drcluster}' with {n_configurations} configurations")
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
           Note: this method is cheaper but less informative than decomposition.
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
        # find coincident points
        points = list(self.problem.get_coincident_points(var))

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
        self._update_variable(line)

    def _rem_variable(self, var):
        if var in self._map:
            self.dr.remove(self._map[var])
            # Note: CLSolver automatically removes variables with no dependent clusters
            del self._map[var]

    def _add_constraint(self, con):
        if isinstance(con, AngleConstraint):
            # map to hedgdehog
            vars = list(con.variables)
            hog = Hedgehog(vars[1],[vars[0],vars[2]])
            self._map[con] = hog
            self._map[hog] = con
            self.dr.add(hog)
            # set configuration
            self._update_constraint(con)
        elif isinstance(con, DistanceConstraint):
            # map to rigid
            vars = list(con.variables)
            rig = Rigid([vars[0],vars[1]])
            self._map[con] = rig
            self._map[rig] = con
            self.dr.add(rig)
            # set configuration
            self._update_constraint(con)
        elif isinstance(con, RigidConstraint):
            # map to rigid
            vars = list(con.variables)
            rig = Rigid(vars)
            self._map[con] = rig
            self._map[rig] = con
            self.dr.add(rig)
            # set configuration
            self._update_constraint(con)
        elif isinstance(con, FixConstraint):
            if self.fixcluster != None:
                self.dr.remove(self.fixcluster)
            self.fixvars.append(con.variables[0])
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
            lines = [var for var in con.variables if isinstance(var,Line)]
            points = [var for var in con.variables if isinstance(var,Point)]
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
                    LOGGER.debug(f"mapped constraint '{con}' to cluster '{angle_hog}'")
                    self._update_constraint(con)
        else:
            raise Exception("unknown constraint type")
            pass

    def _rem_constraint(self, con):
        if isinstance(con,FixConstraint):
            if self.fixcluster != None:
                self.dr.remove(self.fixcluster)
            var = con.variables[0]
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
            vars = list(con.variables)
            v0 = vars[0]
            v1 = vars[1]
            v2 = vars[2]
            angle = con.angle
            p0 = Vector([1.0,0.0])
            p1 = Vector.origin()
            p2 = Vector([np.cos(angle), np.sin(angle)])
            conf = Configuration({v0:p0,v1:p1,v2:p2})
            self.dr.set(hog, [conf])
            assert con.satisfied(conf.map)
        elif isinstance(con, DistanceConstraint):
            # set configuration
            rig = self._map[con]
            vars = list(con.variables)
            v0 = vars[0]
            v1 = vars[1]
            dist = con.distance
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
            vars = list(con.variables)
            conf = con.configuration
            self.dr.set(rig, [conf])
            assert con.satisfied(conf.map)
        elif isinstance(con, FixConstraint):
            self._update_fix()
        elif isinstance(con, CoincidenceConstraint):
            lines = [var for var in con.variables if isinstance(var,Line)]
            points = [var for var in con.variables if isinstance(var,Point)]
            if len(lines)==1 and len(points)==1:
                line = next(iter(lines))
                point = next(iter(points))
                line_rigid = self._map[line]
                point_rigid = self._map[point]
                point_vertex = next(iter(point_rigid.vars))
                LOGGER.debug(f"point_vertex: {point_vertex}")
                line_vertex = line_rigid.vertex
                line_normal = line_rigid.normal
                angle_hog = self._map[con]
                pv = Vector([1.0,0.0])
                lv = Vector.origin()
                ln = Vector([0.0,1.0])
                conf1 = Configuration({line_vertex:lv, line_normal:ln, point_vertex: 1.0*pv})
                conf2 = Configuration({line_vertex:lv, line_normal:ln, point_vertex:-1.0*pv})
                self.dr.set(angle_hog, [conf1,conf2])
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
        if line_normal in self._map:
            normal_rigid = self._map[line_normal]
            conf = Configuration({line_normal: n})
            self.dr.set(normal_rigid, [conf])
        # update line configuration
        conf = Configuration({line_vertex:v, line_normal:n})
        self.dr.set(cluster, [conf])

    def _update_fix(self):
        if self.fixcluster:
            vars = self.fixcluster.vars
            map = {}
            for var in vars:
                map[var] = self.problem.get_fix(var).position
            conf = Configuration(map)
            self.dr.set(self.fixcluster, [conf])
        else:
            LOGGER.debug("no fixcluster to update")
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
