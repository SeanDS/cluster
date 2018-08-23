import logging
import copy
import numpy as np

from .event import Observable, Observer, UnknownEventException
from .graph import Graph
from .geometry import Vector
from .methods import Method, Variable, MethodGraph, PrototypeMethod
from .methods.merges import (Merge, MergeRRR, MergeRRH, MergeRHR, MergePR, MergeRR, MergeRH,
                             MergeBH, BalloonFromHogs, BalloonMerge, BalloonRigidMerge, MergeHogs)
from .methods.derives import SubHog, RigidToHog, BalloonToHog
from .configuration import Configuration
from .decomposition import Decomposition
from .clusters import Distance, Angle, Rigid, Hedgehog, Balloon
from .constraints import DistanceConstraint, AngleConstraint, FixConstraint

LOGGER = logging.getLogger(__name__)

class GeometricSolver(Observer, Observable):
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

        # map from variables to their associated rigid, single-variable point clusters
        self._variable_point_map = {}

        # map from constraints to their associated clusters
        self._constraint_cluster_map = {}

        # register listeners
        self.constraint_graph.add_observer(self)
        self.problem.add_observer(self)

        # create an initial fix cluster
        self.fixvars = []
        self.fixcluster = None

        # map current constraint graph variables
        for var in self.constraint_graph.variables:
            self._add_variable(var)

        # list of constraints from graph
        constraints = self.constraint_graph.constraints

        # add fixed constraints first (keeps fixed points where they are; keeps solutions valid)
        for constraint in constraints:
            if isinstance(constraint, FixConstraint):
                self._add_constraint(constraint)

        # add distances next (doing this before angles results in simpler and faster decompositions)
        for constraint in constraints:
            if isinstance(constraint, DistanceConstraint):
                self._add_constraint(constraint)

        # add everything else
        for constraint in constraints:
            if not isinstance(constraint, FixConstraint) and not isinstance(constraint, DistanceConstraint):
                self._add_constraint(constraint)

    def get_constrainedness(self):
        # get cluster solver's top level solution(s)
        toplevel = self.solver.top_level()

        if len(toplevel) > 1:
            return "under-constrained"

        elif len(toplevel) == 1:
            cluster = toplevel[0]

            if isinstance(cluster, Rigid):
                configurations = self.solver.get_configurations(cluster)

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
            geocluster = Decomposition()

            # create map from geometric cluster to drcluster (and vice versa)
            mapping[drcluster] = geocluster
            mapping[geocluster] = drcluster

            # determine variables
            for variables in drcluster.variables:
                geocluster.variables.append(variables)

            # determine solutions
            solutions = self.solver.get_configurations(drcluster)

            underconstrained = False

            if solutions is not None:
                for solution in solutions:
                    geocluster.solutions.append(solution.mapping)

                    if solution.underconstrained:
                        underconstrained = True

            # determine flag
            if drcluster.overconstrained:
                geocluster.flag = Decomposition.S_OVER
            elif len(geocluster.solutions) == 0:
                geocluster.flag = Decomposition.I_OVER
            elif underconstrained:
                geocluster.flag = Decomposition.I_UNDER
            else:
                geocluster.flag = Decomposition.OK

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
            result = Decomposition()

            result.variables = []
            result.subs = []
            result.solutions = []
            result.flag = Decomposition.UNSOLVED
        elif len(rigids) == 1:
            # structurally well constrained
            result = mapping[rigids[0]]
        else:
            # structurally underconstrained cluster
            result = Decomposition()
            result.flag = Decomposition.S_UNDER

            for rigid in rigids:
                result.subs.append(mapping[rigid])

        return result

    @property
    def variables(self):
        return self._variable_point_map.keys()

    @property
    def constraints(self):
        return self._constraint_cluster_map.keys()

    def _handle_event(self, event):
        """Take notice of changes in constraint graph"""
        # constraint graph events
        if event.message == "add_constraint":
            self._add_constraint(event.data["constraint"])
        elif event.message == "remove_constraint":
            self._remove_constraint(event.data["constraint"])
        elif event.message == "add_variable":
            self._add_variable(event.data["variable"])
        elif event.message == "remove_variable":
            self._remove_variable(event.data["variable"])
        # problem events
        elif event.message == "set_point":
            self._update_variable(event.data["variable"])
        elif event.message == "set_parameter":
            if "constraint" in event.data:
                # a constraint's parameter has been changed
                self._update_constraint(event.data["constraint"])
        else:
            raise UnknownEventException(event)

    def _add_variable(self, variable):
        if variable in self.variables:
            raise ValueError(f"variable '{variable}' already in problem")

        LOGGER.debug(f"adding variable '{variable}")

        rigid = Rigid([variable])

        self._variable_point_map[variable] = rigid
        self.solver.add(rigid)

        self._update_variable(variable)

    def _remove_variable(self, variable):
        if variable not in self.variables:
            raise ValueError(f"variable '{variable}' not in problem")

        LOGGER.debug(f"removing variable '{variable}'")
        self.solver.remove(self._variable_point_map[variable])
        del(self._variable_point_map[variable])

    def _add_constraint(self, constraint):
        LOGGER.debug(f"adding constraint '{constraint}'")

        if isinstance(constraint, (DistanceConstraint, AngleConstraint)):
            cluster = constraint.default_cluster()

            # map constraint to cluster
            self._constraint_cluster_map[constraint] = cluster

            self.solver.add(cluster)

            # set configuration
            self._update_constraint(constraint)
        elif isinstance(constraint, FixConstraint):
            if self.fixcluster is not None:
                # remove existing fix cluster
                self.solver.remove(self.fixcluster)
                self.fixcluster = None

            self.fixvars.extend(constraint.variables)

            # check if there are more fixed variables than dimensions
            if len(self.fixvars) >= 2:
                # TODO: check that a Rigid() is always correct to use here
                self.fixcluster = Rigid(self.fixvars)
                self.solver.add(self.fixcluster)
                self.solver.set_root(self.fixcluster)

            self._update_fix()
        else:
            raise ValueError(f"unrecognised constraint '{constraint}'")

    def _remove_constraint(self, constraint):
        LOGGER.debug(f"removing constraint '{constraint}'")

        if isinstance(constraint, FixConstraint):
            if self.fixcluster is not None:
                self.solver.remove(self.fixcluster)

            fixed_point = self.solver.get_configurations(constraint.point)

            if fixed_point in self.fixvars:
                self.fixvars.remove(fixed_point)

            # check if there are less fixed variables than dimensions
            if len(self.fixvars) < 2:
                self.fixcluster = None
            else:
                self.fixcluster = Rigid(self.fixvars)
                self.solver.add(self.fixcluster)
                self.solver.set_root(self.fixcluster)
        elif constraint in self.constraints:
            self.solver.remove(self._constraint_cluster_map[constraint])
            del(self._constraint_cluster_map[constraint])
        else:
            raise ValueError(f"constraint '{constraint}' not in problem")

    def _update_constraint(self, constraint):
        if isinstance(constraint, FixConstraint):
            self._update_fix()
        else:
            cluster = self._constraint_cluster_map[constraint]
            configuration = constraint.default_config(self.problem)

            self.solver.set_configurations(cluster, [configuration])

            assert constraint.satisfied(configuration.mapping)

    def _update_fix(self):
        if not self.fixcluster:
            LOGGER.info("no fix cluster to update")
            return

        mapping = {variable: self.problem.get_fix(variable).value for variable in self.fixcluster.variables}
        conf = Configuration(mapping)

        self.solver.set_configurations(self.fixcluster, [conf])

    def _update_variable(self, variable):
        LOGGER.debug(f"updating variable '{variable}'")

        # point cluster
        point = self._variable_point_map[variable]

        # prototype point
        proto = self.problem.get_point(variable)

        conf = Configuration({variable: proto})

        self.solver.set_configurations(point, [conf])

    def __str__(self):
        return f"GeometricSolver({self.problem})"


class ClusterSolver:
    """A generic 2D geometric constraint solver

    Finds a generic solution for problems formulated by cluster constraints.

    Constraints are Clusters: Rigids, Hedgehogs and Balloons.
    Cluster are added and removed using the add and remove methods.
    After adding each Cluster, the solver tries to merge it with
    other clusters, resulting in new Clusters and Methods.

    For each Cluster a set of Configurations can be set using the
    set method. Configurations are propagated via Methods and can
    be retrieved with the get method."""

    def __init__(self, *args, **kwargs):
        """Create a new empty solver"""
        super().__init__(*args, **kwargs)

        self._graph = Graph()
        self._graph.add_node("_root")
        self._graph.add_node("_toplevel")
        self._graph.add_node("_variables")
        self._graph.add_node("_distances")
        self._graph.add_node("_angles")
        self._graph.add_node("_rigids")
        self._graph.add_node("_hedgehogs")
        self._graph.add_node("_balloons")
        self._graph.add_node("_methods")

        # queue of new objects to process
        self._new = []

        # methodgraph
        self._mg = MethodGraph()

    def variables(self):
        """get list of variables"""
        return self._graph.successors("_variables")

    def distances(self):
        """get list of distances"""
        return self._graph.successors("_distances")

    def angles(self):
        """get list of angles"""
        return self._graph.successors("_angles")

    def rigids(self):
        """get list of rigids"""
        return self._graph.successors("_rigids")

    def hedgehogs(self):
        """get list of hedgehogs"""
        return self._graph.successors("_hedgehogs")

    def balloons(self):
        """get list of balloons"""
        return self._graph.successors("_balloons")

    def methods(self):
        """get list of methods"""
        return self._graph.successors("_methods")

    def top_level(self):
        """get top-level objects"""
        return self._graph.successors("_toplevel")

    def is_top_level(self, obj):
        return self._graph.has_edge("_toplevel", obj)

    def add(self, cluster):
        """Add a cluster.

           arguments:
              cluster: A Rigid
           """

        LOGGER.debug("Adding cluster %s", cluster)

        self._add_cluster(cluster)
        self._process_new()

    def remove(self, cluster):
        """Remove a cluster

        All dependent objects are also removed.
        """

        self._remove(cluster)
        self._process_new()

    def set_configurations(self, cluster, configurations):
        """Associate a list of configurations with a cluster"""
        self._mg.set_node_value(cluster, configurations)

    def get_configurations(self, cluster):
        """Return a set of configurations associated with a cluster"""
        return self._mg.get_node_value(cluster)

    def set_root(self, rigid):
        """Make given rigid cluster the root cluster

           arguments:
              cluster: A Rigid
           """
        LOGGER.debug("Setting root to %s", rigid)
        self._graph.remove_node("_root")
        self._graph.add_edge("_root", rigid)

    def find_dependent(self, obj):
        """Return a list of objects that depend on given object directly."""
        l = self._graph.successors(obj)
        return [x for x in l if self._graph.get_edge_value(obj, x) == "dependency"]

    def find_depends(self, obj):
        """Return a list of objects that the given object depends on directly"""
        l = self._graph.predecessors(obj)
        return [x for x in l if self._graph.get_edge_value(x, obj) == "dependency"]

    def contains(self, obj):
        return self._graph.has_node(obj)

    def _add_dependency(self, on, dependend):
        """Add a dependence for second object on first object"""
        self._graph.add_edge(on, dependend, value="dependency")

    def _add_to_group(self, group, obj):
        """Add object to group"""
        self._graph.add_edge(group, obj, value="contains")

    def _add_needed_by(self, needed, by):
        """Add relation 'needed' object is needed 'by'"""
        self._graph.add_edge(needed, by, value="needed_by")

    def _objects_that_need(self, needed):
        """Return objects needed by given object"""
        return [x for x in self._graph.successors(needed) if self._graph.get_edge_value(needed, x) == "needed_by"]

    def _objects_needed_by(self, needer):
        """Return objects needed by given object"""
        return [x for x in self._graph.predecessors(needer) if self._graph.get_edge_value(x, needer) == "needed_by"]

    def _add_top_level(self, obj):
        self._graph.add_edge("_toplevel", obj)
        self._new.append(obj)

    def _rem_top_level(self, obj):
        self._graph.remove_edge("_toplevel", obj)

        if obj in self._new:
            self._new.remove(obj)

    def _remove(self, obj):
        # find all indirectly dependend objects
        to_delete = [obj] + self._find_descendent(obj)

        to_restore = set()

        # remove all objects
        for item in to_delete:
            # delete it from graph
            LOGGER.debug("Deleting %s", item)
            self._graph.remove_node(item)

            # remove from _new list
            if item in self._new:
                self._new.remove(item)

            # remove from methodgraph
            if isinstance(item, Method):
                # note: method may have been removed because variable removed
                try:
                    self._mg.remove_method(item)
                except:
                    pass
            elif isinstance(item, Variable):
                self._mg.remove_variable(item)

        # restore top level (also added to _new)
        for cluster in to_restore:
            if self._graph.has_node(cluster):
                self._add_top_level(cluster)

        # re-solve
        self._process_new()

    def _find_descendent(self,v):
        """find all descendend objects of v (directly or indirectly dependent)"""

        front = [v]
        result = {}

        while len(front) > 0:
            x = front.pop()

            if x not in result:
                result[x] = 1
                front += self.find_dependent(x)

        del(result[v])

        return list(result)

    def _add_variable(self, var):
        """Add a variable if not already in system

           arguments:
              var: any hashable object
        """

        if not self._graph.has_node(var):
            LOGGER.debug("Adding variable %s", var)
            self._add_to_group("_variables", var)

    def _add_cluster(self, cluster):
        if isinstance(cluster, Rigid):
            self._add_rigid(cluster)
        elif isinstance(cluster, Hedgehog):
            self._add_hog(cluster)
        elif isinstance(cluster, Balloon):
            self._add_balloon(cluster)
        else:
            raise Exception("unsupported type {0}".format(type(cluster)))

    def _add_rigid(self, newcluster):
        """add a rigid cluster if not already in system"""

        LOGGER.debug("Adding rigid %s", newcluster)

        # check if not already exists
        if self._graph.has_node(newcluster):
            raise Exception("rigid already in clsolver")

        # update graph
        self._add_to_group("_rigids", newcluster)

        for variable in newcluster.variables:
            self._add_variable(variable)
            self._add_dependency(variable, newcluster)

        # if there is no root cluster, this one will be it
        if len(list(self._graph.successors("_root"))) == 0:
            self._graph.add_edge("_root", newcluster)

        # add to top level
        self._add_top_level(newcluster)

        # add to methodgraph
        self._mg.add_variable(newcluster)

    def _add_hog(self, hog):
        LOGGER.debug("Adding hedgehog: %s", hog)

        # check if not already exists
        if self._graph.has_node(hog):
            raise Exception("hedgehog already in clsolver")

        # update graph
        self._add_to_group("_hedgehogs",hog)

        for var in list(hog.xvars) + [hog.cvar]:
            self._add_variable(var)
            self._add_dependency(var, hog)

        # add to top level
        self._add_top_level(hog)

        # add to methodgraph
        self._mg.add_variable(hog)

    def _add_balloon(self, newballoon):
        """add a cluster if not already in system"""
        LOGGER.debug("Adding balloon %s", newballoon)

        # check if not already exists
        if self._graph.has_node(newballoon):
            raise Exception("balloon already in clsolver")

        # update graph
        self._add_to_group("_balloons", newballoon)

        for variable in newballoon.variables:
            self._add_variable(variable)
            self._add_dependency(variable, newballoon)

        # add to top level
        self._add_top_level(newballoon)

         # add to methodgraph
        self._mg.add_variable(newballoon)

    def _add_merge(self, merge):
        # structural check that method has one output
        if len(merge.outputs) != 1:
            raise Exception("merge number of outputs != 1")

        # get only output of merge
        output = merge.outputs[0]

        # consistent merge?
        consistent = True

        # check if all combinations of the inputs are consistent
        for i1 in range(len(merge.inputs)):
            for i2 in range(i1 + 1, len(merge.inputs)):
                c1 = merge.inputs[i1]
                c2 = merge.inputs[i2]

                consistent = consistent and self._is_consistent_pair(c1, c2)

        # set merge consistency
        merge.consistent = consistent

        # merge is overconstrained if all of the inputs are overconstrained
        overconstrained = not consistent
        for cluster in merge.inputs:
            overconstrained = overconstrained and cluster.overconstrained
        output.overconstrained = overconstrained

        # add to graph
        self._add_cluster(output)
        self._add_method(merge)

        # remove inputs from toplevel
        for merge_input in merge.inputs:
            self._rem_top_level(merge_input)

        # add prototype selection method
        self._add_prototype_selector(merge)

    def _add_prototype_selector(self, merge):
        incluster = merge.outputs[0]
        constraints = merge.prototype_constraints()

        if len(constraints) == 0:
            return

        variables = set()

        for constraint in constraints:
            variables.update(constraint.variables)

        selclusters = []

        for variable in variables:
            clusters = self._graph.successors(variable)
            clusters = [c for c in clusters if isinstance(c, Rigid)]
            clusters = [c for c in clusters if len(c.variables) == 1]

            if len(clusters) != 1:
                raise Exception(f"no prototype cluster for variable '{variable}'")

            selclusters.append(clusters[0])

        outcluster = copy.copy(incluster)

        # Rick 20090519 - copy does not copy structural overconstrained flag?
        outcluster.overconstrained = incluster.overconstrained

        selector = PrototypeMethod(incluster, selclusters, outcluster, constraints)

        self._add_cluster(outcluster)
        self._add_method(selector)
        self._rem_top_level(incluster)

    def _add_method(self, method):
        LOGGER.debug("Adding method %s", method)

        self._add_to_group("_methods", method)

        for obj in method.inputs:
            self._add_dependency(obj, method)

        for obj in method.outputs:
            self._add_dependency(method, obj)
            self._add_dependency(obj, method)

        self._mg.add_method(method)

    def _process_new(self):
        while len(self._new) > 0:
            new_obj = self._new.pop()
            LOGGER.debug("Searching from %s", new_obj)

            # do search
            search = self._search(new_obj)

            if search and self.is_top_level(new_obj):
                # maybe more rules applicable.... push back on stack
                self._new.append(new_obj)

    def _search(self, new_cluster):
        if isinstance(new_cluster, Rigid):
            self._search_from_rigid(new_cluster)
        elif isinstance(new_cluster, Hedgehog):
            self._search_from_hog(new_cluster)
        elif isinstance(new_cluster, Balloon):
            self._search_from_balloon(new_cluster)
        else:
            raise Exception("don't know how to search from {0}".format(new_cluster))

    def _search_from_balloon(self, balloon):
        if self._search_absorb_from_balloon(balloon):
            return
        if self._search_balloon_from_balloon(balloon):
            return
        if self._search_rigid_from_balloon(balloon):
            return
        self._search_hogs_from_balloon(balloon)

    def _search_from_hog(self, hog):
        if self._search_absorb_from_hog(hog):
            return
        if self._search_merge_from_hog(hog):
            return
        if self._search_balloon_from_hog(hog):
            return
        self._search_hogs_from_hog(hog)

    def _search_from_rigid(self, rigid):
        if self._search_absorb_from_rigid(rigid):
            return
        if self._search_balloonrigidmerge_from_rigid(rigid):
            return
        if self._search_merge_from_rigid(rigid):
            return
        self._search_hogs_from_rigid(rigid)

    def _search_absorb_from_balloon(self, balloon):
        for cvar in balloon.variables:
            # find all incident hedgehogs
            hogs = self._find_hogs(cvar)

            # determine shared vertices per hedgehog
            for hog in hogs:
                shared = set(hog.xvars).intersection(balloon.variables)

                if len(shared) == len(hog.xvars):
                    return self._merge_balloon_hog(balloon, hog)

    def _search_absorb_from_rigid(self, cluster):
        for cvar in cluster.variables:
            # find all incident hedgehogs
            hogs = self._find_hogs(cvar)

            # determine shared vertices per hedgehog
            for hog in hogs:
                shared = set(hog.xvars).intersection(cluster.variables)

                if len(shared) == len(hog.xvars):
                    return self._merge_rigid_hog(cluster, hog)

    def _search_absorb_from_hog(self, hog):
        dep = self.find_dependent(hog.cvar)

        # case BH (overconstrained):
        balloons = [x for x in dep if isinstance(x, Balloon) and self.is_top_level(x)]
        sharecx = [x for x in balloons if len(set(hog.xvars).intersection(x.variables)) >= 1]

        for balloon in sharecx:
            sharedcx = set(balloon.variables).intersection(hog.xvars)

            if len(sharedcx) == len(hog.xvars):
                return self._merge_balloon_hog(balloon, hog)

        # case CH (overconstrained)
        clusters = [x for x in dep if isinstance(x, Rigid) and self.is_top_level(x)]
        sharecx = [x for x in clusters if len(set(hog.xvars).intersection(x.variables)) >= 1]

        for cluster in sharecx:
            sharedcx = set(cluster.variables).intersection(hog.xvars)

            if len(sharedcx) == len(hog.xvars):
                return self._merge_rigid_hog(cluster, hog)

    def _find_balloons(self, variables):
        balloons = set()

        for var in variables:
            deps = self.find_dependent(var)
            balls = [x for x in deps if isinstance(x, Balloon)]
            balloons = balloons.intersection(balls)

        return balloons

    def _make_balloon(self, var1, var2, var3, hog1, hog2):
        LOGGER.debug("Making balloon %s, %s, %s", var1, var2, var3)

        # derive sub-hogs if necessary
        variables = set([var1, var2, var3])

        if len(hog1.xvars) > 2:
            hog1 = self._derive_subhog(hog1, variables.intersection(hog1.xvars))

        if len(hog2.xvars) > 2:
            hog2 = self._derive_subhog(hog2, variables.intersection(hog2.xvars))

        # create balloon
        balloon = Balloon([var1, var2, var3])

        # create balloon method
        balloon_method = BalloonFromHogs(hog1, hog2, balloon)

        # add the new merge
        self._add_merge(balloon_method)

        return balloon

    def _search_balloon_from_hog(self, hog):
        new_balloons = []

        var1 = hog.cvar

        for var2 in hog.xvars:
            hogs = self._find_hogs(var2)

            for hog2 in hogs:
                if var1 in hog2.xvars:
                    for var3 in hog2.xvars:
                        if var3 != var2 and var3 in hog.xvars:
                            if not self._known_angle(var1, var3, var2):
                                new_balloons.append(self._make_balloon(var1, var2, var3, hog, hog2))

        if len(new_balloons) > 0:
            return new_balloons

        return None

    def _search_balloon_from_balloon(self, balloon):
        # map from adjacent balloons to variables shared with input balloon
        mapping = {}

        for var in balloon.variables:
            deps = self.find_dependent(var)

            balloons = [x for x in deps if isinstance(x, Balloon)]
            balloons = [x for x in balloons if self.is_top_level(x)]

            for bal2 in balloons:
                if bal2 != balloon:
                    if bal2 in mapping:
                        mapping[bal2].update([var])
                    else:
                        mapping[bal2] = set([var])

        for bal2 in mapping:
            if len(mapping[bal2]) >= 2:
                return self._merge_balloons(balloon, bal2)

        return None

    def _search_rigid_from_balloon(self, balloon):
        LOGGER.debug("Searching for rigid from balloon")

        # map from adjacent clusters to variables shared with input balloon
        mapping = {}

        for var in balloon.variables:
            deps = self.find_dependent(var)

            clusters = [x for x in deps if isinstance(x, Rigid) or isinstance(x, Distance)]
            clusters = [x for x in clusters if self.is_top_level(x)]

            for c in clusters:
                if c in mapping:
                    mapping[c].update([var])
                else:
                    mapping[c] = set([var])

        for cluster in mapping:
            if len(mapping[cluster]) >= 2:
                return self._merge_balloon_rigid(balloon, cluster)

        return None

    def _search_balloonrigidmerge_from_rigid(self, rigid):
        LOGGER.debug("Searching for balloon-rigid merge from rigid")

        # map from adjacent clusters to variables shared with input balloon
        mapping = {}

        for var in rigid.variables:
            deps = self.find_dependent(var)

            balloons = [x for x in deps if isinstance(x, Balloon)]
            balloons = [x for x in balloons if self.is_top_level(x)]

            for b in balloons:
                if b in mapping:
                    mapping[b].update([var])
                else:
                    mapping[b] = set([var])

        for balloon in mapping:
            if len(mapping[balloon]) >= 2:
                return self._merge_balloon_rigid(balloon, rigid)

        return None

    def _merge_balloons(self, balloon_a, balloon_b):
        # create new balloon and merge method

        # get variables in both balloons
        variables = set(balloon_a.variables).union(balloon_b.variables)

        # create new balloon cluster
        new_balloon = Balloon(variables)

        # add new balloon merge
        self._add_merge(BalloonMerge(balloon_a, balloon_b, new_balloon))

        return new_balloon

    def _merge_balloon_rigid(self, balloon, rigid):
        # create new rigid and method

        # get variables in both the balloon and the rigid
        variables = set(balloon.variables).union(rigid.variables)

        # create new rigid cluster
        new_cluster = Rigid(list(variables))

        # add new balloon-rigid merge
        self._add_merge(BalloonRigidMerge(balloon, rigid, new_cluster))

        return new_cluster

    def _find_hogs(self, cvar):
        deps = self.find_dependent(cvar)

        hogs = [x for x in deps if isinstance(x, Hedgehog)]
        hogs = [x for x in hogs if x.cvar == cvar]
        hogs = [x for x in hogs if self.is_top_level(x)]

        return hogs

    def _make_hog_from_rigid(self, cvar, rigid):
        # outer variables of hedgehog are the rigid's variables
        xvars = set(rigid.variables)

        # remove the central value of the hedgehog from the list
        xvars.remove(cvar)

        # create the new hedgehog from the central and outer variables
        hog = Hedgehog(cvar, xvars)

        # add the hedgehog
        self._add_hog(hog)

        # add new rigid-to-hedgehog method
        self._add_method(RigidToHog(rigid, hog))

        return hog

    def _make_hog_from_balloon(self, cvar, balloon):
        xvars = set(balloon.variables)

        xvars.remove(cvar)

        hog = Hedgehog(cvar, xvars)

        self._add_hog(hog)

        method = BalloonToHog(balloon, hog)

        self._add_method(method)

        return hog

    def _search_hogs_from_balloon(self, newballoon):
        if len(newballoon.variables) <= 2:
            return None

        # create/merge hogs
        for cvar in newballoon.variables:
            # potential new hog
            xvars = set(newballoon.variables)

            xvars.remove(cvar)

            # find all incident hogs
            hogs = self._find_hogs(cvar)

            # determine shared vertices per hog
            for hog in hogs:
                shared = set(hog.xvars).intersection(xvars)
                if len(shared) >= 1 and len(shared) < len(hog.xvars) and len(shared) < len(xvars):
                    tmphog = Hedgehog(cvar, xvars)
                    if not self._graph.has_node(tmphog):
                        newhog = self._make_hog_from_balloon(cvar, newballoon)
                        self._merge_hogs(hog, newhog)

    def _search_hogs_from_rigid(self, newcluster):
        if len(newcluster.variables) <= 2:
            return None

        # create/merge hogs
        for cvar in newcluster.variables:
            # potential new hog
            xvars = set(newcluster.variables)
            xvars.remove(cvar)

            # find all incident hogs
            hogs = self._find_hogs(cvar)

            # determine shared vertices per hog
            for hog in hogs:
                shared = set(hog.xvars).intersection(xvars)

                if len(shared) >= 1 and len(shared) < len(hog.xvars) and len(shared) < len(xvars):
                    tmphog = Hedgehog(cvar, xvars)
                    if not self._graph.has_node(tmphog):
                        newhog = self._make_hog_from_rigid(cvar, newcluster)
                        self._merge_hogs(hog, newhog)

    def _search_hogs_from_hog(self, newhog):
        # find adjacent clusters
        dep = self.find_dependent(newhog.cvar)

        top = [c for c in dep if self.is_top_level(c)]
        clusters = [x for x in top if isinstance(x,Rigid)]
        balloons = [x for x in top if isinstance(x,Balloon)]
        hogs = self._find_hogs(newhog.cvar)

        tomerge = []

        for cluster in clusters:
            if len(cluster.variables) < 3:
                continue

            # determine shared vars
            xvars = set(cluster.variables)
            xvars.remove(newhog.cvar)

            shared = set(newhog.xvars).intersection(xvars)

            if len(shared) >= 1 and len(shared) < len(xvars) and len(shared) < len(newhog.xvars):
                tmphog = Hedgehog(newhog.cvar, xvars)

                if not self._graph.has_node(tmphog):
                    newnewhog = self._make_hog_from_rigid(newhog.cvar, cluster)

                    tomerge.append(newnewhog)

        for balloon in balloons:
            # determine shared vars
            xvars = set(balloon.variables)
            xvars.remove(newhog.cvar)

            shared = set(newhog.xvars).intersection(xvars)

            if len(shared) >= 1 and len(shared) < len(xvars) and len(shared) < len(newhog.xvars):
                tmphog = Hedgehog(newhog.cvar, xvars)

                if not self._graph.has_node(tmphog):
                    newnewhog = self._make_hog_from_balloon(newhog.cvar, balloon)

                    tomerge.append(newnewhog)

        for hog in hogs:
            if hog == newhog:
                continue

            # determine shared vars
            shared = set(newhog.xvars).intersection(hog.xvars)

            if len(shared) >= 1 and len(shared) < len(hog.xvars) and len(shared) < len(newhog.xvars):
                # if mergeable, then create new hog
                tomerge.append(hog)

        if len(tomerge) == 0:
            return None
        else:
            lasthog = newhog

            for hog in tomerge:
                lasthog = self._merge_hogs(lasthog, hog)

            return lasthog

    def _merge_hogs(self, hog1, hog2):
        LOGGER.debug("Merging hedgehogs %s and %s", hog1, hog2)

        # create new hog and method
        xvars = set(hog1.xvars).union(hog2.xvars)

        mergedhog = Hedgehog(hog1.cvar, xvars)

        method = MergeHogs(hog1, hog2, mergedhog)

        self._add_merge(method)

        return mergedhog

    def _search_merge_from_hog(self, hog):
        # case CH (overconstrained)
        dep = self.find_dependent(hog.cvar)

        clusters = [x for x in dep if isinstance(x,Rigid) and self.is_top_level(x)]
        sharecx = [x for x in clusters if len(set(hog.xvars).intersection(x.variables)) >=1]

        for cluster in sharecx:
            sharedcx = set(cluster.variables).intersection(hog.xvars)

            if len(sharedcx) == len(hog.xvars):
                return self._merge_rigid_hog(cluster, hog)

        # case CHC
        for i in range(len(sharecx)):
            c1 = sharecx[i]

            for j in range(i + 1, len(sharecx)):
                c2 = sharecx[j]

                return self._merge_rigid_hog_rigid(c1, hog, c2)

        # case CCH
        sharex = set()

        for var in hog.xvars:
            dep = self.find_dependent(var)

            sharex.update([x for x in dep if isinstance(x,Rigid) and self.is_top_level(x)])

        for c1 in sharecx:
            for c2 in sharex:
                if c1 == c2: continue

                shared12 = set(c1.variables).intersection(c2.variables)
                sharedh2 = set(hog.xvars).intersection(c2.variables)
                shared2 = shared12.union(sharedh2)

                if len(shared12) >= 1 and len(sharedh2) >= 1 and len(shared2) == 2:
                    return self._merge_rigid_rigid_hog(c1, c2, hog)

        return None

    def _search_merge_from_rigid(self, newcluster):
        LOGGER.debug("Searching for merge from rigid %s", newcluster)

        # find clusters overlapping with new cluster
        overlap = {}
        for var in newcluster.variables:
            # get dependent objects
            dep = self._graph.successors(var)

            # only clusters
            dep = [c for c in dep if self._graph.has_edge("_rigids",c)]

            # only top level
            dep = list([c for c in dep if self.is_top_level(c)])

            # remove newcluster
            if newcluster in dep:
                dep.remove(newcluster)

            for cluster in dep:
                if cluster in overlap:
                    overlap[cluster].append(var)
                else:
                    overlap[cluster] = [var]

        # point-cluster merge
        for cluster in overlap:
            if len(overlap[cluster]) == 1:
                if len(cluster.variables) == 1:
                    return self._merge_point_rigid(cluster, newcluster)
                elif len(newcluster.variables) == 1:
                    return self._merge_point_rigid(newcluster, cluster)

        # two cluster merge (overconstrained)
        for cluster in overlap:
            if len(overlap[cluster]) >= 2:
                return self._merge_rigid_pair(cluster, newcluster)

        # three cluster merge
        clusterlist = list(overlap.keys())

        for i in range(len(clusterlist)):
            c1 = clusterlist[i]
            for j in range(i + 1, len(clusterlist)):
                c2 = clusterlist[j]

                shared12 = set(c1.variables).intersection(c2.variables)
                shared13 = set(c1.variables).intersection(newcluster.variables)
                shared23 = set(c2.variables).intersection(newcluster.variables)
                shared1 = shared12.union(shared13)
                shared2 = shared12.union(shared23)
                shared3 = shared13.union(shared23)

                if len(shared1) == 2 and len(shared1) == 2 and len(shared2) == 2:
                    return self._merge_rigid_triple(c1, c2, newcluster)

        # merge with an angle, case 1
        for cluster in overlap:
            ovars = overlap[cluster]

            if len(ovars) == 1:
                cvar = ovars[0]
            else:
                raise Exception("unexpected case")

            hogs = self._find_hogs(cvar)

            for hog in hogs:
                sharedch = set(cluster.variables).intersection(hog.xvars)
                sharednh = set(newcluster.variables).intersection(hog.xvars)
                sharedh = sharedch.union(sharednh)

                if len(sharedch) >= 1 and len(sharednh) >= 1 and len(sharedh) >= 2:
                    return self._merge_rigid_hog_rigid(cluster, hog, newcluster)

        # merge with an angle, case 2
        for var in newcluster.variables:
            hogs = self._find_hogs(var)

            for hog in hogs:
                sharednh = set(newcluster.variables).intersection(hog.xvars)

                if len(sharednh) < 1:
                    continue

                for cluster in overlap:
                    sharednc = set(newcluster.variables).intersection(cluster.variables)

                    if len(sharednc) != 1:
                        raise Exception("unexpected case")

                    if hog.cvar in cluster.variables:
                        continue

                    sharedch = set(cluster.variables).intersection(hog.xvars)
                    sharedc = sharedch.union(sharednc)

                    if len(sharedch) >= 1 and len(sharedc) >= 2:
                        return self._merge_rigid_rigid_hog(newcluster, cluster, hog)

        # merge with an angle, case 3
        for cluster in overlap:
            sharednc = set(newcluster.variables).intersection(cluster.variables)

            if len(sharednc) != 1:
                raise Exception("unexpected case")

            for var in cluster.variables:
                hogs = self._find_hogs(var)

                for hog in hogs:
                    if hog.cvar in newcluster.variables:
                        continue

                    sharedhc = set(newcluster.variables).intersection(hog.xvars)
                    sharedhn = set(cluster.variables).intersection(hog.xvars)
                    sharedh = sharedhn.union(sharedhc)
                    sharedc = sharedhc.union(sharednc)

                    if len(sharedhc) >= 1 and len(sharedhn) >= 1 and len(sharedh) >= 2 and len(sharedc) == 2:
                        return self._merge_rigid_rigid_hog(cluster, newcluster, hog)

    def _merge_point_rigid(self, point, rigid):
        LOGGER.debug("Merging point %s with rigid %s", point, rigid)

        # get variables from point and rigid
        variables = set(point.variables).union(rigid.variables)

        # create new rigid from variables
        new_cluster = Rigid(variables)

        # add new point-to-rigid merge
        self._add_merge(MergePR(point, rigid, new_cluster))

        return new_cluster

    def _merge_rigid_pair(self, r1, r2):
        """Merge a pair of clusters, structurally overconstrained.
           Rigid which contains root is used as origin.
           Returns resulting cluster.
        """

        r1_contains_root = self._contains_root(r1)
        r2_contains_root = self._contains_root(r2)

        LOGGER.debug("Merging rigid pair %s (root derived = %s) and %s (root derived = %s)",
                     r1, r1_contains_root, r2, r2_contains_root)

        # always use root cluster as first cluster, swap if needed
        if r1_contains_root and r2_contains_root:
            raise Exception("two root clusters")
        elif r2_contains_root:
            LOGGER.debug("Swapping rigid order")

            return self._merge_rigid_pair(r2, r1)

        # create new cluster and merge
        variables = set(r1.variables).union(r2.variables)

        # create new rigid cluster from variables
        new_cluster = Rigid(variables)

        # add new two-rigid merge
        self._add_merge(MergeRR(r1, r2, new_cluster))

        return new_cluster

    def _merge_rigid_hog(self, rigid, hog):
        """merge rigid and hog (absorb hog, overconstrained)"""

        LOGGER.debug("Merging rigid %s with hedgehog %s", rigid, hog)

        # create new rigid from variables
        new_cluster = Rigid(rigid.variables)

        # add new rigid-hedgehog merge
        self._add_merge(MergeRH(rigid, hog, new_cluster))

        return new_cluster

    def _merge_balloon_hog(self, balloon, hog):
        """merge balloon and hog (absorb hog, overconstrained)"""

        LOGGER.debug("Merging balloon %s with hedgehog %s", balloon, hog)

        # create new balloon and merge
        newballoon = Balloon(balloon.variables)

        merge = MergeBH(balloon, hog, newballoon)

        self._add_merge(merge)

        return newballoon

    def _merge_rigid_triple(self, r1, r2, r3):
        """Merge a triple of clusters.
           Rigid which contains root is used as origin.
           Returns resulting cluster.
        """

        r1_contains_root = self._contains_root(r1)
        r2_contains_root = self._contains_root(r2)
        r3_contains_root = self._contains_root(r3)

        LOGGER.debug("Merging rigids %s (root derived = %s), %s (root derived = %s) and %s "
                     "(root derived = %s)", r1, r1_contains_root, r2, r2_contains_root, r3,
                     r3_contains_root)

        # always use root rigid as first cluster, swap if needed
        if r2_contains_root:
            LOGGER.debug("Swapping rigid order")

            return self._merge_rigid_triple(r2, r1, r3)
        elif r3_contains_root:
            LOGGER.debug("Swapping rigid order")

            return self._merge_rigid_triple(r3, r1, r2)

        # create new cluster and method
        allvars = set(r1.variables).union(r2.variables).union(r3.variables)

        newcluster = Rigid(allvars)

        merge = MergeRRR(r1,r2,r3,newcluster)

        self._add_merge(merge)

        return newcluster

    def _merge_rigid_hog_rigid(self, r1, hog, r2):
        """merge r1 and r2 with a hog, with hog center in r1 and r2"""

        r1_contains_root = self._contains_root(r1)
        r2_contains_root = self._contains_root(r2)

        LOGGER.debug("Merging rigid %s (root derived = %s), hedgehog %s and rigid %s "
                     "(root derived = %s)", r1, r1_contains_root, hog, r2, r2_contains_root)

        # always use root rigid as first cluster, swap if needed
        if r2_contains_root:
            LOGGER.debug("Swapping rigid order")

            return self._merge_rigid_hog_rigid(r2, hog, r1)

        # derive sub-hog if nessecairy
        allvars = set(r1.variables).union(r2.variables)
        xvars = set(hog.xvars).intersection(allvars)

        if len(xvars) < len(hog.xvars):
            LOGGER.debug("Deriving sub-hedgehog")

            hog = self._derive_subhog(hog, xvars)

        #create new cluster and merge
        allvars = set(r1.variables).union(r2.variables)

        newcluster = Rigid(allvars)

        merge = MergeRHR(r1, hog, r2, newcluster)

        self._add_merge(merge)

        return newcluster

    def _derive_subhog(self, hog, xvars):
        subvars = set(hog.xvars).intersection(xvars)

        assert len(subvars) == len(xvars)

        subhog = Hedgehog(hog.cvar, xvars)
        method = SubHog(hog, subhog)

        self._add_hog(subhog)
        self._add_method(method)

        return subhog

    def _merge_rigid_rigid_hog(self, r1, r2, hog):
        """merge c1 and c2 with a hog, with hog center only in c1"""

        r1_contains_root = self._contains_root(r1)
        r2_contains_root = self._contains_root(r2)

        LOGGER.debug("Merging rigid %s (root derived = %s), rigid %s (root derived = %s) and "
                     "hedgehog %s", r1, r1_contains_root, r2, r2_contains_root, hog)

        # always use root rigid as first cluster, swap if needed
        if r1_contains_root and r2_contains_root:
            raise Exception("two root clusters!")
        elif r2_contains_root:
            LOGGER.debug("Swapping rigid order")

            return self._merge_rigid_rigid_hog(r2, r1, hog)

        # derive subhog if necessary
        allvars = set(r1.variables).union(r2.variables)
        xvars = set(hog.xvars).intersection(allvars)

        if len(xvars) < len(hog.xvars):
            LOGGER.debug("Deriving sub-hedgehog")

            hog = self._derive_subhog(hog, xvars)

        # create new cluster and method
        newcluster = Rigid(allvars)

        merge = MergeRRH(r1, r2, hog, newcluster)

        self._add_merge(merge)

        return newcluster

    def _contains_root(self, input_cluster):
        """returns True iff input_cluster is root cluster or was determined by
        merging with the root cluster."""

        # start from root cluster. Follow merges upwards until:
        #  - input cluster found -> True
        #  - no more merges -> False

        # get the vertices attached to root
        successors = list(self._graph.successors("_root"))

        # number of root clusters
        num_roots = len(successors)

        if num_roots > 1:
            raise Exception("more than one root cluster")
        if num_roots == 1:
            cluster = successors[0]
        else:
            cluster = None

        while (cluster != None):
            if cluster is input_cluster:
                return True

            # list of vertices this cluster is linked to
            fr = self._graph.successors(cluster)

            # get vertices that are merges with this cluster as an input
            me = [x for x in fr if isinstance(x, Merge) and cluster in x.inputs]

            num_merges = len(me)

            if num_merges > 1:
                raise Exception("cluster merged more than once")
            elif num_merges == 0:
                cluster = None
            elif len(me[0].outputs) != 1:
                raise Exception("a merge with number of outputs != 1")
            else:
                cluster = me[0].outputs[0]

        return False

    def _is_consistent_pair(self, object1, object2):
        LOGGER.debug("Checking if %s and %s are a consistent pair", object1, object2)

        oc = object1.over_constraints(object2)

        if len(oc):
            LOGGER.debug(f"overconstraints of consistent pair: {oc}")

        # calculate consistency (True if no overconstraints)
        consistent = True
        for constraint in oc:
            consistent = consistent and self._consistent_overconstraint_in_pair(constraint, object1, object2)

        if consistent:
            LOGGER.debug(f"pair is globally consistent")
        else:
            LOGGER.debug(f"pair is not globally consistent")

        return consistent

    def _consistent_overconstraint_in_pair(self, overconstraint, object1, object2):
        LOGGER.debug(f"checking if '{object1}' and '{object2}' have consistent overconstraint"
                     f"'{overconstraint}'")

        # get sources for constraint in given clusters
        s1 = self._source_constraint_in_cluster(overconstraint, object1)
        s2 = self._source_constraint_in_cluster(overconstraint, object2)

        if s1 is None:
            consistent = False
        elif s2 is None:
            consistent = False
        elif s1 == s2:
            consistent = True
        else:
            if self._is_atomic(s1) and not self._is_atomic(s2):
                consistent = False
            elif self._is_atomic(s2) and not self._is_atomic(s1):
                consistent = False
            else:
                consistent = True

        LOGGER.debug("Consistent? %s", consistent)

        return consistent

    def _source_constraint_in_cluster(self, constraint, cluster):
        if not self._contains_constraint(cluster, constraint):
            raise Exception("constraint not in cluster")
        elif self._is_atomic(cluster):
            return cluster
        else:
            method = self._determining_method(cluster)
            inputs = method.inputs
            down = [x for x in inputs if self._contains_constraint(x, constraint)]

            if len(down) == 0:
                return cluster
            elif len(down) > 1:
                if method.consistent:
                    return self._source_constraint_in_cluster(constraint, down[0])
                else:
                    LOGGER.warning("Source is inconsistent")
                    return None
            else:
                return self._source_constraint_in_cluster(constraint, down[0])

    def _is_atomic(self, object):
        return self._determining_method(object) is None

    def _determining_method(self, object):
        depends = self.find_depends(object)
        methods = [x for x in depends if isinstance(x, Method)]

        if len(methods) == 0:
            return None
        elif len(methods) > 1:
            raise Exception("object determined by more than one method")
        else:
            return methods[0]

    def _contains_constraint(self, object, constraint):
        if isinstance(constraint, Distance):
            return self._contains_distance(object, constraint)
        elif isinstance(constraint, Angle):
            return self._contains_angle(object, constraint)
        else:
            raise Exception("unexpected case")

    def _contains_distance(self,object, distance):
        if isinstance(object, Rigid):
            return (distance.points[0] in object.variables and distance.points[1] in object.variables)
        elif isinstance(object, Distance):
            return (distance.points[0] in object.variables and distance.points[1] in object.variables)
        else:
            return False

    def _contains_angle(self, object, angle):
        if isinstance(object, Rigid) or isinstance(object, Balloon):
            return (angle.points[0] in object.variables
            and angle.points[1] in object.variables
            and angle.points[2] in object.variables)
        elif isinstance(object, Hedgehog):
            return (angle.points[1] == object.cvar and
            angle.points[0] in object.xvars and
            angle.points[2] in object.xvars)
        elif isinstance(object, Angle):
            return (angle.points[1] == object.variables[1] and
            angle.points[0] in object.variables and
            angle.points[2] in object.variables)
        else:
            return False

    def __str__(self):
        distances = "\n\t".join([str(x) for x in self.distances()])
        angles = "\n\t".join([str(x) for x in self.angles()])
        rigids = "\n\t".join([str(x) for x in self.rigids()])
        hedgehogs = "\n\t".join([str(x) for x in self.hedgehogs()])
        balloons = "\n\t".join([str(x) for x in self.balloons()])
        methods = "\n\t".join([str(x) for x in self.methods()])

        return f"""Distances:
\t{distances}
Angles:
\t{angles}
Rigids:
\t{rigids}
Hedgehogs:
\t{hedgehogs}
Balloons:
\t{balloons}
Methods:
\t{methods}"""

    def _known_angle(self, a, b, c):
        """returns Balloon, Rigid or Hedgehog that contains angle(a, b, c)"""

        if a == b or a == c or b == c:
            raise Exception("all vars in angle must be different")

        # get objects dependend on a, b and c
        dep_a = self._graph.successors(a)
        dep_b = self._graph.successors(b)
        dep_c = self._graph.successors(c)

        dependend = []

        for obj in dep_a:
            if obj in dep_b and obj in dep_c:
                dependend.append(obj)

        # find a hedgehog
        hogs = [x for x in dependend if isinstance(x, Hedgehog)]
        hogs = [hog for hog in hogs if hog.cvar == b]
        hogs = [x for x in hogs if self.is_top_level(x)]

        if len(hogs) == 1: return hogs[0]
        if len(hogs) > 1: raise Exception("angle in more than one hedgehog")

        # or find a cluster
        clusters = [x for x in dependend if isinstance(x, Rigid)]
        clusters = [x for x in clusters if self.is_top_level(x)]

        if len(clusters) == 1: return clusters[0]
        if len(clusters) > 1: raise Exception("angle in more than one Rigid")

        # or find a balloon
        balloons = [x for x in dependend if isinstance(x, Balloon)]
        balloons = [x for x in balloons if self.is_top_level(x)]

        if len(balloons) == 1: return balloons[0]
        if len(balloons) > 1: raise Exception("angle in more than one Balloon")

        return None
