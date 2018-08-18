"""A generic 2D geometric constraint solver.

The solver finds a generic solution for problems formulated by Clusters. The
generic solution is a directed acyclic graph of Clusters and Methods. Particular
problems and solutions are represented by a Configuration for each cluster.
"""

import abc
import logging
import numpy as np
import numpy.linalg as linalg

from ..graph import Graph, MethodGraph
from ..method import Method
from ..notify import Notifier
from ..multimethod import MultiVariable, MultiMethod
from ..cluster import *
from ..configuration import Configuration
from ..selconstr import NotCounterClockwiseConstraint, NotClockwiseConstraint, NotAcuteConstraint, NotObtuseConstraint
from ..geometry import Vector, rr_int, cr_int, cc_int, tol_zero

LOGGER = logging.getLogger(__name__)

class ClusterSolver(Notifier):
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

    def set(self, cluster, configurations):
        """Associate a list of configurations with a cluster"""
        self._mg.set_node_value(cluster, configurations)

    def get(self, cluster):
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
            # if merge removed items from toplevel then add them back to top level
            if hasattr(item, "restore_toplevel"):
                for cluster in item.restore_toplevel:
                    to_restore.add(cluster)

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
                    self._mg.rem_method(item)
                except:
                    pass
            elif isinstance(item, MultiVariable):
                self._mg.rem_variable(item)

            # notify listeners
            self.send_notify(("remove", item))

        # restore top level (also added to _new)
        for cluster in to_restore:
            if self._graph.has_node(cluster):
                self._add_top_level(cluster)

        # re-solve
        self._process_new()

    def _find_descendent(self,v):
        """find all descendend objects of v (directly or indirectly \
        dependent)"""

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

        for var in newcluster.vars:
            self._add_variable(var)
            self._add_dependency(var, newcluster)

        # if there is no root cluster, this one will be it
        if len(list(self._graph.successors("_root"))) == 0:
            self._graph.add_edge("_root", newcluster)

        # add to top level
        self._add_top_level(newcluster)

        # add to methodgraph
        self._mg.add_variable(newcluster)

        # notify
        self.send_notify(("add", newcluster))

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

        # notify
        self.send_notify(("add", hog))

    def _add_balloon(self, newballoon):
        """add a cluster if not already in system"""
        LOGGER.debug("Adding balloon %s", newballoon)

        # check if not already exists
        if self._graph.has_node(newballoon):
            raise Exception("balloon already in clsolver")

        # update graph
        self._add_to_group("_balloons", newballoon)

        for var in newballoon.vars:
            self._add_variable(var)
            self._add_dependency(var, newballoon)

        # add to top level
        self._add_top_level(newballoon)

         # add to methodgraph
        self._mg.add_variable(newballoon)

        # notify
        self.send_notify(("add", newballoon))

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

        for var in variables:
            clusters = self._graph.successors(var)
            clusters = [c for c in clusters if isinstance(c, Rigid)]
            clusters = [c for c in clusters if len(c.vars) == 1]

            if len(clusters) != 1:
                raise Exception("no prototype cluster for variable \
{0}".format(var))

            selclusters.append(clusters[0])

        outcluster = incluster.copy()

        # Rick 20090519 - copy does not copy structural overconstrained flag?
        outcluster.overconstrained = incluster.overconstrained

        selector = PrototypeMethod(incluster, selclusters, outcluster, \
        constraints)

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
        self.send_notify(("add", method))

    def _process_new(self):
        while len(self._new) > 0:
            new_obj = self._new.pop()
            LOGGER.debug("Searching from %s", \
            new_obj)

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
        for cvar in balloon.vars:
            # find all incident hedgehogs
            hogs = self._find_hogs(cvar)

            # determine shared vertices per hedgehog
            for hog in hogs:
                shared = set(hog.xvars).intersection(balloon.vars)

                if len(shared) == len(hog.xvars):
                    return self._merge_balloon_hog(balloon, hog)

    def _search_absorb_from_rigid(self, cluster):
        for cvar in cluster.vars:
            # find all incident hedgehogs
            hogs = self._find_hogs(cvar)

            # determine shared vertices per hedgehog
            for hog in hogs:
                shared = set(hog.xvars).intersection(cluster.vars)

                if len(shared) == len(hog.xvars):
                    return self._merge_rigid_hog(cluster, hog)

    def _search_absorb_from_hog(self, hog):
        dep = self.find_dependent(hog.cvar)

        # case BH (overconstrained):
        balloons = [x for x in dep if isinstance(x, Balloon) \
        and self.is_top_level(x)]
        sharecx = [x for x in balloons if len(set(hog.xvars).intersection(x.vars)) \
        >= 1]

        for balloon in sharecx:
            sharedcx = set(balloon.vars).intersection(hog.xvars)

            if len(sharedcx) == len(hog.xvars):
                return self._merge_balloon_hog(balloon, hog)

        # case CH (overconstrained)
        clusters = [x for x in dep if isinstance(x, Rigid) \
        and self.is_top_level(x)]
        sharecx = [x for x in clusters if len(set(hog.xvars).intersection(x.vars)) \
        >= 1]

        for cluster in sharecx:
            sharedcx = set(cluster.vars).intersection(hog.xvars)

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
        LOGGER.debug("Making balloon %s, %s, %s", \
        var1, var2, var3)

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
                                new_balloons.append(self._make_balloon(var1, \
                                var2, var3, hog, hog2))

        if len(new_balloons) > 0:
            return new_balloons

        return None

    def _search_balloon_from_balloon(self, balloon):
        # map from adjacent balloons to variables shared with input balloon
        mapping = {}

        for var in balloon.vars:
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

        for var in balloon.vars:
            deps = self.find_dependent(var)

            clusters = [x for x in deps if isinstance(x, Rigid) \
            or isinstance(x, Distance)]
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
        LOGGER.debug("Searching for balloon-rigid \
merge from rigid")

        # map from adjacent clusters to variables shared with input balloon
        mapping = {}

        for var in rigid.vars:
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
        variables = set(balloon_a.vars).union(balloon_b.vars)

        # create new balloon cluster
        new_balloon = Balloon(variables)

        # add new balloon merge
        self._add_merge(BalloonMerge(balloon_a, balloon_b, new_balloon))

        return new_balloon

    def _merge_balloon_rigid(self, balloon, rigid):
        # create new rigid and method

        # get variables in both the balloon and the rigid
        variables = set(balloon.vars).union(rigid.vars)

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
        xvars = set(rigid.vars)

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
        xvars = set(balloon.vars)

        xvars.remove(cvar)

        hog = Hedgehog(cvar, xvars)

        self._add_hog(hog)

        method = BalloonToHog(balloon, hog)

        self._add_method(method)

        return hog

    def _search_hogs_from_balloon(self, newballoon):
        if len(newballoon.vars) <= 2:
            return None

        # create/merge hogs
        for cvar in newballoon.vars:
            # potential new hog
            xvars = set(newballoon.vars)

            xvars.remove(cvar)

            # find all incident hogs
            hogs = self._find_hogs(cvar)

            # determine shared vertices per hog
            for hog in hogs:
                shared = set(hog.xvars).intersection(xvars)
                if len(shared) >= 1 and len(shared) \
                < len(hog.xvars) and len(shared) < len(xvars):
                    tmphog = Hedgehog(cvar, xvars)
                    if not self._graph.has_node(tmphog):
                        newhog = self._make_hog_from_balloon(cvar, newballoon)
                        self._merge_hogs(hog, newhog)

    def _search_hogs_from_rigid(self, newcluster):
        if len(newcluster.vars) <= 2:
            return None

        # create/merge hogs
        for cvar in newcluster.vars:
            # potential new hog
            xvars = set(newcluster.vars)
            xvars.remove(cvar)

            # find all incident hogs
            hogs = self._find_hogs(cvar)

            # determine shared vertices per hog
            for hog in hogs:
                shared = set(hog.xvars).intersection(xvars)

                if len(shared) >= 1 and len(shared) \
                < len(hog.xvars) and len(shared) < len(xvars):
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
            if len(cluster.vars) < 3:
                continue

            # determine shared vars
            xvars = set(cluster.vars)
            xvars.remove(newhog.cvar)

            shared = set(newhog.xvars).intersection(xvars)

            if len(shared) >= 1 and len(shared) < len(xvars) \
            and len(shared) < len(newhog.xvars):
                tmphog = Hedgehog(newhog.cvar, xvars)

                if not self._graph.has_node(tmphog):
                    newnewhog = self._make_hog_from_rigid(newhog.cvar, \
                    cluster)

                    tomerge.append(newnewhog)

        for balloon in balloons:
            # determine shared vars
            xvars = set(balloon.vars)
            xvars.remove(newhog.cvar)

            shared = set(newhog.xvars).intersection(xvars)

            if len(shared) >= 1 and len(shared) \
            < len(xvars) and len(shared) < len(newhog.xvars):
                tmphog = Hedgehog(newhog.cvar, xvars)

                if not self._graph.has_node(tmphog):
                    newnewhog = self._make_hog_from_balloon(newhog.cvar, \
                    balloon)

                    tomerge.append(newnewhog)

        for hog in hogs:
            if hog == newhog:
                continue

            # determine shared vars
            shared = set(newhog.xvars).intersection(hog.xvars)

            if len(shared) >= 1 and len(shared) \
            < len(hog.xvars) and len(shared) < len(newhog.xvars):
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
        LOGGER.debug("Merging hedgehogs %s and \
%s", hog1, hog2)

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
        sharecx = [x for x in clusters if len(set(hog.xvars).intersection(x.vars)) >=1]

        for cluster in sharecx:
            sharedcx = set(cluster.vars).intersection(hog.xvars)

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

            sharex.update([x for x in dep if isinstance(x,Rigid) \
            and self.is_top_level(x)])

        for c1 in sharecx:
            for c2 in sharex:
                if c1 == c2: continue

                shared12 = set(c1.vars).intersection(c2.vars)
                sharedh2 = set(hog.xvars).intersection(c2.vars)
                shared2 = shared12.union(sharedh2)

                if len(shared12) >= 1 and len(sharedh2) >= 1 \
                and len(shared2) == 2:
                    return self._merge_rigid_rigid_hog(c1, c2, hog)

        return None

    def _search_merge_from_rigid(self, newcluster):
        LOGGER.debug("Searching for merge from \
rigid %s", newcluster)

        # find clusters overlapping with new cluster
        overlap = {}
        for var in newcluster.vars:
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
                if len(cluster.vars) == 1:
                    return self._merge_point_rigid(cluster, newcluster)
                elif len(newcluster.vars) == 1:
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

                shared12 = set(c1.vars).intersection(c2.vars)
                shared13 = set(c1.vars).intersection(newcluster.vars)
                shared23 = set(c2.vars).intersection(newcluster.vars)
                shared1 = shared12.union(shared13)
                shared2 = shared12.union(shared23)
                shared3 = shared13.union(shared23)

                if len(shared1) == 2 and len(shared1) == 2 and \
                   len(shared2) == 2:
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
                sharedch = set(cluster.vars).intersection(hog.xvars)
                sharednh = set(newcluster.vars).intersection(hog.xvars)
                sharedh = sharedch.union(sharednh)

                if len(sharedch) >= 1 and len(sharednh) >= 1 \
                and len(sharedh) >= 2:
                    return self._merge_rigid_hog_rigid(cluster, hog, \
                    newcluster)

        # merge with an angle, case 2
        for var in newcluster.vars:
            hogs = self._find_hogs(var)

            for hog in hogs:
                sharednh = set(newcluster.vars).intersection(hog.xvars)

                if len(sharednh) < 1:
                    continue

                for cluster in overlap:
                    sharednc = set(newcluster.vars).intersection(cluster.vars)

                    if len(sharednc) != 1:
                        raise Exception("unexpected case")

                    if hog.cvar in cluster.vars:
                        continue

                    sharedch = set(cluster.vars).intersection(hog.xvars)
                    sharedc = sharedch.union(sharednc)

                    if len(sharedch) >= 1 and len(sharedc) >= 2:
                        return self._merge_rigid_rigid_hog(newcluster, \
                        cluster, hog)

        # merge with an angle, case 3
        for cluster in overlap:
            sharednc = set(newcluster.vars).intersection(cluster.vars)

            if len(sharednc) != 1:
                raise Exception("unexpected case")

            for var in cluster.vars:
                hogs = self._find_hogs(var)

                for hog in hogs:
                    if hog.cvar in newcluster.vars:
                        continue

                    sharedhc = set(newcluster.vars).intersection(hog.xvars)
                    sharedhn = set(cluster.vars).intersection(hog.xvars)
                    sharedh = sharedhn.union(sharedhc)
                    sharedc = sharedhc.union(sharednc)

                    if len(sharedhc) >= 1 and len(sharedhn) >= 1 \
                    and len(sharedh) >= 2 and len(sharedc) == 2:
                        return self._merge_rigid_rigid_hog(cluster, \
                        newcluster, hog)

    def _merge_point_rigid(self, point, rigid):
        LOGGER.debug("Merging point %s with rigid \
%s", point, rigid)

        # get variables from point and rigid
        variables = set(point.vars).union(rigid.vars)

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

        LOGGER.debug("Merging rigid pair %s (root \
derived = %s) and %s (root derived = %s)", r1, r1_contains_root, r2, \
        r2_contains_root)

        # always use root cluster as first cluster, swap if needed
        if r1_contains_root and r2_contains_root:
            raise Exception("two root clusters")
        elif r2_contains_root:
            LOGGER.debug("Swapping rigid order")

            return self._merge_rigid_pair(r2, r1)

        # create new cluster and merge
        variables = set(r1.vars).union(r2.vars)

        # create new rigid cluster from variables
        new_cluster = Rigid(variables)

        # add new two-rigid merge
        self._add_merge(MergeRR(r1, r2, new_cluster))

        return new_cluster

    def _merge_rigid_hog(self, rigid, hog):
        """merge rigid and hog (absorb hog, overconstrained)"""

        LOGGER.debug("Merging rigid %s with \
hedgehog %s", rigid, hog)

        # create new rigid from variables
        new_cluster = Rigid(rigid.vars)

        # add new rigid-hedgehog merge
        self._add_merge(MergeRH(rigid, hog, new_cluster))

        return new_cluster

    def _merge_balloon_hog(self, balloon, hog):
        """merge balloon and hog (absorb hog, overconstrained)"""

        LOGGER.debug("Merging balloon %s with \
hedgehog %s", balloon, hog)

        # create new balloon and merge
        newballoon = Balloon(balloon.vars)

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

        LOGGER.debug("Merging rigids %s (root \
derived = %s), %s (root derived = %s) and %s (root derived = %s)", r1, \
r1_contains_root, r2, r2_contains_root, r3, r3_contains_root)

        # always use root rigid as first cluster, swap if needed
        if r2_contains_root:
            LOGGER.debug("Swapping rigid order")

            return self._merge_rigid_triple(r2, r1, r3)
        elif r3_contains_root:
            LOGGER.debug("Swapping rigid order")

            return self._merge_rigid_triple(r3, r1, r2)

        # create new cluster and method
        allvars = set(r1.vars).union(r2.vars).union(r3.vars)

        newcluster = Rigid(allvars)

        merge = MergeRRR(r1,r2,r3,newcluster)

        self._add_merge(merge)

        return newcluster

    def _merge_rigid_hog_rigid(self, r1, hog, r2):
        """merge r1 and r2 with a hog, with hog center in r1 and r2"""

        r1_contains_root = self._contains_root(r1)
        r2_contains_root = self._contains_root(r2)

        LOGGER.debug("Merging rigid %s (root \
derived = %s), hedgehog %s and rigid %s (root derived = %s)", r1, \
        r1_contains_root, hog, r2, r2_contains_root)

        # always use root rigid as first cluster, swap if needed
        if r2_contains_root:
            LOGGER.debug("Swapping rigid order")

            return self._merge_rigid_hog_rigid(r2, hog, r1)

        # derive sub-hog if nessecairy
        allvars = set(r1.vars).union(r2.vars)
        xvars = set(hog.xvars).intersection(allvars)

        if len(xvars) < len(hog.xvars):
            LOGGER.debug("Deriving sub-hedgehog")

            hog = self._derive_subhog(hog, xvars)

        #create new cluster and merge
        allvars = set(r1.vars).union(r2.vars)

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

        LOGGER.debug("Merging rigid %s (root \
derived = %s), rigid %s (root derived = %s) and hedgehog %s", r1, \
        r1_contains_root, r2, r2_contains_root, hog)

        # always use root rigid as first cluster, swap if needed
        if r1_contains_root and r2_contains_root:
            raise Exception("two root clusters!")
        elif r2_contains_root:
            LOGGER.debug("Swapping rigid order")

            return self._merge_rigid_rigid_hog(r2, r1, hog)

        # derive subhog if necessary
        allvars = set(r1.vars).union(r2.vars)
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
        LOGGER.debug("Checking if %s and %s are a \
consistent pair", object1, object2)

        oc = over_constraints(object1, object2)

        LOGGER.debug("Overconstraints of consistent pair: %s", [str(c) for c in oc])

        # calculate consistency (True if no overconstraints)
        consistent = True
        for constraint in oc:
            consistent = consistent and self._consistent_overconstraint_in_pair(constraint, object1, object2)

        LOGGER.debug("Global consistent? %s", consistent)

        return consistent

    def _consistent_overconstraint_in_pair(self, overconstraint, object1, \
    object2):
        LOGGER.debug("Checking if %s and %s have \
a consistent overconstraint %s", object1, object2, overconstraint)

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
                    return self._source_constraint_in_cluster(constraint, \
                    down[0])
                else:
                    LOGGER.warning("Source is \
inconsistent")
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
            return (distance.points[0] in object.vars and distance.points[1] \
            in object.vars)
        elif isinstance(object, Distance):
            return (distance.points[0] in object.vars and distance.points[1] \
            in object.vars)
        else:
            return False

    def _contains_angle(self, object, angle):
        if isinstance(object, Rigid) or isinstance(object, Balloon):
            return (angle.points[0] in object.vars
            and angle.points[1] in object.vars
            and angle.points[2] in object.vars)
        elif isinstance(object, Hedgehog):
            return (angle.points[1] == object.cvar and
            angle.points[0] in object.xvars and
            angle.points[2] in object.xvars)
        elif isinstance(object, Angle):
            return (angle.points[1] == object.vars[1] and
            angle.points[0] in object.vars and
            angle.points[2] in object.vars)
        else:
            return False

    def __str__(self):
        return "Distances:\n\t{0}\nAngles:\n\t{1}\nRigids:\n\t{2}\n\
Hedgehogs:\n\t{3}\nBalloons:\n\t{4}\nMethods:\n\t{5}".format(\
        "\n\t".join([str(x) for x in self.distances()]), \
        "\n\t".join([str(x) for x in self.angles()]), \
        "\n\t".join([str(x) for x in self.rigids()]), \
        "\n\t".join([str(x) for x in self.hedgehogs()]), \
        "\n\t".join([str(x) for x in self.balloons()]), \
        "\n\t".join([str(x) for x in self.methods()]))

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
        hogs = [x for x in dependend if isinstance(x,Hedgehog)]
        hogs = [hog for hog in hogs if hog.cvar == b]
        hogs = [x for x in hogs if self.is_top_level(x)]

        if len(hogs) == 1: return hogs[0]
        if len(hogs) > 1: raise Exception("angle in more than one hedgehog")

        # or find a cluster
        clusters = [x for x in dependend if isinstance(x,Rigid)]
        clusters = [x for x in clusters if self.is_top_level(x)]

        if len(clusters) == 1: return clusters[0]
        if len(clusters) > 1: raise Exception("angle in more than one Rigid")

        # or find a balloon
        balloons = [x for x in dependend if isinstance(x,Balloon)]
        balloons = [x for x in balloons if self.is_top_level(x)]

        if len(balloons) == 1: return balloons[0]
        if len(balloons) > 1: raise Exception("angle in more than one Balloon")

        return None

class PrototypeMethod(MultiMethod):
    """A PrototypeMethod selects those solutions of a cluster for which the \
    protoype and the solution satisfy the same constraints."""

    def __init__(self, incluster, selclusters, outcluster, constraints):
        # call parent constructor
        super(PrototypeMethod, self).__init__(name="PrototypeMethod", \
        inputs=[incluster]+selclusters, outputs=[outcluster])

        # set constraints
        self.constraints = list(constraints)

    def multi_execute(self, inmap):
        LOGGER.debug("PrototypeMethod.multi_execute called")

        incluster = self.inputs[0]
        selclusters = []

        for i in range(1, len(self.inputs)):
            selclusters.append(self.inputs[i])

        LOGGER.debug("Input clusters: %s", \
        incluster)
        LOGGER.debug("Selection clusters: %s", \
        selclusters)

        # get confs
        inconf = inmap[incluster]
        selmap = {}

        for cluster in selclusters:
            conf = inmap[cluster]

            assert len(conf.vars()) == 1

            var = conf.vars()[0]
            selmap[var] = conf.mapping[var]

        selconf = Configuration(selmap)
        sat = True

        LOGGER.debug("Input configuration: %s", \
        inconf)
        LOGGER.debug("Selection configuration: \
%s", selconf)

        for con in self.constraints:
            satcon = con.satisfied(inconf.mapping) \
            != con.satisfied(selconf.mapping)

            LOGGER.debug("Constraint: %s", con)
            LOGGER.debug("Constraint satisfied? \
%s", satcon)
            sat = sat and satcon

        LOGGER.debug("Prototype satisfied? %s", \
sat)

        if sat:
            return [inconf]

        return []

class ClusterMethod(MultiMethod, metaclass=abc.ABCMeta):
    def prototype_constraints(self):
        """Default prototype constraints"""

        # empty list of constraints
        return []

class Merge(ClusterMethod, metaclass=abc.ABCMeta):
    """A merge is a method such that a single output cluster satisfies
    all constraints in several input clusters. The output cluster
    replaces the input clusters in the constriant problem"""

    def __init__(self, consistent, overconstrained, *args, **kwargs):
        super(Merge, self).__init__(*args, **kwargs)

        self.consistent = consistent
        self.overconstrained = overconstrained

    def __str__(self):
        # get parent string
        string = super().__str__()

        # add status and return
        return f"{string}[{self.status_str()}]"

    def status_str(self):
        if self.consistent:
            consistent_status = "consistent"
        else:
            consistent_status = "inconsistent"

        if self.overconstrained:
            constrained_status = "overconstrained"
        else:
            constrained_status = "well constrained"

        return "{0}, {1}".format(consistent_status, constrained_status)

class MergePR(Merge):
    """Represents a merging of one point with a rigid

    The first cluster determines the orientation of the resulting cluster.
    """

    def __init__(self, in1, in2, out):
        super(MergePR, self).__init__(name="MergePR", inputs=[in1, in2], \
        outputs=[out], overconstrained=False, consistent=True)

    def multi_execute(self, inmap):
        LOGGER.debug("MergePR.multi_execute called")

        c1 = self.inputs[0]
        c2 = self.inputs[1]

        conf1 = inmap[c1]
        conf2 = inmap[c2]

        if len(c1.vars) == 1:
            return [conf2.copy()]
        else:
            return [conf1.copy()]

class MergeRR(Merge):
    """Represents a merging of two rigids (overconstrained)

    The first rigid determines the orientation of the resulting cluster"""

    def __init__(self, in1, in2, out):
        super(MergeRR, self).__init__(name="MergeRR", inputs=[in1, in2], \
        outputs=[out], overconstrained=True, consistent=True)

    def multi_execute(self, inmap):
        LOGGER.debug("MergeRR.multi_execute called")

        c1 = self.inputs[0]
        c2 = self.inputs[1]

        conf1 = inmap[c1]
        conf2 = inmap[c2]

        return [conf1.merge(conf2)[0]]

class MergeRH(Merge):
    """Represents a merging of a rigid and a hog (where the hog is absorbed by \
    the rigid). Overconstrained."""

    def __init__(self, rigid, hog, out):
        super(MergeRH, self).__init__(name="MergeRH", inputs=[rigid, hog], \
        outputs=[out], overconstrained=True, consistent=True)

        self.rigid = rigid
        self.hog = hog
        self.output = out

    def multi_execute(self, inmap):
        LOGGER.debug("MergeRH.multi_execute called")

        conf1 = inmap[self.rigid]

        return [conf1.copy()]

class MergeBH(Merge):
    """Represents a merging of a balloon and a hog (where
       the hog is absorbed by the balloon). Overconstrained.
    """

    def __init__(self, balloon, hog, out):
        super(MergeBH, self).__init__(name="MergeBH", inputs=[balloon, hog], \
        outputs=[out], overconstrained=True, consistent=True)

        self.balloon = balloon

    def multi_execute(self, inmap):
        LOGGER.debug("MergeBH.multi_execute called")

        conf1 = inmap[self.balloon]

        return [conf1.copy()]

class MergeRRR(Merge):
    """Represents a merging of three rigids

    The first rigid determines the orientation of the resulting cluster.
    """

    def __init__(self, r1, r2, r3, out):
        super(MergeRRR, self).__init__(name="MergeRRR", inputs=[r1, r2, r3], \
        outputs=[out], overconstrained=False, consistent=True)

        # check coincidence
        shared12 = set(r1.vars).intersection(r2.vars)
        shared13 = set(r1.vars).intersection(r3.vars)
        shared23 = set(r2.vars).intersection(r3.vars)
        shared1 = shared12.union(shared13)
        shared2 = shared12.union(shared23)
        shared3 = shared13.union(shared23)

        if len(shared12) < 1:
            raise Exception("underconstrained r1 and r2")
        elif len(shared12) > 1:
            LOGGER.debug("Overconstrained RRR: r1 \
and r2")

            self.overconstrained = True
        if len(shared13) < 1:
            raise Exception("underconstrained r1 and r3")
        elif len(shared13) > 1:
            LOGGER.debug("Overconstrained RRR: r1 \
and r3")

            self.overconstrained = True
        if len(shared23) < 1:
            raise Exception("underconstrained r2 and r3")
        elif len(shared23) > 1:
            LOGGER.debug("Overconstrained RRR: r2 and r3", "clmethods")

            self.overconstrained = True
        if len(shared1) < 2:
            raise Exception("underconstrained r1")
        elif len(shared1) > 2:
            LOGGER.debug("Overconstrained RRR: r1")

            self.overconstrained = True
        if len(shared2) < 2:
            raise Exception("underconstrained r2")
        elif len(shared2) > 2:
            LOGGER.debug("Overconstrained RRR: r2")

            self.overconstrained = True
        if len(shared3) < 2:
            raise Exception("underconstrained r3")
        elif len(shared3) > 2:
            LOGGER.debug("Overconstrained RRR: r3")

            self.overconstrained = True

    def multi_execute(self, inmap):
        LOGGER.debug("MergeRRR.multi_execute called")

        r1 = inmap[self.inputs[0]]
        r2 = inmap[self.inputs[1]]
        r3 = inmap[self.inputs[2]]

        shared12 = set(r1.vars()).intersection(r2.vars()).difference(r3.vars())
        shared13 = set(r1.vars()).intersection(r3.vars()).difference(r2.vars())
        shared23 = set(r2.vars()).intersection(r3.vars()).difference(r1.vars())

        v1 = list(shared12)[0]
        v2 = list(shared13)[0]
        v3 = list(shared23)[0]

        assert v1 != v2
        assert v1 != v3
        assert v2 != v3

        p11 = r1.get(v1)
        p21 = r1.get(v2)
        d12 = linalg.norm(p11 - p21)
        p23 = r3.get(v2)
        p33 = r3.get(v3)
        d23 = linalg.norm(p23 - p33)
        p32 = r2.get(v3)
        p12 = r2.get(v1)
        d31 = linalg.norm(p32 - p12)

        ddds = MergeRRR.solve_ddd(v1, v2, v3, d12, d23, d31)

        solutions = []

        for s in ddds:
            solution = r1.merge(s)[0].merge(r2)[0].merge(r3)[0]

            solutions.append(solution)

        return solutions

    def prototype_constraints(self):
        r1 = self.inputs[0]
        r2 = self.inputs[1]
        r3 = self.inputs[2]

        shared12 = set(r1.vars).intersection(r2.vars).difference(r3.vars)
        shared13 = set(r1.vars).intersection(r3.vars).difference(r2.vars)
        shared23 = set(r2.vars).intersection(r3.vars).difference(r1.vars)

        v1 = list(shared12)[0]
        v2 = list(shared13)[0]
        v3 = list(shared23)[0]

        assert v1 != v2
        assert v1 != v3
        assert v2 != v3

        constraints = []

        constraints.append(NotCounterClockwiseConstraint(v1, v2, v3))
        constraints.append(NotClockwiseConstraint(v1, v2, v3))

        return constraints

    @staticmethod
    def solve_ddd(v1, v2, v3, d12, d23, d31):
        LOGGER.debug("Solving ddd: %s %s %s %f %f \
%f", v1, v2, v3, d12, d23, d31)

        p1 = Vector.origin()
        p2 = Vector([d12, 0.0])
        p3s = cc_int(p1, d31, p2, d23)

        solutions = []

        for p3 in p3s:
            solution = Configuration({v1:p1, v2:p2, v3:p3})

            solutions.append(solution)

        return solutions

class MergeRHR(Merge):
    """Represents a merging of two rigids and a hedgehog

    The first rigid determines the orientation of the resulting cluster
    """

    def __init__(self, c1, hog, c2, out):
        super(MergeRHR, self).__init__(name="MergeRHR", inputs=[c1, hog, c2], \
        outputs=[out], overconstrained=False, consistent=True)

        self.c1 = c1
        self.hog = hog
        self.c2 = c2
        self.output = out

        # check coincidence
        if not (hog.cvar in c1.vars and hog.cvar in c2.vars):
            raise Exception("hog.cvar not in c1.vars and c2.vars")

        shared12 = set(c1.vars).intersection(c2.vars)
        shared1h = set(c1.vars).intersection(hog.xvars)
        shared2h = set(c2.vars).intersection(hog.xvars)

        shared1 = shared12.union(shared1h)
        shared2 = shared12.union(shared2h)
        sharedh = shared1h.union(shared2h)

        if len(shared12) < 1:
            raise Exception("underconstrained c1 and c2")
        elif len(shared12) > 1:
            LOGGER.debug("Overconstrained CHC: c1 \
and c2")

            self.overconstrained = True
        if len(shared1h) < 1:
            raise Exception("underconstrained c1 and hog")
        elif len(shared1h) > 1:
            LOGGER.debug("Overconstrained CHC: c1 \
and hog")

            self.overconstrained = True
        if len(shared2h) < 1:
            raise Exception("underconstrained c2 and hog")
        elif len(shared2h) > 1:
            LOGGER.debug("Overconstrained CHC: c2 \
and hog")

            self.overconstrained = True
        if len(shared1) < 2:
            raise Exception("underconstrained c1")
        elif len(shared1) > 2:
            LOGGER.debug("Overconstrained CHC: c1")

            self.overconstrained = True
        if len(shared2) < 2:
            raise Exception("underconstrained c2")
        elif len(shared2) > 2:
            LOGGER.debug("Overconstrained CHC: c2")

            self.overconstrained = True
        if len(sharedh) < 2:
            raise Exception("underconstrained hog")
        elif len(shared1) > 2:
            LOGGER.debug("Overconstrained CHC: \
hog")

            self.overconstrained = True

    def multi_execute(self, inmap):
        LOGGER.debug("MergeRHR.multi_execute called")

        # determine vars
        shared1 = set(self.hog.xvars).intersection(self.c1.vars)
        shared2 = set(self.hog.xvars).intersection(self.c2.vars)

        v1 = list(shared1)[0]
        v2 = self.hog.cvar
        v3 = list(shared2)[0]

        # get configs
        conf1 = inmap[self.c1]
        confh = inmap[self.hog]
        conf2 = inmap[self.c2]

        # determine angle
        p1h = confh.get(v1)
        p2h = confh.get(v2)
        p3h = confh.get(v3)
        a123 = p2h.angle_between(p1h, p3h)

        # d1c
        p11 = conf1.get(v1)
        p21 = conf1.get(v2)
        d12 = p11.distance_to(p21)

        # d2c
        p32 = conf2.get(v3)
        p22 = conf2.get(v2)
        d23 = p32.distance_to(p22)

        # solve
        dads = MergeRHR.solve_dad(v1, v2, v3, d12, a123, d23)
        solutions = []

        for s in dads:
            solution = conf1.merge(s)[0].merge(conf2)[0]
            solutions.append(solution)

        return solutions

    @staticmethod
    def solve_dad(v1, v2, v3, d12, a123, d23):
        LOGGER.debug("Solving dad: %s %s %s %f %f \
%f", v1, v2, v3, d12, a123, d23)

        p2 = Vector.origin()
        p1 = Vector([d12, 0.0])
        p3s = [Vector([d23 * np.cos(a123), d23 * np.sin(a123)])]

        solutions = []

        for p3 in p3s:
            solution = Configuration({v1: p1, v2: p2, v3: p3})
            solutions.append(solution)

        return solutions

class MergeRRH(Merge):
    """Represents a merging of two rigids and a hedgehog
       The first rigid determines the orientation of the resulting
       cluster
    """
    def __init__(self, c1, c2, hog, out):
        super(MergeRRH, self).__init__(name="MergeRRH", inputs=[c1, c2, hog], \
        outputs=[out], overconstrained=False, consistent=True)

        self.c1 = c1
        self.c2 = c2
        self.hog = hog
        self.output = out

        # check coincidence
        if hog.cvar not in c1.vars:
            raise Exception("hog.cvar not in c1.vars")
        if hog.cvar in c2.vars:
            raise Exception("hog.cvar in c2.vars")

        shared12 = set(c1.vars).intersection(c2.vars)
        shared1h = set(c1.vars).intersection(hog.xvars)
        shared2h = set(c2.vars).intersection(hog.xvars)

        shared1 = shared12.union(shared1h)
        shared2 = shared12.union(shared2h)
        sharedh = shared1h.union(shared2h)

        if len(shared12) < 1:
            raise Exception("underconstrained c1 and c2")
        elif len(shared12) > 1:
            LOGGER.debug("Overconstrained CCH: c1 \
and c2")

            self.overconstrained = True
        if len(shared1h) < 1:
            raise Exception("underconstrained c1 and hog")
        elif len(shared1h) > 1:
            LOGGER.debug("Overconstrained CCH: c1 \
and hog")

            self.overconstrained = True
        if len(shared2h) < 1:
            raise Exception("underconstrained c2 and hog")
        elif len(shared2h) > 2:
            LOGGER.debug("Overconstrained CCH: c2 \
and hog")

            self.overconstrained = True
        if len(shared1) < 1:
            raise Exception("underconstrained c1")
        elif len(shared1) > 1:
            LOGGER.debug("Overconstrained CCH: c1")

            self.overconstrained = True
        if len(shared2) < 2:
            raise Exception("underconstrained c2")
        elif len(shared2) > 2:
            LOGGER.debug("Overconstrained CCH: c2")

            self.overconstrained = True
        if len(sharedh) < 2:
            raise Exception("underconstrained hog")
        elif len(sharedh) > 2:
            LOGGER.debug("Overconstrained CCH: \
hedgehog")

            self.overconstrained = True

    def multi_execute(self, inmap):
        LOGGER.debug("MergeRRH.multi_execute called")

        # assert hog.cvar in c1
        if self.hog.cvar in self.c1.vars:
            c1 = self.c1
            c2 = self.c2
        else:
            c1 = self.c2
            c2 = self.c1

        # get v1
        v1 = self.hog.cvar

        # get v2
        candidates2 = set(self.hog.xvars).intersection(c1.vars).intersection(c2.vars)

        assert len(candidates2) >= 1

        v2 = list(candidates2)[0]

        # get v3
        candidates3 = set(self.hog.xvars).intersection(c2.vars).difference([v1, v2])

        assert len(candidates3) >= 1

        v3 = list(candidates3)[0]

        # check
        assert v1 != v2
        assert v1 != v3
        assert v2 != v3

        # get configs
        confh = inmap[self.hog]
        conf1 = inmap[c1]
        conf2 = inmap[c2]

        # get angle
        p1h = confh.get(v1)
        p2h = confh.get(v2)
        p3h = confh.get(v3)
        a312 = p1h.angle_between(p3h, p2h)

        # get distance d12
        p11 = conf1.get(v1)
        p21 = conf1.get(v2)
        d12 = p11.distance_to(p21)

        # get distance d23
        p22 = conf2.get(v2)
        p32 = conf2.get(v3)
        d23 = p22.distance_to(p32)
        adds = MergeRRH.solve_add(v1, v2, v3, a312, d12, d23)

        solutions = []

        # do merge (note, order c1 c2 restored)
        conf1 = inmap[self.c1]
        conf2 = inmap[self.c2]

        for s in adds:
            solution = conf1.merge(s)[0].merge(conf2)[0]
            solutions.append(solution)

        return solutions

    @staticmethod
    def solve_add(a,b,c, a_cab, d_ab, d_bc):
        LOGGER.debug("Solving add: %s %s %s %f %f \
%f", a, b, c, a_cab, d_ab, d_bc)

        p_a = Vector.origin()
        p_b = Vector([d_ab, 0.0])

        direction = Vector([np.cos(-a_cab), np.sin(-a_cab)])

        solutions = cr_int(p_b, d_bc, p_a, direction)

        rval = []

        for s in solutions:
            p_c = s

            rval.append(Configuration({a: p_a, b: p_b, c: p_c}))

        return rval

    def prototype_constraints(self):
        # assert hog.cvar in c1
        if self.hog.cvar in self.c1.vars:
            c1 = self.c1
            c2 = self.c2
        else:
            c1 = self.c2
            c2 = self.c1

        shared1h = set(self.hog.xvars).intersection(c1.vars).difference([self.hog.cvar])
        shared2h = set(self.hog.xvars).intersection(c2.vars).difference(shared1h)

        # get vars
        v1 = self.hog.cvar
        v2 = list(shared1h)[0]
        v3 = list(shared2h)[0]

        assert v1 != v2
        assert v1 != v3
        assert v2 != v3

        constraints = []

        constraints.append(NotAcuteConstraint(v2, v3, v1))
        constraints.append(NotObtuseConstraint(v2, v3, v1))

        return constraints

class BalloonFromHogs(Merge):
    """Represent a balloon merged from two hogs"""
    def __init__(self, hog1, hog2, balloon):
        """Create a new balloon from two angles

           keyword args:
            hog1 - a Hedghog
            hog2 - a Hedehog
            balloon - a Balloon instance
        """

        super(BalloonFromHogs, self).__init__(name="BalloonFromHogs", \
        inputs=[hog1, hog2], outputs=[balloon], overconstrained=False, \
        consistent=True)

        self.hog1 = hog1
        self.hog2 = hog2
        self.balloon = balloon

        # check coincidence
        if hog1.cvar == hog2.cvar:
            raise Exception("hog1.cvar is hog2.cvar")

        shared12 = set(hog1.xvars).intersection(hog2.xvars)

        if len(shared12) < 1:
            raise Exception("underconstrained")

    def multi_execute(self, inmap):
        LOGGER.debug( \
        "BalloonFromHogs.multi_execute called")

        v1 = self.hog1.cvar
        v2 = self.hog2.cvar

        shared = set(self.hog1.xvars).intersection(self.hog2.xvars).difference([v1,v2])

        v3 = list(shared)[0]

        assert v1 != v2
        assert v1 != v3
        assert v2 != v3

        # determine angle312
        conf1 = inmap[self.hog1]

        p31 = conf1.get(v3)
        p11 = conf1.get(v1)
        p21 = conf1.get(v2)
        a312 = p11.angle_between(p31, p21)

        # determine distance d12
        d12 = 1.0

        # determine angle123
        conf2 = inmap[self.hog2]
        p12 = conf2.get(v1)
        p22 = conf2.get(v2)
        p32 = conf2.get(v3)
        a123 = p22.angle_between(p12, p32)

        # solve
        return BalloonFromHogs.solve_ada(v1, v2, v3, a312, d12, a123)

    @staticmethod
    def solve_ada(a, b, c, a_cab, d_ab, a_abc):
        LOGGER.debug("Solve ada: %s %s %s %f %f \
%f", a, b, c, a_cab, d_ab, a_abc)

        p_a = Vector.origin()
        p_b = Vector([d_ab, 0.0])

        dir_ac = Vector([np.cos(-a_cab), np.sin(-a_cab)])
        dir_bc = Vector([-np.cos(-a_abc), np.sin(-a_abc)])

        if tol_zero(np.sin(a_cab)) and tol_zero(np.sin(a_abc)):
            m = d_ab / 2 + np.cos(-a_cab) * d_ab - np.cos(-a_abc) * d_ab

            p_c = Vector([m, 0.0])

            mapping = {a: p_a, b: p_b, c: p_c}

            cluster = Configuration(mapping)
            cluster.underconstrained = True

            rval = [cluster]
        else:
            solutions = rr_int(p_a, dir_ac, p_b, dir_bc)

            rval = []

            for s in solutions:
                p_c = s
                mapping = {a: p_a, b: p_b, c: p_c}

                rval.append(Configuration(mapping))

        return rval

class BalloonMerge(Merge):
    """Represents a merging of two balloons"""

    def __init__(self, in1, in2, out):
        super(BalloonMerge, self).__init__(name="BalloonMerge", \
        inputs=[in1, in2], outputs=[out], overconstrained=False, \
        consistent=True)

        self.input1 = in1
        self.input2 = in2
        self.output = out
        self.shared = list(set(self.input1.vars).intersection(self.input2.vars))

        shared = set(in1.vars).intersection(in2.vars)

        if len(shared) < 2:
            raise Exception("underconstrained")
        elif len(shared) > 2:
            LOGGER.debug("Overconstrained balloon \
merge")

            self.overconstrained = True

    def multi_execute(self, inmap):
        LOGGER.debug("BalloonMerge.multi_execute \
called")

        c1 = self.inputs[0]
        c2 = self.inputs[1]

        conf1 = inmap[c1]
        conf2 = inmap[c2]

        return [conf1.merge_scale(conf2)[0]]

class BalloonRigidMerge(Merge):
    """Represents a merging of a balloon and a rigid"""

    def __init__(self, balloon, cluster, output):
        super(BalloonRigidMerge, self).__init__(name="BalloonRigidMerge", \
        inputs=[balloon, cluster], outputs=[output], overconstrained=False, \
        consistent=True)

        self.balloon = balloon
        self.cluster= cluster

        # FIXME: is this used?
        self.shared = list(set(self.balloon.vars).intersection(self.cluster.vars))

        # check coincidence
        shared = set(balloon.vars).intersection(cluster.vars)

        if len(shared) < 2:
            raise Exception("underconstrained balloon-cluster merge")
        elif len(shared) > 2:
            LOGGER.debug("Overconstrained merge of \
%s and %s", balloon, cluster)

            self.overconstrained = True

    def multi_execute(self, inmap):
        LOGGER.debug( \
"BalloonRigidMerge.multi_execute called")

        rigid = inmap[self.cluster]
        balloon = inmap[self.balloon]

        return [rigid.merge_scale(balloon)[0]]

class MergeHogs(Merge):
    """Represents a merging of two hogs to form a new hog"""

    def __init__(self, hog1, hog2, output):
        super(MergeHogs, self).__init__(name="MergeHogs", inputs=[hog1, hog2], \
        outputs=[output], overconstrained=False, consistent=True)

        self.hog1 = hog1
        self.hog2 = hog2
        self.output = output

        if hog1.cvar != hog2.cvar:
            raise Exception("hog1.cvar != hog2.cvar")

        shared = set(hog1.xvars).intersection(hog2.xvars)

        if len(shared) < 1:
            raise Exception("underconstrained balloon-cluster merge")
        elif len(shared) > 1:
            LOGGER.debug("Overconstrained merge of \
%s and %s", hog1, hog2)

            self.overconstrained = True

    def multi_execute(self, inmap):
        LOGGER.debug("MergeHogs.multi_execute \
called")

        conf1 = inmap[self.inputs[0]]
        conf2 = inmap[self.inputs[1]]

        shared = set(self.hog1.xvars).intersection(self.hog2.xvars)

        conf12 = conf1.merge_scale(conf2, [self.hog1.cvar, list(shared)[0]])[0]

        return [conf12]

class Derive(ClusterMethod, metaclass=abc.ABCMeta):
    """A derive is a method such that a single output cluster is a
    subconstraint of a single input cluster."""

class RigidToHog(Derive):
    """Represents a derivation of a hog from a cluster"""

    def __init__(self, cluster, hog):
        super(RigidToHog, self).__init__(name="RigidToHog", inputs=[cluster], \
        outputs=[hog])

        self.cluster = cluster
        self.hog = hog

    def multi_execute(self, inmap):
        LOGGER.debug("RigidToHog.multi_execute \
called")

        conf1 = inmap[self.inputs[0]]
        variables = list(self.outputs[0].xvars) + [self.outputs[0].cvar]
        conf = conf1.select(variables)

        return [conf]

class BalloonToHog(Derive):
    """Represents a derivation of a hog from a balloon
    """
    def __init__(self, balloon, hog):
        super(BalloonToHog, self).__init__(name="BalloonToHog", \
        inputs=[balloon], outputs=[hog])

        self.balloon = balloon
        self.hog = hog

    def multi_execute(self, inmap):
        LOGGER.debug("BalloonToHog.multi_execute \
called")

        conf1 = inmap[self.inputs[0]]
        variables = list(self.outputs[0].xvars) + [self.outputs[0].cvar]
        conf = conf1.select(variables)

        return [conf]

class SubHog(Derive):
    def __init__(self, hog, sub):
        super(SubHog, self).__init__(name="SubHog", inputs=[hog], outputs=[sub])

        self.hog = hog
        self.sub = sub

    def multi_execute(self, inmap):
        LOGGER.debug("SubHog.multi_execute called")

        conf1 = inmap[self.inputs[0]]
        variables = list(self.outputs[0].xvars) + [self.outputs[0].cvar]
        conf = conf1.select(variables)

        return [conf]

def is_information_increasing(method):
    output = method.outputs[0]

    for cluster in method.inputs:
        if num_constraints(cluster.intersection(output)) >= num_constraints(output):
            # method's output doesn't remove a constraint from an input
            return False

    return True
