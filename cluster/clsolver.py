"""A generic geometric constraint solver.

This module provides basic functionality for
ClusterSolver2D and ClusterSolver3D.
"""

import logging

from .graph import Graph
from .method import Method, MethodGraph
from .notify import Notifier
from .multimethod import MultiVariable, MultiMethod
from .cluster import *
from .configuration import Configuration
from .gmatch import gmatch
from .method import OrMethod,SetMethod
from .incremental import IncrementalSet,MutableSet,Union,Filter

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------
# ---------- ClusterSolver main class --------------
# --------------------------------------------------

class ClusterSolver(Notifier):
    """
    Finds a generic solution for problems formulated by Clusters.

    Cluster are added and removed using the add and remove methods.
    After adding each Cluster, the solver tries to merge it with
    others, resulting in new Clusters.

    The generic solution is a directed acyclic graph of Clusters and Methods.
    Particilar problems and solutions are represented by a Configuration
    for each cluster.

    For each Cluster a set of Configurations can be set using the
    set method. Configurations are propagated via Methods and can
    be retrieved with the get method.
    """

    # ------- PUBLIC METHODS --------

    def __init__(self, methodclasses):
        """Create a new solver, using the given subclasses of ClusterMethod."""
        # init superclasses
        Notifier.__init__(self)
        # store arguments
        self._methodclasses = methodclasses
        self._pattern_methods = [m for m in self._methodclasses if hasattr(m,"patterngraph")]
        self._handcoded_methods = [m for m in self._methodclasses if hasattr(m,"handcoded_match")]
        self._incremental_methods = [m for m in self._methodclasses if hasattr(m,"incremental_matcher")]
        # init instance vars
        self._graph = Graph()
        self._graph.add_vertex("_variables")
        self._graph.add_vertex("_clusters")
        self._graph.add_vertex("_methods")
        self._new = []
        self._mg = MethodGraph()
        # add prototype_selection boolean var to method graph
        self._prototype_selection_var = "_prototype_selection_enabled"
        self._mg.add_variable(self._prototype_selection_var)
        self._mg.set(self._prototype_selection_var, True)
        # store map of selection_constraints to SelectionMethod (or None)
        self._selection_method = {}
        # store root cluster (will be assigned when first cluster added)
        self._rootcluster = None
        # an incrementally updated toplevel set
        self._toplevel = MutableSet()
        # incrementally updated set of applicable methods
        self._incremental_matchers = [method.incremental_matcher(self) for method in self._incremental_methods]
        #print "incremental matchers:",self._incremental_matchers
        self._applicable_methods = Union(*self._incremental_matchers)

    # ------- methods for setting up constraint problems ------------

    def add(self, cluster):
        """Add a cluster"""
        LOGGER.debug(f"adding cluster '{cluster}'")
        self._add_cluster(cluster)
        self._process_new()

    def remove(self, cluster):
        """Remove a cluster.
           All dependend objects are also removed.
        """
        LOGGER.debug(f"removing cluster '{cluster}'")
        self._remove(cluster)
        self._process_new()

    def set(self, cluster, configurations):
        """Associate a list of configurations with a cluster"""
        LOGGER.debug(f"setting configurations for cluster '{cluster}' to {configurations}")
        self._mg.set(cluster, configurations)

    def get(self, cluster):
        """Return a set of configurations associated with a cluster"""
        return self._mg.get(cluster)

    def set_root(self, cluster):
        """Set root cluster, used for positionig and orienting the solutions"""
        LOGGER.debug(f"set root to '{self._rootcluster}'")
        if self._rootcluster != None:
            oldrootvar = rootname(self._rootcluster)
            self._mg.set(oldrootvar, False)
        newrootvar = rootname(cluster)
        self._mg.set(newrootvar, True)
        self._rootcluster = cluster

    def get_root(self):
        """returns current root cluster or None"""
        return self._rootcluster

    def set_prototype_selection(self, enabled):
        """Enable or disable prototype-based solution selection"""
        self._mg.set(self._prototype_selection_var, enabled)

    def add_selection_constraint(self, con):
        """Add a SelectionConstraint to filter solutions"""
        if con not in self._selection_method:
            selector = self._find_selection_method(con)
            if selector != None:
                selector.add_constraint(con)
                self._selection_method[con] = selector
                self._mg.execute(selector)
            #self._selection_method[con] = None     # this line wrong?
            self._selection_method[con] = selector     # this line better?

    def rem_selection_constraint(self, con):
        """Remove a SelectionConstraint"""
        if con in self._selection_method:
            selector = self._selection_method[con]
            if selector != None:
                selector.rem_constraint(con)
                self._mg.execute(selector)
            del self._selection_method[con]

    # ------- methods for inspecting the state of the solver ------------

    def variables(self):
        """get list of variables"""
        return self._graph.outgoing_vertices("_variables")

    def clusters(self):
        """get list of clusters"""
        return self._graph.outgoing_vertices("_clusters")

    def methods(self):
        """get list of methods"""
        return self._graph.outgoing_vertices("_methods")

    def top_level(self):
        """return IncrementalSet of top-level clusters"""
        return self._toplevel
        # return self._graph.outgoing_vertices("_toplevel")

    def is_top_level(self, object):
        """Returns True iff given cluster is a top-level cluster"""
        #return self._graph.has_edge("_toplevel",object)
        return object in self._toplevel

    def find_dependend(self, object):
        """Return a list of objects that depend on given object directly."""
        l = self._graph.outgoing_vertices(object)
        return [x for x in l if self._graph.get(object,x) == "dependency"]

    def find_depends(self, object):
        """Return a list of objects that the given object depends on directly"""
        l = self._graph.ingoing_vertices(object)
        return [x for x in l if self._graph.get(x,object) == "dependency"]

    def contains(self, obj):
        return self._graph.has_vertex(obj)

    # ------------ INTERNALLY USED METHODS --------

    # --- dependencies and groups

    def _add_dependency(self, on, dependend):
        """Add a dependence for second object on first object

        Used to delete dependent nodes when another node is deleted.
        """
        self._graph.add_edge(on, dependend, "dependency")

    def _add_to_group(self, group, object):
        """Add object to group"""
        self._graph.add_edge(group, object, "contains")

    def _add_needed_by(self, needed, by):
        """Add relation 'needed' object is needed 'by'"""
        self._graph.add_edge(needed, by, "needed_by")

    def _objects_that_need(self, needed):
        """Return objects needed by given object"""
        l = self._graph.outgoing_vertices(needed)
        return [x for x in l if self._graph.get(needed,x) == "needed_by"]

    def _objects_needed_by(self, needer):
        """Return objects needed by given object"""
        l = self._graph.ingoing_vertices(needer)
        return [x for x in l if self._graph.get(x,needer) == "needed_by"]

    def _add_top_level(self, cluster):
        # self._graph.add_edge("_toplevel",cluster)
        self._new.append(cluster)
        self._toplevel.add(cluster)

    def _rem_top_level(self, object):
        # self._graph.rem_edge("_toplevel",object)
        if object in self._new:
            self._new.remove(object)
        self._toplevel.remove(object)

    def _find_descendend(self,v):
        """find all descendend objects of v (i.e.. directly or indirectly dependend)"""
        front = [v]
        result = {}
        while len(front) > 0:
            x = front.pop()
            if x not in result:
                result[x] = 1
                front += self.find_dependend(x)
        del result[v]
        return list(result)


    # -- add object types

    def _add_variable(self, var):
        if not self._graph.has_vertex(var):
            LOGGER.debug(f"_add_variable '{var}'")
            self._add_to_group("_variables", var)

    def _add_cluster(self, newcluster):
        # check if not already exists
        if self._graph.has_vertex(newcluster):
            raise Exception("cluster %s already in clsolver"%(str(newcluster)))
        # update graph
        self._add_to_group("_clusters", newcluster)
        for var in newcluster.vars:
            self._add_variable(var)
            self._add_dependency(var, newcluster)
        # add to top level
        self._add_top_level(newcluster)
        # add to methodgraph
        self._mg.add_variable(newcluster)
        # add root-variable if needed with default value False
        root = rootname(newcluster)
        if not self._mg.contains(root):
            self._mg.add_variable(root, False)
            self._mg.set(root, False)
            # add root-variable to dependency graph
            self._add_dependency(newcluster, root)
        # if there is no root cluster, this one will be it
        if self.get_root() == None:
            self.set_root(newcluster)
        # notify listeners
        self.send_notify(("add", newcluster))

    def _add_method(self, method):
        self._add_to_group("_methods", method)
        for obj in method.inputs():
            self._add_dependency(obj, method)
        for obj in method.outputs():
            self._add_dependency(method, obj)
            self._add_dependency(obj, method)
        self._mg.add_method(method)
        self.send_notify(("add", method))

    # ----- solution selection

    def _add_prototype_selector(self, merge):
        incluster = merge.outputs()[0]
        constraints = merge.prototype_constraints()
        vars = set()
        for con in constraints:
            vars.update(con.variables())
        selclusters = []
        for var in vars:
            clusters = self._graph.outgoing_vertices(var)
            clusters = [c for c in clusters if isinstance(c, Rigid)]
            clusters = [c for c in clusters if len(c.vars) == 1]
            if len(clusters) < 1:
                raise Exception("no prototype cluster for variable "+str(var))
            elif len(clusters) > 1:
                raise Exception("more than one candidate prototype cluster for variable "+str(var))
            selclusters.append(clusters[0])
        outcluster = incluster.copy()
        selector = PrototypeMethod(incluster, selclusters, outcluster, constraints, self._prototype_selection_var)
        self._add_cluster(outcluster)
        self._add_method(selector)
        self._rem_top_level(incluster)
        return outcluster

    def _add_solution_selector(self, incluster):
        outcluster = incluster.copy()
        selector = SelectionMethod(incluster, outcluster)
        constraints = self._find_selection_constraints(incluster)
        for con in constraints:
            selector.add_constraint(con)
            self._selection_method[con] = selector
        self._add_cluster(outcluster)
        self._add_method(selector)
        self._rem_top_level(incluster)
        return selector

    def _find_selection_method(self, con):
        # find clusters containing all constraints vars
        candidates = None
        for var in con.variables():
            # find clusters
            clusters = set(self.find_dependend(var))
            if candidates == None:
                candidates = clusters
            else:
                candidates = candidates.intersection(clusters)
        # get selection methods of clusters
        methods = []
        for cluster in candidates:
            methods += [c for c in self.find_depends(cluster) if isinstance(c,SelectionMethod)]
        # get selection method with smallest cluster
        if len(methods)>0:
            method = min(methods, key=lambda m: len(m.inputs()[0].vars))
            return method
        else:
            return None

        ##slow implementation, better would be to find method via clustering information in graph
        #convars = set(con.variables())
        #selmethods = filter(lambda x: isinstance(x,SelectionMethod), self.methods())
        #for method in selmethods:
        #    incluster = method.inputs()[0]
        #    clvars = set(incluster.vars)
        #    if clvars.intersection(convars) == convars:
        #        return method
        #return None

    def _find_selection_constraints(self, incluster):
        applicable = []
        for con in self._selection_method:
            selector = self._selection_method[con]
            if selector == None:
                convars = set(con.variables())
                clvars = set(incluster.vars)
                if convars.intersection(clvars) == convars:
                    applicable.append(con)
        return applicable



    # --------------
    # isearch methods
    # --------------

    def _process_new(self):
        # try incremental matchers and old style matching alternatingly
        non_redundant_methods = [m for m in self._applicable_methods if not self._is_redundant_method(m)]
        while len(non_redundant_methods) > 0 or len(self._new) > 0:
            # check incremental matches
            if len(non_redundant_methods) > 0:
                method = next(iter(non_redundant_methods))
                #print "applicable methods:", map(str, self._applicable_methods)
                LOGGER.debug(f"incremental search found method '{method}'")
                self._add_method_complete(method)
            else:
                newobject = self._new.pop()
                success = self._search(newobject)
                if success and self.is_top_level(newobject):
                    # maybe more rules applicable.... push back on stack
                    self._new.append(newobject)
                #endif
            # endif
            non_redundant_methods = [m for m in self._applicable_methods if not self._is_redundant_method(m)]
        # endwhile
    #end def

    def _search(self, newcluster):
        LOGGER.debug(f"searching from cluster '{newcluster}'")
        # find all toplevel clusters connected to newcluster
        # via one or more variables
        connected = set()
        for var in newcluster.vars:
            dependend = self.find_dependend(var)
            dependend = [x for x in dependend if self.is_top_level(x)]
            connected.update(dependend)
        LOGGER.debug(f"connected clusters: {connected}")

        # first try handcoded matching
        for methodclass in self._handcoded_methods:
            LOGGER.debug(f"trying handcoded match for method '{methodclass}'")
            matches = methodclass.handcoded_match(self, newcluster, connected)
            if self._try_matches(methodclass, matches):
                return True

        # if incremental matching failed, try full pattern matching
        if self._try_methods(connected):
            return True
        return False


    def _try_methods(self, nlet):
        """finds a possible rewrite rule applications on given set of clusters, applies it
           and returns True iff successfull
        """
        refgraph = reference2graph(nlet)
        for methodclass in self._pattern_methods:
            LOGGER.debug(f"trying generic pattern matching for method '{methodclass}'")
            matches = gmatch(methodclass.patterngraph, refgraph)
            if self._try_matches(methodclass,matches):
                return True
            # end for match
        # end for method
        return False

    def _try_matches(self, methodclass, matches):
        # print "method="+str(methodclass),"number of matches = "+str(len(matches))
        for s in matches:
            LOGGER.debug(f"trying match '{s}'")
            method = methodclass(*[s])
            succes = self._add_method_complete(method)
            if succes:
                #raw_input()
                #print "press key"
                return True
            else:    # WARING: fast bailout, may be incoplete!
                return False
        # end for match
        return False

    def _is_information_increasing(self, merge):
        # check that the method is information increasing (infinc)
        output = merge.outputs()[0]
        infinc = True
        connected = set()
        for var in output.vars:
            dependend = self.find_dependend(var)
            dependend = [x for x in dependend if self.is_top_level(x)]
            connected.update(dependend)
        # NOTE 07-11-2007 (while writing the paper): this  implementation of information increasing may not be correct. We may need to check that the total sum of the information in the overlapping clusters is equal to the information in the output.
        for cluster in connected:
            if num_constraints(cluster.intersection(output)) >= num_constraints(output):
                infinc = False
                break

        LOGGER.debug(f"information increasing: {infinc}")

        return infinc


    def _is_cluster_reducing(self, merge):
        # check if method reduces number of clusters (reduc)
        output = merge.outputs()[0]
        nremove = 0
        for cluster in merge.input_clusters():
            if num_constraints(cluster.intersection(output)) >= num_constraints(cluster):
               # will be removed from toplevel
               nremove += 1
        # exeption if method sets noremove flag
        if hasattr(merge,"noremove") and merge.noremove == True:
            nremove = 0
        reduc = (nremove > 1)

        LOGGER.debug(f"reduce number of clusters: {reduc}")

        return reduc

    def _is_redundant_method(self, merge):
        # check if the method is redundant (not information increasing and not reducing number of clusters)
        infinc = self._is_information_increasing(merge)
        reduc = self._is_cluster_reducing(merge)
        if not infinc and not reduc:
            LOGGER.debug(f"method '{merge}' is redundant")
            return True
        else:
            LOGGER.debug(f"method '{merge}' is not redundant")
            return False

    def _add_method_complete(self, merge):
        # do not add if method is redundant
        if self._is_redundant_method(merge):
            return False

        output = merge.outputs()[0]

        # check consistency and local/global overconstrained
        consistent = True
        local_oc = False
        for i1 in range(0, len(merge.input_clusters())):
            for i2 in range(i1+1, len(merge.input_clusters())):
                c1 = merge.input_clusters()[i1]
                c2 = merge.input_clusters()[i2]
                if num_constraints(c1.intersection(c2)) != 0:
                    local_oc = True
                consistent = consistent and self._is_consistent_pair(c1, c2)
        merge.consistent = consistent
        merge.overconstrained = local_oc

        # global overconstrained? (store in output cluster)
        overconstrained = not consistent
        for cluster in merge.input_clusters():
            overconstrained = overconstrained or cluster.overconstrained
        output.overconstrained = overconstrained

        # determine infinc before adding (used later)
        infinc = self._is_information_increasing(merge)

        # add to graph
        self._add_cluster(output)
        self._add_method(merge)

        # remove input clusters from top_level
        merge.restore_toplevel = []    # make restore list in method
        for cluster in merge.input_clusters():
            # do not remove rigids from toplevel if method does not consider root
            if isinstance(cluster, Rigid):
                if hasattr(merge,"noremove") and merge.noremove == True:
                    LOGGER.debug(f"cluster '{cluster}' is blocking the top-level")
                    break
            # remove input clusters when all its constraints are in output cluster
            if num_constraints(cluster.intersection(output)) >= num_constraints(cluster):
                LOGGER.debug(f"removing cluster '{cluster}' from top-level")
                self._rem_top_level(cluster)
                merge.restore_toplevel.append(cluster)
            else:
                LOGGER.debug(f"keeping cluster '{cluster}' at top-level")

        # add method to determine root-variable
        if hasattr(merge,"noremove") and merge.noremove == True:
            self._add_root_false(merge.outputs()[0])
        else:
            self._add_root_method(merge.input_clusters(),merge.outputs()[0])
        # add solution selection methods, only if information increasing
        if infinc:
            output2 = self._add_prototype_selector(merge)
            output3 = self._add_solution_selector(output2)

        # success
        return True

    def _add_root_method(self,inclusters,outcluster):
        inroots = []
        for cluster in inclusters:
            inroots.append(rootname(cluster))
        outroot = rootname(outcluster)
        method = OrMethod(inroots, outroot)
        # add method
        self._add_method(method)
        # make sure its deleted when cluster is deleted
        self._add_dependency(outcluster, method)

    def _add_root_false(self,outcluster):
        outroot = rootname(outcluster)
        method = SetMethod(outroot, False)
        # add method
        self._add_method(method)
        # make sure its deleted when cluster is deleted
        self._add_dependency(outcluster, method)


    # -- removing objects

    def _remove(self, object):
        # find all indirectly dependend objects
        todelete = [object]+self._find_descendend(object)
        torestore = set()
        # remove all objects
        for item in todelete:
            # if merge removed items from toplevel then add them back to top level
            if hasattr(item, "restore_toplevel"):
                for cluster in item.restore_toplevel:
                    torestore.add(cluster)
            # delete it from graph
            LOGGER.debug(f"deleting '{item}'")
            self._graph.rem_vertex(item)
            # remove from _new list
            if item in self._new:
                self._new.remove(item)
            # remove from incremental top_level
            self._toplevel.remove(item)
            # remove from methodgraph
            if isinstance(item, Method):
                # note: method may have been removed because variable removed
                try:
                    self._mg.rem_method(item)
                except:
                    pass
                # restore SelectionConstraints
                if isinstance(item, SelectionMethod):
                    for con in item.iter_constraints():
                        self._selection_method[con] = None
            if isinstance(item, MultiVariable):
                self._mg.rem_variable(item)
            # remove variables with no dependent clusters
            if isinstance(item, Cluster):
                for var in item.vars:
                    if len(self.find_dependend(var)) == 0:
                        self._graph.rem_vertex(var)
            # notify listeners
            self.send_notify(("remove", item))
        # restore toplevel (also added to _new)
        for cluster in torestore:
            if self._graph.has_vertex(cluster):
                self._add_top_level(cluster)


    ##def _contains_root(self, input_cluster):
    ##   """returns True iff input_cluster is root cluster or was determined by
    ##    merging with the root cluster."""
    ##
    ##    # start from root cluster. Follow merges upwards until:
    ##    #  - input cluster found -> True
    ##    #  - no more merges -> False
    ##
    ##    if len(self._graph.outgoing_vertices("_root")) > 1:
    ##        raise StandardError, "more than one root cluster"
    ##    if len(self._graph.outgoing_vertices("_root")) == 1:
    ##        cluster = self._graph.outgoing_vertices("_root")[0]
    ##    else:
    ##        cluster = None
    ##    while (cluster != None):
    ##        if cluster is input_cluster:
    ##            return True
    ##        fr = self._graph.outgoing_vertices(cluster)
    ##        me = filter(lambda x: isinstance(x, Merge), fr)
    ##        me = filter(lambda x: cluster in x.outputs(), me)
    ##        if len(me) > 1:
    ##            raise StandardError, "root cluster merged more than once"
    ##        elif len(me) == 0:
    ##            cluster = None
    ##        elif len(me[0].outputs()) != 1:
    ##            raise StandardError, "a merge with number of outputs != 1"
    ##        else:
    ##            cluster = me[0].outputs()[0]
    ##    #while
    ##    return False
    #def

    # ---- consistency

    def _is_consistent_pair(self, object1, object2):
        oc = over_constraints(object1, object2)
        oc_list = [str(c) for c in oc]
        LOGGER.debug(f"overconstraints between '{object1}' and '{object2}': {oc_list}")
        consistent = True
        for con in oc:
            consistent = consistent and self._consistent_overconstraint_in_pair(con, object1, object2)
        LOGGER.debug(f"'{object1}' and '{object2}' globally consistent: {consistent}")
        return consistent

    def _consistent_overconstraint_in_pair(self, overconstraint, object1, object2):
        # get sources for constraint in given clusters
        s1 = self._source_constraint_in_cluster(overconstraint, object1)
        s2 = self._source_constraint_in_cluster(overconstraint, object2)

        if s1 == None:
            consistent = False
        elif s2 == None:
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
            #c1to2 = constraits_from_s1_in_s2(s1, s2)
            #if solve(c1to2) contains overconstraint then consistent
            #c2to1 = constraits_from_s1_in_s2(s2, s1)
            #if solve(c2to1) contains overconstraint then consistent
            #raise StandardError, "not yet implemented"

        LOGGER.debug(f"'{object1}' and '{object2}' consistent: {consistent}")
        return consistent

    def _source_constraint_in_cluster(self, constraint, cluster):
        if not self._contains_constraint(cluster, constraint):
            raise Exception("constraint not in cluster")
        elif self._is_atomic(cluster):
            return cluster
        else:
            method = self._determining_method(cluster)
            inputs = method.inputs()
            down = [x for x in inputs if self._contains_constraint(x, constraint)]
            if len(down) == 0:
                return cluster
            elif len(down) > 1:
                if method.consistent == True:
                    return self._source_constraint_in_cluster(constraint, down[0])
                else:
                    LOGGER.warning(f"source method '{method}' is inconsistent")
                    return None
            else:
                return self._source_constraint_in_cluster(constraint, down[0])


    def _is_atomic(self, object):
        method = self._determining_method(object)
        if method == None:
            return True
        #elif isinstance(method, Distance2Rigid) or isinstance(method, Angle2Hog):
        #    return True
        else:
            return False

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
            return (distance.vars[0] in object.vars and distance.vars[1] in object.vars)
        elif isinstance(object, Distance):
            return (distance.vars[0] in object.vars and distance.vars[1] in object.vars)
        else:
            return False

    def _contains_angle(self, object, angle):
        if isinstance(object, Rigid) or isinstance(object, Balloon):
            return (angle.vars[0] in object.vars
            and angle.vars[1] in object.vars
            and angle.vars[2] in object.vars)
        elif isinstance(object, Hedgehog):
            return (angle.vars[1] == object.cvar and
            angle.vars[0] in object.xvars and
            angle.vars[2] in object.xvars)
        elif isinstance(object, Angle):
            return (angle.vars[1] == object.vars[1] and
            angle.vars[0] in object.vars and
            angle.vars[2] in object.vars)
        else:
            return False


    # --------- special methods ------

    def __str__(self):
        s = ""
        s += "Clusters:\n"
        for x in self.clusters():
            s += str(x) + "\n"
        s += "Methods:\n"
        for x in self.methods():
            s += str(x) + "\n"
        return s


# class ClusterSolver

#  -----------------------------------------------------------
#  ----------Method classes used by ClusterSolver -------------
#  -----------------------------------------------------------

class ClusterMethod(MultiMethod):
    """A method that determines a single output cluster from a set of input clusters.

       Subclasses should provide a static class variable 'patterngraph', which is a graph,
       describing the pattern that is matched by the solver and used to instantiate the Method.
       (see function pattern2graph)

       Alternatively, subclasses may implement the static class method 'handcoded_match', which should
       return a list of matches (given a new cluster and all connected clusters).

       Subclasses should implement function _multi_execute such that the output cluster satisfies all
       the constraints in the input clusters.

       instance vars:
        overconstrained - True iff the merge locally overconstrained
        consistent - True iff the merge is generically consistent
        (these variables are automatically set by the solver for debugging purposes )
    """

    def __init__(self):
        self.overconstrained = None
        self.consistent = None
        MultiMethod.__init__(self)

    def prototype_constraints(self):
        """Return a list of SelectionConstraint"""
        return []

    def status_str(self):
        s = ""
        if self.consistent == True:
            s += "consistent "
        elif self.consistent == False:
            s += "inconsistent "
        if self.overconstrained == True:
            s += "overconstrained"
        elif self.overconstrained == False:
            s += "well-constrained"
        return s

    def input_clusters(self):
        return [var for var in self.inputs() if isinstance(var, Cluster)]

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self._inputs == other._inputs
        else:
            return False

    def __hash__(self):
        return hash(tuple(self._inputs)+tuple([self.__class__]))


class PrototypeMethod(MultiMethod):
    """A PrototypeMethod selects those solutions of a cluster for which
       the protoype and the solution satisfy the same constraints.
    """

    def __init__(self, incluster, selclusters, outcluster, constraints, enabled):
        self._inputs = [incluster]+selclusters+[enabled]
        self._outputs = [outcluster]
        self._constraints = constraints
        MultiMethod.__init__(self)

    def multi_execute(self, inmap):
        LOGGER.debug("multi execute called")
        incluster = self._inputs[0]
        selclusters = []
        for i in range(1,len(self._inputs)-1):
            selclusters.append(self._inputs[i])
        enabledvar = self._inputs[-1]

        LOGGER.debug(f"input cluster: {incluster}")
        LOGGER.debug(f"selection clusters: {selclusters}")
        LOGGER.debug(f"enabled variable: {enabledvar}")

        # get confs/values
        enabledval = inmap[enabledvar]
        inconf = inmap[incluster]
        selmap = {}
        for cluster in selclusters:
            conf = inmap[cluster]
            assert len(conf.vars()) == 1
            var = conf.vars()[0]
            selmap[var] = conf.map[var]
        if len(selmap) == 0:
            selconf = {}
        else:
            selconf = Configuration(selmap)

        LOGGER.debug(f"input configuration: {inconf}")
        LOGGER.debug(f"selection configurations: {selconf}")
        LOGGER.debug(f"enabled value: {enabledval}")

        # do test
        if enabledval == True:
            sat = True
            for con in self._constraints:
                satcon = con.satisfied(inconf.map) == con.satisfied(selconf.map)
                LOGGER.debug(f"constraint '{con}' satisifed: {satcon}")
                sat = sat and satcon

            LOGGER.debug(f"prototype satisfied: {sat}")

            if sat:
                return [inconf]
            else:
                return []
        else:
            return [inconf]

    def __str__(self):
        return "PrototypeMethod#%d(%s->%s)"%(id(self),str(self._inputs[0]), str(self._outputs[0]))

class SelectionMethod(MultiMethod):
    """A SelectionMethod selects those solutions of a cluster for which
       all selectionconstraints are satisfied.
    """

    def __init__(self, incluster, outcluster):
        self._inputs = [incluster]
        self._outputs = [outcluster]
        self._constraints = []
        MultiMethod.__init__(self)

    def add_constraint(self, con):
        self._constraints.append(con)

    def rem_constraint(self, con):
        self._constraints.remove(con)

    def iter_constraints(self):
        return iter(self._constraints)

    def multi_execute(self, inmap):
        incluster = self._inputs[0]
        inconf = inmap[incluster]
        LOGGER.debug(f"input configuration: {inconf}")
        sat = True

        for con in self._constraints:
            LOGGER.debug(f"constraint '{con}' satisfied: {satcon}")
            satcon = con.satisfied(inconf.map)
            sat = sat and satcon

        LOGGER.debug(f"all satisfied: {sat}")

        if sat:
            return [inconf]
        else:
            return []


    def __str__(self):
        return "SelectionMethod#%d(%s & %s ->%s)"%(id(self),str(self._inputs[0]), str(self._constraints), str(self._outputs[0]))

# --------------------------------------
# helper functions for pattern matching
# --------------------------------------

def pattern2graph(pattern):
    """Convert a pattern to a pattern graph, used before graph based matching.
       The pattern is a list of tuples (pattype, patname, patvars), where
       pattype is one of "point", "distance", "rigid", "balloon" or "hedgehog"
       patname is a string, which is the name of a variable which will be associated with a cluster
       patvars is a list of strings, where each string is a variable to be associated with a point variable
       If pattype is point or distance, then the length of the cluster is fixed to 1 or 2 points.
       Otherwise, clusters with any number of variables are matched.
       If pattype is hedgehog, then the first variable in patvars is the center variable.
    """
    pgraph = Graph()
    pgraph.add_vertex("point")
    pgraph.add_vertex("distance")
    pgraph.add_vertex("rigid")
    pgraph.add_vertex("balloon")
    pgraph.add_vertex("hedgehog")
    for clpattern in pattern:
        (pattype, patname, patvars) = clpattern
        pgraph.add_edge(pattype, patname)
        for var in patvars:
            pgraph.add_edge(patname, var)
        if pattype == "hedgehog":
            pgraph.add_edge("cvar"+"#"+patname, patvars[0])
            pgraph.add_edge(patname, "cvar"+"#"+patname)
    return pgraph

def reference2graph(nlet):
    """Convert a set of (supposedly connected) clusters to a reference graph, used before graph-based matching."""
    rgraph = Graph()
    rgraph.add_vertex("point")
    rgraph.add_vertex("distance")
    rgraph.add_vertex("rigid")
    rgraph.add_vertex("balloon")
    rgraph.add_vertex("hedgehog")
    for cluster in nlet:
        for var in cluster.vars:
            rgraph.add_edge(cluster, var)
        if isinstance(cluster, Rigid):
            rgraph.add_edge("rigid", cluster)
            if len(cluster.vars) == 1:
                rgraph.add_edge("point", cluster)
            elif len(cluster.vars) == 2:
                rgraph.add_edge("distance", cluster)
        if isinstance(cluster, Balloon):
            rgraph.add_edge("balloon", cluster)
        if isinstance(cluster, Hedgehog):
            rgraph.add_edge("hedgehog", cluster)
            rgraph.add_edge("cvar"+"#"+str(id(cluster)), cluster.cvar)
            rgraph.add_edge(cluster, "cvar"+"#"+str(id(cluster)))
    return rgraph

def rootname(cluster):
    """returns the name of the root variable associated with the name of a cluster variable"""
    return "root#"+str(id(cluster))

