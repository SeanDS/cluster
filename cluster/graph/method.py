import logging

LOGGER = logging.getLogger(__name__)

class OldGraph:
    """A weighted directed graph"""

    def __init__(self, graph=None):
        # forward and reverse edges are stored in a dictionary of dictionaries
        self._forward = {}
        self._reverse = {}

    def nodes(self):
        """Get a list of vertices in this graph"""

        return list(self._forward.keys())

    def add_node(self, vertex):
        """Add vertex to graph

        :param vertex: vertex to add
        """

        if vertex in self._forward:
            # already in dict
            return

        # add vertex to each dict
        self._forward[vertex] = {}
        self._reverse[vertex] = {}

    def remove_node(self, node):
        """Remove node and incident edges

        :param node: node to remove
        """

        if node not in self._forward:
            raise Exception("node not in graph")

        # remove edges going to and from vertex
        for predecessor in self.predecessors(node):
            self.remove_edge(predecessor, node)

        for successor in self.successors(node):
            self.remove_edge(node, successor)

        # remove vertex in dicts
        del self._forward[node]
        del self._reverse[node]

    def has_node(self, vertex):
        """Check if this graph contains the specified vertex

        :param vertex: vertex to check
        :returns: True if the graph contains the vertex, False otherwise
        :rtype: boolean
        """

        return vertex in self._forward

    def successors(self, vertex):
        """Get a list of vertices connected from the specified vertex via an \
        edge

        :param vertex: vertex to use as a reference
        """

        # look up forward graph
        return list(self._forward[vertex].keys())

    def predecessors(self, vertex):
        """Get a list of vertices connected to the specified vertex via an \
        edge

        :param vertex: vertex to use as a reference
        """

        # look up the reverse graph
        return list(self._reverse[vertex].keys())

    def edges(self):
        """Get a list of the edges in this graph"""

        # empty list
        l = []

        for i in self._forward:
            for j in self._forward[i]:
                l.append((i, j))

        return l

    def add_edge(self, v1, v2, value=1):
        """Add edge with optional value

        :param v1: first vertex
        :param v2: second vertex
        :param value: optional value
        """

        # add vertices
        if v1 not in self._forward:
            self.add_node(v1)

        if v2 not in self._forward:
            self.add_node(v2)

        # add edge
        if v2 not in self._forward[v1]:
            self._forward[v1][v2] = value

        # and the reverse edge
        if v1 not in self._reverse[v2]:
            self._reverse[v2][v1] = value

    def remove_edge(self, v1, v2):
        """Remove edge

        :param v1: first vertex
        :param v2: second vertex
        """

        if not self.has_edge(v1,v2):
            raise Exception("Edge not in graph")

        # remove edges from dicts
        del self._forward[v1][v2]
        del self._reverse[v2][v1]

    def has_edge(self, v1, v2):
        """Check if this graph contains the edge specified by the two vertices

        :param v1: first vertex
        :param v2: second vertex
        :returns: True if the graph contains the edge, False otherwise
        :rtype: boolean
        """

        if v1 not in self._forward:
            return False

        return v2 in self._forward[v1]

    def in_edges(self, vertex):
        """Get a list of edges connecting towards the specified vertex

        :param vertex: vertex to retrieve edges for
        """

        return [(v, vertex) for v in self.predecessors(vertex)]

    def out_edges(self, vertex):
        """Get a list of edges connecting away from the specified vertex

        :param vertex: vertex to retrieve edges for
        """

        return [(v, vertex) for v in self.successors(vertex)]

    def get(self, v1, v2):
        """Get value of edge

        :param v1: first vertex
        :param v2: second vertex
        :returns: value of edge
        :rtype: hashable
        """

        return self._forward[v1][v2]

    def set(self, v1, v2, value):
        """Set value of edge, adding it if it doesn't yet exist

        :param v1: first vertex
        :param v2: second vertex
        :param value: vertex value
        """

        if not self.has_edge(v1, v2):
            # add new edge and set value
            self.add_edge(v1, v2, value=value)
        else:
            # set edge value
            self._forward[v1][v2] = value
            self._reverse[v2][v1] = value

    def path(self, start, end):
        """Gets an arbitrary path (list of vertices) from start to end

        If start is equal to end, then the path is a cycle
        If there is no path between start and end, then an empty list is
        returned

        :param start: start vertex
        :param end: end vertex
        :returns: list of vertices connecting start and end
        :rtype: list
        """

        # map from vertices to shortest path to that key vertex
        trails = {}

        # set start vertex
        trails[start] = [start]

        # list of vertices to consider
        consider = [start]

        # loop until there are no vertices to consider
        while len(consider) > 0:
            # next key to consider
            key = consider.pop()

            # current path
            this_path = trails[key]

            for v in self.successors(key):
                if v == end:
                    # found the end, so return the path we have taken
                    return this_path + [v]
                elif v not in trails:
                    # add vertex to trails taken
                    trails[v] = this_path + [v]

                    # add the vertex to the list to be searched
                    consider.append(v)
                elif len(this_path) + 1 < len(trails[v]):
                    # this trail is shorter than a previous one, so overwrite
                    trails[v] = this_path + [v]

        # no path found
        return []

    def __str__(self):
        s = ""

        for i in self._forward:
            v = ", ".join(["{0}: {1}".format(str(j), str(self.get(i, j))) for j in self._forward[i]])

            s += "%s: {%s}" % (str(i), v)

        return "{%s}" % s

class MethodGraph:
    """A method graph

    A method graph is represented by a directed bi-partite graph: nodes are
    either varables or methods. Edges run from input variables to methods and
    from methods to output variables.

    A method graph must not contain cycles. Every variable must be determined by
    at most one constraint.

    Methods must be instances of :class:`~.Method`. Variables are basically just
    names, and may be any immutable, hashable object, e.g. strings. Values
    associated with variables may be of any type.

    If no value is explicitly associated with a variable, it defaults to None.
    """

    def __init__(self):
        # the graph structure
        self._graph = OldGraph()

        # map from variable names (the keys) to values
        self._map = {}

        # collection of methods (keys)
        self._methods = {}

        # collection of changed variables since last propagation
        self._changed = {}

    def variables(self):
        """Returns a list of variables in the method graph"""

        return list(self._map.keys())

    def methods(self):
        """Returns a list of methods associated with the graph"""

        return list(self._methods.keys())

    def add_variable(self, varname, value=None):
        """Adds a variable, optionally with a value"""

        if varname not in self._map:
            self._map[varname] = value
            self._graph.add_node(varname)

    def rem_variable(self, varname):
        """Remove a variable and all methods on that variable"""

        if varname not in self._map:
            raise Exception("Variable not in graph")

        del(self._map[varname])

        if varname in self._changed:
            del(self._changed[varname])

        # delete all methods on it
        for method in self._graph.predecessors(varname):
            self.rem_method(method)

        for method in self._graph.successors(varname):
            self.rem_method(method)

        # remove it from graph
        self._graph.remove_node(varname)

    def get(self, variable):
        """Gets the value of a variable"""

        return self._map[variable]

    def set(self, varname, value, prop=True):
        """Sets the value of a variable.

        :param prop: whether to propagate changes
        """

        self._map[varname] = value
        self._changed[varname] = 1

        if prop:
            self.propagate()

    def add_method(self, met, prop=True):
        """Adds a method.

        :param prop: whether to propagate changes
        """

        if met in self._methods:
            return

        self._methods[met] = 1

        # update graph
        for var in met.inputs:
            self.add_variable(var)
            self._graph.add_edge(var, met)

        for var in met.outputs:
            self.add_variable(var)
            self._graph.add_edge(met, var)

        # check validity of graph
        for var in met.outputs:
            if len(self._graph.predecessors(var)) > 1:
                self.rem_method(met)

                raise MethodGraphDetermineException("Variable {0} determined \
by multiple methods".format(var))
            elif len(self._graph.path(var, var)) != 0:
                self.rem_method(met)

                raise MethodGraphCycleException("Cycle in graph not allowed \
(variable {0})".format(var))

        if prop:
            # execute includes propagation
            self.execute(met)

    def rem_method(self, met):
        """Removes a method"""

        if met not in self._methods:
            raise Exception("Method not in graph")

        del(self._methods[met])
        self._graph.remove_node(met)

    def propagate(self):
        """Propagates any pending changes

        Changes are propagated until no changes are left or until no more
        changes can be propagated. This method is called from set() and
        add_method() by default. However, if the user so chooses, the methods
        will not call propagate, and the user should call this function at a
        convenient time.
        """

        LOGGER.debug("Propagating changes")

        while len(self._changed) != 0:
            pick = list(self._changed.keys())[0]

            for method in self._graph.successors(pick):
                self._do_execute(method)

            if pick in self._changed:
                del(self._changed[pick])

    def clear(self):
        """Clears the method graph by removing all its variables"""

        while (len(self._map) > 0):
            self.rem_variable(list(self._map.keys())[0])

    def execute(self, method):
        """Executes a method and propagates changes

        Method must be in MethodGraph
        """

        if method not in self._methods:
            raise Exception("Method not in graph")

        self._do_execute(method)
        self.propagate()

    def _do_execute(self, method):
        """Executes a method

        Method is executed only if all input variable values are not None.
        Updates mapping and change flags.
        """

        LOGGER.debug("Executing method %s", method)

        # create input map and check for None values
        inmap = {}
        has_nones = False

        for var in method.inputs:
            value = self._map[var]

            if value == None:
                has_nones = True

            inmap[var] = value

        for var in method.outputs:
            inmap[var] = self._map[var]

        # call method.execute
        if has_nones:
            outmap = {}
        else:
            LOGGER.debug("No None values in map")
            outmap = method.execute(inmap)

        # update values in self._map
        # set output variables changed
        for var in method.outputs:
            if var in outmap:
                self._map[var] = outmap[var]
                self._changed[var] = 1
            else:
                if self._map[var] != None:
                    self._changed[var] = 1
                    self._map[var] = None

        # clear change flag on input variables
        for var in method.inputs:
            if var in self._changed:
                del(self._changed[var])

    def __str__(self):
        variables = ", ".join([str(element) \
        for element in list(self._map.keys())])
        methods = ", ".join([str(element) \
        for element in list(self._methods.keys())])

        return "MethodGraph(variables=[{0}], methods=[{1}])".format(variables, \
        methods)

class MethodGraphCycleException(Exception):
    """Error indicating a cyclic connection in a MethodGraph"""
    pass

class MethodGraphDetermineException(Exception):
    """Error indicating a variable is determined by more than one method in a
    MethodGraph"""
    pass
