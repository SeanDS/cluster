import logging
from .graph import Graph

LOGGER = logging.getLogger(__name__)

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
        self._graph = Graph()

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
