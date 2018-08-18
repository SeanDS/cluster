import logging
from .base import Graph

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

        # collection of changed variables since last propagation
        self._changed = {}

    @property
    def methods(self):
        return [node for node, data in self._graph.nodes(data=True) if data.get("node_type") == "method"]

    @property
    def variables(self):
        return [node for node, data in self._graph.nodes(data=True) if data.get("node_type") == "variable"]

    def add_variable(self, varname, value=None):
        """Adds a variable, optionally with a value"""

        if varname not in self.variables:
            self._graph.add_node(varname, node_type="variable", value=value)

    def rem_variable(self, varname):
        """Remove a variable and all methods on that variable"""

        if varname not in self.variables:
            raise Exception("Variable not in graph")

        if varname in self._changed:
            del(self._changed[varname])

        # delete all methods on it
        for method in self._graph.predecessors(varname):
            self.rem_method(method)

        for method in self._graph.successors(varname):
            self.rem_method(method)

        # remove it from graph
        self._graph.remove_node(varname)

    def get_node_value(self, variable):
        """Gets the value of a variable"""
        return self._graph.get_node_value(variable)

    def set_node_value(self, varname, value, prop=True):
        """Sets the value of a variable.

        :param prop: whether to propagate changes
        """

        self._graph.set_node_value(varname, value)
        self._changed[varname] = 1

        if prop:
            self.propagate()

    def add_method(self, met, prop=True):
        """Adds a method.

        :param prop: whether to propagate changes
        """

        if met in self.methods:
            return

        self._graph.add_node(met, node_type="method", value=1)

        # update graph
        for var in met.inputs:
            self.add_variable(var)
            self._graph.add_edge(var, met)

        for var in met.outputs:
            self.add_variable(var)
            self._graph.add_edge(met, var)

        # check validity of graph
        for var in met.outputs:
            if len(list(self._graph.predecessors(var))) > 1:
                self.rem_method(met)

                raise MethodGraphDetermineException("Variable {0} determined \
by multiple methods".format(var))
            elif self._graph.has_cycle(var):
                self.rem_method(met)

                raise MethodGraphCycleException("Cycle in graph not allowed \
(variable {0})".format(var))

        if prop:
            # execute includes propagation
            self.execute(met)

    def rem_method(self, met):
        """Removes a method"""

        if met not in self.methods:
            raise Exception("Method not in graph")

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

    def execute(self, method):
        """Executes a method and propagates changes

        Method must be in MethodGraph
        """

        if method not in self.methods:
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
            value = self.get_node_value(var)

            if value == None:
                has_nones = True

            inmap[var] = value

        for var in method.outputs:
            inmap[var] = self.get_node_value(var)

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
                self.set_node_value(var, outmap[var], prop=False)
                self._changed[var] = 1
            else:
                if self.get_node_value(var) is not None:
                    self._changed[var] = 1
                    self.set_node_value(var, None, prop=False)

        # clear change flag on input variables
        for var in method.inputs:
            if var in self._changed:
                del(self._changed[var])

class MethodGraphCycleException(Exception):
    """Error indicating a cyclic connection in a MethodGraph"""
    pass

class MethodGraphDetermineException(Exception):
    """Error indicating a variable is determined by more than one method in a
    MethodGraph"""
    pass
