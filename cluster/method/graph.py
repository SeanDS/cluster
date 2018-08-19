import logging
from ..graph import Graph

LOGGER = logging.getLogger(__name__)

class MethodGraph(Graph):
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
        super().__init__()

        # variables changed since last propagation
        self._changed = []

    @property
    def variables(self):
        return [node for node, data in self.nodes(data=True)
                if data.get("node_type") == "variable"]

    @property
    def methods(self):
        return [node for node, data in self.nodes(data=True)
                if data.get("node_type") == "method"]

    def set_node_value(self, variable, value, propagate=True):
        """Sets the value of a variable.

        :param propagate: whether to propagate changes
        """

        super().set_node_value(variable, value)
        # flag that a change must be propagated
        self._changed.append(variable)

        if propagate:
            self.propagate()

    def add_variable(self, variable, value=None):
        """Adds a variable, optionally with a value"""
        if variable not in self.variables:
            self.add_node(variable, node_type="variable", value=value)

    def remove_variable(self, variable):
        """Remove a variable and all methods on that variable"""

        if variable not in self.variables:
            raise ValueError(f"variable '{variable}' not in graph")

        if variable in self._changed:
            # remove variable from pending changes
            self._changed.remove(variable)

        # delete methods
        # must convert nodes to list because the graph changes
        for method in list(self.predecessors(variable)):
            self.remove_method(method)
        for method in list(self.successors(variable)):
            self.remove_method(method)

        # remove it from graph
        self.remove_node(variable)

    def add_method(self, method, propagate=True):
        """Adds a method.

        :param propagate: whether to propagate changes
        """

        if method in self.methods:
            return

        self.add_node(method, node_type="method", value=1)

        # update graph
        for variable in method.inputs:
            self.add_variable(variable)
            self.add_edge(variable, method)

        for variable in method.outputs:
            self.add_variable(variable)
            self.add_edge(method, variable)

        # check validity of graph
        for variable in method.outputs:
            if len(list(self.predecessors(variable))) > 1:
                self.remove_method(method)
                raise MethodGraphDetermineException(f"variable '{variable}' determined by multiple "
                                                    "methods")
            elif self.has_cycle(variable):
                self.remove_method(method)
                raise MethodGraphCycleException(f"adding variable '{variable}' results in a cycle")

        if propagate:
            # execute includes propagation
            self.execute(method)

    def remove_method(self, method):
        """Removes a method"""
        if method not in self.methods:
            raise ValueError(f"method '{method}' not in graph")

        self.remove_node(method)

    def propagate(self):
        """Propagates any pending changes

        Changes are propagated until no changes are left or until no more
        changes can be propagated. This method is called from set() and
        add_method() by default. However, if the user so chooses, the methods
        will not call propagate, and the user should call this function at a
        convenient time.
        """
        LOGGER.debug("propagating changes")

        while len(self._changed) != 0:
            # get a pending change
            pick = self._changed.pop()

            for method in self.successors(pick):
                self._do_execute(method)

    def execute(self, method):
        """Executes a method and propagates changes

        Method must be in MethodGraph
        """

        if method not in self.methods:
            raise ValueError(f"method '{method}' not in graph")

        self._do_execute(method)
        self.propagate()

    def _do_execute(self, method):
        """Executes a method

        Method is executed only if all input variable values are not None.
        Updates mapping and changes flag.
        """
        LOGGER.debug(f"executing method '{method}'")

        # create input map and check for None values
        input_map = {}
        has_nones = False

        for variable in method.inputs:
            value = self.get_node_value(variable)

            if value is None:
                has_nones = True

            input_map[variable] = value

        for variable in method.outputs:
            input_map[variable] = self.get_node_value(variable)

        # call method.execute
        if has_nones:
            output_map = {}
        else:
            LOGGER.debug("no None values in input map")
            output_map = method.execute(input_map)

        # update variable values and set changed flags
        for variable in method.outputs:
            if variable in output_map:
                self.set_node_value(variable, output_map[variable], propagate=False)
            else:
                if self.get_node_value(variable) is not None:
                    self.set_node_value(variable, None, propagate=False)

        # clear change flag on input variables
        for variable in method.inputs:
            if variable in self._changed:
                self._changed.remove(variable)


class MethodGraphCycleException(Exception):
    """Error indicating a cyclic connection in a MethodGraph"""
    pass


class MethodGraphDetermineException(Exception):
    """Error indicating a variable is determined by more than one method in a
    MethodGraph"""
    pass
