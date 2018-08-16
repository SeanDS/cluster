"""Graph data structures"""

import networkx as nx

class ConstraintGraph:
    """Constraint graph

    The constraint graph defines the relationships between variables and constraints using a
    directed graph data structure. Constraints and variables represent nodes of the graph, and if a
    constraint is imposed upon a variable, this is represented by an edge between these nodes.
    """
    def __init__(self):
        # variables
        self._variables = {}

        # constraints
        self._constraints = {}

        # empty graph
        self._graph = nx.DiGraph()

    @property
    def variables(self):
        """Constraint variables"""
        return list(self._variables)

    @property
    def constraints(self):
        """Constraints"""
        return list(self._constraints)

    def add_variable(self, name):
        """Adds the specified variable to the graph

        Parameters
        ----------
        name : hashable
            The name of the variable to add.
        Raises
        ------
        DuplicateError
            If the variable is already in the graph.
        """
        if name in self.variables:
            raise DuplicateError("variable name '%s' already exists" % name)

        # create entry with no coordinate
        self._variables[name] = None

        # create a vertex in the graph for the variable
        self._graph.add_node(name)

    def remove_variable(self, name):
        """Removes the specified variable to the graph

        Parameters
        ----------
        name : hashable
            The name of the variable to remove.
        Raises
        ------
        ValueError
            If the variable is not in the graph.
        """
        if name not in self.variables:
            raise ValueError("variable name '%s' doesn't exist" % name)

        # remove constraints on this variable
        for constraint in self.get_constraints_on(name):
            self.remove_constraint(constraint)

        # remove variable
        del(self._variables[name])

        # remove variable's graph edge
        self._graph.remove_node(name)

    def add_constraint(self, constraint):
        """Adds the specified constraint to the graph

        Parameters
        ----------
        constraint :
            The constraint to add.
        Raises
        ------
        DuplicateError
            If the constraint is already in the graph.
        ValueError
            If the constraint contains a variable not already present in the graph.
        """
        # only add constraint if it isn't already in the graph
        if constraint in self._constraints:
            raise DuplicateError(f"constraint '{constraint}' already exists")

        # create entry
        self._constraints[constraint] = None

        # process the constraint's variables
        for variable in constraint.variables:
            if variable not in self.variables:
                raise ValueError(f"variable '{variable}' in constraint '{constraint}' is not "
                                 "present in graph")

            # create edge in the graph for the variable
            self._graph.add_edge(variable, constraint)

    def remove_constraint(self, constraint):
        """Removes the specified constraint from the graph

        Parameters
        ----------
        constraint :
            The constraint to remove.
        Raises
        ------
        ValueError
            If the constraint is not in the graph.
        """
        # only remove constraint if it already exists in the graph
        if constraint not in self._constraints:
            raise Exception("constraint '%s' doesn't exist" % constraint)

        # remove constraint
        del(self._constraints[constraint])

        # remove graph vertex associated with the constraint
        self._graph.remove_node(constraint)

    def get_constraints_on(self, variable):
        """Returns a list of all constraints on the specified variable

        Parameters
        ----------
        variable : hashable
            The variable to get constraints for.
        """
        if variable not in self._graph:
            # variable is not in the graph
            return []

        # return nodes connected to this variable via an edge
        return self._graph.neighbors(variable)

    def get_constraints_on_all(self, variables):
        """Returns a list of the constraints shared by all specified variables

        Parameters
        ----------
        variables : sequence of hashable
            The variables to get shared constraints for.
        """
        # if no variables were specified, there are no shared constraints
        if len(variables) == 0:
            # return  an empty list
            return []

        # initially set constraints to those of first variable
        constraints = set(self.get_constraints_on(variables[0]))

        # find all shared constraints
        for variable in variables[1:]:
            constraints.intersection_update(self.get_constraints_on(variable))

        return list(constraints)

    def get_constraints_on_any(self, variables):
        """Returns a list of the constraints on any of the specified variables

        Parameters
        ----------
        variables : sequence of hashable
            The variables to get constraints for.
        """
        # if no variables were specified, there are no constraints
        if len(variables) == 0:
            # return  an empty list
            return []

        # empty set of constraints
        constraints = set([])

        for variable in variables:
            constraints.update(self.get_constraints_on(variable))

        # return constraints set as a list
        return list(constraints)

    def __repr__(self):
        variables = ", ".join([str(variable) for variable in self._variables])
        constraints = ", ".join([str(constraint) for constraint in self._constraints])

        return f"ConstraintGraph(variables=[{variables}], constraints=[{constraints}])"

    def __str__(self):
        return repr(self)


class DuplicateError(ValueError):
    pass
