import logging

from ..graph import Graph
from ..event import Observable, Event

LOGGER = logging.getLogger(__name__)

# Note: since NetworkX doesn't call super(), it must be second here
# in order to achieve the correct MRO.
class ConstraintGraph(Observable, Graph):
    """A constraint graph"""
    @property
    def variables(self):
        return [node for node, data in self.nodes(data=True)
                if data.get("node_type") == "variable"]

    @property
    def constraints(self):
        return [node for node, data in self.nodes(data=True)
                if data.get("node_type") == "constraint"]

    def add_variable(self, variable):
        """Adds the specified variable to the graph

        :param var_name: name of the variable to add
        """
        # only add if it doesn't already exist
        if variable in self.variables:
            return

        # create a vertex in the graph for the variable
        self.add_node(variable, node_type="variable")

        # notify observers that a new variable has been added
        self.fire(Event("add_variable", variable=variable))

    def rem_variable(self, variable):
        """Removes the specified variable from the graph

        :param var_name: name of the variable to remove
        """
        # only remove if it already exists
        if variable not in self.variables:
            LOGGER.warning("Trying to remove variable that isn't in graph")

            return

        # remove variable's constraints
        for constraint in self.constraints_on(variable):
            self.rem_constraint(constraint)

        # remove graph vertex associated with the variable
        self.remove_node(variable)

        # notify observers that a variable has been removed
        self.fire(Event("rem_variable", variable=variable))

    def add_constraint(self, constraint):
        """Adds the specified constraint to the graph

        :param constraint: constraint to add
        """

        # only add constraint if it isn't already in the graph
        if constraint in self.constraints:
            return

        self.add_node(constraint, node_type="constraint")

        # process the constraint's variables
        for variable in constraint.variables:
            # add the variable to the graph
            self.add_variable(variable)

            # create edge in the graph for the variable
            self.add_edge(variable, constraint)

        # notify observers that a constraint has been added
        self.fire(Event("add_constraint", constraint=constraint))

    def rem_constraint(self, constraint):
        """Removes the specified constraint from the graph

        :param constraint: constraint to remove
        """

        # only remove constraint if it already exists in the graph
        if constraint not in self.constraints:
            LOGGER.warning("Trying to remove constraint that isn't in graph")

            return

        # remove graph vertex associated with the constraint
        self.remove_node(constraint)

        # notify observers that a constraint was removed
        self.fire(Event("remove_constraint", constraint=constraint))

    def constraints_on(self, variable):
        """Returns a list of all constraints on the specified variable

        :param variable: variable to get constraints for
        """

        # check if variable is in the graph
        if not self.has_node(variable):
            # default to empty list
            return []

        # return the variable's outgoing vertices
        return self.successors(variable)

    def constraints_on_all(self, variables):
        """Gets a list of the constraints shared by all of the variables \
        specified in the sequence

        :param variables: variables to find constraints for"""

        # if no variables were specified, there are no shared constraints
        if len(variables) == 0:
            # return  an empty list
            return []

        # empty list of shared constraints
        shared_constraints = []

        # loop over the constraints of the first variable in the list
        for constraint in self.constraints_on(variables[0]):
            # default flag
            shared_constraint = True

            # loop over the variables in the rest of the list
            for var in variables[1:]:
                # is this variable constrained by this constraint?
                if var not in constraint.variables:
                    # this variable doesn't share the constraint
                    shared_constraint = False

                    # no point checking the others
                    break

            if shared_constraint:
                # add constraint to list of shared constraints
                shared_constraints.append(constraint)

        return shared_constraints

    def constraints_on_any(self, variables):
        """Gets a list of the constraints on any of the specified variables

        :param variables: variables to get constraints for"""

        # if no variables were specified, there are no constraints
        if len(variables) == 0:
            # return  an empty list
            return []

        # empty set of constraints
        constraints = set([])

        for variable in variables:
            constraint = set(self.constraints_on(variable))
            constraints.update(constraint)

        # return constraints set as a list
        return list(constraints)

    def __str__(self):
        variables = ", ".join([str(variable) for variable in self.variables])
        constraints = ", ".join([str(constraint) for constraint in self.constraints])
        return f"ConstraintGraph(variables=[{variables}], constraints=[{constraints}])"
