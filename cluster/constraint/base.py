"""Constraint graphs

A constraint graph represents a constraint problem. A constraint defines a
number of variables, and a relation between those variables that must be
satisfied.

Note that no values are associated with variables in the constraint graph, i.e.
satisfying constraints is not considered in this module.

The constraint graph is (internally) represented by a directed bi-partite
graph; nodes are variables or constraints and edges run from variables to
constraints.

Variables are just names; any non-mutable hashable object, e.g. a string,
qualifies for a variable. Constraints must be instances of (suclasses of) class
Constraint, and must also be non-mutable, hashable objects.
"""

import abc
import logging

from ..notify import Notifier

LOGGER = logging.getLogger(__name__)

class Constraint(metaclass=abc.ABCMeta):
    """Abstract constraint

    A constraint defines a relation between variables that should be satisfied.

    Subclasses must define proper __init__(), variables() and satisfied()
    methods.

    Constraints must be immutable, hashable objects.
    """

    def variables(self):
        """Returns a list of variables in this constraint

        If an attribute '_variables' has been defined, a new list with the
        contents of that attribute will be returned.

        Subclasses may choose to initialise this variable or to override this
        function.
        """

        # check if there are explicit variables defined
        if hasattr(self, "_variables"):
            # return list of variables
            return list(self._variables)
        else:
            # subclass hasn't set _variables or overridden this function
            raise NotImplementedError

    @abc.abstractmethod
    def satisfied(self, mapping):
        """Returns true if this constraint is satisfied by the specified \
        mapping from variables to values

        :param mapping: dict containing mapping
        """

        pass

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class ParametricConstraint(Constraint, Notifier, metaclass=abc.ABCMeta):
    """A constraint with a parameter and notification when parameter changes"""

    def __init__(self):
        """initialize ParametricConstraint"""

        Constraint.__init__(self)
        Notifier.__init__(self)

        self._value = None

    def get_parameter(self):
        """get parameter value"""

        return self._value

    def set_parameter(self, value):
        """set parameter value and notify any listeners"""

        self._value = value
        self.send_notify(("set_parameter", value))


class PlusConstraint(Constraint):
    """Constraint for testing purposes"""

    def __init__(self, a, b, c):
        self._variables = [a, b, c]

    def __str__(self):
        return "PlusConstraint({0})".format(", ".join(self.variables))

    def satisfied(self, mapping):
        return mapping[self._variables[0]] + mapping[self._variables[1]] \
         == mapping[self._variables[2]]
