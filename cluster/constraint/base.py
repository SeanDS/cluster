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

from ..geometry import tol_eq
from ..notify import Notifier

LOGGER = logging.getLogger(__name__)

class Constraint(metaclass=abc.ABCMeta):
    """Abstract constraint

    A constraint defines a relation between variables that should be satisfied.

    Subclasses must define proper __init__(), variables() and satisfied()
    methods.

    Constraints must be immutable, hashable objects.
    """

    NAME = "Constraint"

    def __init__(self, variables=None):
        super().__init__()

        self.variables = variables

    @abc.abstractmethod
    def satisfied(self, mapping):
        """Returns true if this constraint is satisfied by the specified \
        mapping from variables to values

        :param mapping: dict containing mapping
        """
        raise NotImplementedError

    @property
    def _variable_str(self):
        return ", ".join(self.variables)

    def __str__(self):
        return f"{self.NAME}({self._variable_str})"

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash((self.NAME, tuple(self.variables)))

class ParametricConstraint(Constraint, Notifier, metaclass=abc.ABCMeta):
    """A constraint with a parameter and notification when parameter changes"""

    NAME = "ParametricConstraint"

    def __init__(self, value=None, *args, **kwargs):
        """initialize ParametricConstraint"""
        super().__init__(*args, **kwargs)

        self.value = value

    def get_parameter(self):
        """get parameter value"""

        return self.value

    def set_parameter(self, value):
        """set parameter value and notify any listeners"""

        self.value = value
        self.send_notify(("set_parameter", value))

    @abc.abstractmethod
    def mapped_value(self, mapping):
        raise NotImplementedError

    def satisfied(self, mapping):
        """return True iff mapping from variable names to points satisfies constraint"""
        return tol_eq(self.mapped_value(mapping), self.value)

    def __str__(self):
        return f"{self.NAME}(({self._variable_str}) = {self.value})"


class PlusConstraint(Constraint):
    """Constraint for testing purposes"""

    def __init__(self, a, b, c):
        super().__init__([a, b, c])

    def satisfied(self, mapping):
        return mapping[self.variables[0]] + mapping[self.variables[1]] == mapping[self.variables[2]]
