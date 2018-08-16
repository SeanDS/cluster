"""Geometric problems"""

from .graphs import ConstraintGraph
from .geometry import Vector

class Problem:
    """Geometric constraint problem

    A GeometricProblem consists of point `variables`, `prototype` coordinates for each variable,
    and `constraints` on those variables.

    `Variables` are just names and can be identified by any hashable object (:class:`str`
    recommended).

    `Constraints` define relations required between points such as distances and angles.

    `Prototype` points are the current solution's suggested coordinates for those points.
    """
    def __init__(self):
        # dictionary mapping variables to points
        self.prototype = {}

        # constraint graph
        self.cg = ConstraintGraph()

    def add_point(self, variable, position=None):
        """Add point variable with optional prototype position"""
        if variable in self.prototype:
            raise Exception("point '%s' already in problem" % variable)

        if position is None:
            # assume position is at origin
            position = Vector.origin()

        # add to prototype
        self.prototype[variable] = position

        # add to constraint graph
        self.cg.add_variable(variable)

    def set_point(self, variable, position):
        """Set prototype position of point variable"""
        if variable not in self.prototype:
            raise Exception("unknown point variable '%s'" % variable)

        self.prototype[variable] = position

    def get_point(self, variable):
        """Get prototype position of point variable"""
        if variable not in self.prototype:
            raise Exception("unknown point variable '%s'" % variable)

        return self.prototype[variable]

    def has_point(self, variable):
        return variable in self.prototype

    def __repr__(self):
        # variable list on separate lines
        variables = "\n".join(["{0} = {1}".format(variable, self.prototype[variable])
                               for variable in self.prototype])

        # constraints on separate lines
        constraints = "\n".join([str(constraint) for constraint in self.cg.constraints])

        return "{0}\n{1}".format(variables, constraints)

    def __str__(self):
        return repr(self)
