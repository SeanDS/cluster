"""Constraint graph tests"""

from unittest import TestCase

from cluster.graphs import ConstraintGraph, DuplicateError
from cluster.constraints import FixConstraint, DistanceConstraint, AngleConstraint

class ConstraintGraphTestCase(TestCase):
    """Constraint graph tests"""
    def test_variables(self):
        """Test graph variables"""
        graph = ConstraintGraph()
        graph.add_variable("a")
        graph.add_variable("b")
        graph.add_variable("c")

        self.assertSetEqual(set(["a", "b", "c"]), set(graph.variables))

        graph.remove_variable("b")

        self.assertSetEqual(set(["a", "c"]), set(graph.variables))

    def test_constraints(self):
        """Test graph constraints"""
        graph = ConstraintGraph()
        graph.add_variable("a")
        graph.add_variable("b")
        graph.add_variable("c")

        c1 = DistanceConstraint("a", "b", 10)
        c2 = DistanceConstraint("b", "c", 20)
        c3 = DistanceConstraint("c", "a", 30)

        graph.add_constraint(c1)
        graph.add_constraint(c2)
        graph.add_constraint(c3)

        self.assertSetEqual(set([c1, c2, c3]), set(graph.constraints))

        graph.remove_constraint(c2)

        self.assertSetEqual(set([c1, c3]), set(graph.constraints))

    def test_duplicate_constraint_raises_error(self):
        """Test that adding a duplicate constraint to the graph raises an error"""
        graph = ConstraintGraph()
        graph.add_variable("a")
        graph.add_variable("b")

        c1 = DistanceConstraint("a", "b", 10)
        # same points and length
        c2 = DistanceConstraint("a", "b", 10)

        graph.add_constraint(c1)

        self.assertRaises(DuplicateError, graph.add_constraint, c2)

    def test_constraints_on_points_not_in_graph_raises_error(self):
        """Test constraints on points not in graph raises error"""
        graph = ConstraintGraph()
        graph.add_variable("a")

        c1 = DistanceConstraint("a", "b", 10)

        self.assertRaises(ValueError, graph.add_constraint, c1)
