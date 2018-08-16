"""Geometry tests"""

from unittest import TestCase
import numpy as np

from cluster.geometry import Vector

class GeometryTestCase(TestCase):
    """Geometry tests"""
    def assertVectorEqual(self, a, b):
        self.assertTrue(a.tol_eq(b))

    def test_xy(self):
        """Test x- and y-values"""
        vector = Vector([1, 2])

        self.assertEqual(vector.x, 1)
        self.assertEqual(vector.y, 2)

    def test_origin(self):
        self.assertVectorEqual(Vector([0, 0]), Vector.origin())

    def test_length(self):
        self.assertEqual(Vector([0, 0]).length, 0)
        self.assertEqual(Vector([1, 0]).length, 1)
        self.assertEqual(Vector([1, 1]).length, np.sqrt(2))
        self.assertEqual(Vector([2, 1]).length, np.sqrt(5))
        self.assertEqual(Vector([3, 4]).length, 5)
        self.assertEqual(Vector([12, 5]).length, 13)

        # ignores signs
        self.assertEqual(Vector([1, 0]).length, Vector([0, 1]).length)
        self.assertEqual(Vector([5.5, 9.7]).length, Vector([9.7, 5.5]).length)

        # commutes
        self.assertEqual(Vector([1, 2]).length, Vector([2, 1]).length)

    def test_angle(self):
        # zero angle
        self.assertEqual(Vector([0, 0]).angle_between(Vector([1, 0]), Vector([1, 0])), 0)
        self.assertEqual(Vector([0, 0]).angle_between(Vector([1, 0]), Vector([1, -0])), 0)

        # right angles
        self.assertEqual(Vector([0, 0]).angle_between(Vector([1, 0]), Vector([0, 1])), np.pi / 2)
        self.assertEqual(Vector([0, 0]).angle_between(Vector([0, 1]), Vector([1, 0])), -np.pi / 2)

    def test_clockwise(self):
        self.assertTrue(Vector.is_clockwise(Vector([1, 0]), Vector([0, 0]), Vector([0, 1])))
        self.assertTrue(Vector.is_clockwise(Vector([1, 0]), Vector([0, 0]), Vector([1, 1])))

    def test_counterclockwise(self):
        self.assertTrue(Vector.is_counterclockwise(Vector([0, 1]), Vector([0, 0]), Vector([1, 0])))
        self.assertTrue(Vector.is_counterclockwise(Vector([0, 1]), Vector([0, 0]), Vector([1, 1])))
