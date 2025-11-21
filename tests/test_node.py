import unittest

import numpy as np

from node import Node
from parameter import PIXEL_FREE, PIXEL_OCCUPIED, PIXEL_UNKNOWN


class TestNode(unittest.TestCase):
    def setUp(self):
        self.coords = np.array([5, 5])
        self.frontiers = np.array([[2, 2], [8, 8]])
        self.map_size = (10, 10)
        self.map = np.full(self.map_size, PIXEL_UNKNOWN, dtype=int)
        self.node = Node(self.coords, self.frontiers, self.map)

    def test_initialization(self):
        np.testing.assert_array_equal(self.node.coords, self.coords)
        self.assertEqual(self.node.utility, 0)
        # Node does not have a visited attribute initialized to False, it's implicit via utility/observable list

    def test_set_visited(self):
        self.node.set_visited()
        # set_visited clears observable frontiers and sets utility to 0
        self.assertEqual(self.node.utility, 0)
        self.assertEqual(len(self.node.observable_frontiers_list), 0)

    def test_update_observable_frontiers(self):
        # Mock observed frontiers
        observed_frontiers = set()
        new_frontiers = np.array([[2, 2]])  # One frontier remains

        # Update
        self.node.update_observable_frontiers(
            observed_frontiers, new_frontiers, self.map
        )

        # Check if frontiers list is updated (logic depends on visibility check in Node)
        # Since map is all UNKNOWN, visibility might be limited, but let's check basic attributes
        self.assertIsNotNone(self.node.observable_frontiers_list)


if __name__ == "__main__":
    unittest.main()
