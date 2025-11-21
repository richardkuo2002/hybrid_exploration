import unittest
import numpy as np
from parameter import PIXEL_FREE, PIXEL_OCCUPIED
from utils import check_collision

class TestUtils(unittest.TestCase):
    def setUp(self):
        # Create a simple 10x10 map
        self.map_size = (10, 10)
        self.map = np.full(self.map_size, PIXEL_FREE, dtype=int)
        
        # Add an obstacle in the middle
        self.map[5, 5] = PIXEL_OCCUPIED

    def test_check_collision_free_path(self):
        # Path from (0,0) to (0,9) should be free
        start = np.array([0, 0])
        end = np.array([0, 9])
        self.assertFalse(check_collision(start, end, self.map))

    def test_check_collision_blocked_path(self):
        # Path from (0,0) to (9,9) passes through (5,5) which is occupied
        start = np.array([0, 0])
        end = np.array([9, 9])
        self.assertTrue(check_collision(start, end, self.map))

    def test_check_collision_boundary(self):
        # Path along the edge
        start = np.array([0, 0])
        end = np.array([9, 0])
        self.assertFalse(check_collision(start, end, self.map))

    def test_check_collision_start_end_same(self):
        start = np.array([2, 2])
        end = np.array([2, 2])
        self.assertFalse(check_collision(start, end, self.map))

if __name__ == '__main__':
    unittest.main()
