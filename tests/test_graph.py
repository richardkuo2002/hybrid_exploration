import unittest
import math
from graph import Graph, a_star

class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        # Create a simple graph:
        # (0,0) --1--> (1,0) --1--> (2,0)
        #   |            |
        #   1            1
        #   v            v
        # (0,1) --1--> (1,1)
        
        self.n00 = (0, 0)
        self.n10 = (1, 0)
        self.n20 = (2, 0)
        self.n01 = (0, 1)
        self.n11 = (1, 1)

        self.graph.add_node(self.n00)
        self.graph.add_node(self.n10)
        self.graph.add_node(self.n20)
        self.graph.add_node(self.n01)
        self.graph.add_node(self.n11)

        self.graph.add_edge(self.n00, self.n10, 1.0)
        self.graph.add_edge(self.n10, self.n20, 1.0)
        self.graph.add_edge(self.n00, self.n01, 1.0)
        self.graph.add_edge(self.n01, self.n11, 1.0)
        self.graph.add_edge(self.n10, self.n11, 1.0)

    def test_add_node(self):
        new_node = (3, 3)
        self.graph.add_node(new_node)
        self.assertIn(new_node, self.graph.nodes)

    def test_add_edge(self):
        self.assertIn(self.n00, self.graph.edges)
        self.assertIn(self.n10, self.graph.edges[self.n00])
        self.assertEqual(self.graph.edges[self.n00][self.n10].length, 1.0)

    def test_a_star_path_exists(self):
        # Path from (0,0) to (2,0) should be length 2
        path, cost, _, _ = a_star(self.n00, self.n20, self.graph)
        self.assertIsNotNone(path)
        self.assertEqual(cost, 2.0)
        self.assertEqual(path, [self.n00, self.n10, self.n20])

    def test_a_star_path_alternative(self):
        # Path from (0,0) to (1,1) can be (0,0)->(1,0)->(1,1) or (0,0)->(0,1)->(1,1)
        # Both have cost 2.0
        path, cost, _, _ = a_star(self.n00, self.n11, self.graph)
        self.assertIsNotNone(path)
        self.assertEqual(cost, 2.0)

    def test_a_star_no_path(self):
        isolated_node = (5, 5)
        self.graph.add_node(isolated_node)
        path, cost, _, _ = a_star(self.n00, isolated_node, self.graph)
        self.assertIsNone(path)
        self.assertEqual(cost, 1e5)

if __name__ == '__main__':
    unittest.main()
