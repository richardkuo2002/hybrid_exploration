import unittest
import numpy as np
from graph_generator import Graph_generator
from parameter import PIXEL_FREE, PIXEL_OCCUPIED, PIXEL_UNKNOWN

class TestGraphOptimization(unittest.TestCase):
    def setUp(self):
        self.map_size = (100, 100)
        self.k_size = 4
        self.sensor_range = 20
        self.gg = Graph_generator(self.map_size, self.k_size, self.sensor_range)
        
        # Create a simple empty map
        self.map = np.full(self.map_size, PIXEL_FREE, dtype=np.uint8)
        # Add borders
        self.map[0, :] = PIXEL_OCCUPIED
        self.map[-1, :] = PIXEL_OCCUPIED
        self.map[:, 0] = PIXEL_OCCUPIED
        self.map[:, -1] = PIXEL_OCCUPIED

    def test_incremental_update_obstacle(self):
        # 1. Generate initial graph
        start_pos = np.array([50, 50])
        nodes, edges, _, _ = self.gg.generate_graph(start_pos, self.map, None)
        
        # Verify we have nodes and edges
        print(f"DEBUG: Nodes generated: {len(nodes)}")
        print(f"DEBUG: Edges generated: {len(edges)}")
        self.assertTrue(len(nodes) > 0, f"Nodes should be > 0, got {len(nodes)}")
        self.assertTrue(len(edges) > 0)
        
        # Find two connected nodes near (50, 50)
        center_node = tuple(nodes[self.gg.find_closest_index_from_coords(nodes, start_pos)])
        neighbors = list(edges[center_node].keys())
        self.assertTrue(len(neighbors) > 0)
        neighbor_node = neighbors[0]
        
        # 2. Add an obstacle between them
        # Calculate midpoint
        mid_x = int((center_node[0] + neighbor_node[0]) / 2)
        mid_y = int((center_node[1] + neighbor_node[1]) / 2)
        
        # Place a block of obstacle
        self.map[mid_y-1:mid_y+2, mid_x-1:mid_x+2] = PIXEL_OCCUPIED
        
        # 3. Rebuild graph with robot near the obstacle
        # This should trigger incremental update for nodes near start_pos
        self.gg.rebuild_graph_structure(
            self.map, 
            None, 
            None, 
            start_pos, 
            all_robot_positions=[start_pos]
        )
        
        # 4. Verify the edge is gone
        # Note: node coords might have changed slightly or re-indexed, so we need to find closest nodes again
        new_nodes = self.gg.node_coords
        new_edges = self.gg.graph.edges
        
        idx1 = self.gg.find_closest_index_from_coords(new_nodes, np.array(center_node))
        idx2 = self.gg.find_closest_index_from_coords(new_nodes, np.array(neighbor_node))
        
        node1 = tuple(new_nodes[idx1])
        node2 = tuple(new_nodes[idx2])
        
        # Check if edge exists
        edge_exists = False
        if node1 in new_edges and node2 in new_edges[node1]:
            edge_exists = True
            
        self.assertFalse(edge_exists, f"Edge between {node1} and {node2} should be removed by obstacle at ({mid_x}, {mid_y})")

        # 5. Verify far away edge is preserved
        # Find a node far from start_pos (e.g., at 10, 10)
        far_pos = np.array([10, 10])
        idx_far = self.gg.find_closest_index_from_coords(new_nodes, far_pos)
        if idx_far is not None:
            node_far = tuple(new_nodes[idx_far])
            # It should have neighbors
            if node_far in new_edges:
                self.assertTrue(len(new_edges[node_far]) > 0, "Far away node should still have edges")

if __name__ == '__main__':
    unittest.main()
