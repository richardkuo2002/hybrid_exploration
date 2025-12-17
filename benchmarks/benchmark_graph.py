import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import numpy as np

from graph_generator import Graph_generator
from parameter import PIXEL_FREE


def benchmark_find_k_neighbor():
    # Setup
    map_size = (100, 100)
    robot_map = np.full(map_size, PIXEL_FREE)
    gg = Graph_generator(map_size, k_size=15, sensor_range=80)

    # Generate random nodes
    num_nodes = 1000
    gg.node_coords = np.random.rand(num_nodes, 2) * 100

    print(f"Benchmarking find_k_neighbor_all_nodes with {num_nodes} nodes...")

    start_time = time.time()
    gg.find_k_neighbor_all_nodes(robot_map)
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    benchmark_find_k_neighbor()
