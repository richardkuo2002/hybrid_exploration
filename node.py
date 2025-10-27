#######################################################################
# Name: node.py
# Initialize and update nodes in the collision-free graph.
#######################################################################

import sys
from parameter import *
import numpy as np
from utils import check_collision  # <--- 修改點：從 utils 匯入

class Node():
    def __init__(self, coords, frontiers, robot_map):
        self.coords = coords
        self.observable_frontiers = set()
        self.sensor_range = SENSOR_RANGE 
        self.initialize_observable_frontiers(frontiers, robot_map)
        self.utility = self.get_node_utility()
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def initialize_observable_frontiers(self, frontiers, robot_map):
        """
        依據目前節點位置，初始化可觀測的探索邊界（frontiers）。
        """
        dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
        frontiers_in_range = frontiers[dist_list < UTILITY_CALC_RANGE]      
        for point in frontiers_in_range:
            collision = check_collision(self.coords, point, robot_map) # <--- 修改點
            if not collision:
                self.observable_frontiers.add(tuple(point))

    def get_node_utility(self):
        """
        取得目前節點的探索效益（utility）。
        """
        return len(self.observable_frontiers)

    def update_observable_frontiers(self, observed_frontiers_set, new_frontiers, robot_map):
        """
        更新目前節點可觀測的探索邊界（observable frontiers）。
        """
        if len(observed_frontiers_set) > 0:
            self.observable_frontiers -= observed_frontiers_set

        if len(new_frontiers) > 0:
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < UTILITY_CALC_RANGE]     
            for point in new_frontiers_in_range:
                collision = check_collision(self.coords, point, robot_map) # <--- 修改點
                if not collision:
                    self.observable_frontiers.add(tuple(point))

        self.utility = self.get_node_utility()
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def reset_observable_frontiers(self, new_frontiers, robot_map):
        """ Reset observable frontiers from node position """
        self.observable_frontiers = []

        if len(new_frontiers) > 0:
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < UTILITY_CALC_RANGE]    
            for point in new_frontiers_in_range:
                collision = check_collision(self.coords, point, robot_map) # <--- 修改點
                if not collision:
                    self.observable_frontiers.add(tuple(point))

        self.utility = self.get_node_utility()
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def set_visited(self):
        """ Set node to be visited """
        self.observable_frontiers = set()
        self.utility = 0
        self.zero_utility_node = True

    def frontiers_within_utility_calc_range(self, frontiers):
        """ Check frontiers only within specified threshold radius """
        dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
        return len(dist_list[dist_list < GLOBAL_NODES_TO_FRONTIER_AVOID_SPARSE_RAD]) > 0

    # <--- 修改點：移除了 check_collision 函式 --- >