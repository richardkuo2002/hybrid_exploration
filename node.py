#######################################################################
# Name: node.py
# Initialize and update nodes in the coliision-free graph.
#######################################################################

import sys

from parameter import *


import numpy as np


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

        Args:
            frontiers (ndarray): 所有候選探索邊界座標，shape=(N,2)。
            robot_map (ndarray): 機器人本地地圖（障礙物分布）。

        Returns:
            None。作用：self.observable_frontiers 會加上無碰撞、在觀測範圍內的邊界座標。
        """
        dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
        frontiers_in_range = frontiers[dist_list < UTILITY_CALC_RANGE]      
        for point in frontiers_in_range:
            collision = self.check_collision(self.coords, point, robot_map)
            if not collision:
                self.observable_frontiers.add(tuple(point))

    def get_node_utility(self):
        """
        取得目前節點的探索效益（utility）。

        Returns:
            utility (int): 可觀測邊界的數量，代表該節點的探索效益。
        """
        return len(self.observable_frontiers)

    def update_observable_frontiers(self, observed_frontiers_set, new_frontiers, robot_map):
        """
        更新目前節點可觀測的探索邊界（observable frontiers）。

        Args:
            observed_frontiers_set (set): 已經被觀測過的邊界座標集合（需移除）。
            new_frontiers (ndarray): 新增探索邊界座標，shape=(N,2)。
            robot_map (ndarray): 機器人的本地地圖（障礙物分布）。

        Returns:
            None。作用：
                - self.observable_frontiers 會移除已觀測邊界並加入符合條件的新邊界座標。
                - self.utility 更新，表示目前節點探索效益。
                - self.zero_utility_node 更新，True 代表此節點已無有效可探索邊界。
        """
        if len(observed_frontiers_set) > 0:
            self.observable_frontiers -= observed_frontiers_set

        if len(new_frontiers) > 0:
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < UTILITY_CALC_RANGE]     
            for point in new_frontiers_in_range:
                collision = self.check_collision(self.coords, point, robot_map)
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
                collision = self.check_collision(self.coords, point, robot_map)
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


    def check_collision(self, start, end, robot_map):
        """
        檢查兩點間路徑在地圖上是否有碰撞（障礙物），以 Bresenham 演算法沿線偵測。

        Args:
            start (array-like): 起點座標 [x, y]。
            end (array-like): 終點座標 [x, y]。
            robot_map (ndarray): 機器人本地地圖（陣列），其中 1 或 127 代表障礙物，其他代表可通行。

        Returns:
            collision (bool): True 代表路徑有碰撞（遇到障礙物），False 代表路徑暢通。
        """
        collision = False
        map = robot_map 

        x0 = start[0]
        y0 = start[1]
        x1 = end[0]
        y1 = end[1]
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
            k = map.item(int(y), int(x))
            if x == x1 and y == y1:
                break
            if k == 1:
                collision = True
                break
            if k == 127:
                collision = True
                break
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return collision

