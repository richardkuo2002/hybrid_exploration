import copy
import numpy as np
from scipy.spatial import KDTree
from time import time

from parameter import *
from graph import Graph, a_star
from node import Node
from utils import check_collision

class Graph_generator:
    def __init__(self, map_size, k_size, sensor_range, plot=False):
        self.k_size = k_size
        self.graph = Graph()
        self.node_coords = None # ndarray (N, 2)
        self.nodes_list: list[Node] = [] # list of Node objects
        self.node_utility = None # ndarray (N,)
        self.target_candidates = np.array([]).reshape(0, 2) # ndarray (M, 2)
        self.candidates_utility = np.array([]) # ndarray (M,)
        self.guidepost = None # ndarray (N, 1)

        self.plot = plot
        self.x = [] # For plotting edges
        self.y = [] # For plotting edges
        self.map_x = map_size[1]
        self.map_y = map_size[0]
        self.uniform_points = self.generate_uniform_points()
        self.sensor_range = sensor_range
        self.route_node = [] # list of coords added manually

    def edge_clear_all_nodes(self):
        self.graph = Graph()
        self.x = []; self.y = []

    def edge_clear(self, coords):
        self.graph.clear_edge(tuple(coords))

    def node_clear(self, coords, remove_bidirectional_edges=False):
        self.graph.clear_node(tuple(coords), remove_bidirectional_edges=remove_bidirectional_edges)

    def generate_graph(self, robot_location, robot_map, frontiers):
        """
        (初始化時呼叫) 建立初始的無碰撞圖。
        """
        self.edge_clear_all_nodes()
        free_area = self.free_area(robot_map)

        free_area_complex = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_complex = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_complex, uniform_points_complex, return_indices=True)

        valid_candidate_indices = candidate_indices[candidate_indices < len(self.uniform_points)]
        node_coords = self.uniform_points[valid_candidate_indices]

        node_coords = np.concatenate((np.array(robot_location).reshape(1, 2), node_coords))
        self.node_coords = np.unique(node_coords, axis=0)

        if len(self.node_coords) > 0:
            self.find_k_neighbor_all_nodes(robot_map, update_dense=True)
        else:
             print("Warning: No nodes generated in generate_graph.")
             self.graph = Graph()

        self._update_nodes_and_utilities(frontiers, robot_map)
        self._update_guidepost()

        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost

    def update_graph(self, robot_map, frontiers, old_frontiers, position, all_robot_positions=None):
        """
        (每一步呼叫) 完整更新圖結構和節點效益。
        """
        # 1. 計算新的有效節點座標
        free_area = self.free_area(robot_map)
        free_area_complex = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_complex = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_complex, uniform_points_complex, return_indices=True)

        valid_candidate_indices = candidate_indices[candidate_indices < len(self.uniform_points)]
        new_potential_nodes = self.uniform_points[valid_candidate_indices]
        current_pos_arr = np.array(position).reshape(1, 2)

        new_node_coords = np.concatenate((current_pos_arr, new_potential_nodes))
        new_node_coords = np.unique(new_node_coords, axis=0)

        # 2. 比較新舊節點
        old_node_coords = self.node_coords if self.node_coords is not None else np.array([]).reshape(0, 2)
        old_nodes_set = set(map(tuple, old_node_coords))
        new_nodes_set = set(map(tuple, new_node_coords))

        nodes_to_remove = old_nodes_set - new_nodes_set
        nodes_to_add = new_nodes_set - old_nodes_set

        # 3. 更新 self.node_coords
        self.node_coords = new_node_coords

        # 4. 移除失效 Node 物件
        if self.nodes_list is None: self.nodes_list = []
        self.nodes_list = [node for node in self.nodes_list if hasattr(node, 'coords') and tuple(node.coords) not in nodes_to_remove]

        # 5. 創建新增 Node 物件
        existing_coords_set = set(tuple(node.coords) for node in self.nodes_list if hasattr(node, 'coords'))
        current_frontiers = frontiers if frontiers is not None else np.array([]).reshape(0, 2) # 確保 frontiers 有效
        for coords_tuple in nodes_to_add:
            if coords_tuple not in existing_coords_set:
                coords = np.array(coords_tuple)
                new_node = Node(coords, current_frontiers, robot_map) # 使用 current_frontiers
                self.nodes_list.append(new_node)

        # 6. 重建圖邊緣
        self.graph = Graph()
        if len(self.node_coords) > 0:
            self.find_k_neighbor_all_nodes(robot_map, update_dense=True)
        else:
             print("Warning: No nodes available in update_graph after coordinate update.")

        # 7. 更新所有節點效益
        self._update_nodes_and_utilities(frontiers, robot_map, old_frontiers=old_frontiers, all_robot_positions=all_robot_positions)

        # 8. 更新 guidepost
        self._update_guidepost()

        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost

    # --- 輔助函式 ---

    def _update_nodes_and_utilities(self, frontiers, robot_map, old_frontiers=None, all_robot_positions=None):
        """
        (內部函式) 更新節點效益並重新生成相關列表/陣列。
        """
        if self.nodes_list is None: self.nodes_list = []
        if frontiers is None: frontiers = np.array([]).reshape(0, 2)
        if old_frontiers is None: old_frontiers = np.array([]).reshape(0, 2)

        observed_frontiers_set = set()
        new_frontiers_only = frontiers
        if len(old_frontiers) > 0 and len(frontiers) > 0:
            old_frontiers_complex = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
            new_frontiers_complex = frontiers[:, 0] + frontiers[:, 1] * 1j
            observed_mask = ~np.isin(old_frontiers_complex, new_frontiers_complex)
            new_only_mask = ~np.isin(new_frontiers_complex, old_frontiers_complex)
            observed_frontiers = old_frontiers[observed_mask]
            new_frontiers_only = frontiers[new_only_mask]
            observed_frontiers_set = set(map(tuple, observed_frontiers))
        elif len(old_frontiers) == 0:
             new_frontiers_only = frontiers

        for node in self.nodes_list:
             if hasattr(node, 'update_observable_frontiers'):
                if len(old_frontiers) > 0:
                     node.update_observable_frontiers(observed_frontiers_set, new_frontiers_only, robot_map)
                else:
                     node.reset_observable_frontiers(frontiers, robot_map)

        if all_robot_positions is not None:
            for robot_pos in all_robot_positions:
                if robot_pos is not None:
                    for node in self.nodes_list:
                         if hasattr(node, 'coords'):
                            dist = np.linalg.norm(node.coords - robot_pos)
                            if dist < 10: node.set_visited()

        self.target_candidates = []
        self.candidates_utility = []
        node_utilities_list = []

        coords_map = {tuple(node.coords): node for node in self.nodes_list if hasattr(node, 'coords')}
        final_nodes_list = []

        if self.node_coords is not None:
            for i in range(len(self.node_coords)):
                coords = self.node_coords[i]
                node = coords_map.get(tuple(coords))
                if node is None:
                     node = Node(coords, frontiers, robot_map)

                final_nodes_list.append(node)

                if hasattr(node, 'utility'):
                    utility = node.utility
                    node_utilities_list.append(utility)
                    if utility > 0:
                        self.target_candidates.append(coords)
                        self.candidates_utility.append(utility)
                else:
                     node_utilities_list.append(0)

        self.nodes_list = final_nodes_list
        self.node_utility = np.array(node_utilities_list) if node_utilities_list else np.array([])
        self.target_candidates = np.array(self.target_candidates) if self.target_candidates else np.array([]).reshape(0, 2)
        self.candidates_utility = np.array(self.candidates_utility) if self.candidates_utility else np.array([])

    def _update_guidepost(self):
        """
        (內部函式) 更新 guidepost。
        """
        if self.node_coords is not None and len(self.node_coords) > 0:
            self.guidepost = np.zeros((len(self.node_coords), 1))
            for node_pos in self.route_node:
                index = self.find_closest_index_from_coords(self.node_coords, node_pos)
                if index is not None and index < len(self.guidepost):
                    self.guidepost[index] += 1
        else:
            self.guidepost = np.array([]).reshape(0, 1)


    def find_shortest_path(self, current, destination, node_coords, graph):
        if node_coords is None or len(node_coords) == 0 or graph is None or not hasattr(graph, 'nodes'):
            return 1e5, None

        start_index = self.find_closest_index_from_coords(node_coords, current)
        end_index = self.find_closest_index_from_coords(node_coords, destination)

        if start_index is None or end_index is None: return 1e5, None

        start_node = tuple(node_coords[start_index])
        end_node = tuple(node_coords[end_index])

        if start_node not in graph.nodes or end_node not in graph.nodes: return 1e5, None
        if start_node not in graph.edges or not graph.edges[start_node]:
             if start_node == end_node: return 0, [start_node]
             return 1e5, None

        route, dist, _, _ = a_star(start_node, end_node, graph)

        if start_node != end_node and route is None: return dist, None
        if route is not None: route = list(map(tuple, route))

        return dist, route


    def generate_uniform_points(self):
        x = np.linspace(0, self.map_x - 1, NUM_DENSE_COORDS_WIDTH).round().astype(int)
        y = np.linspace(0, self.map_y - 1, NUM_DENSE_COORDS_WIDTH).round().astype(int)
        t1, t2 = np.meshgrid(x, y); points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points

    def free_area(self, robot_map):
        index = np.where(robot_map == 255); free = np.asarray([index[1], index[0]]).T
        return free

    def find_closest_index_from_coords(self, node_coords, p):
        if node_coords is None or len(node_coords) == 0: return None
        if not isinstance(p, np.ndarray): p = np.array(p)
        if p.shape != (2,): return None
        try: return np.argmin(np.linalg.norm(node_coords - p, axis=1))
        except ValueError: return None


    def find_k_neighbor_all_nodes(self, robot_map, update_dense=True, global_graph:Graph=None, global_graph_knn_dist_max=GLOBAL_GRAPH_KNN_RAD, global_graph_knn_dist_min=CUR_AGENT_KNN_RAD):
        if self.node_coords is None or len(self.node_coords) == 0: return
        try: kd_tree = KDTree(self.node_coords)
        except ValueError: return

        self.x = []; self.y = [] # Reset plot lists

        for i, p in enumerate(self.node_coords):
            p_tuple = tuple(p)
            num_global_neighbours = 0
            # --- Global Graph Logic ---
            topk_global_graph_nodes = set() # <--- 修改點：在這裡初始化

            if global_graph is not None and hasattr(global_graph, 'edges') and p_tuple in global_graph.edges:
                global_graph_edges = global_graph.edges[p_tuple].values()
                global_graph_nodes_arr = np.array([edge.to_node for edge in global_graph_edges])
                global_graph_dist_arr = np.array([edge.length for edge in global_graph_edges])

                if global_graph_nodes_arr.ndim == 2 and global_graph_dist_arr.ndim == 1 and len(global_graph_nodes_arr) == len(global_graph_dist_arr):
                    filtered_idx = (global_graph_dist_arr <= global_graph_knn_dist_max) & (global_graph_dist_arr > global_graph_knn_dist_min)
                    filtered_nodes = global_graph_nodes_arr[filtered_idx]
                    filtered_dist = global_graph_dist_arr[filtered_idx]

                    num_available = len(filtered_nodes)
                    num_global_neighbours = min(num_available, self.k_size)

                    if num_global_neighbours > 0:
                        topk_indices = np.argsort(filtered_dist)[:num_global_neighbours]
                        # <--- 修改點：賦值給 topk_global_graph_nodes ---
                        topk_global_graph_nodes = set(map(tuple, filtered_nodes[topk_indices]))
                        # --- ---

            # 現在 topk_global_graph_nodes 總是被定義了 (即使是空集合)
            for neighbour_node in topk_global_graph_nodes:
                self.graph.add_node(p_tuple)
                neighbour_tuple = tuple(neighbour_node)
                dist = np.linalg.norm(np.array(p_tuple) - np.array(neighbour_tuple))
                self.graph.add_edge(p_tuple, neighbour_tuple, dist)
            # --- End Global Graph Logic ---

            # --- Local Graph Logic (KDTree) ---
            max_neighbours = self.k_size - num_global_neighbours
            num_neighbours_available = len(self.node_coords)
            k_query = min(max(1, max_neighbours), num_neighbours_available)
            indices = []
            if k_query > 0:
                try:
                    distances, indices = kd_tree.query(p, k=k_query)
                    if np.isscalar(indices): indices = np.array([indices])
                except ValueError: indices = []

            for index in indices:
                if index < 0 or index >= len(self.node_coords): continue
                neighbour = self.node_coords[index]
                if np.array_equal(p, neighbour): continue
                start = p; end = neighbour
                if start.shape == (2,) and end.shape == (2,):
                    if not check_collision(start, end, robot_map):
                        start_tuple = tuple(start); end_tuple = tuple(end)
                        if update_dense:
                            self.graph.add_node(start_tuple)
                            dist = np.linalg.norm(start-end)
                            self.graph.add_edge(start_tuple, end_tuple, dist)
                            self.graph.add_node(end_tuple)
                            self.graph.add_edge(end_tuple, start_tuple, dist)
                        if self.plot:
                            self.x.append([p[0], neighbour[0]])
                            self.y.append([p[1], neighbour[1]])