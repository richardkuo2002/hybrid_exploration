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

        # 1. 找出有效的初始節點座標
        # 使用複數技巧找出 free_area 和 uniform_points 的交集
        free_area_complex = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_complex = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_complex, uniform_points_complex, return_indices=True)

        # 確保 candidate_indices 在範圍內
        valid_candidate_indices = candidate_indices[candidate_indices < len(self.uniform_points)]
        node_coords = self.uniform_points[valid_candidate_indices]

        # 將機器人當前位置加入節點
        node_coords = np.concatenate((np.array(robot_location).reshape(1, 2), node_coords))
        self.node_coords = np.unique(node_coords, axis=0) # 確保唯一性

        # 2. 建立圖的邊緣 (K鄰近 + 碰撞檢查)
        if len(self.node_coords) > 0:
            self.find_k_neighbor_all_nodes(robot_map, update_dense=True)
        else:
             print("Warning: No nodes generated in generate_graph.")
             self.graph = Graph() # 確保 graph 是空的

        # 3. 初始化節點物件 (Node) 和效益 (Utility)
        self._update_nodes_and_utilities(frontiers, robot_map)

        # 4. 初始化 guidepost
        self._update_guidepost()

        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost

    # --- 合併後的單一更新函式 ---
    def update_graph(self, robot_map, frontiers, old_frontiers, position, all_robot_positions=None):
        """
        (每一步呼叫) 完整更新圖結構和節點效益。
        合併了 rebuild_graph_structure 和 update_node_utilities 的邏輯。
        """
        # 1. 計算新的有效節點座標 (包含當前位置)
        free_area = self.free_area(robot_map)
        free_area_complex = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_complex = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_complex, uniform_points_complex, return_indices=True)

        valid_candidate_indices = candidate_indices[candidate_indices < len(self.uniform_points)]
        new_potential_nodes = self.uniform_points[valid_candidate_indices]
        current_pos_arr = np.array(position).reshape(1, 2)

        # 合併並確保唯一性
        new_node_coords = np.concatenate((current_pos_arr, new_potential_nodes))
        new_node_coords = np.unique(new_node_coords, axis=0)

        # 2. 比較新舊節點，找出需要增刪的節點
        old_node_coords = self.node_coords if self.node_coords is not None else np.array([]).reshape(0, 2)
        old_nodes_set = set(map(tuple, old_node_coords))
        new_nodes_set = set(map(tuple, new_node_coords))

        nodes_to_remove = old_nodes_set - new_nodes_set
        nodes_to_add = new_nodes_set - old_nodes_set
        # nodes_to_keep = old_nodes_set & new_nodes_set # 這個暫時不用

        # 3. 更新 self.node_coords
        self.node_coords = new_node_coords

        # 4. 從 self.nodes_list 中移除失效的 Node 物件
        if self.nodes_list is None: self.nodes_list = [] # 初始化保護
        self.nodes_list = [node for node in self.nodes_list if hasattr(node, 'coords') and tuple(node.coords) not in nodes_to_remove]

        # 5. 為新增的座標點創建 Node 物件
        existing_coords_set = set(tuple(node.coords) for node in self.nodes_list if hasattr(node, 'coords'))
        for coords_tuple in nodes_to_add:
            if coords_tuple not in existing_coords_set:
                coords = np.array(coords_tuple)
                # 只有在 frontiers 有效時才創建 Node
                if frontiers is not None and len(frontiers) > 0:
                    new_node = Node(coords, frontiers, robot_map)
                    self.nodes_list.append(new_node)
                else:
                    # 如果沒有 frontiers，創建一個 utility=0 的 Node
                    new_node = Node(coords, np.array([]).reshape(0,2), robot_map)
                    self.nodes_list.append(new_node)


        # 6. 重建圖的邊緣 (find_k_neighbor_all_nodes)
        self.graph = Graph() # 清空舊圖
        if len(self.node_coords) > 0:
            self.find_k_neighbor_all_nodes(robot_map, update_dense=True)
        else:
             print("Warning: No nodes available in update_graph after coordinate update.")

        # 7. 更新所有節點的效益 (Utility)
        self._update_nodes_and_utilities(frontiers, robot_map, old_frontiers=old_frontiers, all_robot_positions=all_robot_positions)

        # 8. 更新 guidepost
        self._update_guidepost()

        # 9. 最終狀態檢查 (可選，用於 debug)
        # assert self.node_coords is not None and self.nodes_list is not None and self.node_utility is not None and \
        #        len(self.node_coords) == len(self.nodes_list) == len(self.node_utility), \
        #        f"Length mismatch after update! coords:{len(self.node_coords)}, list:{len(self.nodes_list)}, util:{len(self.node_utility)}"

        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost

    # --- 輔助函式 ---

    def _update_nodes_and_utilities(self, frontiers, robot_map, old_frontiers=None, all_robot_positions=None):
        """
        (內部函式) 根據最新的 frontiers 和 robot_map 更新 self.nodes_list 中所有節點的效益，
        並重新生成 self.node_utility, self.target_candidates, self.candidates_utility。
        """
        if self.nodes_list is None: self.nodes_list = []
        if frontiers is None: frontiers = np.array([]).reshape(0, 2)
        if old_frontiers is None: old_frontiers = np.array([]).reshape(0, 2)

        # 1. 計算邊界變化 (優化：只有在 old_frontiers 存在時才計算)
        observed_frontiers_set = set()
        new_frontiers_only = frontiers
        if len(old_frontiers) > 0 and len(frontiers) > 0:
            # (計算 observed 和 new_only 的邏輯不變)
            old_frontiers_to_check = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
            new_frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
            observed_frontiers_index = np.where(np.isin(old_frontiers_to_check, new_frontiers_to_check, assume_unique=True) == False)
            new_frontiers_index = np.where(np.isin(new_frontiers_to_check, old_frontiers_to_check, assume_unique=True) == False)
            if len(observed_frontiers_index[0]) > 0:
                observed_frontiers = old_frontiers[observed_frontiers_index]
                observed_frontiers_set = set(map(tuple, observed_frontiers))
            if len(new_frontiers_index[0]) > 0:
                new_frontiers_only = frontiers[new_frontiers_index]
            else:
                new_frontiers_only = np.array([]).reshape(0, 2)
        elif len(old_frontiers) == 0:
             # 如果沒有舊 frontiers，所有 new frontiers 都是 new_only
             new_frontiers_only = frontiers


        # 2. 更新每個節點的效益
        for node in self.nodes_list:
             if hasattr(node, 'update_observable_frontiers'): # 確保 node 有效
                # 如果 old_frontiers 存在，則進行增量更新
                if len(old_frontiers) > 0:
                     node.update_observable_frontiers(observed_frontiers_set, new_frontiers_only, robot_map)
                else:
                     # 否則，進行完全重置 (更安全)
                     node.reset_observable_frontiers(frontiers, robot_map)


        # 3. 處理機器人佔用
        if all_robot_positions is not None:
            for robot_pos in all_robot_positions:
                if robot_pos is not None:
                    for node in self.nodes_list:
                         if hasattr(node, 'coords'):
                            dist = np.linalg.norm(node.coords - robot_pos)
                            if dist < 10:
                                node.set_visited() # set_visited 會處理 utility 和 zero_utility_node

        # 4. 重新生成列表/陣列
        self.target_candidates = []
        self.candidates_utility = []
        node_utilities_list = []

        # 確保 node_coords 和 nodes_list 長度一致 (理論上應該一致)
        # 如果不一致，以較短的為準或報錯
        min_len = 0
        if self.node_coords is not None and self.nodes_list is not None:
             min_len = min(len(self.node_coords), len(self.nodes_list))
             if len(self.node_coords) != len(self.nodes_list):
                  print(f"Warning: _update_nodes_and_utilities length mismatch! coords:{len(self.node_coords)}, list:{len(self.nodes_list)}. Using {min_len}")
                  # 強制截斷以保持一致
                  # self.node_coords = self.node_coords[:min_len]
                  # self.nodes_list = self.nodes_list[:min_len]
                  # 更好的做法是找到不一致的原因，這裡先用簡單方法避免崩潰
                  # 改為只迭代 min_len 次
        elif self.nodes_list is not None:
             min_len = len(self.nodes_list) # 如果只有 nodes_list
        # 如果兩者都 None 或空，min_len 保持 0


        coords_map = {tuple(node.coords): node for node in self.nodes_list if hasattr(node, 'coords')}
        final_nodes_list = []

        # 以 self.node_coords 為基準重建 self.nodes_list 和 utility
        if self.node_coords is not None:
            for i in range(len(self.node_coords)): # 只迭代有效長度
                coords = self.node_coords[i]
                node = coords_map.get(tuple(coords))

                # 如果 node 丢失 (理論上不應發生)，創建一個新的
                if node is None:
                     # print(f"Warning: Node for coord {tuple(coords)} not found in nodes_list. Creating new.")
                     node = Node(coords, frontiers, robot_map)
                     # 需要將新 node 加回 self.nodes_list 嗎？
                     # 這裡的邏輯是重建，所以加到 final_nodes_list
                
                final_nodes_list.append(node) # 建立新的 node list

                if hasattr(node, 'utility'):
                    utility = node.utility
                    node_utilities_list.append(utility)
                    if utility > 0:
                        self.target_candidates.append(coords)
                        self.candidates_utility.append(utility)
                else:
                     node_utilities_list.append(0) # 無效 node utility 為 0
        
        # 更新 self.nodes_list
        self.nodes_list = final_nodes_list

        # 轉換為 numpy array
        self.node_utility = np.array(node_utilities_list) if node_utilities_list else np.array([])
        self.target_candidates = np.array(self.target_candidates) if self.target_candidates else np.array([]).reshape(0, 2)
        self.candidates_utility = np.array(self.candidates_utility) if self.candidates_utility else np.array([])


    def _update_guidepost(self):
        """
        (內部函式) 根據 self.route_node 更新 self.guidepost。
        """
        if self.node_coords is not None and len(self.node_coords) > 0:
            self.guidepost = np.zeros((len(self.node_coords), 1))
            for node_pos in self.route_node:
                index = self.find_closest_index_from_coords(self.node_coords, node_pos)
                if index is not None and index < len(self.guidepost):
                    self.guidepost[index] += 1
        else:
            self.guidepost = np.array([]).reshape(0, 1) # 維持 (N, 1) 的形狀


    def find_shortest_path(self, current, destination, node_coords, graph):
        # ... (此函式保持不變，但依賴於 update_graph 正確建立 graph) ...
        # 安全檢查
        if node_coords is None or len(node_coords) == 0 or graph is None or not hasattr(graph, 'nodes'):
            # print(f"Warning: Invalid input for find_shortest_path. nc:{node_coords is not None}, graph:{graph is not None}")
            return 1e5, None # 返回高成本和 None 路徑

        t1 = time()
        start_index = self.find_closest_index_from_coords(node_coords, current)
        end_index = self.find_closest_index_from_coords(node_coords, destination)

        if start_index is None or end_index is None:
            # print(f"Warning: Cannot find closest node index. start:{start_index}, end:{end_index}")
            return 1e5, None

        start_node = tuple(node_coords[start_index])
        end_node = tuple(node_coords[end_index])

        if start_node not in graph.nodes or end_node not in graph.nodes:
            # print(f"Warning: Start ({start_node}) or End ({end_node}) node not in graph nodes set.")
            return 1e5, None

        if start_node not in graph.edges or not graph.edges[start_node]:
             # print(f"Warning: Start node {start_node} has no outgoing edges.")
             # 如果起點等於終點，返回空路徑
             if start_node == end_node:
                  return 0, [start_node]
             return 1e5, None


        route, dist, _, _ = a_star(start_node, end_node, graph)

        if start_node != end_node and route is None:
             # print(f"Warning: A* failed path from {start_node} to {end_node}")
             return dist, None # dist 可能是 1e5

        if route is not None:
            route = list(map(tuple, route))

        return dist, route


    def generate_uniform_points(self):
        # ... (保持不變) ...
        x = np.linspace(0, self.map_x - 1, NUM_DENSE_COORDS_WIDTH).round().astype(int)
        y = np.linspace(0, self.map_y - 1, NUM_DENSE_COORDS_WIDTH).round().astype(int)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points

    def free_area(self, robot_map):
        # ... (保持不變) ...
        index = np.where(robot_map == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def find_closest_index_from_coords(self, node_coords, p):
        # ... (保持不變) ...
        if node_coords is None or len(node_coords) == 0: return None
        if not isinstance(p, np.ndarray): p = np.array(p)
        if p.shape != (2,): return None
        try: return np.argmin(np.linalg.norm(node_coords - p, axis=1))
        except ValueError: return None


    def find_k_neighbor_all_nodes(self, robot_map, update_dense=True, global_graph:Graph=None, global_graph_knn_dist_max=GLOBAL_GRAPH_KNN_RAD, global_graph_knn_dist_min=CUR_AGENT_KNN_RAD):
        # ... (此函式保持不變，但依賴於 self.node_coords 正確) ...
        if self.node_coords is None or len(self.node_coords) == 0: return
        try: kd_tree = KDTree(self.node_coords)
        except ValueError: return

        self.x = [] # 重置繪圖列表
        self.y = []

        for i, p in enumerate(self.node_coords):
            p_tuple = tuple(p)
            num_global_neighbours = 0
            # (global graph logic 保持不變) ...
            if global_graph is not None and hasattr(global_graph, 'edges') and p_tuple in global_graph.edges:
                # ... (略) ...
                for neighbour_node in topk_global_graph_nodes:
                    self.graph.add_node(p_tuple)
                    neighbour_tuple = tuple(neighbour_node)
                    dist = np.linalg.norm(np.array(p_tuple) - np.array(neighbour_tuple))
                    self.graph.add_edge(p_tuple, neighbour_tuple, dist)

            max_neighbours = self.k_size - num_global_neighbours
            num_neighbours_available = len(self.node_coords)
            # <--- 修正：查詢的 k 數必須 > 0 --- >
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
                            # <--- 修正：確保雙向邊也被加入 --- >
                            self.graph.add_node(end_tuple)
                            self.graph.add_edge(end_tuple, start_tuple, dist)
                        if self.plot:
                            self.x.append([p[0], neighbour[0]])
                            self.y.append([p[1], neighbour[1]])