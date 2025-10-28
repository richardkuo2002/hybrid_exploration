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
        self.node_coords = None
        self.plot = plot
        self.x = []
        self.y = []
        self.map_x = map_size[1]
        self.map_y = map_size[0]
        self.uniform_points = self.generate_uniform_points()
        self.sensor_range = sensor_range
        self.route_node = []
        self.nodes_list:list[Node] = []
        self.target_candidates = []
        self.candidates_utility = []
        # <--- 修正點：初始化 guidepost ---
        self.guidepost = None # 或者 self.guidepost = np.array([])
        # --- ---

    # ... (其餘程式碼不變) ...
    def edge_clear_all_nodes(self):
        self.graph = Graph()
        self.x = []
        self.y = []
    
    def edge_clear(self, coords):
        self.graph.clear_edge(tuple(coords))

    def node_clear(self, coords, remove_bidirectional_edges=False):
        self.graph.clear_node(tuple(coords), remove_bidirectional_edges=remove_bidirectional_edges)
    
    def generate_graph(self, robot_location, robot_map, frontiers):
        """
        ... (generate_graph 保持不變) ...
        """
        
        self.edge_clear_all_nodes()
        free_area = self.free_area(robot_map)
        
        free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        node_coords = self.uniform_points[candidate_indices]
        node_coords = np.concatenate((robot_location.reshape(1, 2), node_coords))
        self.node_coords = node_coords
        
        self.find_k_neighbor_all_nodes(robot_map, update_dense=True)
        
        self.node_utility = []
        self.nodes_list = [] # 確保清空
        self.target_candidates = [] # 確保清空
        self.candidates_utility = [] # 確保清空

        for coords in self.node_coords:
            node = Node(coords, frontiers, robot_map)
            self.nodes_list.append(node)
            utility = node.utility
            if utility > 0: # 修正: 檢查是否 > 0
                self.target_candidates.append(coords)
                self.candidates_utility.append(utility)
            self.node_utility.append(utility)
        
        # 確保總是轉換為 numpy array
        self.target_candidates = np.array(self.target_candidates) if self.target_candidates else np.array([]).reshape(0,2)
        self.candidates_utility = np.array(self.candidates_utility) if self.candidates_utility else np.array([])
        self.node_utility = np.array(self.node_utility) if self.node_utility else np.array([])

        # 初始化 guidepost
        self.guidepost = np.zeros((self.node_coords.shape[0], 1))
        # (這裡不需要 complex 轉換)
        for node_pos in self.route_node:
             # 安全檢查 node_coords 是否為空
            if self.node_coords is not None and len(self.node_coords) > 0:
                index = self.find_closest_index_from_coords(self.node_coords, node_pos)
                if index is not None and index < len(self.guidepost): # 邊界檢查
                    self.guidepost[index] += 1
        
        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost

    def update_node_utilities(self, robot_map, frontiers, old_frontiers, all_robot_positions=None):
        """
        僅更新節點的探索效益 (utility)，不重新計算圖結構。
        """
        
        # 1. 計算邊界變化
        observed_frontiers_set = set()
        new_frontiers_only = frontiers
        
        if len(old_frontiers) > 0 and len(frontiers) > 0:
            old_frontiers_to_check = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
            new_frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
            
            observed_frontiers_index = np.where(
                np.isin(old_frontiers_to_check, new_frontiers_to_check, assume_unique=True) == False)
            new_frontiers_index = np.where(
                np.isin(new_frontiers_to_check, old_frontiers_to_check, assume_unique=True) == False)
            
            if len(observed_frontiers_index[0]) > 0:
                observed_frontiers = old_frontiers[observed_frontiers_index]
                observed_frontiers_set = set(map(tuple, observed_frontiers))
            
            if len(new_frontiers_index[0]) > 0:
                new_frontiers_only = frontiers[new_frontiers_index]
            else:
                new_frontiers_only = np.array([]).reshape(0, 2)
        
        # 2. 更新所有現有節點的邊界資訊
        # 安全檢查 self.nodes_list 是否存在
        if not hasattr(self, 'nodes_list') or self.nodes_list is None:
             self.nodes_list = [] # 如果不存在則初始化
        for node in self.nodes_list:
            node.update_observable_frontiers(observed_frontiers_set, new_frontiers_only, robot_map)

        # 3. 處理機器人佔用節點
        if all_robot_positions is not None:
            for robot_pos in all_robot_positions:
                if robot_pos is not None:
                    for node in self.nodes_list:
                         if hasattr(node, 'coords'): # 檢查 node 是否有效
                            dist = np.linalg.norm(node.coords - robot_pos)
                            if dist < 10:
                                node.observable_frontiers_list.clear()
                                node.utility = 0
                                node.zero_utility_node = True
        
        # 4. 重新生成 node_utility 陣列和 candidates
        self.target_candidates = []
        self.candidates_utility = []
        
        # 安全檢查 self.nodes_list 和長度
        if not hasattr(self, 'nodes_list') or self.nodes_list is None or len(self.nodes_list) == 0:
            self.node_utility = np.array([])
        else:
             # 預分配，如果 nodes_list 有效
            self.node_utility = np.zeros(len(self.nodes_list))
            for i, node in enumerate(self.nodes_list):
                 # 再次檢查 node 有效性
                 if hasattr(node, 'utility'):
                    utility = node.utility
                    self.node_utility[i] = utility
                    if utility > 0 and hasattr(node, 'coords'): # 確保 coords 存在
                        self.target_candidates.append(node.coords)
                        self.candidates_utility.append(utility)
                 else:
                     # 如果 node 無效，utility 設為 0
                     self.node_utility[i] = 0

        # 確保轉換為 numpy array
        self.target_candidates = np.array(self.target_candidates) if self.target_candidates else np.array([]).reshape(0,2)
        self.candidates_utility = np.array(self.candidates_utility) if self.candidates_utility else np.array([])
        
        # 5. 更新 guidepost (僅在 node_coords 有效時)
        if self.node_coords is not None and len(self.node_coords) > 0:
            if not hasattr(self, 'guidepost') or self.guidepost is None or len(self.guidepost) != len(self.node_coords):
                self.guidepost = np.zeros((self.node_coords.shape[0], 1))
            else:
                self.guidepost.fill(0) # 重置為 0

            for node_pos in self.route_node:
                index = self.find_closest_index_from_coords(self.node_coords, node_pos)
                if index is not None and index < len(self.guidepost): # 邊界檢查
                    self.guidepost[index] += 1
        else:
            self.guidepost = np.array([]) # 如果 node_coords 無效，設為空

        return self.node_utility, self.guidepost


    def rebuild_graph_structure(self, robot_map, frontiers, old_frontiers, position, all_robot_positions=None):
        """
        (昂貴的函式) 完整重建圖結構和節點效益。
        """
        
        # 1. 重新計算有效節點座標
        free_area = self.free_area(robot_map)
        free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        new_node_coords = self.uniform_points[candidate_indices]
        # 安全檢查：確保 position 是 ndarray
        if isinstance(position, (list, tuple)):
            position = np.array(position)
        # 安全檢查：確保 position shape 正確
        if position.shape != (2,):
             # 提供一個預設值或報錯
             print(f"Warning: Invalid position shape {position.shape} in rebuild_graph_structure. Using default [0,0].")
             position = np.array([0,0])

        new_node_coords = np.concatenate((position.reshape(1, 2), new_node_coords))
        
        # 2. 比較新舊節點座標
        old_node_coords = self.node_coords if self.node_coords is not None else np.array([]).reshape(0, 2)
        old_nodes_set = set(map(tuple, old_node_coords))
        new_nodes_set = set(map(tuple, new_node_coords))
        
        nodes_to_remove = old_nodes_set - new_nodes_set
        nodes_to_add = new_nodes_set - old_nodes_set
        nodes_to_keep = old_nodes_set & new_nodes_set
        
        # 3. 更新 self.node_coords
        self.node_coords = new_node_coords
        
        # 4. 移除無效節點
        # 安全檢查 self.nodes_list 是否存在
        if not hasattr(self, 'nodes_list') or self.nodes_list is None:
             self.nodes_list = []
        self.nodes_list = [node for node in self.nodes_list 
                        if hasattr(node, 'coords') and tuple(node.coords) not in nodes_to_remove]
        
        # 清理圖中被移除的節點
        for coords in nodes_to_remove:
            self.node_clear(coords, remove_bidirectional_edges=True)
        
        # 5. 計算邊界變化
        observed_frontiers_set = set()
        new_frontiers_only = frontiers
        
        # 安全檢查 frontiers 和 old_frontiers 是否為 ndarray 且有內容
        if isinstance(old_frontiers, np.ndarray) and old_frontiers.ndim == 2 and old_frontiers.shape[0] > 0 and \
           isinstance(frontiers, np.ndarray) and frontiers.ndim == 2 and frontiers.shape[0] > 0:
            old_frontiers_to_check = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
            new_frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
            
            observed_frontiers_index = np.where(
                np.isin(old_frontiers_to_check, new_frontiers_to_check, assume_unique=True) == False)
            new_frontiers_index = np.where(
                np.isin(new_frontiers_to_check, old_frontiers_to_check, assume_unique=True) == False)
            
            if len(observed_frontiers_index[0]) > 0:
                observed_frontiers = old_frontiers[observed_frontiers_index]
                observed_frontiers_set = set(map(tuple, observed_frontiers))
            
            if len(new_frontiers_index[0]) > 0:
                new_frontiers_only = frontiers[new_frontiers_index]
            else:
                new_frontiers_only = np.array([]).reshape(0, 2)
        elif isinstance(frontiers, np.ndarray) and frontiers.ndim == 2:
             # 如果只有 new frontiers
             new_frontiers_only = frontiers
        else:
             # 如果兩者都無效
             new_frontiers_only = np.array([]).reshape(0, 2)


        # 6. 更新保留節點的邊界資訊
        for node in self.nodes_list:
             # 確保 node 有 coords 屬性
             if hasattr(node, 'coords'):
                node_coords_tuple = tuple(node.coords)
                if node_coords_tuple in nodes_to_keep:
                    node.update_observable_frontiers(observed_frontiers_set, new_frontiers_only, robot_map)
        
        # 7. 為新增節點建立 Node 物件
        existing_coords_set = set(tuple(node.coords) for node in self.nodes_list if hasattr(node, 'coords'))
        for coords_tuple in nodes_to_add:
            if coords_tuple not in existing_coords_set:
                coords = np.array(coords_tuple)
                new_node = Node(coords, frontiers, robot_map)
                self.nodes_list.append(new_node)
        
        # 8. 重建圖邊連線 (昂貴!)
        self.graph = Graph()
        # 安全檢查：確保 self.node_coords 有效才執行
        if self.node_coords is not None and len(self.node_coords) > 0:
            self.find_k_neighbor_all_nodes(robot_map, update_dense=True)
        
        # 9. 重新建立完整的資料一致性
        coords_to_node_map = {tuple(node.coords): node for node in self.nodes_list if hasattr(node, 'coords')}
        
        new_nodes_list = []
        new_node_utility = []
        self.target_candidates = []
        self.candidates_utility = []
        
        # 安全檢查：確保 self.node_coords 有效
        if self.node_coords is not None and len(self.node_coords) > 0:
            for coords in self.node_coords:
                coords_tuple = tuple(coords)
                
                if coords_tuple in coords_to_node_map:
                    node = coords_to_node_map[coords_tuple]
                    # 確保 node 有效
                    if hasattr(node, 'utility'):
                        utility = node.utility
                        if utility > 0:
                            self.target_candidates.append(coords)
                            self.candidates_utility.append(utility)
                        new_nodes_list.append(node)
                        new_node_utility.append(utility)
                    else: # 如果 node 無效
                         new_nodes_list.append(None) # 或其他標記
                         new_node_utility.append(0)

                else:
                    new_node = Node(coords, frontiers, robot_map)
                    utility = new_node.utility
                    if utility > 0:
                        self.target_candidates.append(coords)
                        self.candidates_utility.append(utility)
                    new_nodes_list.append(new_node)
                    new_node_utility.append(utility)
            
            # 過濾掉可能的 None
            valid_indices = [i for i, node in enumerate(new_nodes_list) if node is not None]
            self.nodes_list = [new_nodes_list[i] for i in valid_indices]
            # 如果 node_coords 和 utility list 需要同步，也要過濾
            # self.node_coords = self.node_coords[valid_indices]
            self.node_utility = np.array([new_node_utility[i] for i in valid_indices])


        # 確保總是轉換為 numpy array
        self.target_candidates = np.array(self.target_candidates) if self.target_candidates else np.array([]).reshape(0,2)
        self.candidates_utility = np.array(self.candidates_utility) if self.candidates_utility else np.array([])
        # self.node_utility 已在上面處理


        # 10. 處理機器人佔用節點
        if all_robot_positions is not None:
            for robot_pos in all_robot_positions:
                if robot_pos is not None:
                    for node in self.nodes_list:
                         if hasattr(node, 'coords'):
                            dist = np.linalg.norm(node.coords - robot_pos)
                            if dist < 10: # 假設靠近就算佔用
                                # <--- 修正點：使用正確的屬性名稱 ---
                                if hasattr(node, 'observable_frontiers_list'):
                                    node.observable_frontiers_list.clear() # 清空 list
                                # --- ---
                                node.utility = 0
                                node.zero_utility_node = True
        
        # 11. 重新生成 node_utility（因為可能有修改）
        # 安全檢查 self.nodes_list
        if hasattr(self, 'nodes_list') and self.nodes_list is not None:
            self.node_utility = np.array([node.utility for node in self.nodes_list if hasattr(node, 'utility')])
        else:
            self.node_utility = np.array([])

        
        # 12. 最終檢查 (保持不變)
        # (這裡的檢查可能因為上面的過濾而觸發，需要注意)
        if self.node_coords is not None and len(self.node_coords) != len(self.nodes_list) or \
           self.node_coords is not None and len(self.node_coords) != len(self.node_utility):
            print(f"Warning: Length mismatch after rebuild:")
            # 進行修正 (保持不變)
            min_len = min(len(self.node_coords) if self.node_coords is not None else 0,
                          len(self.nodes_list) if self.nodes_list is not None else 0,
                          len(self.node_utility) if self.node_utility is not None else 0)

            if self.node_coords is not None: self.node_coords = self.node_coords[:min_len]
            if self.nodes_list is not None: self.nodes_list = self.nodes_list[:min_len]
            if self.node_utility is not None: self.node_utility = self.node_utility[:min_len]
            print(f"  Corrected to length: {min_len}")


        # 13. 更新 guidepost (僅在 node_coords 有效時)
        if self.node_coords is not None and len(self.node_coords) > 0:
            self.guidepost = np.zeros((self.node_coords.shape[0], 1))
            for node_pos in self.route_node:
                index = self.find_closest_index_from_coords(self.node_coords, node_pos)
                if index is not None and index < len(self.guidepost): # 邊界檢查
                    self.guidepost[index] += 1
        else:
             self.guidepost = np.array([]) # 如果 node_coords 無效，設為空


        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost


    def find_shortest_path(self, current, destination, node_coords, graph):
        # 安全檢查
        if node_coords is None or len(node_coords) == 0 or graph is None:
            return 1e5, None # 返回高成本和 None 路徑

        t1 = time()
        start_index = self.find_closest_index_from_coords(node_coords, current)
        end_index = self.find_closest_index_from_coords(node_coords, destination)

        # 再次檢查索引有效性
        if start_index is None or end_index is None:
            return 1e5, None

        start_node = tuple(node_coords[start_index])
        end_node = tuple(node_coords[end_index])

        # 檢查節點是否存在於圖中
        if start_node not in graph.nodes or end_node not in graph.nodes:
            print(f"Warning: Start ({start_node}) or End ({end_node}) node not in graph.")
            return 1e5, None

        # 檢查 start_node 是否有邊
        if start_node not in graph.edges or not graph.edges[start_node]:
             print(f"Warning: Start node {start_node} has no outgoing edges.")
             # 嘗試找到一個最近的有邊的節點作為起點？(複雜)
             # 或者直接返回失敗
             return 1e5, None


        route, dist, _, _ = a_star(start_node, end_node, graph)

        # 原來的 assert 可能過於嚴格，改為檢查 route 是否為 None
        if start_node != end_node and route is None:
             print(f"Warning: A* failed to find path from {start_node} to {end_node}")
             # 可以嘗試返回 [start_node] 讓機器人停在原地？
             return dist, None # dist 可能是 1e5

        if route is not None:
            # A* 返回的是 tuple list, 我們需要 ndarray list?
            # 保持 tuple list 應該可以
            route = list(map(tuple, route))

        return dist, route

    
    def generate_uniform_points(self):
        # ... (generate_uniform_points 保持不變) ...
        x = np.linspace(0, self.map_x - 1, NUM_DENSE_COORDS_WIDTH).round().astype(int)
        y = np.linspace(0, self.map_y - 1, NUM_DENSE_COORDS_WIDTH).round().astype(int) 
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points
    
    def free_area(self, robot_map):
        # ... (free_area 保持不變) ...
        index = np.where(robot_map == 255)
        free = np.asarray([index[1], index[0]]).T
        return free
    
    def find_closest_index_from_coords(self, node_coords, p):
        # 安全檢查
        if node_coords is None or len(node_coords) == 0:
            return None # 返回 None 表示找不到
        # 確保 p 是 ndarray
        if not isinstance(p, np.ndarray):
             p = np.array(p)
        # 確保 p 的 shape
        if p.shape != (2,):
             print(f"Warning: Invalid point shape {p.shape} in find_closest_index. Returning None.")
             return None
        try:
            return np.argmin(np.linalg.norm(node_coords - p, axis=1))
        except ValueError: # 如果 node_coords 維度不對可能報錯
            print(f"Warning: ValueError in find_closest_index. node_coords shape: {node_coords.shape}, p shape: {p.shape}")
            return None

    
    def find_k_neighbor_all_nodes(self, robot_map, update_dense=True, global_graph:Graph=None, global_graph_knn_dist_max=GLOBAL_GRAPH_KNN_RAD, global_graph_knn_dist_min=CUR_AGENT_KNN_RAD):
        # 安全檢查
        if self.node_coords is None or len(self.node_coords) == 0:
            print("Warning: node_coords is empty in find_k_neighbor_all_nodes. Skipping.")
            return

        try:
            kd_tree = KDTree(self.node_coords)
        except ValueError:
            print(f"Warning: ValueError creating KDTree. node_coords shape: {self.node_coords.shape}. Skipping neighbor search.")
            return

        for i, p in enumerate(self.node_coords):
            # 確保 p 是 tuple 以便在 graph.edges 中查找
            p_tuple = tuple(p)

            num_global_neighbours = 0
            # 安全檢查 global_graph 和 p_tuple
            if global_graph is not None and hasattr(global_graph, 'edges') and p_tuple in global_graph.edges:
                global_graph_edges = global_graph.edges[p_tuple].values()
                global_graph_nodes = np.array([edge.to_node for edge in global_graph_edges])
                global_graph_dist = np.array([edge.length for edge in global_graph_edges])

                # 安全檢查 shape
                if global_graph_nodes.ndim == 2 and global_graph_dist.ndim == 1 and len(global_graph_nodes) == len(global_graph_dist):
                    filtered_global_graph_idx = (global_graph_dist <= global_graph_knn_dist_max) & (global_graph_dist > global_graph_knn_dist_min)   
                    filtered_global_graph_nodes = global_graph_nodes[filtered_global_graph_idx]
                    filtered_global_graph_dist = global_graph_dist[filtered_global_graph_idx]
                    
                    num_available = len(filtered_global_graph_nodes)
                    num_global_neighbours = min(num_available, self.k_size) # 不能超過 k_size

                    if num_global_neighbours > 0:
                        # 僅在有鄰居時排序
                        topk_indices = np.argsort(filtered_global_graph_dist)[:num_global_neighbours]
                        topk_global_graph_nodes = filtered_global_graph_nodes[topk_indices]
                        topk_global_graph_nodes = set(map(tuple, topk_global_graph_nodes))  

                        for neighbour_node in topk_global_graph_nodes:
                            self.graph.add_node(p_tuple)
                            # 確保 neighbour_node 也是 tuple
                            neighbour_tuple = tuple(neighbour_node)
                            # 計算實際距離
                            dist = np.linalg.norm(np.array(p_tuple) - np.array(neighbour_tuple))
                            self.graph.add_edge(p_tuple, neighbour_tuple, dist)

            
            max_neighbours = self.k_size - num_global_neighbours
            # 確保 num_neighbours > 0 且 <= node_coords 長度
            num_neighbours_available = len(self.node_coords)
            num_neighbours = min(max(1, max_neighbours), num_neighbours_available) # 至少查詢 1 個

            if num_neighbours > 0:
                 # 安全地執行查詢
                 # k 不能大於資料點數量
                k_query = min(num_neighbours, num_neighbours_available)
                if k_query > 0:
                    try:
                        distances, indices = kd_tree.query(p, k=k_query)
                        if np.isscalar(indices): # 如果只返回一個鄰居
                            indices = np.array([indices])
                    except ValueError as e:
                         print(f"Warning: KDTree query failed for point {p} with k={k_query}. Error: {e}")
                         indices = [] # 設為空，跳過後續處理

                else:
                    indices = []


                for index in indices:
                    # 邊界檢查
                    if index < 0 or index >= len(self.node_coords):
                        continue # 跳過無效索引

                    neighbour = self.node_coords[index]
                    # 避免自己連自己
                    if np.array_equal(p, neighbour):
                         continue

                    start = p
                    end = neighbour
                    # 安全檢查：確保 start, end shape 正確
                    if start.shape == (2,) and end.shape == (2,):
                        if not check_collision(start, end, robot_map):
                            start_tuple = tuple(start)
                            end_tuple = tuple(end)
                            if update_dense:
                                self.graph.add_node(start_tuple)
                                dist = np.linalg.norm(start-end)
                                self.graph.add_edge(start_tuple, end_tuple, dist)
                            if self.plot:
                                self.x.append([p[0], neighbour[0]])
                                self.y.append([p[1], neighbour[1]])