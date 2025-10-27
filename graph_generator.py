import copy
import numpy as np
from scipy.spatial import KDTree
from time import time

from parameter import *
from graph import Graph, a_star
from node import Node
from utils import check_collision  # <--- 修改點：從 utils 匯入

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
    
    # ... (edge_clear_all_nodes, edge_clear, node_clear 保持不變) ...
    def edge_clear_all_nodes(self):
        """
        初始化所有節點的圖結構與資料。

        說明：
            建立空的 Graph 物件，並重設 x、y 兩個座標儲存陣列，通常用於重新開始路徑搜尋或繪圖前的狀態清空。

        Returns:
            None
        """
        self.graph = Graph()
        self.x = []
        self.y = []
    
    def edge_clear(self, coords):
        """
        清除指定座標的圖邊（edge）。

        Args:
            coords (array-like): 要清除的邊所在座標 [x, y]。

        Returns:
            None。作用：self.graph 內該座標的邊會被移除。
        """
        self.graph.clear_edge(tuple(coords))

    def node_clear(self, coords, remove_bidirectional_edges=False):
        """
        清除指定座標的圖節點（node），可選是否同時移除雙向邊。

        Args:
            coords (array-like): 要清除的節點座標 [x, y]。
            remove_bidirectional_edges (bool, optional): 是否同時移除該節點所有雙向邊，預設為 False。

        Returns:
            None。作用：self.graph 內該座標的節點及相關邊會被移除。
        """
        self.graph.clear_node(tuple(coords), remove_bidirectional_edges=remove_bidirectional_edges)

    def generate_graph(self, robot_location, robot_map, frontiers):
        """
        根據地圖資訊、機器人位置及探索邊界，初始化圖結構並建立所有節點與其鄰邊。

        Args:
            robot_location (ndarray): 機器人目前座標，shape=(2,)。
            robot_map (ndarray): 機器人的本地地圖陣列，包含自由空間與障礙物資訊。
            frontiers (ndarray): 探索邊界座標陣列，shape=(N,2)。

        Returns:
            node_coords (ndarray): 所有有效節點的座標組合，形狀為 (M,2)。
            graph.edges (dict 或集合): 已生成的節點間連線（邊）資訊。
            node_utility (ndarray): 各節點的探索效益（observable frontiers 數量）。
            guidepost (ndarray): 路徑上節點的指標陣列，用於輔助導航等應用。
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
        for coords in self.node_coords:
            node = Node(coords, frontiers, robot_map)
            self.nodes_list.append(node)
            utility = node.utility
            if utility:
                self.target_candidates.append(coords)
                self.candidates_utility.append(utility)
            self.node_utility.append(utility)
        
        self.target_candidates = np.array(self.target_candidates)
        self.candidates_utility = np.array(self.candidates_utility)
        self.node_utility = np.array(self.node_utility)
        self.guidepost = np.zeros((self.node_coords.shape[0], 1))
        x = self.node_coords[:,0] + self.node_coords[:,1]*1j
        for node in self.route_node:
            index = self.find_closest_index_from_coords(self.node_coords, node)     
            self.guidepost[index] += 1    # = 1
        
        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost

    def update_graph(self, robot_map, frontiers, old_frontiers, position, all_robot_positions=None):
        """
        更新圖結構，包含節點座標、邊連線及探索效益。

        Args:
            robot_map (ndarray): 機器人目前的本地地圖陣列。
            frontiers (ndarray): 新探索邊界座標陣列，shape=(N,2)。
            old_frontiers (ndarray): 先前探索邊界座標陣列，shape=(M,2)。
            position (ndarray): 機器人目前位置座標 [x, y]。

        Returns:
            node_coords (ndarray): 更新後的節點座標。
            graph.edges (dict): 更新後的圖邊資訊。
            node_utility (ndarray): 各節點的探索效益。
            guidepost (ndarray): 路徑節點指標陣列。
        """
        
        # 1. 重新計算有效節點座標
        free_area = self.free_area(robot_map)
        free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        new_node_coords = self.uniform_points[candidate_indices]
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
        self.nodes_list = [node for node in self.nodes_list 
                        if tuple(node.coords) not in nodes_to_remove]
        
        # 清理圖中被移除的節點
        for coords in nodes_to_remove:
            self.node_clear(coords, remove_bidirectional_edges=True)
        
        # 5. 計算邊界變化
        observed_frontiers_set = set()
        new_frontiers_only = frontiers
        
        if len(old_frontiers) > 0 and len(frontiers) > 0:
            old_frontiers_to_check = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
            new_frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
            
            # 找出已被觀測的邊界
            observed_frontiers_index = np.where(
                np.isin(old_frontiers_to_check, new_frontiers_to_check, assume_unique=True) == False)
            # 找出新增的邊界
            new_frontiers_index = np.where(
                np.isin(new_frontiers_to_check, old_frontiers_to_check, assume_unique=True) == False)
            
            if len(observed_frontiers_index[0]) > 0:
                observed_frontiers = old_frontiers[observed_frontiers_index]
                observed_frontiers_set = set(map(tuple, observed_frontiers))
            
            if len(new_frontiers_index[0]) > 0:
                new_frontiers_only = frontiers[new_frontiers_index]
            else:
                new_frontiers_only = np.array([]).reshape(0, 2)
        
        # 6. 更新保留節點的邊界資訊
        for node in self.nodes_list:
            node_coords_tuple = tuple(node.coords)
            if node_coords_tuple in nodes_to_keep:
                # 使用 Node 的 update_observable_frontiers 方法
                node.update_observable_frontiers(observed_frontiers_set, new_frontiers_only, robot_map)
        
        # 7. 為新增節點建立 Node 物件
        existing_coords_set = set(tuple(node.coords) for node in self.nodes_list)
        for coords_tuple in nodes_to_add:
            if coords_tuple not in existing_coords_set:
                coords = np.array(coords_tuple)
                new_node = Node(coords, frontiers, robot_map)
                self.nodes_list.append(new_node)
        
        # 8. 重建圖邊連線
        self.graph = Graph()
        self.find_k_neighbor_all_nodes(robot_map, update_dense=True)
        
        # 9. 重新建立完整的資料一致性
        # 先清空現有的 nodes_list，重新建立
        coords_to_node_map = {tuple(node.coords): node for node in self.nodes_list}
        
        new_nodes_list = []
        new_node_utility = []
        self.target_candidates = []
        self.candidates_utility = []
        
        for coords in self.node_coords:
            coords_tuple = tuple(coords)
            
            # 查找現有節點
            if coords_tuple in coords_to_node_map:
                node = coords_to_node_map[coords_tuple]
                if node.utility:
                    self.target_candidates.append(coords)
                    self.candidates_utility.append(node.utility)
                new_nodes_list.append(node)
                new_node_utility.append(node.utility)
            else:
                # 建立新節點
                new_node = Node(coords, frontiers, robot_map)
                if node.utility:
                    self.target_candidates.append(coords)
                    self.candidates_utility.append(node.utility)
                new_nodes_list.append(new_node)
                new_node_utility.append(new_node.utility)
        
        self.target_candidates = np.array(self.target_candidates)
        self.candidates_utility = np.array(self.candidates_utility)
        # 更新 nodes_list 和 node_utility
        self.nodes_list = new_nodes_list
        self.node_utility = np.array(new_node_utility)
        
        # 10. 處理機器人佔用節點
        if all_robot_positions is not None:
            for robot_pos in all_robot_positions:
                if robot_pos is not None:
                    for node in self.nodes_list:
                        dist = np.linalg.norm(node.coords - robot_pos)
                        if dist < 10:
                            node.observable_frontiers.clear()
                            node.utility = 0
                            node.zero_utility_node = True
        
        # 11. 重新生成 node_utility（因為可能有修改）
        self.node_utility = np.array([node.utility for node in self.nodes_list])
        
        # 12. 最終檢查
        if len(self.node_coords) != len(self.nodes_list) or len(self.node_coords) != len(self.node_utility):
            print(f"Final check failed:")
            print(f"  node_coords: {len(self.node_coords)}")
            print(f"  nodes_list: {len(self.nodes_list)}")
            print(f"  node_utility: {len(self.node_utility)}")
            
            # 強制修正到最短長度
            min_len = min(len(self.node_coords), len(self.nodes_list))
            self.node_coords = self.node_coords[:min_len]
            self.nodes_list = self.nodes_list[:min_len]
            self.node_utility = self.node_utility[:min_len]
            print(f"  Corrected to length: {min_len}")
        
        # 13. 更新 guidepost
        self.guidepost = np.zeros((self.node_coords.shape[0], 1))
        for node in self.route_node:
            index = self.find_closest_index_from_coords(self.node_coords, node)     
            self.guidepost[index] += 1
        
        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost

    def find_shortest_path(self, current, destination, node_coords, graph):
        t1 = time()
        start_node = tuple(node_coords[self.find_closest_index_from_coords(node_coords, current)])        
        end_node = tuple(node_coords[self.find_closest_index_from_coords(node_coords, destination)])      
        route, dist, _, _ = a_star(start_node, end_node, graph) 

        if start_node != end_node:
            assert route != []

        elif route is not None:
            route = list(map(tuple, route))
        return dist, route
    
    def generate_uniform_points(self):
        """
        在地圖範圍內產生均勻分布的座標點

        Args:
            無參數，直接使用 class 內部的 map_x, map_y 作為地圖尺寸

        Returns:
            points (ndarray): 均勻分布於地圖範圍內的座標點，shape 為 (N, 2)
        """
        x = np.linspace(0, self.map_x - 1, NUM_DENSE_COORDS_WIDTH).round().astype(int)
        y = np.linspace(0, self.map_y - 1, NUM_DENSE_COORDS_WIDTH).round().astype(int) 
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points
    
    def free_area(self, robot_map):
        """
        從 Robot map 中取得自由空間座標

        Args:
            robot_map (ndarray): 機器人的本地地圖陣列

        Returns:
            free (ndarray): 地圖中自由空間的座標點 (N, 2)，每行是 [x, y]
        """
        index = np.where(robot_map == 255)
        free = np.asarray([index[1], index[0]]).T
        return free
    
    def find_closest_index_from_coords(self, node_coords, p):
        return np.argmin(np.linalg.norm(node_coords - p, axis=1))

    # <--- 修改點：移除了 check_collision 函式 --- >

    def find_k_neighbor_all_nodes(self, robot_map, update_dense=True, global_graph:Graph=None, global_graph_knn_dist_max=GLOBAL_GRAPH_KNN_RAD, global_graph_knn_dist_min=CUR_AGENT_KNN_RAD):
        """
        對所有節點尋找前 k 個鄰居 (最近鄰)，並根據全域圖與當前機器人地圖動態建立圖結構。
        """

        kd_tree = KDTree(self.node_coords)
        for i, p in enumerate(self.node_coords):

            # Append global graph edges to each node first (to ensure connectivity)
            num_global_neighbors = 0
            if global_graph is not None and tuple(p) in global_graph.edges:
                global_graph_edges = global_graph.edges[tuple(p)].values()
                global_graph_nodes = np.array([edge.to_node for edge in global_graph_edges])
                global_graph_dist = np.array([edge.length for edge in global_graph_edges])
                filtered_global_graph_idx = (global_graph_dist <= global_graph_knn_dist_max) & (global_graph_dist > global_graph_knn_dist_min)   
                filtered_global_graph_nodes = global_graph_nodes[filtered_global_graph_idx]
                filtered_global_graph_dist = global_graph_dist[filtered_global_graph_idx]
                num_global_neighbors = len(filtered_global_graph_nodes) if len(filtered_global_graph_nodes) < self.k_size else self.k_size
                topk_global_graph_nodes = filtered_global_graph_nodes[np.argsort(filtered_global_graph_dist)[:num_global_neighbors]]
                topk_global_graph_nodes = set(map(tuple, topk_global_graph_nodes))  

                for neighbor_node in topk_global_graph_nodes:
                    self.graph.add_node(tuple(p))
                    self.graph.add_edge(tuple(p), tuple(neighbor_node), np.linalg.norm(p-neighbor_node))
            
            max_neighbors = self.k_size - num_global_neighbors
            num_neighbors = len(self.node_coords) if len(self.node_coords) < max_neighbors else max_neighbors
            if num_neighbors > 0:
                _, indices = kd_tree.query(p, k=num_neighbors)
                if np.isscalar(indices):
                    indices = np.array([indices])
                for j, neighbor in enumerate(self.node_coords[indices]):                
                    start = p
                    end = neighbor
                    if not check_collision(start, end, robot_map): # <--- 修改點
                        if update_dense:
                            self.graph.add_node(tuple(start))
                            self.graph.add_edge(tuple(start), tuple(end), np.linalg.norm(start-end))
                        if self.plot:
                            self.x.append([p[0], neighbor[0]])
                            self.y.append([p[1], neighbor[1]])