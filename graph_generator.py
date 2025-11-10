import copy
import numpy as np
from scipy.spatial import KDTree
from time import time
import logging

from parameter import *
from graph import Graph, a_star
from node import Node
from utils import check_collision

logger = logging.getLogger(__name__)

class Graph_generator:
    def __init__(self, map_size, k_size, sensor_range, plot=False):
        """初始化 Graph_generator。

        Args:
            map_size (tuple): 地圖大小 (height, width)。
            k_size (int): K 個鄰居數。
            sensor_range (int): 感測半徑。
            plot (bool): 是否啟用繪圖。

        Returns:
            None
        """
        self.k_size = k_size
        self.graph = Graph()
        self.node_coords = None
        self.nodes_list: list[Node] = []
        self.node_utility = None
        self.target_candidates = np.array([]).reshape(0, 2)
        self.candidates_utility = np.array([])
        self.guidepost = None
        self.plot = plot
        self.x = []; self.y = []
        self.map_x = map_size[1]; self.map_y = map_size[0]
        self.uniform_points = self.generate_uniform_points()
        self.sensor_range = sensor_range
        self.route_node = []

    def edge_clear_all_nodes(self): self.graph = Graph(); self.x = []; self.y = []
    def edge_clear(self, coords): self.graph.clear_edge(tuple(coords))
    def node_clear(self, coords, remove_bidirectional_edges=False): self.graph.clear_node(tuple(coords), remove_bidirectional_edges=remove_bidirectional_edges)

    def generate_graph(self, robot_location, robot_map, frontiers):
        """建立初始圖結構並回傳 node/edges/utility/guidepost。

        Args:
            robot_location (array-like[2]): 機器人位置 (x,y)。
            robot_map (ndarray): 地圖 (y,x)。
            frontiers (ndarray): frontier 陣列 (N,2)。

        Returns:
            tuple: (node_coords (ndarray), graph_edges (dict), node_utility (ndarray), guidepost)
        """
        self.edge_clear_all_nodes()
        free_area = self.free_area(robot_map)
        if len(free_area) == 0: # 如果一開始就沒有自由空間
             logger.warning("No free area found in initial map for graph generation.")
             self.node_coords = np.array(robot_location).reshape(1, 2) # 至少包含起始點
             self.graph = Graph(); self.graph.add_node(tuple(robot_location)) # 空圖
        else:
            free_area_complex = free_area[:, 0] + free_area[:, 1] * 1j
            uniform_points_complex = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
            # assume_unique=True 可能導致問題，移除
            _, _, candidate_indices = np.intersect1d(free_area_complex, uniform_points_complex, return_indices=True)
            valid_candidate_indices = candidate_indices[candidate_indices < len(self.uniform_points)]
            node_coords = self.uniform_points[valid_candidate_indices]
            node_coords = np.concatenate((np.array(robot_location).reshape(1, 2), node_coords))
            self.node_coords = np.unique(node_coords, axis=0)

        if len(self.node_coords) > 0:
            self.find_k_neighbor_all_nodes(robot_map, update_dense=True)
        else:
             logger.warning("No nodes generated in generate_graph.")
             self.graph = Graph()

        current_frontiers = frontiers if frontiers is not None else np.array([]).reshape(0, 2)
        self._update_nodes_and_utilities(current_frontiers, robot_map, old_frontiers=None, all_robot_positions=None)
        self._update_guidepost()
        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost

    def update_node_utilities(self, robot_map, frontiers, old_frontiers, all_robot_positions=None):
        """輕量級更新節點效用（不重建整個圖）。

        Args:
            robot_map (ndarray): 地圖 (y,x)。
            frontiers (ndarray): 新的 frontier 陣列 (N,2)。
            old_frontiers (ndarray): 先前的 frontier 陣列 (M,2)。
            all_robot_positions (list|None): 其他機器人位置清單或 None。

        Returns:
            tuple: (node_utility (ndarray), guidepost)
        """
        logger.debug("Updating node utilities...")
        # Diagnostic: report current sizes (temporary)
        try:
            coords_len = len(self.node_coords) if hasattr(self, 'node_coords') and self.node_coords is not None else 0
        except Exception:
            coords_len = -1
        try:
            targ_len = len(self.target_candidates) if hasattr(self, 'target_candidates') and self.target_candidates is not None else 0
        except Exception:
            targ_len = -1
        try:
            util_len = len(self.candidates_utility) if hasattr(self, 'candidates_utility') and self.candidates_utility is not None else 0
        except Exception:
            util_len = -1
        logger.debug(f"[Diag graph] node_coords_len={coords_len} target_candidates_len={targ_len} candidates_utility_len={util_len}")
        current_frontiers = frontiers if frontiers is not None else np.array([]).reshape(0, 2)
        current_old_frontiers = old_frontiers if old_frontiers is not None else np.array([]).reshape(0, 2)
        self._update_nodes_and_utilities(current_frontiers, robot_map, old_frontiers=current_old_frontiers, all_robot_positions=all_robot_positions)
        self._update_guidepost()
        return self.node_utility, self.guidepost

    def rebuild_graph_structure(self, robot_map, frontiers, old_frontiers, position, all_robot_positions=None):
        """重建整個圖結構並更新效用（較昂貴）。

        Args:
            robot_map (ndarray): 地圖 (y,x)。
            frontiers (ndarray): 新的 frontier 陣列。
            old_frontiers (ndarray): 舊的 frontier 陣列。
            position (array-like[2]): 當前位置。
            all_robot_positions (list|None): 其他機器人位置清單或 None。

        Returns:
            tuple: (node_coords, graph_edges, node_utility, guidepost)
        """
        logger.debug("Rebuilding graph structure...")
        # 1. 計算新節點座標
        free_area = self.free_area(robot_map)
        if len(free_area) == 0:
             logger.warning("No free area found during rebuild.")
             new_node_coords = np.array(position).reshape(1, 2) # 至少包含當前位置
        else:
            free_area_complex = free_area[:, 0] + free_area[:, 1] * 1j
            uniform_points_complex = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
            _, _, candidate_indices = np.intersect1d(free_area_complex, uniform_points_complex, return_indices=True) # assume_unique 移除
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
        current_frontiers = frontiers if frontiers is not None else np.array([]).reshape(0, 2)
        for coords_tuple in nodes_to_add:
            if coords_tuple not in existing_coords_set:
                coords = np.array(coords_tuple)
                new_node = Node(coords, current_frontiers, robot_map)
                self.nodes_list.append(new_node)

        # 6. 重建圖邊緣
        self.graph = Graph()
        if len(self.node_coords) > 0:
            self.find_k_neighbor_all_nodes(robot_map, update_dense=True)
        else:
             logger.warning("No nodes available in rebuild_graph_structure.")

        # 7. 重建後立即更新效益
        current_old_frontiers = old_frontiers if old_frontiers is not None else np.array([]).reshape(0, 2)
        self._update_nodes_and_utilities(current_frontiers, robot_map, old_frontiers=current_old_frontiers, all_robot_positions=all_robot_positions)

        # 8. 更新 guidepost
        self._update_guidepost()
        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost

    def _update_nodes_and_utilities(self, frontiers, robot_map, old_frontiers=None, all_robot_positions=None):
        """內部：根據提供的 frontiers 與地圖更新 nodes_list 的 observable/frontier 及 candidates。

        Args:
            frontiers (ndarray): 當前 frontier 陣列。
            robot_map (ndarray): 地圖 (y,x)。
            old_frontiers (ndarray|None): 先前 frontier。
            all_robot_positions (list|None): 其他機器人位置清單或 None。

        Returns:
            None
        """
        # ...existing code...
        if self.nodes_list is None: self.nodes_list = []
        if frontiers is None: frontiers = np.array([]).reshape(0, 2)
        if old_frontiers is None: old_frontiers = np.array([]).reshape(0, 2)
        observed_frontiers_set = set(); new_frontiers_only = frontiers
        if len(old_frontiers) > 0 and len(frontiers) > 0:
            old_set = set(map(tuple, old_frontiers)); new_set = set(map(tuple, frontiers))
            observed_frontiers_set = old_set - new_set
            new_frontiers_only_set = new_set - old_set
            new_frontiers_only = np.array(list(new_frontiers_only_set)) if new_frontiers_only_set else np.array([]).reshape(0, 2)
        elif len(old_frontiers) == 0: new_frontiers_only = frontiers

        # 更新效益
        for node in self.nodes_list:
             if hasattr(node, 'update_observable_frontiers'):
                 # 檢查傳入的 new_frontiers_only 是否有效
                 valid_new_frontiers = new_frontiers_only if isinstance(new_frontiers_only, np.ndarray) and new_frontiers_only.ndim == 2 else np.array([]).reshape(0,2)
                 node.update_observable_frontiers(observed_frontiers_set, valid_new_frontiers, robot_map)

        # 處理佔用
        if all_robot_positions is not None:
            for robot_pos in all_robot_positions:
                if robot_pos is not None:
                    for node in self.nodes_list:
                         if hasattr(node, 'coords'):
                            dist = np.linalg.norm(node.coords - robot_pos)
                            if dist < 10: node.set_visited()

        # 重新生成列表
        self.target_candidates = []; self.candidates_utility = []; node_utilities_list = []
        final_node_coords = []
        if self.nodes_list is not None:
            for node in self.nodes_list:
                if hasattr(node, 'coords') and hasattr(node, 'utility'):
                     final_node_coords.append(node.coords)
                     utility = node.utility
                     node_utilities_list.append(utility)
                     if utility > 0:
                         self.target_candidates.append(node.coords)
                         self.candidates_utility.append(utility)

        # 與 node_coords 同步 (關鍵)
        self.node_coords = np.array(final_node_coords) if final_node_coords else np.array([]).reshape(0, 2)
        self.node_utility = np.array(node_utilities_list) if node_utilities_list else np.array([])
        self.target_candidates = np.array(self.target_candidates) if self.target_candidates else np.array([]).reshape(0, 2)
        self.candidates_utility = np.array(self.candidates_utility) if self.candidates_utility else np.array([])

        # 長度檢查 (Debug 用)
        if len(self.node_coords) != len(self.nodes_list) or len(self.node_coords) != len(self.node_utility):
             logger.warning(f"_update_nodes_and_utilities length mismatch! coords:{len(self.node_coords)}, list:{len(self.nodes_list)}, util:{len(self.node_utility)}")
             # 不再強制截斷，讓上層處理

    def _update_guidepost(self):
        """更新 guidepost（內部），回傳 None，更新 self.guidepost。"""
        if self.node_coords is not None and len(self.node_coords) > 0:
            self.guidepost = np.zeros((len(self.node_coords), 1))
            for node_pos in self.route_node:
                index = self.find_closest_index_from_coords(self.node_coords, node_pos)
                if index is not None and index < len(self.guidepost): self.guidepost[index] += 1
        else: self.guidepost = np.array([]).reshape(0, 1)

    # --- find_shortest_path, generate_uniform_points, free_area, find_closest_index_from_coords ---
    # --- find_k_neighbor_all_nodes ---
    # (保持不變)
    def find_shortest_path(self, current, destination, node_coords, graph):
        """以 Graph 執行 A* 搜尋最短路徑，回傳距離與節點路徑。

        Args:
            current (array-like[2]): 起點。
            destination (array-like[2]): 目標點。
            node_coords (ndarray): 節點座標陣列。
            graph (Graph): Graph 物件。

        Returns:
            tuple: (dist (float), route (list|None))；失敗回傳大距離與 None。
        """
        if node_coords is None or len(node_coords) == 0 or graph is None or not hasattr(graph, 'nodes'):
            logger.warning(f"Invalid input for find_shortest_path. Target:{destination}")
            return 1e5, None
        start_index = self.find_closest_index_from_coords(node_coords, current)
        end_index = self.find_closest_index_from_coords(node_coords, destination)
        if start_index is None or end_index is None:
             logger.warning(f"Cannot find node index. Start:{start_index}, End:{end_index}")
             return 1e5, None
        start_node = tuple(node_coords[start_index]); end_node = tuple(node_coords[end_index])
        if start_node not in graph.nodes or end_node not in graph.nodes:
             logger.warning(f"Start ({start_node}) or End ({end_node}) node not in graph nodes set.")
             return 1e5, None
        if start_node not in graph.edges or not graph.edges.get(start_node):
             logger.warning(f"Start node {start_node} has no outgoing edges.")
             if start_node == end_node: return 0, [start_node]
             return 1e5, None
        route, dist, _, _ = a_star(start_node, end_node, graph)
        if start_node != end_node and route is None:
             logger.warning(f"A* failed path from {start_node} to {end_node}")
             return dist, None
        if route is not None: route = list(map(tuple, route))
        return dist, route

    def generate_uniform_points(self):
        """產生均勻格點 (internal helper)。

        Returns:
            ndarray: 均勻點集合 (M,2)。
        """
        x = np.linspace(0, self.map_x - 1, NUM_DENSE_COORDS_WIDTH).round().astype(int)
        y = np.linspace(0, self.map_y - 1, NUM_DENSE_COORDS_WIDTH).round().astype(int)
        t1, t2 = np.meshgrid(x, y); points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points

    def free_area(self, robot_map):
        """回傳地圖中 free pixel 的座標陣列 (x,y)。

        Args:
            robot_map (ndarray): 地圖 (y,x)。

        Returns:
            ndarray: free pixels 座標 (N,2)。
        """
        index = np.where(robot_map == 255); free = np.asarray([index[1], index[0]]).T
        return free

    def find_closest_index_from_coords(self, node_coords, p):
        """在 node_coords 中找到距離 p 最近的索引。

        Args:
            node_coords (ndarray): 節點陣列。
            p (array-like[2]): 參考點。

        Returns:
            int|None: 最近節點索引，找不到則回傳 None。
        """
        if node_coords is None or len(node_coords) == 0: return None
        if not isinstance(p, np.ndarray): p = np.array(p)
        if p.shape != (2,): logger.warning(f"Invalid point shape {p.shape} in find_closest"); return None
        try: return np.argmin(np.linalg.norm(node_coords - p, axis=1))
        except ValueError: logger.error("ValueError in find_closest_index", exc_info=True); return None

    def find_k_neighbor_all_nodes(self, robot_map, update_dense=True, global_graph:Graph=None, global_graph_knn_dist_max=GLOBAL_GRAPH_KNN_RAD, global_graph_knn_dist_min=CUR_AGENT_KNN_RAD):
        """為每個 node 找 k 個鄰居並建立 graph.edges（包含 global graph 的補充）。

        Args:
            robot_map (ndarray): 地圖 (y,x)。
            update_dense (bool): 是否更新雙向邊。
            global_graph (Graph|None): 全域 graph 以考慮 global edges。
            global_graph_knn_dist_max (float): global 邊距離上限。
            global_graph_knn_dist_min (float): global 邊距離下限。

        Returns:
            None
        """
        if self.node_coords is None or len(self.node_coords) == 0: return
        try: kd_tree = KDTree(self.node_coords)
        except (ValueError, IndexError) as e: logger.error(f"KDTree Error: {e}"); return

        self.x = []; self.y = []

        for i, p in enumerate(self.node_coords):
            p_tuple = tuple(p)
            num_global_neighbours = 0
            topk_global_graph_nodes = set()
            if global_graph is not None and hasattr(global_graph, 'edges') and p_tuple in global_graph.edges:
                # ... (global graph logic) ...
                try: # 加強保護
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
                            topk_global_graph_nodes = set(map(tuple, filtered_nodes[topk_indices]))
                except Exception as e:
                     logger.warning(f"Error processing global graph edges for {p_tuple}: {e}")

            for neighbour_node in topk_global_graph_nodes:
                self.graph.add_node(p_tuple)
                neighbour_tuple = tuple(neighbour_node)
                dist = np.linalg.norm(np.array(p_tuple) - np.array(neighbour_tuple))
                self.graph.add_edge(p_tuple, neighbour_tuple, dist)

            max_neighbours = self.k_size - num_global_neighbours
            num_neighbours_available = len(self.node_coords)
            k_query = min(max(1, max_neighbours), num_neighbours_available)
            indices = []
            if k_query > 0:
                try:
                    distances, indices = kd_tree.query(p, k=k_query)
                    if np.isscalar(indices): indices = np.array([indices])
                except (ValueError, IndexError) as e: logger.warning(f"KDTree query failed k={k_query}: {e}"); indices = []

            for index in indices:
                if index < 0 or index >= len(self.node_coords): continue
                neighbour = self.node_coords[index]
                if np.array_equal(p, neighbour): continue
                start = p; end = neighbour
                if start.shape == (2,) and end.shape == (2,):
                    if not check_collision(start, end, robot_map):
                        start_tuple = tuple(start); end_tuple = tuple(end)
                        if update_dense:
                            dist = np.linalg.norm(start-end)
                            if start_tuple not in self.graph.nodes: self.graph.add_node(start_tuple)
                            if end_tuple not in self.graph.nodes: self.graph.add_node(end_tuple)
                            self.graph.add_edge(start_tuple, end_tuple, dist)
                            self.graph.add_edge(end_tuple, start_tuple, dist)
                        if self.plot:
                            self.x.append([p[0], neighbour[0]])
                            self.y.append([p[1], neighbour[1]])