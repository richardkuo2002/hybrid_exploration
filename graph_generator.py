import copy
import logging
from time import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.spatial import KDTree

from graph import Graph, a_star
from node import Node
from parameter import *
from utils import check_collision

logger = logging.getLogger(__name__)


class Graph_generator:
    def __init__(
        self,
        map_size: Tuple[int, int],
        k_size: int,
        sensor_range: int,
        plot: bool = False,
        debug_mode: bool = False,
    ) -> None:
        """初始化 Graph_generator。

        Args:
            map_size (Tuple[int, int]): 地圖大小 (height, width)。
            k_size (int): K 個鄰居數。
            sensor_range (int): 感測半徑。
            plot (bool): 是否啟用繪圖。
            debug_mode (bool): 是否啟用除錯模式。

        Returns:
            None
        """
        self.k_size = k_size
        self.graph = Graph()
        self.node_coords: Optional[np.ndarray] = None
        self.nodes_list: List[Node] = []
        self.node_utility: Optional[np.ndarray] = None
        self.target_candidates = np.array([]).reshape(0, 2)
        self.candidates_utility = np.array([])
        self.guidepost: Optional[np.ndarray] = None
        self.plot = plot
        self.debug_mode = debug_mode
        self.x: List[List[int]] = []
        self.y: List[List[int]] = []
        self.map_x = map_size[1]
        self.map_y = map_size[0]
        self.uniform_points = self.generate_uniform_points()
        self.sensor_range = sensor_range
        self.route_node: List[np.ndarray] = []

    def edge_clear_all_nodes(self) -> None:
        self.graph = Graph()
        self.x = []
        self.y = []

    def edge_clear(self, coords: np.ndarray) -> None:
        self.graph.clear_edge(tuple(coords))

    def node_clear(
        self, coords: np.ndarray, remove_bidirectional_edges: bool = False
    ) -> None:
        self.graph.clear_node(
            tuple(coords), remove_bidirectional_edges=remove_bidirectional_edges
        )

    def generate_graph(
        self, robot_location: np.ndarray, robot_map: np.ndarray, frontiers: np.ndarray
    ) -> Tuple[np.ndarray, Dict, np.ndarray, Optional[np.ndarray]]:
        """建立初始圖結構並回傳 node/edges/utility/guidepost。

        Args:
            robot_location (np.ndarray): 機器人位置 (x,y)。
            robot_map (np.ndarray): 地圖 (y,x)。
            frontiers (np.ndarray): frontier 陣列 (N,2)。

        Returns:
            Tuple[np.ndarray, Dict, np.ndarray, Optional[np.ndarray]]: (node_coords, graph_edges, node_utility, guidepost)
        """
        self.edge_clear_all_nodes()
        from parameter import PIXEL_FREE

        free_area = self.free_area(robot_map)
        if len(free_area) == 0:  # 如果一開始就沒有自由空間
            logger.warning("No free area found in initial map for graph generation.")
            self.node_coords = np.array(robot_location).reshape(1, 2)  # 至少包含起始點
            self.graph = Graph()
            self.graph.add_node(tuple(robot_location))  # 空圖
        else:
            free_area_complex = free_area[:, 0] + free_area[:, 1] * 1j
            uniform_points_complex = (
                self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
            )
            # assume_unique=True 可能導致問題，移除
            _, _, candidate_indices = np.intersect1d(
                free_area_complex, uniform_points_complex, return_indices=True
            )
            valid_candidate_indices = candidate_indices[
                candidate_indices < len(self.uniform_points)
            ]
            node_coords = self.uniform_points[valid_candidate_indices]
            node_coords = np.concatenate(
                (np.array(robot_location).reshape(1, 2), node_coords)
            )
            self.node_coords = np.unique(node_coords, axis=0)
            
            # Fix: Populate nodes_list
            self.nodes_list = []
            current_frontiers = (
                frontiers if frontiers is not None else np.array([]).reshape(0, 2)
            )
            for coords in self.node_coords:
                self.nodes_list.append(Node(coords, current_frontiers, robot_map))

        if len(self.node_coords) > 0:
            self.find_k_neighbor_all_nodes(robot_map, update_dense=True)
        else:
            logger.warning("No nodes generated in generate_graph.")
            self.graph = Graph()

        current_frontiers = (
            frontiers if frontiers is not None else np.array([]).reshape(0, 2)
        )
        all_robot_positions = [robot_location]
        self._update_nodes_and_utilities(
            current_frontiers, robot_map, old_frontiers=None, all_robot_positions=all_robot_positions
        )
        self._update_guidepost()

        # Graph Pruning (Initial)
        if ENABLE_GRAPH_PRUNING:
            self.prune_graph_to_skeleton(all_robot_positions)

        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost

    def update_node_utilities(
        self,
        robot_map: np.ndarray,
        frontiers: np.ndarray,
        old_frontiers: Optional[np.ndarray],
        all_robot_positions: Optional[List[Optional[np.ndarray]]] = None,
        caller: str = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """輕量級更新節點效用（不重建整個圖）。

        Args:
            robot_map (np.ndarray): 地圖 (y,x)。
            frontiers (np.ndarray): 新的 frontier 陣列 (N,2)。
            old_frontiers (Optional[np.ndarray]): 先前的 frontier 陣列 (M,2)。
            all_robot_positions (Optional[List[Optional[np.ndarray]]]): 其他機器人位置清單或 None。
            caller (str): 呼叫者識別字串 (debug用)。

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: (node_utility, guidepost)
        """
        logger.debug(f"Updating node utilities... caller={caller}")
        # Diagnostic: report current sizes (temporary)
        try:
            coords_len = (
                len(self.node_coords)
                if hasattr(self, "node_coords") and self.node_coords is not None
                else 0
            )
        except Exception:
            coords_len = -1
        try:
            targ_len = (
                len(self.target_candidates)
                if hasattr(self, "target_candidates")
                and self.target_candidates is not None
                else 0
            )
        except Exception:
            targ_len = -1
        try:
            util_len = (
                len(self.candidates_utility)
                if hasattr(self, "candidates_utility")
                and self.candidates_utility is not None
                else 0
            )
        except Exception:
            util_len = -1
        logger.debug(
            f"[Diag graph] caller={caller} node_coords_len={coords_len} target_candidates_len={targ_len} candidates_utility_len={util_len}"
        )
        current_frontiers = (
            frontiers if frontiers is not None else np.array([]).reshape(0, 2)
        )
        current_old_frontiers = (
            old_frontiers if old_frontiers is not None else np.array([]).reshape(0, 2)
        )
        self._update_nodes_and_utilities(
            current_frontiers,
            robot_map,
            old_frontiers=current_old_frontiers,
            all_robot_positions=all_robot_positions,
        )
        self._update_guidepost()
        return self.node_utility, self.guidepost

    def rebuild_graph_structure(
        self,
        robot_map: np.ndarray,
        frontiers: np.ndarray,
        old_frontiers: Optional[np.ndarray],
        position: np.ndarray,
        all_robot_positions: Optional[List[Optional[np.ndarray]]] = None,
    ) -> Tuple[np.ndarray, Dict, np.ndarray, Optional[np.ndarray]]:
        """重建整個圖結構並更新效用（較昂貴）。

        Args:
            robot_map (np.ndarray): 地圖 (y,x)。
            frontiers (np.ndarray): 新的 frontier 陣列。
            old_frontiers (Optional[np.ndarray]): 舊的 frontier 陣列。
            position (np.ndarray): 當前位置。
            all_robot_positions (Optional[List[Optional[np.ndarray]]]): 其他機器人位置清單或 None。

        Returns:
            Tuple[np.ndarray, Dict, np.ndarray, Optional[np.ndarray]]: (node_coords, graph_edges, node_utility, guidepost)
        """
        logger.debug("Rebuilding graph structure...")
        # 1. 計算新節點座標
        free_area = self.free_area(robot_map)
        if len(free_area) == 0:
            # This is an informative message — not an error. Lower severity to DEBUG to avoid cluttering batch run logs.
            logger.debug("No free area found during rebuild.")
            new_node_coords = np.array(position).reshape(1, 2)  # 至少包含當前位置
        else:
            free_area_complex = free_area[:, 0] + free_area[:, 1] * 1j
            uniform_points_complex = (
                self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
            )
            _, _, candidate_indices = np.intersect1d(
                free_area_complex, uniform_points_complex, return_indices=True
            )  # assume_unique 移除
            valid_candidate_indices = candidate_indices[
                candidate_indices < len(self.uniform_points)
            ]
            new_potential_nodes = self.uniform_points[valid_candidate_indices]
            current_pos_arr = np.array(position).reshape(1, 2)
            new_node_coords = np.concatenate((current_pos_arr, new_potential_nodes))
            new_node_coords = np.unique(new_node_coords, axis=0)

        # 2. 比較新舊節點
        old_node_coords = (
            self.node_coords
            if self.node_coords is not None
            else np.array([]).reshape(0, 2)
        )
        old_nodes_set = set(map(tuple, old_node_coords))
        new_nodes_set = set(map(tuple, new_node_coords))
        nodes_to_remove = old_nodes_set - new_nodes_set
        nodes_to_add = new_nodes_set - old_nodes_set

        # 3. 更新 self.node_coords
        self.node_coords = new_node_coords

        # 4. 移除失效 Node 物件
        if self.nodes_list is None:
            self.nodes_list = []
        self.nodes_list = [
            node
            for node in self.nodes_list
            if hasattr(node, "coords") and tuple(node.coords) not in nodes_to_remove
        ]

        # 5. 創建新增 Node 物件
        existing_coords_set = set(
            tuple(node.coords) for node in self.nodes_list if hasattr(node, "coords")
        )
        current_frontiers = (
            frontiers if frontiers is not None else np.array([]).reshape(0, 2)
        )
        for coords_tuple in nodes_to_add:
            if coords_tuple not in existing_coords_set:
                coords = np.array(coords_tuple)
                new_node = Node(coords, current_frontiers, robot_map)
                self.nodes_list.append(new_node)

        # 6. 增量更新圖邊緣 (Incremental Edge Update)
        # 找出需要更新邊緣的節點：
        # 1. 新增的節點 (nodes_to_add)
        # 2. 位於機器人感測範圍附近的節點 (可能因障礙物變動而受影響)
        
        affected_indices = []
        nodes_to_add_set = set(map(tuple, nodes_to_add))
        
        # 定義受影響半徑 (感測半徑 + 餘裕)
        affected_radius = self.sensor_range + 10
        
        robot_positions_arr = []
        if all_robot_positions:
            robot_positions_arr = [p for p in all_robot_positions if p is not None]
        
        if len(self.node_coords) > 0:
            # 優化：一次性遍歷找出所有受影響節點索引
            for i, coord in enumerate(self.node_coords):
                coord_tuple = tuple(coord)
                is_affected = False
                
                # 條件 1: 新增節點
                if coord_tuple in nodes_to_add_set:
                    is_affected = True
                
                # 條件 2: 附近有機器人 (僅對非新增節點檢查，避免重複)
                if not is_affected and robot_positions_arr:
                    # 向量化計算到所有機器人的距離
                    dists = np.linalg.norm(coord - robot_positions_arr, axis=1)
                    if np.any(dists < affected_radius):
                        is_affected = True
                
                if is_affected:
                    affected_indices.append(i)
                    # 清除舊邊緣 (雙向)，因為障礙物可能改變了連通性
                    # 注意：對於新增節點，clear_edge 無副作用
                    if coord_tuple in self.graph.nodes:
                        self.graph.clear_edge(coord_tuple, remove_bidirectional_edges=True)

            # 執行局部 KNN 更新
            if affected_indices:
                logger.debug(f"Incremental graph update: {len(affected_indices)}/{len(self.node_coords)} nodes affected.")
                self.find_k_neighbor_all_nodes(
                    robot_map, 
                    update_dense=True, 
                    target_indices=affected_indices
                )
            else:
                logger.debug("No nodes affected for incremental update.")
        else:
            logger.warning("No nodes available in rebuild_graph_structure.")

        # 7. 重建後立即更新效益
        current_old_frontiers = (
            old_frontiers if old_frontiers is not None else np.array([]).reshape(0, 2)
        )
        self._update_nodes_and_utilities(
            current_frontiers,
            robot_map,
            old_frontiers=current_old_frontiers,
            all_robot_positions=all_robot_positions,
        )

        # 8. 更新 guidepost
        self._update_guidepost()

        # 9. Graph Pruning (New Feature)
        if self.debug_mode:
            print(f"[DEBUG] ENABLE_GRAPH_PRUNING: {ENABLE_GRAPH_PRUNING}")
        if ENABLE_GRAPH_PRUNING:
            self.prune_graph_to_skeleton(all_robot_positions)

        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost

    def prune_graph_to_skeleton(
        self, all_robot_positions: Optional[List[Optional[np.ndarray]]] = None
    ) -> None:
        """Active Pruning: Keep only nodes that are part of paths to frontiers or near robots.
        
        mimics IR2's efficiency strategy by removing "dead branches".
        """
        if self.node_coords is None or len(self.node_coords) == 0:
            return

        # 1. Identify Target Nodes (Utility > 0)
        target_indices = np.where(self.node_utility > 0)[0]
        if len(target_indices) == 0:
            logger.debug("No targets for pruning. Skipping.")
            return
        
        # 2. Identify Source Nodes (Robot Allocations)
        robot_indices = []
        if all_robot_positions:
            for i, pos in enumerate(all_robot_positions):
                if pos is not None:
                    idx = self.find_closest_index_from_coords(self.node_coords, pos)
                    if idx is not None:
                        robot_indices.append(idx)
        
        if not robot_indices:
             logger.debug("No robots for pruning sources. Skipping.")
             return

        # 3. Batch Pathfinding (Union of Shortest Paths)
        # We want to keep any node that is part of a path from ANY robot to ANY target.
        # Ideally, we should use Dijkstra from each frontier backwards, or just A* for assigned tasks.
        # For simplicity and robustness, we'll keep paths from NEAREST robot to each target.
        
        nodes_to_keep = set()
        
        # Add robot safety zones (KeepAlive)
        robot_positions_arr = np.array([p for p in all_robot_positions if p is not None])
        if len(robot_positions_arr) > 0:
            for i, coord in enumerate(self.node_coords):
                dists = np.linalg.norm(coord - robot_positions_arr, axis=1)
                if np.any(dists < PRUNING_KEEPALIVE_RADIUS):
                     nodes_to_keep.add(tuple(coord))

        # Add paths
        # To avoid N_robots * N_targets A* calls, we can:
        # Strategy A: Only plan for assigned targets (if we had assignment info here).
        # Strategy B: Plan for ALL targets from their closest robot.
        
        for t_idx in target_indices:
            target_coord = self.node_coords[t_idx]
            
            # Find closest robot to this target (Euclidean heuristic is fine for selection)
            # This mimics "this target belongs to this region"
            dists_to_robots = np.linalg.norm(target_coord - robot_positions_arr, axis=1)
            closest_robot_idx = np.argmin(dists_to_robots)
            source_pos = robot_positions_arr[closest_robot_idx]
            
            # Run A*
            dist, route = self.find_shortest_path(source_pos, target_coord, self.node_coords, self.graph)
            if route:
                for p in route:
                    nodes_to_keep.add(tuple(p))

        # 4. Filter Nodes and Rebuild
        logger.debug(f"Pruning: Keeping {len(nodes_to_keep)} / {len(self.node_coords)} nodes.")
        
        old_nodes_list = self.node_coords.tolist()
        new_nodes_list = [p for p in old_nodes_list if tuple(p) in nodes_to_keep]
        
        if len(new_nodes_list) == len(old_nodes_list):
            return # No change

        self.node_coords = np.array(new_nodes_list) if new_nodes_list else np.array([]).reshape(0, 2)
        
        # Re-sync lists (utility, Nodes object)
        # It's expensive to filter everything, but necessary.
        keep_set = set(map(tuple, self.node_coords))
        
        self.nodes_list = [n for n in self.nodes_list if tuple(n.coords) in keep_set]
        
        # Re-calc utility array (faster than full update loop)
        self.node_utility = np.array([n.utility for n in self.nodes_list])
        
        # Rebuild Graph Edges (Subset)
        # We can just clear edges for removed nodes, or rebuild the whole graph object to be clean.
        # Clearing edges is safer? 
        # Actually, self.graph stores edges in a dict. We should remove keys for removed nodes.
        
        nodes_to_remove = set(map(tuple, old_nodes_list)) - keep_set
        for p in nodes_to_remove:
            self.graph.clear_node(p, remove_bidirectional_edges=True)
            
        # Update Guidepost
        self._update_guidepost()

    def _update_nodes_and_utilities(
        self,
        frontiers: np.ndarray,
        robot_map: np.ndarray,
        old_frontiers: Optional[np.ndarray] = None,
        all_robot_positions: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        """內部：根據提供的 frontiers 與地圖更新 nodes_list 的 observable/frontier 及 candidates。

        Args:
            frontiers (np.ndarray): 當前 frontier 陣列。
            robot_map (np.ndarray): 地圖 (y,x)。
            old_frontiers (Optional[np.ndarray]): 先前 frontier。
            all_robot_positions (Optional[List[Optional[np.ndarray]]]): 其他機器人位置清單或 None。

        Returns:
            None
        """
        if self.nodes_list is None:
            self.nodes_list = []
        if frontiers is None:
            frontiers = np.array([]).reshape(0, 2)
        if old_frontiers is None:
            old_frontiers = np.array([]).reshape(0, 2)
        # Diagnostic: log inputs and current internal lists to help trace empty-results issue
        try:
            logger.debug(
                f"[Diag _update] frontiers_shape={getattr(frontiers, 'shape', None)} old_frontiers_shape={getattr(old_frontiers, 'shape', None)} robot_map_shape={getattr(robot_map, 'shape', None)} node_coords_len={len(self.node_coords) if hasattr(self, 'node_coords') and self.node_coords is not None else 0} nodes_list_len={len(self.nodes_list)}"
            )
            # sample first few nodes if present
            if len(self.nodes_list) > 0:
                sample = []
                for n in self.nodes_list[:5]:
                    try:
                        sample.append(
                            (
                                tuple(n.coords) if hasattr(n, "coords") else None,
                                getattr(n, "utility", None),
                            )
                        )
                    except Exception:
                        sample.append((None, None))
                logger.debug(f"[Diag _update] nodes_list_sample={sample}")
        except Exception:
            logger.exception("Error logging diagnostics in _update_nodes_and_utilities")
        observed_frontiers_set = set()
        new_frontiers_only = frontiers
        if len(old_frontiers) > 0 and len(frontiers) > 0:
            old_set = set(map(tuple, old_frontiers))
            new_set = set(map(tuple, frontiers))
            observed_frontiers_set = old_set - new_set
            new_frontiers_only_set = new_set - old_set
            new_frontiers_only = (
                np.array(list(new_frontiers_only_set))
                if new_frontiers_only_set
                else np.array([]).reshape(0, 2)
            )
        elif len(old_frontiers) == 0:
            new_frontiers_only = frontiers

        # 更新效益
        for node in self.nodes_list:
            if hasattr(node, "update_observable_frontiers"):
                # 檢查傳入的 new_frontiers_only 是否有效
                valid_new_frontiers = (
                    new_frontiers_only
                    if isinstance(new_frontiers_only, np.ndarray)
                    and new_frontiers_only.ndim == 2
                    else np.array([]).reshape(0, 2)
                )
                node.update_observable_frontiers(
                    observed_frontiers_set, valid_new_frontiers, robot_map
                )

        # 處理佔用
        if all_robot_positions is not None:
            for robot_pos in all_robot_positions:
                if robot_pos is not None:
                    for node in self.nodes_list:
                        if hasattr(node, "coords"):
                            dist = np.linalg.norm(node.coords - robot_pos)
                            if dist < VISITED_DIST_THRESHOLD:
                                node.set_visited()

        # 重新生成列表
        self.target_candidates = []
        self.candidates_utility = []
        node_utilities_list = []
        final_node_coords = []
        if self.nodes_list is not None:
            for node in self.nodes_list:
                if hasattr(node, "coords") and hasattr(node, "utility"):
                    final_node_coords.append(node.coords)
                    utility = node.utility
                    node_utilities_list.append(utility)
                    if utility > 0:
                        self.target_candidates.append(node.coords)
                        self.candidates_utility.append(utility)

        # 與 node_coords 同步 (關鍵)
        self.node_coords = (
            np.array(final_node_coords)
            if final_node_coords
            else np.array([]).reshape(0, 2)
        )
        self.node_utility = (
            np.array(node_utilities_list) if node_utilities_list else np.array([])
        )
        self.target_candidates = (
            np.array(self.target_candidates)
            if self.target_candidates
            else np.array([]).reshape(0, 2)
        )
        self.candidates_utility = (
            np.array(self.candidates_utility)
            if self.candidates_utility
            else np.array([])
        )

        # 長度檢查 (Debug 用)
        if len(self.node_coords) != len(self.nodes_list) or len(
            self.node_coords
        ) != len(self.node_utility):
            logger.warning(
                f"_update_nodes_and_utilities length mismatch! coords:{len(self.node_coords)}, list:{len(self.nodes_list)}, util:{len(self.node_utility)}"
            )
            # 不再強制截斷，讓上層處理

    def _update_guidepost(self) -> None:
        """更新 guidepost（內部），回傳 None，更新 self.guidepost。"""
        if self.node_coords is not None and len(self.node_coords) > 0:
            self.guidepost = np.zeros((len(self.node_coords), 1))
            for node_pos in self.route_node:
                index = self.find_closest_index_from_coords(self.node_coords, node_pos)
                if index is not None and index < len(self.guidepost):
                    self.guidepost[index] += 1
        else:
            self.guidepost = np.array([]).reshape(0, 1)

    def find_shortest_path(
        self,
        current: np.ndarray,
        destination: np.ndarray,
        node_coords: np.ndarray,
        graph: Graph,
    ) -> Tuple[float, Optional[List[Tuple[int, int]]]]:
        """以 Graph 執行 A* 搜尋最短路徑，回傳距離與節點路徑。

        Args:
            current (np.ndarray): 起點。
            destination (np.ndarray): 目標點。
            node_coords (np.ndarray): 節點座標陣列。
            graph (Graph): Graph 物件。

        Returns:
            Tuple[float, Optional[List[Tuple[int, int]]]]: (dist (float), route (list|None))；失敗回傳大距離與 None。
        """
        if (
            node_coords is None
            or len(node_coords) == 0
            or graph is None
            or not hasattr(graph, "nodes")
        ):
            logger.warning(
                f"Invalid input for find_shortest_path. Target:{destination}"
            )
            return LARGE_DISTANCE, None
        start_index = self.find_closest_index_from_coords(node_coords, current)
        end_index = self.find_closest_index_from_coords(node_coords, destination)
        if start_index is None or end_index is None:
            # These conditions can occur due to invalid or missing current/destination
            # (e.g., None or outside node set). Treat as non-fatal; log at debug level
            # to avoid noisy logs during batch runs, but keep the information under DEBUG.
            logger.debug(
                f"Cannot find node index. Start:{start_index}, End:{end_index}"
            )
            return LARGE_DISTANCE, None
        start_node = tuple(node_coords[start_index])
        end_node = tuple(node_coords[end_index])
        if start_node not in graph.nodes or end_node not in graph.nodes:
            logger.debug(
                f"Start ({start_node}) or End ({end_node}) node not in graph nodes set."
            )
            return LARGE_DISTANCE, None
        if start_node not in graph.edges or not graph.edges.get(start_node):
            logger.debug(f"Start node {start_node} has no outgoing edges.")
            if start_node == end_node:
                return 0, [start_node]
            return LARGE_DISTANCE, None
        route, dist, _, _ = a_star(start_node, end_node, graph)
        if start_node != end_node and route is None:
            logger.debug(f"A* failed path from {start_node} to {end_node}")
            return dist, None
        if route is not None:
            route = list(map(tuple, route))
        return dist, route

    def generate_uniform_points(self) -> np.ndarray:
        """產生均勻格點 (internal helper)。

        Returns:
            np.ndarray: 均勻點集合 (M,2)。
        """
        x = np.linspace(0, self.map_x - 1, NUM_DENSE_COORDS_WIDTH).round().astype(int)
        y = np.linspace(0, self.map_y - 1, NUM_DENSE_COORDS_WIDTH).round().astype(int)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points

    def free_area(self, robot_map: np.ndarray) -> np.ndarray:
        """回傳地圖中 free pixel 的座標陣列 (x,y)。

        Args:
            robot_map (np.ndarray): 地圖 (y,x)。

        Returns:
            np.ndarray: free pixels 座標 (N,2)。
        """
        from parameter import PIXEL_FREE

        index = np.where(robot_map == PIXEL_FREE)
        free = np.asarray([index[1], index[0]]).T
        return free

    def find_closest_index_from_coords(
        self, node_coords: np.ndarray, p: np.ndarray
    ) -> Optional[int]:
        """在 node_coords 中找到距離 p 最近的索引。

        Args:
            node_coords (np.ndarray): 節點陣列。
            p (np.ndarray): 參考點。

        Returns:
            Optional[int]: 最近節點索引，找不到則回傳 None。
        """
        if node_coords is None or len(node_coords) == 0:
            return None
        if p is None:
            # silently return None — caller will handle
            return None
        if not isinstance(p, np.ndarray):
            try:
                p = np.array(p)
            except Exception:
                # non-convertible: debug log and return None
                logger.debug("Non-array target in find_closest_index_from_coords; returning None")
                return None
        # Ensure shape is (2,); if not, silently return None without warning noise
        if p.shape != (2,):
            logger.debug(f"Invalid point shape {p.shape} in find_closest_index_from_coords; returning None")
            return None
        try:
            return np.argmin(np.linalg.norm(node_coords - p, axis=1))
        except ValueError:
            logger.exception("ValueError in find_closest_index")
            return None

    def find_k_neighbor_all_nodes(
        self,
        robot_map: np.ndarray,
        update_dense: bool = True,
        global_graph: Optional[Graph] = None,
        global_graph_knn_dist_max: float = GLOBAL_GRAPH_KNN_RAD,
        global_graph_knn_dist_min: float = CUR_AGENT_KNN_RAD,
        target_indices: Optional[List[int]] = None,
    ) -> None:
        """為每個 node 找 k 個鄰居並建立 graph.edges（包含 global graph 的補充）。
        
        Args:
            target_indices: 若提供，僅更新這些索引對應的節點 (Incremental Update)。
        
        Optimized version: Vectorized KDTree query.
        """
        if self.node_coords is None or len(self.node_coords) == 0:
            return
        try:
            kd_tree = KDTree(self.node_coords)
        except (ValueError, IndexError) as e:
            logger.error(f"KDTree Error: {e}")
            return

        self.x = []
        self.y = []

        # 1. Global Graph Edges Processing (Keep existing logic but maybe clean up if possible, for now keep as is or slight optimize)
        # Since global graph logic is per-node specific and depends on external graph, it's hard to fully vectorize without major refactor.
        # We will keep the loop for global graph part but optimize the local KNN part.

        # Pre-calculate global neighbors to avoid mixing logic too much
        # Actually, let's do the vectorized KNN first, then add global edges.

        num_nodes = len(self.node_coords)
        k_query = min(self.k_size + 1, num_nodes)  # +1 because it finds itself

        # Determine query points and loop range based on target_indices
        if target_indices is not None:
            query_points = self.node_coords[target_indices]
            loop_indices = target_indices
        else:
            query_points = self.node_coords
            loop_indices = range(num_nodes)

        if k_query > 1:
            try:
                # Vectorized query: (N_query, k)
                dists_all, indices_all = kd_tree.query(query_points, k=k_query)
            except (ValueError, IndexError) as e:
                logger.warning(f"KDTree query failed k={k_query}: {e}")
                return
        else:
            return

        # Iterate over nodes to process edges
        # 注意：enumerate(loop_indices) 的 i 是 0..M，idx 是真實索引
        for i, idx in enumerate(loop_indices):
            p = self.node_coords[idx]
            p_tuple = tuple(p)

            # --- Global Graph Logic (Preserved) ---
            num_global_neighbours = 0
            topk_global_graph_nodes = set()
            if (
                global_graph is not None
                and hasattr(global_graph, "edges")
                and p_tuple in global_graph.edges
            ):
                try:
                    global_graph_edges = global_graph.edges[p_tuple].values()
                    # Optimization: Generator to list is slow if large, but usually small degree
                    global_graph_nodes_arr = np.array(
                        [edge.to_node for edge in global_graph_edges]
                    )
                    global_graph_dist_arr = np.array(
                        [edge.length for edge in global_graph_edges]
                    )

                    if len(global_graph_nodes_arr) > 0:
                        filtered_idx = (
                            global_graph_dist_arr <= global_graph_knn_dist_max
                        ) & (global_graph_dist_arr > global_graph_knn_dist_min)
                        filtered_nodes = global_graph_nodes_arr[filtered_idx]
                        filtered_dist = global_graph_dist_arr[filtered_idx]

                        num_available = len(filtered_nodes)
                        num_global_neighbours = min(num_available, self.k_size)

                        if num_global_neighbours > 0:
                            # partial sort is faster than full sort for top k
                            if num_global_neighbours < len(filtered_dist):
                                topk_indices = np.argpartition(
                                    filtered_dist, num_global_neighbours
                                )[:num_global_neighbours]
                            else:
                                topk_indices = np.arange(len(filtered_dist))

                            topk_global_graph_nodes = set(
                                map(tuple, filtered_nodes[topk_indices])
                            )
                except Exception as e:
                    logger.warning(
                        f"Error processing global graph edges for {p_tuple}: {e}"
                    )

            for neighbour_node in topk_global_graph_nodes:
                self.graph.add_node(p_tuple)
                neighbour_tuple = tuple(neighbour_node)
                # Use pre-calculated dist if possible, but here we recalculate to be safe
                dist = np.linalg.norm(p - np.array(neighbour_tuple))
                self.graph.add_edge(p_tuple, neighbour_tuple, dist)

            # --- Local KNN Logic (Using Vectorized Results) ---
            # We need to filter out 'self' (distance ~ 0) and check collisions

            # indices_all[i] contains k indices. One of them is i itself.
            # We want (k_size - num_global_neighbours) from local
            target_k_local = self.k_size - num_global_neighbours
            if target_k_local <= 0:
                continue

            current_indices = indices_all[i]

            count_added = 0
            for idx in current_indices:
                if count_added >= target_k_local:
                    break

                if idx == i or idx < 0 or idx >= num_nodes:
                    continue

                neighbour = self.node_coords[idx]

                # Collision check
                if not check_collision(p, neighbour, robot_map):
                    neighbour_tuple = tuple(neighbour)

                    if update_dense:
                        dist = np.linalg.norm(p - neighbour)
                        if p_tuple not in self.graph.nodes:
                            self.graph.add_node(p_tuple)
                        if neighbour_tuple not in self.graph.nodes:
                            self.graph.add_node(neighbour_tuple)
                        self.graph.add_edge(p_tuple, neighbour_tuple, dist)
                        self.graph.add_edge(neighbour_tuple, p_tuple, dist)

                    if self.plot:
                        self.x.append([p[0], neighbour[0]])
                        self.y.append([p[1], neighbour[1]])

                    count_added += 1
