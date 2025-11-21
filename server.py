import copy
import itertools
import logging
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import block_reduce

from graph import Edge, Graph, a_star
from graph_generator import Graph_generator
from parameter import *

logger = logging.getLogger(__name__)


class Server:
    def __init__(
        self,
        start_position: np.ndarray,
        real_map_size: Tuple[int, int],
        resolution: int,
        k_size: int,
        plot: bool = False,
        force_sync_debug: bool = False,
        graph_update_interval: Optional[int] = None,
    ) -> None:
        """初始化 Server，管理全域地圖與任務分派狀態。

        Args:
            start_position (np.ndarray): 伺服器起始位置。
            real_map_size (Tuple[int, int]): 真實地圖大小 (height, width)。
            resolution (int): 下採樣比例。
            k_size (int): Graph generator 使用的 k。
            plot (bool): 是否啟用繪圖相關功能。
            force_sync_debug (bool): 是否強制同步除錯。
            graph_update_interval (Optional[int]): Graph 更新間隔。

        Returns:
            None
        """
        self.position = start_position
        from parameter import PIXEL_UNKNOWN

        self.global_map = np.ones(real_map_size) * PIXEL_UNKNOWN
        self.downsampled_map = None
        self.comm_range = SERVER_COMM_RANGE
        self.all_robot_position: List[Optional[np.ndarray]] = []
        self.robot_in_range: List[bool] = []
        self.graph_generator: Graph_generator = Graph_generator(
            map_size=real_map_size, sensor_range=SENSOR_RANGE, k_size=k_size, plot=plot
        )
        self.graph_generator.route_node.append(start_position)
        self.frontiers: List[np.ndarray] = []
        self.node_coords: Optional[np.ndarray] = None
        self.local_map_graph: Optional[
            Dict[Tuple[int, int], Dict[Tuple[int, int], Edge]]
        ] = None  # graph.edges dict
        self.node_utility: Optional[np.ndarray] = None
        self.guidepost: Optional[Any] = None
        self.resolution = resolution
        self.graph_update_counter = 0  # Re-added counter
        # graph update interval: if not provided, fallback to parameter default
        from parameter import GRAPH_UPDATE_INTERVAL as DEFAULT_GRAPH_UPDATE_INTERVAL

        self.graph_update_interval = (
            graph_update_interval
            if graph_update_interval is not None
            else DEFAULT_GRAPH_UPDATE_INTERVAL
        )
        self.force_sync_debug = force_sync_debug
        self.last_rebuild_time: Optional[float] = None
        self.last_rebuild_step: Optional[int] = None

    def update_and_assign_tasks(
        self, robot_list: List[Any], real_map: np.ndarray, find_frontier_func: Any
    ) -> Tuple[bool, float]:
        """伺服器在一步之內更新圖、計算效用並指派任務。

        Args:
            robot_list (List[Robot]): 所有 Robot 物件清單。
            real_map (np.ndarray): 真實地圖 (y,x)。
            find_frontier_func (callable): 用於尋找 frontier 的函式，接受 downsampled map 並回傳 frontier 陣列。

        Returns:
            Tuple[bool, float]:
                done (bool): 是否完成探索。
                coverage (float): 目前覆蓋率 (0.0-1.0)。
        """
        logger.debug(f"[Server Step] Updating graph and assigning tasks...")

        # --- 1. 更新全局地圖與圖結構 ---
        self.downsampled_map = block_reduce(
            self.global_map.copy(),
            block_size=(self.resolution, self.resolution),
            func=np.min,
        )
        new_frontiers = find_frontier_func(self.downsampled_map)

        # --- 間歇性更新 ---
        self.graph_update_counter += 1
        # decide whether to do a full rebuild: if interval hit OR frontier change count large
        try:
            old_frontier_set = (
                set(map(tuple, self.frontiers)) if self.frontiers is not None else set()
            )
            new_frontier_set = (
                set(map(tuple, new_frontiers)) if new_frontiers is not None else set()
            )
            frontier_change_count = len(
                old_frontier_set.symmetric_difference(new_frontier_set)
            )
        except Exception:
            frontier_change_count = 0

        need_rebuild = False
        if self.graph_update_counter % self.graph_update_interval == 0:
            need_rebuild = True
            logger.debug(
                f"[Server Step] GRAPH_UPDATE_INTERVAL reached -> scheduling full rebuild."
            )
        elif frontier_change_count >= FRONTIER_REBUILD_THRESHOLD:
            need_rebuild = True
            logger.debug(
                f"[Server Step] Frontier change count {frontier_change_count} >= threshold {FRONTIER_REBUILD_THRESHOLD} -> scheduling full rebuild."
            )

        if need_rebuild:
            logger.debug(
                f"[Server Step] Rebuilding graph structure (expensive)... Frontier change count={frontier_change_count}"
            )
            import time as _time

            t0 = _time.time()
            try:
                node_coords, graph_edges, node_utility, guidepost = (
                    self.graph_generator.rebuild_graph_structure(
                        self.global_map,
                        new_frontiers,
                        self.frontiers,
                        self.position,
                        self.all_robot_position,
                    )
                )
                self.node_coords = node_coords
                self.local_map_graph = graph_edges  # 儲存 graph.edges
                self.node_utility = node_utility
                self.guidepost = guidepost
            except Exception as e:
                logger.error(
                    f"[Server Step] Failed rebuild_graph_structure: {e}", exc_info=True
                )
            finally:
                t1 = _time.time()
                elapsed = t1 - t0
                logger.info(f"[Server Step] Full rebuild took {elapsed:.3f}s")
                # record last rebuild info for external measurement
                try:
                    self.last_rebuild_time = elapsed
                    self.last_rebuild_step = self.graph_update_counter
                except Exception:
                    pass
        else:
            # clear last_rebuild_time to indicate no full rebuild this step
            try:
                self.last_rebuild_time = None
            except Exception:
                pass
            logger.debug(
                f"[Server Step] Updating node utilities (lightweight)... Frontier change count={frontier_change_count}"
            )
            try:
                node_utility, guidepost = self.graph_generator.update_node_utilities(
                    self.global_map,
                    new_frontiers,
                    self.frontiers,
                    self.all_robot_position,
                    caller="server",
                )
                # 只更新這兩個
                self.node_utility = node_utility
                self.guidepost = guidepost
            except Exception as e:
                logger.error(
                    f"[Server Step] Failed update_node_utilities: {e}", exc_info=True
                )

        self.frontiers = new_frontiers

        # --- 同步 server-side graph 給 robot（深拷貝，避免共享引用） ---
        try:
            if (
                hasattr(self, "node_coords")
                and self.node_coords is not None
                and hasattr(self, "local_map_graph")
                and self.local_map_graph is not None
            ):
                for i, robot in enumerate(robot_list):
                    in_range = False
                    if i < len(self.robot_in_range):
                        in_range = self.robot_in_range[i]
                    # 如果啟用了 debug 強制同步，則無條件同步給所有 robots（用於觸發 mismatch diagnostics）
                    do_sync = self.force_sync_debug or in_range
                    if do_sync:
                        try:
                            # 同步機器人可用的 graph 資訊
                            robot.node_coords = copy.deepcopy(self.node_coords)
                            robot.local_map_graph = copy.deepcopy(self.local_map_graph)
                            robot.node_utility = copy.deepcopy(self.node_utility)
                            robot.guidepost = copy.deepcopy(self.guidepost)
                            # 同步 graph_generator 內部結構，確保 robots 的輕量更新可以作用於完整 nodes_list
                            try:
                                robot.graph_generator.node_coords = copy.deepcopy(
                                    self.graph_generator.node_coords
                                )
                                robot.graph_generator.nodes_list = copy.deepcopy(
                                    self.graph_generator.nodes_list
                                )
                                robot.graph_generator.graph = copy.deepcopy(
                                    self.graph_generator.graph
                                )
                                robot.graph_generator.target_candidates = copy.deepcopy(
                                    self.graph_generator.target_candidates
                                )
                                robot.graph_generator.candidates_utility = (
                                    copy.deepcopy(
                                        self.graph_generator.candidates_utility
                                    )
                                )
                            except Exception:
                                logger.exception(
                                    f"[Server Step] Partial graph_generator sync failed for robot {i}"
                                )
                        except Exception:
                            logger.exception(
                                f"[Server Step] Failed to sync graph to robot {i}"
                            )
        except Exception:
            logger.exception("[Server Step] Failed to sync graphs to robots")

        # --- 2. 篩選需要任務的機器人 ---
        # ... (邏輯不變) ...
        robots_need_assignment = []
        robot_positions = []
        if hasattr(self, "robot_in_range") and self.robot_in_range is not None:
            for i, in_range in enumerate(self.robot_in_range):
                if (
                    in_range
                    and i < len(self.all_robot_position)
                    and self.all_robot_position[i] is not None
                ):
                    if i < len(robot_list):
                        robot = robot_list[i]
                        if robot.needs_new_target():
                            robots_need_assignment.append(i)
                            robot_positions.append(self.all_robot_position[i])
        else:
            logger.error("[Server Step] self.robot_in_range missing!")
        logger.debug(
            f"[Server Step] Robots needing assignment: {robots_need_assignment}"
        )

        # --- 檢查探索完成 ---
        # ... (邏輯不變) ...
        total_frontiers = len(self.frontiers) if self.frontiers is not None else 0
        from parameter import PIXEL_FREE

        total_free = np.sum(real_map == PIXEL_FREE)
        explored = np.sum(self.global_map == PIXEL_FREE)
        coverage = explored / total_free if total_free > 0 else 0.0
        done = (total_frontiers == 0) or (coverage >= 0.95)
        if not robots_need_assignment:
            logger.debug("[Server Step] No robots need assignment.")
            return done, coverage

        # --- 3. 準備候選目標 ---
        # ... (邏輯不變) ...
        num_raw_candidates = 0
        current_candidates = None
        current_utilities = None
        if (
            hasattr(self.graph_generator, "target_candidates")
            and self.graph_generator.target_candidates is not None
        ):
            current_candidates = self.graph_generator.target_candidates
            num_raw_candidates = len(current_candidates)
        logger.debug(f"[Server Step] Raw target candidates: {num_raw_candidates}")
        if num_raw_candidates < 1:
            logger.debug("[Server Step] No target candidates.")
            return done, coverage
        candidates = np.array(current_candidates)
        utilities = np.array([])
        if (
            hasattr(self.graph_generator, "candidates_utility")
            and self.graph_generator.candidates_utility is not None
            and len(self.graph_generator.candidates_utility) == num_raw_candidates
        ):
            utilities = np.array(self.graph_generator.candidates_utility)
        else:
            logger.warning(
                f"[Server Step] Mismatch/missing candidates_utility! Using self.node_utility fallback."
            )
            if (
                self.node_utility is not None
                and self.node_coords is not None
                and len(self.node_utility) == len(self.node_coords)
            ):
                utils_dict = {
                    tuple(c): u for c, u in zip(self.node_coords, self.node_utility)
                }
                utilities = np.array(
                    [utils_dict.get(tuple(cand), 0) for cand in candidates]
                )
                if len(utilities) != len(candidates):
                    logger.error("[Server Step] Fallback failed!")
                    return done, coverage
            else:
                logger.error("[Server Step] Cannot gen fallback!")
                return done, coverage

        available_candidates, available_utilities = self._filter_occupied_targets(
            candidates, utilities, robot_list, robots_need_assignment
        )
        num_available_candidates = len(available_candidates)
        logger.debug(
            f"[Server Step] Available candidates after filtering: {num_available_candidates}"
        )
        if num_available_candidates == 0:
            logger.debug("[Server Step] No available targets after filtering.")
            return done, coverage

        # --- 4. 準備匈牙利 ---
        # ... (邏輯不變) ...
        m = len(robots_need_assignment)
        k = len(available_candidates)
        if k < m:
            sorted_indices = np.argsort(-available_utilities)
            extended_candidates = list(available_candidates)
            extended_utilities = list(available_utilities)
            while len(extended_candidates) < m:
                idx_to_add = sorted_indices[
                    (len(extended_candidates) - k) % len(sorted_indices)
                ]
                extended_candidates.append(available_candidates[idx_to_add])
                extended_utilities.append(available_utilities[idx_to_add])
            available_candidates = np.array(extended_candidates)
            available_utilities = np.array(extended_utilities)
            k = m

        # --- 5. 成本矩陣 ---
        # ... (邏輯不變) ...
        cost_matrix = np.zeros((m, k))
        lambda_dist = 1.1
        for i, robot_pos in enumerate(robot_positions):
            for j, candidate in enumerate(available_candidates):
                distance = np.linalg.norm(np.array(robot_pos) - np.array(candidate))
                utility = available_utilities[j]
                cost_matrix[i, j] = -utility + lambda_dist * distance

        # --- 6. 執行匈牙利 ---
        # ... (邏輯不變) ...
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        except Exception as e:
            logger.error(f"[Server Step] Hungarian failed: {e}", exc_info=True)
            return done, coverage

        # --- 7. 衝突預防 ---
        # ... (邏輯不變) ...
        potential_assignments = []
        assignment_map = {}
        for i, j in zip(row_indices, col_indices):
            if i < m and j < k:
                robot_idx = robots_need_assignment[i]
                target_pos = available_candidates[j]
                cost = cost_matrix[i, j]
                potential_assignments.append((robot_idx, target_pos, cost))
                assignment_map[robot_idx] = (target_pos, cost)
            else:
                logger.warning(f"[Server Step] Hungarian index OOB (i={i}, j={j})")
        rejected_indices = set()
        MIN_TARGET_DISTANCE = ROBOT_COMM_RANGE / 2
        for idx1_tuple, idx2_tuple in itertools.combinations(potential_assignments, 2):
            robot_idx1, target1, cost1 = idx1_tuple
            robot_idx2, target2, cost2 = idx2_tuple
            if robot_idx1 in rejected_indices or robot_idx2 in rejected_indices:
                continue
            distance = np.linalg.norm(np.array(target1) - np.array(target2))
            if distance < MIN_TARGET_DISTANCE:
                reject_idx = robot_idx1 if cost1 >= cost2 else robot_idx2
                rejected_indices.add(reject_idx)
                logger.debug(
                    f"[Server Step] Assign Conflict: R{robot_idx1}->{target1}({cost1:.1f}) vs R{robot_idx2}->{target2}({cost2:.1f}). Dist={distance:.1f}. Reject R{reject_idx}."
                )

        # --- 8. 執行指派 ---
        # ... (邏輯不變) ...
        assigned_count = 0
        for robot_idx, target_pos, cost in potential_assignments:
            if robot_idx in rejected_indices:
                continue
            if robot_idx < len(robot_list):
                robot = robot_list[robot_idx]
                robot.target_pos = np.array(target_pos)
                logger.debug(
                    f"[Server Step] Assign R{robot_idx} -> Target {target_pos} (Cost:{cost:.1f}). Planning global path..."
                )
                try:
                    robot.planned_path = self._plan_global_path(
                        robot.position, robot.target_pos
                    )
                    robot.target_gived_by_server = True
                    path_len = len(robot.planned_path) if robot.planned_path else 0
                    if (
                        not robot.planned_path
                        or path_len == 0
                        or (
                            path_len == 1
                            and np.array_equal(robot.planned_path[0], robot.position)
                        )
                    ):
                        # Path planning failed — lower to debug to avoid noisy output during batch runs
                        logger.debug(
                            f"[Server Step] Path plan FAILED for R{robot_idx} to {target_pos}"
                        )
                        robot.target_pos = None
                        robot.target_gived_by_server = False
                    else:
                        logger.debug(
                            f"[Server Step] Path plan SUCCESS for R{robot_idx}, Len: {path_len}"
                        )
                        assigned_count += 1
                except Exception as e:
                    logger.error(
                        f"[Server Step] Path plan EXCEPTION for R{robot_idx}: {e}",
                        exc_info=True,
                    )
                    robot.target_pos = None
                    robot.target_gived_by_server = False
            else:
                logger.error(
                    f"[Server Step] Invalid robot_idx {robot_idx} during assignment."
                )
        num_requested = len(robots_need_assignment)
        logger.debug(
            f"[Server Step] Assigned {assigned_count}/{num_requested} robots ({len(rejected_indices)} rejected)."
        )
        return done, coverage

    # ... (_filter_occupied_targets 和 _plan_global_path 保持不變) ...
    def _filter_occupied_targets(
        self,
        candidates: np.ndarray,
        utilities: np.ndarray,
        robot_list: List[Any],
        requesting_robots: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """過濾已被其他機器人佔用或太接近的候選目標。

        Args:
            candidates (np.ndarray): 候選目標陣列 (K,2)。
            utilities (np.ndarray): 每個候選的效用陣列 (K,)。
            robot_list (List[Robot]): 所有機器人清單。
            requesting_robots (List[int]): 正在請求任務的機器人索引。

        Returns:
            Tuple[np.ndarray, np.ndarray]: (filtered_candidates (ndarray), filtered_utilities (ndarray))
        """
        available_mask = np.ones(len(candidates), dtype=bool)
        initial_count = len(candidates)
        for i, robot in enumerate(robot_list):
            if i in requesting_robots:
                continue
            target_pos = getattr(robot, "target_pos", None)
            robot_pos = getattr(robot, "position", None)
            planned_path = getattr(robot, "planned_path", [])
            if target_pos is not None:
                distances = np.linalg.norm(candidates - target_pos, axis=1)
                available_mask &= distances > 20
            if robot_pos is not None:
                distances_to_robot = np.linalg.norm(candidates - robot_pos, axis=1)
                available_mask &= distances_to_robot > 15
            if planned_path and len(planned_path) > 0:
                for idx, planned_pos in enumerate(planned_path[:3]):
                    if planned_pos is not None:
                        planned_pos = np.array(planned_pos)
                        if planned_pos.shape == (2,):
                            distances_to_planned = np.linalg.norm(
                                candidates - planned_pos, axis=1
                            )
                            available_mask &= distances_to_planned > 10
        final_count = np.sum(available_mask)
        logger.debug(
            f"[Server Filter] Filtered targets: {initial_count} -> {final_count}"
        )
        return candidates[available_mask], utilities[available_mask]

    def _plan_global_path(
        self, current_pos: np.ndarray, target_pos: np.ndarray
    ) -> List[np.ndarray]:
        """為 robot 計算全域路徑（使用最近一次的節點座標與 graph edges）。

        Args:
            current_pos (np.ndarray): 機器人當前位置。
            target_pos (np.ndarray): 目標位置。

        Returns:
            List[np.ndarray]: 路徑座標列表（至少包含 current_pos），若失敗則回傳 [current_pos]。
        """
        gen = self.graph_generator
        current = current_pos
        target = target_pos
        # <--- 使用 self.node_coords 和 self.local_map_graph ---
        coords = self.node_coords
        graph_edges = self.local_map_graph  # 這是 dict
        # --- ---
        if (
            coords is None
            or len(coords) == 0
            or graph_edges is None
            or not isinstance(graph_edges, dict)
        ):
            logger.warning(
                f"[Server PlanGlobal] Invalid graph/nodes for target {target}. Coords:{coords is not None}, Graph:{graph_edges is not None}"
            )
            return [current]

        # <--- 需要將 graph_edges (dict) 轉換為 Graph 物件 ---
        graph_obj = Graph()
        if coords is not None:
            for node_coord in coords:
                graph_obj.add_node(tuple(node_coord))  # 添加所有節點
        if isinstance(graph_edges, dict):
            for from_node_tuple, edges_dict in graph_edges.items():
                # 確保 from_node 存在
                if from_node_tuple not in graph_obj.nodes:
                    graph_obj.add_node(from_node_tuple)

                if isinstance(edges_dict, dict):
                    for to_node_tuple, edge_obj in edges_dict.items():
                        # 確保 to_node 存在
                        if to_node_tuple not in graph_obj.nodes:
                            graph_obj.add_node(to_node_tuple)

                        if isinstance(edge_obj, Edge):
                            graph_obj.add_edge(
                                from_node_tuple, to_node_tuple, edge_obj.length
                            )
                        else:
                            logger.warning(
                                f"Unexpected edge data type: {type(edge_obj)} from {from_node_tuple} to {to_node_tuple}"
                            )
        # --- ---

        dist, route = gen.find_shortest_path(
            current, target, coords, graph_obj
        )  # <--- 使用 graph_obj
        path_len = len(route) if route is not None else 0
        logger.debug(
            f"[Server PlanGlobal] Target:{target}. A* Result:{'Success' if route is not None else 'Failed!'}, Dist:{dist:.1f}, PathLen:{path_len}"
        )
        # 返回原始 A* 路徑
        return route if route is not None else [current]

    def calculate_coverage_ratio(self, real_map: np.ndarray) -> float:
        """計算目前全域地圖的覆蓋率。

        Args:
            real_map (np.ndarray): 真實地圖 (y,x)。

        Returns:
            float: 覆蓋率（0.0 到 1.0）。
        """
        from parameter import PIXEL_FREE

        explored_pixels = np.sum(self.global_map == PIXEL_FREE)
        total_free_pixels = np.sum(real_map == PIXEL_FREE)
        return (
            min(explored_pixels / total_free_pixels, 1.0)
            if total_free_pixels > 0
            else 0.0
        )
