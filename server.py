import numpy as np
from skimage.measure import block_reduce
from scipy.optimize import linear_sum_assignment
import sys
import itertools
import logging # <--- 匯入 logging

from parameter import *
from graph_generator import Graph_generator
from graph import Graph, a_star

# <--- 獲取 logger ---
logger = logging.getLogger(__name__)

class Server():
    # <--- 移除 debug 參數 ---
    def __init__(self, start_position, real_map_size, resolution, k_size, plot=False):
        self.position = start_position
        # ... (其他屬性初始化不變) ...
        self.global_map = np.ones(real_map_size) * 127
        self.downsampled_map = None
        self.comm_range = SERVER_COMM_RANGE
        self.all_robot_position = []
        self.robot_in_range = []
        self.graph_generator:Graph_generator = Graph_generator(map_size=real_map_size, sensor_range=SENSOR_RANGE, k_size=k_size, plot=plot)
        self.graph_generator.route_node.append(start_position)
        self.frontiers = []
        self.node_coords = []
        self.local_map_graph = []
        self.node_utility = []
        self.guidepost = []
        self.resolution = resolution
        # self.debug = debug # 移除


    def update_and_assign_tasks(self, robot_list, real_map, find_frontier_func):
        """
        執行一步的伺服器決策 (使用 logger)。
        """
        logger.debug(f"[Server Step] Updating graph and assigning tasks...")

        # --- 1. 更新全局地圖與圖結構 ---
        # ... (更新邏輯不變) ...
        self.downsampled_map = block_reduce(self.global_map.copy(), block_size=(self.resolution, self.resolution), func=np.min)
        new_frontiers = find_frontier_func(self.downsampled_map)
        logger.debug(f"[Server Step] Updating full graph...")
        try: # 增加錯誤處理
            node_coords, graph, node_utility, guidepost = self.graph_generator.update_graph(
                self.global_map, new_frontiers, self.frontiers, self.position, self.all_robot_position
            )
            self.node_coords = node_coords
            self.local_map_graph = graph
            self.node_utility = node_utility
            self.guidepost = guidepost
        except Exception as e:
            logger.error(f"[Server Step] Failed update_graph: {e}")
            # 如何處理？可能需要返回錯誤或保持舊狀態
            done = False; coverage = self.calculate_coverage_ratio(real_map) # 計算當前覆蓋率
            return done, coverage # 提前返回

        self.frontiers = new_frontiers

        # --- 2. 篩選需要任務的機器人 ---
        # ... (篩選邏輯不變) ...
        robots_need_assignment = []
        robot_positions = []
        for i, in_range in enumerate(self.robot_in_range):
            if in_range and i < len(self.all_robot_position) and self.all_robot_position[i] is not None:
                if i < len(robot_list):
                    robot = robot_list[i]
                    if robot.needs_new_target():
                        robots_need_assignment.append(i)
                        robot_positions.append(self.all_robot_position[i])
        logger.debug(f"[Server Step] Robots needing assignment: {robots_need_assignment}")


        # --- 檢查探索是否完成 ---
        # ... (檢查邏輯不變) ...
        total_frontiers = len(self.frontiers) if self.frontiers is not None else 0
        total_free = np.sum(real_map == 255)
        explored = np.sum(self.global_map == 255)
        coverage = explored / total_free if total_free > 0 else 0.0
        done = (total_frontiers == 0) or (coverage >= 0.95)


        if not robots_need_assignment:
            logger.debug("[Server Step] No robots need assignment.")
            return done, coverage

        # --- 3. 準備候選目標 ---
        # ... (準備目標邏輯不變，加入 logger) ...
        num_raw_candidates = 0; current_candidates = None; current_utilities = None
        if hasattr(self.graph_generator, 'target_candidates') and self.graph_generator.target_candidates is not None:
             current_candidates = self.graph_generator.target_candidates
             num_raw_candidates = len(current_candidates)
        logger.debug(f"[Server Step] Raw target candidates: {num_raw_candidates}")

        if num_raw_candidates < 1:
            logger.debug("[Server Step] No target candidates available.")
            return done, coverage

        candidates = np.array(current_candidates)
        utilities = np.array([])
        if hasattr(self.graph_generator, 'candidates_utility') and self.graph_generator.candidates_utility is not None and len(self.graph_generator.candidates_utility) == num_raw_candidates:
            utilities = np.array(self.graph_generator.candidates_utility)
        else:
             logger.warning(f"[Server Step] Mismatch or missing candidates_utility! Rebuilding...")
             try: # 增加錯誤處理
                 self.graph_generator._update_nodes_and_utilities(new_frontiers, self.global_map, self.frontiers, self.all_robot_position)
                 candidates = np.array(self.graph_generator.target_candidates)
                 if len(candidates) == 0: return done, coverage
                 utilities = np.array(self.graph_generator.candidates_utility)
                 if len(utilities) != len(candidates):
                      logger.error("[Server Step] Failed to rebuild consistent utilities!")
                      return done, coverage
             except Exception as e:
                  logger.error(f"[Server Step] Error during utility rebuild: {e}")
                  return done, coverage

        available_candidates, available_utilities = self._filter_occupied_targets(
            candidates, utilities, robot_list, robots_need_assignment
        )
        num_available_candidates = len(available_candidates)
        logger.debug(f"[Server Step] Available candidates after filtering: {num_available_candidates}")

        if num_available_candidates == 0:
            logger.debug("[Server Step] No available targets after filtering.")
            return done, coverage


        # --- 4. 準備匈牙利演算法 ---
        # ... (準備邏輯不變) ...
        m = len(robots_need_assignment); k = len(available_candidates)
        if k < m:
            sorted_indices = np.argsort(-available_utilities)
            extended_candidates = list(available_candidates)
            extended_utilities = list(available_utilities)
            while len(extended_candidates) < m:
                idx_to_add = sorted_indices[(len(extended_candidates) - k) % len(sorted_indices)]
                extended_candidates.append(available_candidates[idx_to_add])
                extended_utilities.append(available_utilities[idx_to_add])
            available_candidates = np.array(extended_candidates)
            available_utilities = np.array(extended_utilities)
            k = m


        # --- 5. 建立成本矩陣 ---
        # ... (建立矩陣邏輯不變) ...
        cost_matrix = np.zeros((m, k)); lambda_dist = 1.1
        for i, robot_pos in enumerate(robot_positions):
            for j, candidate in enumerate(available_candidates):
                distance = np.linalg.norm(np.array(robot_pos) - np.array(candidate))
                utility = available_utilities[j]
                cost_matrix[i, j] = -utility + lambda_dist * distance


        # --- 6. 執行匈牙利演算法 ---
        # ... (執行邏輯不變) ...
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        except Exception as e:
            logger.error(f"[Server Step] Hungarian algorithm failed: {e}")
            return done, coverage


        # --- 7. 指派後衝突預防 (使用 logger) ---
        potential_assignments = []; assignment_map = {}
        for i, j in zip(row_indices, col_indices):
            if i < m and j < k:
                robot_idx = robots_need_assignment[i]
                target_pos = available_candidates[j]
                cost = cost_matrix[i, j]
                potential_assignments.append((robot_idx, target_pos, cost))
                assignment_map[robot_idx] = (target_pos, cost)
            else:
                 logger.warning(f"[Server Step] Hungarian index OOB (i={i}, j={j})")

        rejected_indices = set(); MIN_TARGET_DISTANCE = ROBOT_COMM_RANGE / 2
        for idx1_tuple, idx2_tuple in itertools.combinations(potential_assignments, 2):
            robot_idx1, target1, cost1 = idx1_tuple
            robot_idx2, target2, cost2 = idx2_tuple
            if robot_idx1 in rejected_indices or robot_idx2 in rejected_indices: continue
            distance = np.linalg.norm(np.array(target1) - np.array(target2))
            if distance < MIN_TARGET_DISTANCE:
                reject_idx = robot_idx1 if cost1 >= cost2 else robot_idx2
                rejected_indices.add(reject_idx)
                logger.debug(f"[Server Step] Assign Conflict: R{robot_idx1}->{target1}({cost1:.1f}) vs R{robot_idx2}->{target2}({cost2:.1f}). Dist={distance:.1f}. Reject R{reject_idx}.")


        # --- 8. 執行有效指派 (使用 logger) ---
        assigned_count = 0
        for robot_idx, target_pos, cost in potential_assignments:
            if robot_idx in rejected_indices: continue

            if robot_idx < len(robot_list):
                robot = robot_list[robot_idx]
                robot.target_pos = np.array(target_pos)
                logger.debug(f"[Server Step] Assign R{robot_idx} -> Target {target_pos} (Cost:{cost:.1f}). Planning global path...")
                try:
                    robot.planned_path = self._plan_global_path(robot.position, robot.target_pos)
                    robot.target_gived_by_server = True
                    path_len = len(robot.planned_path) if robot.planned_path else 0
                    if not robot.planned_path or path_len == 0 or (path_len == 1 and np.array_equal(robot.planned_path[0], robot.position)):
                        logger.warning(f"[Server Step] Path plan FAILED for R{robot_idx} to {target_pos}")
                        robot.target_pos = None; robot.target_gived_by_server = False
                    else:
                        logger.debug(f"[Server Step] Path plan SUCCESS for R{robot_idx}, Len: {path_len}")
                        assigned_count += 1
                except Exception as e:
                    logger.error(f"[Server Step] Path plan EXCEPTION for R{robot_idx}: {e}")
                    robot.target_pos = None; robot.target_gived_by_server = False
            else:
                 logger.error(f"[Server Step] Invalid robot_idx {robot_idx} during assignment.")

        num_requested = len(robots_need_assignment)
        logger.debug(f"[Server Step] Assigned {assigned_count}/{num_requested} robots ({len(rejected_indices)} rejected).")
        return done, coverage

    # 輔助函式加入 logger (可選)
    def _filter_occupied_targets(self, candidates, utilities, robot_list, requesting_robots):
        # ... (邏輯不變，可以在內部加入 logger.debug 輸出過濾原因) ...
        available_mask = np.ones(len(candidates), dtype=bool)
        initial_count = len(candidates)
        for i, robot in enumerate(robot_list):
            if i in requesting_robots: continue
            target_pos = getattr(robot, 'target_pos', None)
            robot_pos = getattr(robot, 'position', None)
            planned_path = getattr(robot, 'planned_path', [])
            if target_pos is not None:
                distances = np.linalg.norm(candidates - target_pos, axis=1)
                mask_target = (distances > 20)
                # logger.debug(f"Filtering vs R{i} target: {np.sum(~mask_target)} removed")
                available_mask &= mask_target
            if robot_pos is not None:
                distances_to_robot = np.linalg.norm(candidates - robot_pos, axis=1)
                mask_pos = (distances_to_robot > 15)
                # logger.debug(f"Filtering vs R{i} pos: {np.sum(~mask_pos)} removed")
                available_mask &= mask_pos
            if planned_path and len(planned_path) > 0:
                for idx, planned_pos in enumerate(planned_path[:3]):
                    if planned_pos is not None:
                        planned_pos = np.array(planned_pos)
                        if planned_pos.shape == (2,):
                             distances_to_planned = np.linalg.norm(candidates - planned_pos, axis=1)
                             mask_plan = (distances_to_planned > 10)
                             # logger.debug(f"Filtering vs R{i} plan[{idx}]: {np.sum(~mask_plan)} removed")
                             available_mask &= mask_plan
        final_count = np.sum(available_mask)
        logger.debug(f"[Server Filter] Filtered targets: {initial_count} -> {final_count}")
        return candidates[available_mask], utilities[available_mask]


    def _plan_global_path(self, current_pos, target_pos):
        # ... (邏輯不變，內部 find_shortest_path 可能有 debug log) ...
        gen = self.graph_generator; current = current_pos; target = target_pos
        coords = gen.node_coords; graph = gen.graph
        if coords is None or len(coords) == 0 or graph is None:
            logger.warning(f"[Server PlanGlobal] Invalid graph/nodes. Target:{target}")
            return [current]
        dist, route = gen.find_shortest_path(current, target, coords, graph) # find_shortest_path 內部有 logger
        # path_len = len(route) if route is not None else 0
        # logger.debug(f"[Server PlanGlobal] Target:{target}. A* Result:{'Success' if route is not None else 'Failed!'}, Dist:{dist:.1f}, PathLen:{path_len}")
        return route if route is not None else [current]

    # 加入 calculate_coverage_ratio (之前可能在 env.py)
    def calculate_coverage_ratio(self, real_map):
        """計算地圖探索覆蓋率 (基於 server 的全局地圖)"""
        explored_pixels = np.sum(self.global_map == 255)
        total_free_pixels = np.sum(real_map == 255)
        return min(explored_pixels / total_free_pixels, 1.0) if total_free_pixels > 0 else 0.0