import numpy as np
from skimage.measure import block_reduce
from scipy.optimize import linear_sum_assignment
import sys
import itertools 

from parameter import *
from graph_generator import Graph_generator
from graph import Graph, a_star

class Server():
    def __init__(self, start_position, real_map_size, resolution, k_size, plot=False, debug=False):
        self.position = start_position
        self.global_map = np.ones(real_map_size) * 127
        self.downsampled_map = None
        self.comm_range = SERVER_COMM_RANGE
        # <--- 修正點：在這裡加入列表初始化 ---
        self.all_robot_position = [] 
        self.robot_in_range = [] 
        # --- ---
        
        self.graph_generator:Graph_generator = Graph_generator(map_size=real_map_size, sensor_range=SENSOR_RANGE, k_size=k_size, plot=plot)
        self.graph_generator.route_node.append(start_position)
        
        self.frontiers = []
        self.node_coords = []
        self.local_map_graph = []
        self.node_utility = []
        self.guidepost = []
        
        self.resolution = resolution
        
        self.graph_update_counter = 0
        self.debug = debug

    # ... (其餘程式碼 update_and_assign_tasks, _filter_occupied_targets, _plan_global_path 保持不變) ...
    def update_and_assign_tasks(self, robot_list, real_map, find_frontier_func):
        """
        執行一步的伺服器決策 (加入 debug 判斷)。
        """
        if self.debug: print(f"\n[Server Step] Updating graph and assigning tasks...", flush=True)

        # --- 1. 更新全局地圖與圖結構 ---
        self.downsampled_map = block_reduce(self.global_map.copy(), block_size=(self.resolution, self.resolution), func=np.min)
        new_frontiers = find_frontier_func(self.downsampled_map)
        
        self.graph_update_counter += 1
        if self.graph_update_counter % GRAPH_UPDATE_INTERVAL == 0:
            if self.debug: print(f"[Server Step] Rebuilding graph structure (expensive)...", flush=True)
            node_coords, graph, node_utility, guidepost = self.graph_generator.rebuild_graph_structure(
                self.global_map, new_frontiers, self.frontiers, self.position, self.all_robot_position
            )
            self.node_coords = node_coords
            self.local_map_graph = graph
            self.node_utility = node_utility
            self.guidepost = guidepost
        else:
            if self.debug: print(f"[Server Step] Updating node utilities (lightweight)...", flush=True)
            node_utility, guidepost = self.graph_generator.update_node_utilities(
                self.global_map, new_frontiers, self.frontiers, self.all_robot_position
            )
            self.node_utility = node_utility
            self.guidepost = guidepost
        self.frontiers = new_frontiers 

        # --- 2. 篩選需要任務的機器人 ---
        robots_need_assignment = []
        robot_positions = []
        for i, in_range in enumerate(self.robot_in_range):
            # 安全檢查 all_robot_position 是否存在且索引有效
            if in_range and i < len(self.all_robot_position) and self.all_robot_position[i] is not None:
                robot = robot_list[i]
                if robot.needs_new_target():
                    robots_need_assignment.append(i)
                    robot_positions.append(self.all_robot_position[i])

        if self.debug: print(f"[Server Step] Robots needing assignment: {robots_need_assignment}", flush=True)

        # --- 檢查探索是否完成 ---
        total_frontiers = len(self.frontiers)
        total_free = np.sum(real_map == 255)
        explored = np.sum(self.global_map == 255)
        coverage = explored / total_free if total_free > 0 else 0.0
        done = (total_frontiers == 0) or (coverage >= 0.95)

        if not robots_need_assignment:
            if self.debug: print("[Server Step] No robots need assignment.", flush=True)
            return done, coverage

        # --- 3. 準備候選目標 ---
        num_raw_candidates = len(self.graph_generator.target_candidates) if hasattr(self.graph_generator, 'target_candidates') and self.graph_generator.target_candidates is not None else 0
        if self.debug: print(f"[Server Step] Raw target candidates: {num_raw_candidates}", flush=True)

        if not hasattr(self.graph_generator, 'target_candidates') or len(self.graph_generator.target_candidates) < 1:
            if self.debug: print("[Server Step] No target candidates available.", flush=True)
            return done, coverage

        candidates = np.array([list(coord) for coord in self.graph_generator.target_candidates])
        utilities = np.array(self.graph_generator.candidates_utility)

        available_candidates, available_utilities = self._filter_occupied_targets(
            candidates, utilities, robot_list, robots_need_assignment
        )
        num_available_candidates = len(available_candidates)
        if self.debug: print(f"[Server Step] Available candidates after filtering: {num_available_candidates}", flush=True)

        if len(available_candidates) == 0:
            if self.debug: print("[Server Step] No available targets after filtering.", flush=True)
            return done, coverage

        # --- 4. 準備匈牙利演算法 ---
        m = len(robots_need_assignment)
        k = len(available_candidates)
        if k < m:
            sorted_indices = np.argsort(-available_utilities)
            extended_candidates = list(available_candidates)
            extended_utilities = list(available_utilities)
            while len(extended_candidates) < m:
                for idx in sorted_indices:
                    if len(extended_candidates) >= m: break
                    extended_candidates.append(available_candidates[idx])
                    extended_utilities.append(available_utilities[idx])
            available_candidates = np.array(extended_candidates)
            available_utilities = np.array(extended_utilities)
            k = m

        # --- 5. 建立成本矩陣 ---
        cost_matrix = np.zeros((m, k))
        lambda_dist = 1.1
        for i, robot_pos in enumerate(robot_positions):
            for j, candidate in enumerate(available_candidates):
                distance = np.linalg.norm(np.array(robot_pos) - np.array(candidate))
                utility = available_utilities[j]
                cost_matrix[i, j] = -utility + lambda_dist * distance

        # --- 6. 執行匈牙利演算法 ---
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        except Exception as e:
            print(f"[Server Step] Hungarian algorithm failed: {e}", flush=True)
            return done, coverage

        # --- 7. 指派後衝突預防 ---
        assignments = {}
        valid_assignments = {}
        MIN_TARGET_DISTANCE = ROBOT_COMM_RANGE / 2

        for i, j in zip(row_indices, col_indices):
            if i < len(robots_need_assignment) and j < len(available_candidates):
                robot_idx = robots_need_assignment[i]
                target = available_candidates[j]
                assignments[robot_idx] = target
            else:
                if self.debug:
                    print(f"[Server Step] Warning: Hungarian index out of bounds (i={i}, j={j}) during initial assignment build", flush=True)

        assigned_robot_indices = sorted(list(assignments.keys()))
        rejected_indices = set()

        for idx1, idx2 in itertools.combinations(assigned_robot_indices, 2):
            if idx1 in rejected_indices or idx2 in rejected_indices: continue

            target1 = assignments[idx1]
            target2 = assignments[idx2]
            distance = np.linalg.norm(np.array(target1) - np.array(target2))

            if distance < MIN_TARGET_DISTANCE:
                reject_idx = max(idx1, idx2)
                rejected_indices.add(reject_idx)
                if self.debug: print(f"[Server Step] Assignment Conflict Detected: R{idx1} -> {target1}, R{idx2} -> {target2}. Distance={distance:.1f} < {MIN_TARGET_DISTANCE}. R{reject_idx} assignment rejected.", flush=True)

        for robot_idx in assigned_robot_indices:
            if robot_idx not in rejected_indices:
                valid_assignments[robot_idx] = assignments[robot_idx]

        # --- 8. 執行有效指派 ---
        assigned_count = 0
        for robot_idx, target_pos in valid_assignments.items():
            robot = robot_list[robot_idx]
            robot.target_pos = np.array(target_pos)

            if self.debug: print(f"[Server Step] Assigning R{robot_idx} -> Target {target_pos}. Planning global path...", flush=True)

            try:
                robot.planned_path = self._plan_global_path(robot.position, robot.target_pos)
                robot.target_gived_by_server = True
                path_len = len(robot.planned_path) if robot.planned_path else 0
                if not robot.planned_path or path_len == 0:
                    if self.debug: print(f"[Server Step] Path planning FAILED for R{robot_idx} to {target_pos}", flush=True)
                    robot.target_pos = None
                    robot.target_gived_by_server = False
                else:
                    if self.debug: print(f"[Server Step] Path planning SUCCESS for R{robot_idx}, Length: {path_len}", flush=True)
                    assigned_count += 1
            except Exception as e:
                print(f"[Server Step] Path planning EXCEPTION for R{robot_idx}: {e}", flush=True)
                robot.target_pos = None
                robot.target_gived_by_server = False

        num_rejected = len(rejected_indices)
        num_requested = len(robots_need_assignment)
        if self.debug: print(f"[Server Step] Assigned {assigned_count}/{num_requested} robots ({num_rejected} rejected due to conflict).", flush=True)
        return done, coverage

    def _filter_occupied_targets(self, candidates, utilities, robot_list, requesting_robots):
        """
        過濾掉已被其他機器人佔據或指派的目標
        """
        available_mask = np.ones(len(candidates), dtype=bool)
        
        for i, robot in enumerate(robot_list):
            if i in requesting_robots: continue
            
            target_pos = getattr(robot, 'target_pos', None)
            robot_pos = getattr(robot, 'position', None)
            planned_path = getattr(robot, 'planned_path', [])

            if target_pos is not None:
                distances = np.linalg.norm(candidates - target_pos, axis=1)
                available_mask &= (distances > 20)
            
            if robot_pos is not None:
                distances_to_robot = np.linalg.norm(candidates - robot_pos, axis=1)
                available_mask &= (distances_to_robot > 15)
            
            if planned_path and len(planned_path) > 0:
                for planned_pos in planned_path[:3]:
                    if planned_pos is not None:
                        planned_pos = np.array(planned_pos)
                        distances_to_planned = np.linalg.norm(candidates - planned_pos, axis=1)
                        available_mask &= (distances_to_planned > 10)
        
        return candidates[available_mask], utilities[available_mask]
        
    def _plan_global_path(self, current_pos, target_pos):
        """
        回傳從當前機器人位置到 target 的完整節點路徑 (加入 debug 判斷)。
        """
        gen = self.graph_generator
        current = current_pos
        target = target_pos
        coords = gen.node_coords
        graph = gen.graph

        if coords is None or len(coords) == 0 or graph is None:
            if self.debug: print(f"[Server PlanGlobal] Error: Invalid graph or node_coords.", flush=True)
            return [current]

        dist, route = gen.find_shortest_path(current, target, coords, graph)
        path_len = len(route) if route is not None else 0

        if self.debug: print(f"[Server PlanGlobal] A* Result: {'Success' if route is not None else 'Failed!'}, Dist: {dist:.1f}, PathLen: {path_len}", flush=True)

        return route if route is not None else [current]