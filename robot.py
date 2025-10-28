import numpy as np
from collections import deque
from parameter import *
from graph_generator import Graph_generator
from sensor import sensor_work
from skimage.measure import block_reduce
import sys

class Robot():
    def __init__(self, start_position, real_map_size, resolution, k_size, plot=False, debug=False):
        self.position = start_position
        self.local_map = np.ones(real_map_size) * 127
        self.downsampled_map = None
        # <--- 修正點：在這裡加入 sensor_range 初始化 ---
        self.sensor_range = SENSOR_RANGE
        # --- ---
        self.frontiers = []
        self.node_coords = []
        self.local_map_graph = []
        self.node_utility = []
        self.guidepost = []
        self.planned_path = []
        self.target_pos = None
        self.movement_history = [start_position.copy()]
        # <--- 修改點：傳遞正確的 sensor_range 給 Graph_generator ---
        self.graph_generator:Graph_generator = Graph_generator(map_size=real_map_size, sensor_range=self.sensor_range, k_size=k_size, plot=plot)
        # --- ---
        self.graph_generator.route_node.append(start_position)

        self.target_gived_by_server = False
        self.last_position_in_server_range = start_position
        self.out_range_step = 0
        self.stay_count = 0

        self.resolution = resolution
        self.is_in_server_range = True

        self.info_gain_history = deque(maxlen=INFO_GAIN_HISTORY_LEN)
        self.current_info_gain = 0
        self.last_explored_area = 0

        self.graph_update_counter = 0
        self.robot_id = -1
        self.debug = debug

    # ... (其餘程式碼不變) ...
    def update_local_awareness(self, real_map, all_robots, server, find_frontier_func, merge_maps_func):
        """
        執行一步的感知和地圖更新。
        """
        
        # --- 1. 感測與地圖合併 (Sensing & Merging) ---
        self.local_map = sensor_work(self.position, self.sensor_range, self.local_map, real_map)
        dist_to_server = np.linalg.norm(self.position - server.position)
        self.is_in_server_range = dist_to_server < SERVER_COMM_RANGE

        for other_robot in all_robots:
            if other_robot is self: continue
            dist = np.linalg.norm(self.position - other_robot.position)
            if dist < ROBOT_COMM_RANGE:
                merged = merge_maps_func([self.local_map, other_robot.local_map])
                self.local_map[:] = merged
                other_robot.local_map[:] = merged
                if not self.is_in_server_range: self.planned_path = []

        if self.is_in_server_range:
            merged = merge_maps_func([self.local_map, server.global_map])
            self.local_map[:] = merged
            server.global_map[:] = merged
            # 安全檢查 ID 是否有效
            if 0 <= self.robot_id < len(server.all_robot_position):
                server.all_robot_position[self.robot_id] = self.position
                server.robot_in_range[self.robot_id] = True
            self.last_position_in_server_range = self.position
            self.out_range_step = 0
            if not self.target_gived_by_server: self.planned_path = []
        else:
            if not self.target_gived_by_server: self.out_range_step += 1
            # 安全檢查 ID 是否有效
            if 0 <= self.robot_id < len(server.robot_in_range):
                server.robot_in_range[self.robot_id] = False
        
        # --- 2. 計算地圖增益 ---
        current_explored_area = np.sum(self.local_map == 255)
        self.current_info_gain = current_explored_area - self.last_explored_area
        self.last_explored_area = current_explored_area
        if not self.is_in_server_range and not self.target_gived_by_server:
             self.info_gain_history.append(self.current_info_gain)
        else:
             self.info_gain_history.clear()

        # --- 3. 更新 Frontiers ---
        self.downsampled_map = block_reduce(self.local_map.copy(), block_size=(self.resolution, self.resolution), func=np.min)
        new_frontiers = find_frontier_func(self.downsampled_map)

        # --- 4. 更新節點圖 ---
        self.graph_update_counter += 1
        if self.graph_update_counter % GRAPH_UPDATE_INTERVAL == 0:
            node_coords, graph, node_utility, guidepost = self.graph_generator.rebuild_graph_structure(
                self.local_map, new_frontiers, self.frontiers, self.position
            )
            self.node_coords = node_coords
            self.local_map_graph = graph
            self.node_utility = node_utility
            self.guidepost = guidepost
        else:
            node_utility, guidepost = self.graph_generator.update_node_utilities(
                self.local_map, new_frontiers, self.frontiers
            )
            self.node_utility = node_utility
            self.guidepost = guidepost
        
        self.frontiers = new_frontiers

    def needs_new_target(self):
         return len(self.planned_path) < 1

    def decide_next_target(self, all_robots):
        """
        決策系統：決定下一個目標點 (加入 debug 判斷)。
        """
        if self.debug:
            print(f"\n[R{self.robot_id} Decide] Pos: {self.position}, InRange: {self.is_in_server_range}, OutSteps: {self.out_range_step}", flush=True)

        target_pos_local, _, min_valid_dists = self._select_node(all_robots)
        max_local_utility = 0
        if self.graph_generator.candidates_utility is not None and len(self.graph_generator.candidates_utility) > 0:
             max_local_utility = np.max(self.graph_generator.candidates_utility)

        recent_info_gain = np.sum(self.info_gain_history)
        is_stagnated = (len(self.info_gain_history) == INFO_GAIN_HISTORY_LEN) and (recent_info_gain < MIN_INFO_GAIN_THRESHOLD)
        is_depleted = max_local_utility < LOCAL_UTILITY_THRESHOLD
        is_timeout = self.out_range_step > OUT_RANGE_STEP

        if self.debug:
            print(f"[R{self.robot_id} Decide] Stagnated: {is_stagnated} (Gain={recent_info_gain}/{MIN_INFO_GAIN_THRESHOLD}), Depleted: {is_depleted} (MaxUtil={max_local_utility}/{LOCAL_UTILITY_THRESHOLD}), Timeout: {is_timeout} (Steps={self.out_range_step}/{OUT_RANGE_STEP})", flush=True)

        decision_reason = "Local Explore"
        if (is_stagnated and is_depleted) or is_timeout:
            target_pos = self.last_position_in_server_range
            decision_reason = f"Return (Stag:{is_stagnated}, Dep:{is_depleted}, Time:{is_timeout})"
        else:
            target_pos = target_pos_local
            if min_valid_dists > (self.sensor_range * 1.5) and self.out_range_step > 5:
                target_pos = self.last_position_in_server_range
                decision_reason = "Return (Local Target Too Far)"

        self.target_pos = target_pos
        self.target_gived_by_server = False

        if self.debug:
            print(f"[R{self.robot_id} Decide] Final Target: {self.target_pos}, Reason: {decision_reason}", flush=True)

        self.planned_path = self._plan_local_path(target_pos)
        path_len = len(self.planned_path) if self.planned_path else 0
        if self.debug:
            print(f"[R{self.robot_id} Decide] Plan Local Path: {'Success' if path_len > 0 else 'Failed!'}, Length: {path_len}", flush=True)

        if not self.planned_path:
             self.planned_path = [self.position]


    def move_one_step(self, all_robots):
        """
        執行一步移動 (加入 debug 判斷)。
        """
        if self.debug:
            path_str = f"[{self.planned_path[0]}]..." if self.planned_path else "[]"
            print(f"\n[R{self.robot_id} Move] Pos: {self.position}, Target: {self.target_pos}, Path(len={len(self.planned_path)}): {path_str}", flush=True)

        if len(self.planned_path) < 1:
            if self.debug: print(f"[R{self.robot_id} Move] No path to move.", flush=True)
            return

        next_step = self.planned_path[0]

        if np.array_equal(next_step, self.position):
            if len(self.planned_path) > 1:
                self.planned_path.pop(0)
                next_step = self.planned_path[0]
                if self.debug: print(f"[R{self.robot_id} Move] Path started with current pos, popped. Next: {next_step}", flush=True)
            else:
                if self.debug: print(f"[R{self.robot_id} Move] Reached target or stuck at current pos {self.position}", flush=True)
                self.planned_path = []
                return

        is_blocked = False
        blocker_id = -1
        for other_robot in all_robots:
            if other_robot is self: continue
            if other_robot.position is not None:
                dist_to_other = np.linalg.norm(next_step - other_robot.position)
                if dist_to_other < 1:
                    is_blocked = True
                    blocker_id = other_robot.robot_id
                    break

        if is_blocked:
            if self.debug: print(f"[R{self.robot_id} Move] Blocked by R{blocker_id} at {all_robots[blocker_id].position}. My next step: {next_step}. Stay count: {self.stay_count}", flush=True)
            self.stay_count += 1
            if self.stay_count > 2 and self.robot_id > blocker_id:
                if self.debug: print(f"[R{self.robot_id} Move] Yielding to R{blocker_id}, replanning...", flush=True)
                self.planned_path = self._plan_local_path(self.target_pos, avoid_pos=next_step)
                path_len = len(self.planned_path) if self.planned_path else 0
                if self.debug: print(f"[R{self.robot_id} Move] Replanned path: {'Success' if path_len > 0 else 'Failed!'}, Length: {path_len}", flush=True)
                if not self.planned_path:
                    self.planned_path = [self.position]
        else:
            if self.debug: print(f"[R{self.robot_id} Move] Moving to {next_step}", flush=True)
            self.stay_count = 0
            self.planned_path.pop(0)
            self.position = np.array(next_step)
            self.movement_history.append(self.position.copy())

        dist_to_server = np.linalg.norm(self.position - self.last_position_in_server_range)
        if dist_to_server < SERVER_COMM_RANGE: self.is_in_server_range = True
        else: self.is_in_server_range = False


    def _select_node(self, all_robots):
        """
        根據演算法選擇下一個圖節點作為移動目標 (加入 debug 判斷)。
        """
        candidates = self.graph_generator.target_candidates
        num_candidates = len(candidates) if candidates is not None else 0

        if candidates is None or len(candidates) == 0:
            if self.debug: print(f"[R{self.robot_id} SelectNode] No candidates available, returning current pos.", flush=True)
            return self.position, 0, 0

        utilities = self.graph_generator.candidates_utility
        dists = np.linalg.norm(candidates - self.position, axis=1)
        valid_mask = utilities > 0

        for robot in all_robots:
             if robot.position is not None:
                same_pos = np.all(candidates == robot.position, axis=1)
                valid_mask &= ~same_pos

        num_valid_after_pos = np.sum(valid_mask)

        if not np.any(valid_mask):
            if self.debug: print(f"[R{self.robot_id} SelectNode] No valid nodes after filtering, returning current pos.", flush=True)
            return self.position, 0, 0

        valid_candidates = candidates[valid_mask]
        valid_utilities = utilities[valid_mask]
        valid_dists = dists[valid_mask]

        if len(valid_dists) == 0:
            if self.debug: print(f"[R{self.robot_id} SelectNode] Valid dists empty, returning current pos.", flush=True)
            return self.position, 0, 0

        λ = 1.0
        epsilon = 1e-6
        min_valid_dists = np.min(valid_dists)
        scores = λ * valid_utilities / (valid_dists + epsilon)
        best_idx_in_valid = np.argmax(scores)
        selected_coord = valid_candidates[best_idx_in_valid]
        original_indices = np.where(valid_mask)[0]
        original_idx = original_indices[best_idx_in_valid]

        if self.debug:
            print(f"[R{self.robot_id} SelectNode] Candidates: {num_candidates}, Valid after filter: {num_valid_after_pos}", flush=True)
            print(f"[R{self.robot_id} SelectNode] Selected: {selected_coord}, Score: {scores[best_idx_in_valid]:.2f}, Utility: {valid_utilities[best_idx_in_valid]}, Dist: {valid_dists[best_idx_in_valid]:.1f}", flush=True)

        return selected_coord, original_idx, min_valid_dists

    def _plan_local_path(self, target, avoid_pos=None):
        """
        回傳從當前機器人位置到 target 的完整節點路徑 (加入 debug 判斷)。
        """
        gen = self.graph_generator
        current = self.position
        if avoid_pos is not None: pass

        coords = gen.node_coords
        graph = gen.graph

        if coords is None or len(coords) == 0 or graph is None or not hasattr(graph, 'edges'):
             if self.debug: print(f"[R{self.robot_id} PlanLocal] Error: Invalid graph or node_coords.", flush=True)
             return [current]

        dist, route = gen.find_shortest_path(current, target, coords, graph)
        path_len = len(route) if route is not None else 0

        if self.debug: print(f"[R{self.robot_id} PlanLocal] A* Result: {'Success' if route is not None else 'Failed!'}, Dist: {dist:.1f}, PathLen: {path_len}", flush=True)

        return route if route is not None else [current]