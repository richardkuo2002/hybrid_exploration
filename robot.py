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
        self.sensor_range = SENSOR_RANGE
        self.frontiers = []
        self.node_coords = []
        self.local_map_graph = []
        self.node_utility = []
        self.guidepost = []
        self.planned_path = []
        self.target_pos = None
        self.movement_history = [start_position.copy()]
        self.graph_generator:Graph_generator = Graph_generator(map_size=real_map_size, sensor_range=self.sensor_range, k_size=k_size, plot=plot)
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

        # <--- 修改點：移除 graph_update_counter ---
        # self.graph_update_counter = 0
        # --- ---
        self.robot_id = -1
        self.debug = debug
        self.is_returning = False # <--- 這一行必須存在！
        self.return_replan_attempts = 0 # <--- 這一行必須存在！
        self.return_fail_cooldown = 0 # <--- 這一行必須存在！

    def update_local_awareness(self, real_map, all_robots, server, find_frontier_func, merge_maps_func):
        """
        執行一步的感知和地圖更新。
        """

        # --- 1. 感測與地圖合併 ---
        # ... (感測和合併邏輯不變) ...
        self.local_map = sensor_work(self.position, self.sensor_range, self.local_map, real_map)
        dist_to_server = np.linalg.norm(self.position - server.position)
        was_in_range = self.is_in_server_range
        self.is_in_server_range = dist_to_server < SERVER_COMM_RANGE

        if self.is_in_server_range and not was_in_range:
             self.is_returning = False
             self.return_replan_attempts = 0
             self.return_fail_cooldown = 0
             if self.debug: print(f"\nRobot {self.robot_id} re-entered server range. State reset.")

        for other_robot in all_robots:
            if other_robot is self: continue
            dist = np.linalg.norm(self.position - other_robot.position)
            if dist < ROBOT_COMM_RANGE:
                merged = merge_maps_func([self.local_map, other_robot.local_map])
                self.local_map[:] = merged; other_robot.local_map[:] = merged
                if not self.is_in_server_range: self.planned_path = []

        if self.is_in_server_range:
            merged = merge_maps_func([self.local_map, server.global_map])
            self.local_map[:] = merged; server.global_map[:] = merged
            if 0 <= self.robot_id < len(server.all_robot_position):
                server.all_robot_position[self.robot_id] = self.position
                server.robot_in_range[self.robot_id] = True
            self.last_position_in_server_range = self.position
            self.out_range_step = 0
            if not self.target_gived_by_server: self.planned_path = []
        else:
            if not self.target_gived_by_server: self.out_range_step += 1
            if 0 <= self.robot_id < len(server.robot_in_range):
                server.robot_in_range[self.robot_id] = False

        # --- 2. 計算地圖增益 ---
        # ... (計算增益邏輯不變) ...
        current_explored_area = np.sum(self.local_map == 255)
        self.current_info_gain = current_explored_area - self.last_explored_area
        self.last_explored_area = current_explored_area
        if not self.is_in_server_range and not self.target_gived_by_server and not self.is_returning:
             self.info_gain_history.append(self.current_info_gain)
        elif self.is_in_server_range:
             self.info_gain_history.clear()


        # --- 3. 更新 Frontiers ---
        # ... (更新 Frontiers 邏輯不變) ...
        self.downsampled_map = block_reduce(self.local_map.copy(), block_size=(self.resolution, self.resolution), func=np.min)
        new_frontiers = find_frontier_func(self.downsampled_map)

        # --- 4. 更新節點圖 ---
        # <--- 修改點：移除間隔邏輯，總是呼叫 update_graph ---
        # self.graph_update_counter += 1
        # if self.graph_update_counter % GRAPH_UPDATE_INTERVAL == 0:
        node_coords, graph, node_utility, guidepost = self.graph_generator.update_graph(
            self.local_map, new_frontiers, self.frontiers, self.position
        )
        self.node_coords = node_coords
        self.local_map_graph = graph
        self.node_utility = node_utility
        self.guidepost = guidepost
        # else: # 這個 else 分支整個移除
            # node_utility, guidepost = self.graph_generator.update_node_utilities(
            #     self.local_map, new_frontiers, self.frontiers
            # )
            # self.node_utility = node_utility
            # self.guidepost = guidepost
        # --- ---

        self.frontiers = new_frontiers # 更新 frontiers 移到最後

    # ... (needs_new_target, decide_next_target, move_one_step, _select_node, _plan_local_path 保持不變) ...
    def needs_new_target(self):
         return len(self.planned_path) < 1

    def decide_next_target(self, all_robots):
        MAX_RETURN_REPLAN_ATTEMPTS = 2
        RETURN_FAIL_COOLDOWN_STEPS = 10

        if self.is_returning:
            if len(self.planned_path) < 1:
                if self.return_replan_attempts < MAX_RETURN_REPLAN_ATTEMPTS:
                    if self.debug: print(f"\n[R{self.robot_id} Decide] Returning but path empty. Re-planning (Attempt {self.return_replan_attempts + 1}).")
                    self.target_pos = self.last_position_in_server_range
                    self.planned_path = self._plan_local_path(self.target_pos)
                    self.return_replan_attempts += 1
                    if not self.planned_path or (len(self.planned_path) == 1 and np.array_equal(self.planned_path[0], self.position)):
                         if self.debug: print(f"[R{self.robot_id} Decide] Replan attempt {self.return_replan_attempts} failed.")
                         self.planned_path = [self.position]
                else:
                    if self.debug: print(f"\n[R{self.robot_id} Decide] Failed return plan after {MAX_RETURN_REPLAN_ATTEMPTS} attempts. Aborting return.")
                    self.is_returning = False
                    self.return_replan_attempts = 0
                    self.return_fail_cooldown = RETURN_FAIL_COOLDOWN_STEPS
            else:
                 return

        if self.return_fail_cooldown > 0:
            if self.debug: print(f"\n[R{self.robot_id} Decide] In return fail cooldown ({self.return_fail_cooldown} steps left). Forcing local exploration.")
            self.return_fail_cooldown -= 1
            target_pos_local, _, _ = self._select_node(all_robots)
            self.target_pos = target_pos_local
            self.planned_path = self._plan_local_path(self.target_pos)
            if not self.planned_path: self.planned_path = [self.position]
            self.is_returning = False
            self.target_gived_by_server = False
            return

        target_pos_local, _, min_valid_dists = self._select_node(all_robots)
        max_local_utility = 0
        # <--- 修正: 檢查 graph_generator 是否存在 --- >
        if hasattr(self.graph_generator, 'candidates_utility') and \
           self.graph_generator.candidates_utility is not None and \
           len(self.graph_generator.candidates_utility) > 0:
             max_local_utility = np.max(self.graph_generator.candidates_utility)

        recent_info_gain = np.sum(self.info_gain_history)
        is_stagnated = (len(self.info_gain_history) == INFO_GAIN_HISTORY_LEN) and (recent_info_gain < MIN_INFO_GAIN_THRESHOLD)
        is_depleted = max_local_utility < LOCAL_UTILITY_THRESHOLD
        is_timeout = self.out_range_step > OUT_RANGE_STEP
        should_return_criteria = (is_stagnated and is_depleted) or is_timeout
        local_target_too_far = min_valid_dists > (self.sensor_range * 1.5) and self.out_range_step > 5

        if self.debug:
            print(f"[R{self.robot_id} Decide] Conditions Check: Stag={is_stagnated}, Dep={is_depleted}, Time={is_timeout}, Far={local_target_too_far}", flush=True)


        decision_reason = "Local Explore"
        if should_return_criteria or local_target_too_far:
            target_pos = self.last_position_in_server_range
            planned_return_path = self._plan_local_path(target_pos)
            if planned_return_path and not (len(planned_return_path) == 1 and np.array_equal(planned_return_path[0], self.position)):
                self.planned_path = planned_return_path
                self.is_returning = True
                self.return_replan_attempts = 0
                self.return_fail_cooldown = 0
                decision_reason = "Return (Criteria Met)" if should_return_criteria else "Return (Local Target Too Far)"
                if self.debug:
                    print(f"\n[R{self.robot_id} Decide] STARTING RETURN ({decision_reason}). PathLen={len(self.planned_path)}", flush=True)

            else:
                 if self.debug: print(f"\n[R{self.robot_id} Decide] Wanted to return ({'Criteria' if should_return_criteria else 'TooFar'}), but failed path plan. Triggering cooldown & exploring locally.", flush=True)
                 self.return_fail_cooldown = RETURN_FAIL_COOLDOWN_STEPS
                 target_pos = target_pos_local
                 self.planned_path = self._plan_local_path(target_pos)
                 if not self.planned_path: self.planned_path = [self.position]
                 self.is_returning = False
                 decision_reason = "Local Explore (Return Failed)"
        else:
            target_pos = target_pos_local
            self.is_returning = False
            self.return_fail_cooldown = 0
            self.planned_path = self._plan_local_path(target_pos)
            if not self.planned_path: self.planned_path = [self.position]
            decision_reason = "Local Explore"


        self.target_pos = target_pos
        self.target_gived_by_server = False

        if self.debug:
             path_len = len(self.planned_path) if self.planned_path else 0
             print(f"[R{self.robot_id} Decide] Final Target: {self.target_pos}, Reason: {decision_reason}, PathLen: {path_len}", flush=True)


    def move_one_step(self, all_robots):
        if self.debug:
            path_str = f"[{self.planned_path[0]}]..." if self.planned_path and len(self.planned_path)>0 else "[]"
            print(f"\n[R{self.robot_id} Move] Pos: {self.position}, Target: {self.target_pos}, IsReturning: {self.is_returning}, Path(len={len(self.planned_path)}): {path_str}", flush=True)

        if len(self.planned_path) < 1:
            if self.debug: print(f"[R{self.robot_id} Move] No path.", flush=True)
            return

        next_step = self.planned_path[0]

        if np.array_equal(next_step, self.position):
            if len(self.planned_path) > 1:
                self.planned_path.pop(0); next_step = self.planned_path[0]
                if self.debug: print(f"[R{self.robot_id} Move] Path started w/ current pos, popped. Next: {next_step}", flush=True)
            else:
                if self.debug: print(f"[R{self.robot_id} Move] Reached target or stuck at {self.position}", flush=True)
                self.planned_path = []; return

        is_blocked = False; blocker_id = -1
        for other_robot in all_robots:
            if other_robot is self: continue
            if other_robot.position is not None:
                dist_to_other = np.linalg.norm(next_step - other_robot.position)
                if dist_to_other < 1.5: # 碰撞半徑
                    is_blocked = True; blocker_id = other_robot.robot_id; break

        if is_blocked:
            if self.debug: print(f"[R{self.robot_id} Move] Blocked by R{blocker_id} @ {all_robots[blocker_id].position}. NextStep: {next_step}. Stay: {self.stay_count}", flush=True)
            self.stay_count += 1; MAX_STAY_COUNT = 5
            if self.stay_count > MAX_STAY_COUNT:
                if self.debug: print(f"[R{self.robot_id} Move] Waited too long, trying back step.", flush=True)
                if len(self.movement_history) > 1:
                    previous_pos = self.movement_history[-2]; can_move_back = True
                    for r in all_robots:
                         if r is not self and np.linalg.norm(previous_pos - r.position) < 1.5: can_move_back = False; break
                    if can_move_back:
                         self.position = np.array(previous_pos); self.planned_path = []; self.stay_count = 0
                         if self.debug: print(f"[R{self.robot_id} Move] Moved back to {self.position}. Path cleared.", flush=True)
                         # 加入移動歷史
                         if not self.movement_history or not np.array_equal(self.position, self.movement_history[-1]):
                              self.movement_history.append(self.position.copy())

                    else:
                         if self.debug: print(f"[R{self.robot_id} Move] Cannot move back.", flush=True)
            elif self.stay_count > 2 and self.robot_id > blocker_id and not self.is_returning:
                if self.debug: print(f"[R{self.robot_id} Move] Yielding to R{blocker_id} & replanning.", flush=True)
                new_path = self._plan_local_path(self.target_pos, avoid_pos=next_step)
                if new_path and not (len(new_path)==1 and np.array_equal(new_path[0], self.position)): self.planned_path = new_path
        else:
            if self.debug: print(f"[R{self.robot_id} Move] Moving to {next_step}", flush=True)
            self.stay_count = 0; popped_step = None
            if len(self.planned_path) > 0:
                popped_step = self.planned_path.pop(0)
                if np.array_equal(popped_step, self.position) and len(self.planned_path) > 0:
                    next_step = self.planned_path.pop(0)
                elif popped_step is not None: next_step = popped_step
            if next_step is not None and not np.array_equal(next_step, self.position):
                self.position = np.array(next_step)
                if not self.movement_history or not np.array_equal(self.position, self.movement_history[-1]):
                    self.movement_history.append(self.position.copy())

        dist_to_server = np.linalg.norm(self.position - self.last_position_in_server_range)
        if dist_to_server < SERVER_COMM_RANGE: self.is_in_server_range = True
        else: self.is_in_server_range = False


    def _select_node(self, all_robots):
        # <--- 修正: 檢查 graph_generator 是否有效 --- >
        if not hasattr(self.graph_generator, 'target_candidates') or self.graph_generator.target_candidates is None:
             if self.debug: print(f"[R{self.robot_id} SelectNode] graph_generator.target_candidates is None!")
             return self.position, 0, 0

        candidates = self.graph_generator.target_candidates
        num_candidates = len(candidates)

        if num_candidates == 0:
            if self.debug: print(f"[R{self.robot_id} SelectNode] No candidates.")
            return self.position, 0, 0

        # <--- 修正: 檢查 utilities 是否有效 --- >
        if not hasattr(self.graph_generator, 'candidates_utility') or \
           self.graph_generator.candidates_utility is None or \
           len(self.graph_generator.candidates_utility) != num_candidates:
            if self.debug: print(f"[R{self.robot_id} SelectNode] Mismatch or missing candidates_utility!")
            # 嘗試重新計算? 或者返回失敗
            return self.position, 0, 0 # 返回失敗

        utilities = self.graph_generator.candidates_utility
        dists = np.linalg.norm(candidates - self.position, axis=1)
        valid_mask = utilities > 0

        for robot in all_robots:
             if robot.position is not None:
                same_pos = np.all(candidates == robot.position, axis=1)
                valid_mask &= ~same_pos
             if robot.target_pos is not None and not robot.is_in_server_range:
                 dist_to_other_target = np.linalg.norm(candidates - robot.target_pos, axis=1)
                 too_close_to_target = dist_to_other_target < 20
                 valid_mask &= ~too_close_to_target

        num_valid_after_filter = np.sum(valid_mask)

        if not np.any(valid_mask):
            if self.debug: print(f"[R{self.robot_id} SelectNode] No valid after filter.")
            return self.position, 0, 0

        valid_candidates = candidates[valid_mask]
        valid_utilities = utilities[valid_mask]
        valid_dists = dists[valid_mask]

        if len(valid_dists) == 0:
             if self.debug: print(f"[R{self.robot_id} SelectNode] Valid dists empty.")
             return self.position, 0, 0

        λ = 1.0; epsilon = 1e-6
        min_valid_dists = np.min(valid_dists)
        scores = λ * valid_utilities / (valid_dists + epsilon)
        best_idx_in_valid = np.argmax(scores)
        selected_coord = valid_candidates[best_idx_in_valid]
        original_indices = np.where(valid_mask)[0]
        if best_idx_in_valid >= len(original_indices):
             if self.debug: print(f"[R{self.robot_id} SelectNode] Index error!")
             return self.position, 0, 0
        original_idx = original_indices[best_idx_in_valid]

        if self.debug:
            print(f"[R{self.robot_id} SelectNode] Cands:{num_candidates}, Valid:{num_valid_after_filter} -> Sel:{selected_coord}, Score:{scores[best_idx_in_valid]:.2f}, Util:{valid_utilities[best_idx_in_valid]}, Dist:{valid_dists[best_idx_in_valid]:.1f}", flush=True)

        return selected_coord, original_idx, min_valid_dists

    def _plan_local_path(self, target, avoid_pos=None):
        gen = self.graph_generator; current = self.position
        if avoid_pos is not None: pass
        coords = gen.node_coords; graph = gen.graph

        # <--- 修正: 更詳細的檢查 --- >
        if coords is None or len(coords) == 0 or graph is None or not hasattr(graph, 'nodes') or not hasattr(graph, 'edges'):
             if self.debug: print(f"[R{self.robot_id} PlanLocal] Invalid graph/nodes. Coords:{coords is not None}, Graph:{graph is not None}", flush=True)
             return [current]

        dist, route = gen.find_shortest_path(current, target, coords, graph)
        path_len = len(route) if route is not None else 0

        if self.debug: print(f"[R{self.robot_id} PlanLocal] Target:{target}. A* Result:{'Success' if route is not None else 'Failed!'}, Dist:{dist:.1f}, PathLen:{path_len}", flush=True)

        if route is None or dist >= 1e5: return [current]
        if len(route) > 1 and np.array_equal(route[0], current): route.pop(0)
        if not route or (len(route) == 1 and np.array_equal(route[0], current)): return [current]
        return route