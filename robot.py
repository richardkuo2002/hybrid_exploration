import numpy as np
from collections import deque
from parameter import *
from graph_generator import Graph_generator
from sensor import sensor_work
from skimage.measure import block_reduce

class Robot():
    def __init__(self, start_position, real_map_size, resolution, k_size, plot=False):
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
        
        # <--- 修改點：新增圖更新計數器 ---
        self.graph_update_counter = 0
        # --- ---
        
    def update_local_awareness(self, real_map, all_robots, server, find_frontier_func, merge_maps_func):
        """
        執行一步的感知和地圖更新。
        """
        
        # --- 1. 感測與地圖合併 (Sensing & Merging) ---
        # ... (這部分邏輯不變) ...
        self.local_map = sensor_work(self.position, self.sensor_range, self.local_map, real_map)
        dist_to_server = np.linalg.norm(self.position - server.position)
        self.is_in_server_range = dist_to_server < SERVER_COMM_RANGE

        for other_robot in all_robots:
            if other_robot is self:
                continue
            dist = np.linalg.norm(self.position - other_robot.position)
            if dist < ROBOT_COMM_RANGE:
                merged = merge_maps_func([self.local_map, other_robot.local_map])
                self.local_map[:] = merged
                other_robot.local_map[:] = merged
                if not self.is_in_server_range:
                    self.planned_path = []

        if self.is_in_server_range:
            merged = merge_maps_func([self.local_map, server.global_map])
            self.local_map[:] = merged
            server.global_map[:] = merged
            server.all_robot_position[self.robot_id] = self.position
            server.robot_in_range[self.robot_id] = True
            self.last_position_in_server_range = self.position
            self.out_range_step = 0
            if not self.target_gived_by_server:
                self.planned_path = []
        else:
            if self.target_gived_by_server:
                 pass
            else:
                self.out_range_step += 1
            server.robot_in_range[self.robot_id] = False
        
        # --- 2. 計算地圖增益 ---
        # ... (這部分邏輯不變) ...
        current_explored_area = np.sum(self.local_map == 255)
        self.current_info_gain = current_explored_area - self.last_explored_area
        self.last_explored_area = current_explored_area
        
        if not self.is_in_server_range and not self.target_gived_by_server:
             self.info_gain_history.append(self.current_info_gain)
        else:
             self.info_gain_history.clear()

        # --- 3. 更新 Frontiers ---
        self.downsampled_map = block_reduce(
            self.local_map.copy(),
            block_size=(self.resolution, self.resolution),
            func=np.min
        )
        new_frontiers = find_frontier_func(self.downsampled_map)

        # <--- 修改點：使用計數器決定更新策略 ---
        # 4. 更新節點圖（帶有頻率控制）
        self.graph_update_counter += 1
        if self.graph_update_counter % GRAPH_UPDATE_INTERVAL == 0:
            # --- 執行昂貴的「結構重建」 ---
            node_coords, graph, node_utility, guidepost = self.graph_generator.rebuild_graph_structure(
                self.local_map,
                new_frontiers,
                self.frontiers, # 傳入舊的 frontiers
                self.position
                # all_robot_positions 設為 None，因為本地圖不需要
            )
            self.node_coords = node_coords
            self.local_map_graph = graph
            self.node_utility = node_utility
            self.guidepost = guidepost
        else:
            # --- 執行輕量的「價值更新」 ---
            node_utility, guidepost = self.graph_generator.update_node_utilities(
                self.local_map,
                new_frontiers,
                self.frontiers # 傳入舊的 frontiers
                # all_robot_positions 設為 None
            )
            # *** 注意：self.node_coords 和 self.local_map_graph 保持不變 ***
            self.node_utility = node_utility
            self.guidepost = guidepost
        
        self.frontiers = new_frontiers # frontiers 必須每一步都更新
        # --- ---

    def needs_new_target(self):
        # ... (needs_new_target 保持不變) ...
        return len(self.planned_path) < 1

    def decide_next_target(self, all_robots):
        # ... (decide_next_target 保持不變) ...
        target_pos_local, _, min_valid_dists = self._select_node(all_robots)
        max_local_utility = 0
        if self.graph_generator.candidates_utility is not None and len(self.graph_generator.candidates_utility) > 0:
             max_local_utility = np.max(self.graph_generator.candidates_utility)
        
        recent_info_gain = np.sum(self.info_gain_history)
        is_stagnated = (len(self.info_gain_history) == INFO_GAIN_HISTORY_LEN) and \
                       (recent_info_gain < MIN_INFO_GAIN_THRESHOLD)
        is_depleted = max_local_utility < LOCAL_UTILITY_THRESHOLD
        is_timeout = self.out_range_step > OUT_RANGE_STEP

        if (is_stagnated and is_depleted) or is_timeout:
            target_pos = self.last_position_in_server_range
        else:
            target_pos = target_pos_local
            if min_valid_dists > (self.sensor_range * 1.5) and self.out_range_step > 5:
                target_pos = self.last_position_in_server_range

        self.target_pos = target_pos
        self.target_gived_by_server = False
        
        self.planned_path = self._plan_local_path(target_pos)
        if not self.planned_path:
             self.planned_path = [self.position]


    def move_one_step(self, all_robots):
        # ... (move_one_step 保持不變) ...
        if len(self.planned_path) < 1:
            return

        next_step = self.planned_path[0]
        
        if np.array_equal(next_step, self.position):
            if len(self.planned_path) > 1:
                self.planned_path.pop(0)
                next_step = self.planned_path[0]
            else:
                self.planned_path = []
                return

        is_blocked = False
        for other_robot in all_robots:
            if other_robot is self:
                continue
            if np.linalg.norm(next_step - other_robot.position) < 1:
                is_blocked = True
                self.stay_count += 1
                if self.stay_count > 2 and self.robot_id > other_robot.robot_id:
                    self.planned_path = self._plan_local_path(self.target_pos, avoid_pos=next_step)
                    if not self.planned_path:
                        self.planned_path = [self.position]
                    next_step = self.planned_path[0]
                    is_blocked = False
                break

        if not is_blocked:
            self.stay_count = 0
            self.planned_path.pop(0)
            self.position = np.array(next_step)
            self.movement_history.append(self.position.copy())
        
        dist_to_server = np.linalg.norm(self.position - self.last_position_in_server_range)
        if dist_to_server < SERVER_COMM_RANGE:
            self.is_in_server_range = True
        else:
            self.is_in_server_range = False


    def _select_node(self, all_robots):
        # ... (_select_node 保持不變) ...
        candidates = self.graph_generator.target_candidates
        
        if candidates is None or len(candidates) == 0:
            return self.position, 0, 0
        
        utilities = self.graph_generator.candidates_utility
        dists = np.linalg.norm(candidates - self.position, axis=1)
        
        valid_mask = utilities > 0
        
        for robot in all_robots:
            same_pos = np.all(candidates == robot.position, axis=1)
            valid_mask &= ~same_pos
        
        if not np.any(valid_mask):
            return self.position, 0, 0
        
        valid_candidates = candidates[valid_mask]
        valid_utilities = utilities[valid_mask]
        valid_dists = dists[valid_mask]
        
        if len(valid_dists) == 0:
            return self.position, 0, 0
            
        λ = 1.0
        epsilon = 1e-6
        min_valid_dists = np.min(valid_dists)
        
        scores = λ * valid_utilities / (valid_dists + epsilon)
        best_idx_in_valid = np.argmax(scores)
        selected_coord = valid_candidates[best_idx_in_valid]
        
        original_indices = np.where(valid_mask)[0]
        original_idx = original_indices[best_idx_in_valid]
        
        return selected_coord, original_idx, min_valid_dists
        
    def _plan_local_path(self, target, avoid_pos=None):
        # ... (_plan_local_path 保持不變) ...
        gen = self.graph_generator
        current = self.position
        
        if avoid_pos is not None:
            pass 

        coords = gen.node_coords
        graph = gen.graph
        
        if coords is None or len(coords) == 0 or graph is None:
            return [current]
            
        _, route = gen.find_shortest_path(current, target, coords, graph)
        return route if route is not None else [current]