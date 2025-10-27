import numpy as np
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
        
        self.target_given_by_server = False
        self.last_position_in_server_range = start_position
        self.out_range_step = 0
        self.stay_count = 0
        
        self.resolution = resolution
        self.is_in_server_range = True # 初始狀態
        
    def update_local_awareness(self, real_map, all_robots, server, find_frontier_func, merge_maps_func):
        """
        執行一步的感知和地圖更新。
        取代舊 env.step() 中的 robot 相關邏輯。
        """
        
        # 1. 感測並更新本地地圖
        self.local_map = sensor_work(self.position, self.sensor_range, self.local_map, real_map)

        # 2. 與其他機器人同步 local_map（基於通訊範圍）
        dist_to_server = np.linalg.norm(self.position - server.position)
        self.is_in_server_range = dist_to_server < SERVER_COMM_RANGE

        for other_robot in all_robots:
            if other_robot is self:
                continue
            dist = np.linalg.norm(self.position - other_robot.position)
            if dist < ROBOT_COMM_RANGE:
                # 雙向合併
                merged = merge_maps_func([self.local_map, other_robot.local_map])
                self.local_map[:] = merged
                other_robot.local_map[:] = merged
                
                if not self.is_in_server_range:
                    self.planned_path = [] # 如果不在伺服器範圍內且遇到隊友，清空路徑以重新決策

        # 3. 與伺服器同步 local_map
        if self.is_in_server_range:
            merged = merge_maps_func([self.local_map, server.global_map])
            self.local_map[:] = merged
            server.global_map[:] = merged
            
            server.all_robot_position[self.robot_id] = self.position # 假設 robot 有 robot_id
            server.robot_in_range[self.robot_id] = True
            self.last_position_in_server_range = self.position
            self.out_range_step = 0
            if not self.target_given_by_server:
                self.planned_path = [] # 在伺服器範圍內且不是伺服器指派的，清空路徑
        else:
            if self.target_given_by_server: # 如果是伺服器指派的任務，跑完為止
                 pass
            else: # 如果是自主探索
                self.out_range_step += 1
            server.robot_in_range[self.robot_id] = False

        # 4. 更新 frontiers
        self.downsampled_map = block_reduce(
            self.local_map.copy(),
            block_size=(self.resolution, self.resolution),
            func=np.min
        )
        new_frontiers = find_frontier_func(self.downsampled_map)

        # 5. 更新節點圖
        node_coords, graph, node_utility, guidepost = self.graph_generator.update_graph(
            self.local_map,
            new_frontiers,
            self.frontiers,
            self.position
        )
        self.node_coords = node_coords
        self.local_map_graph = graph
        self.node_utility = node_utility
        self.guidepost = guidepost
        self.frontiers = new_frontiers

    def needs_new_target(self):
        """檢查是否需要規劃新路徑"""
        return len(self.planned_path) < 1

    def decide_next_target(self, all_robots):
        """
        決策系統：決定下一個目標點。
        包含自主探索選點和返回會合點的邏輯。
        """
        # 1. 檢查是否觸發「返回會合」機制
        # 這裡是你的學術創新點（統一決策框架）可以插入的地方
        # 目前先保留原始邏輯：
        if self.out_range_step > OUT_RANGE_STEP and not self.target_given_by_server:
            target_pos = self.last_position_in_server_range
            # print(f"Robot {self.robot_id} returning to server range at {target_pos}")
        else:
            # 2. 執行自主探索選點
            target_pos, _, min_valid_dists = self._select_node(all_robots)
            
            # 3. 檢查選點是否太遠（可能是孤立點），若是，也返回
            if min_valid_dists > (self.sensor_range * 1.5) and self.out_range_step > 0:
                target_pos = self.last_position_in_server_range
                # print(f"Robot {self.robot_id} selected node too far, returning to server.")

        self.target_pos = target_pos
        self.target_given_by_server = False # 只要是自主決策，就設為 False
        
        # 4. 規劃本地路徑
        self.planned_path = self._plan_local_path(target_pos)
        if not self.planned_path:
             self.planned_path = [self.position]


    def move_one_step(self, all_robots):
        """
        執行一步移動。
        取代舊 worker.py 中的移動邏輯。
        """
        if len(self.planned_path) < 1:
            # print(f"Robot {self.robot_id} has no path to move.")
            return

        next_step = self.planned_path[0]
        
        # 檢查是否卡在當前位置
        if np.array_equal(next_step, self.position):
            if len(self.planned_path) > 1:
                self.planned_path.pop(0)
                next_step = self.planned_path[0]
            else:
                # print(f"Robot {self.robot_id} reached target or is stuck at {self.position}")
                self.planned_path = []
                return

        # 檢查是否被其他機器人阻擋
        is_blocked = False
        for other_robot in all_robots:
            if other_robot is self:
                continue
            if np.linalg.norm(next_step - other_robot.position) < 1: # 碰撞距離
                is_blocked = True
                self.stay_count += 1
                # 簡單的避讓邏輯：ID 大的讓路
                if self.stay_count > 2 and self.robot_id > other_robot.robot_id:
                    # print(f"Robot {self.robot_id} yielding to Robot {other_robot.robot_id}, replanning...")
                    self.planned_path = self._plan_local_path(self.target_pos, avoid_pos=next_step)
                    if not self.planned_path:
                        self.planned_path = [self.position]
                    next_step = self.planned_path[0]
                    is_blocked = False # 假設重規劃的路徑是好的
                # else:
                    # print(f"Robot {self.robot_id} waiting for path to clear.")
                break # 只處理第一個阻擋者

        if not is_blocked:
            self.stay_count = 0
            self.planned_path.pop(0)
            self.position = np.array(next_step)
            self.movement_history.append(self.position.copy())
        
        # 移動後，立即更新與伺服器的連接狀態（但不更新地圖，地圖在下一次 update_local_awareness 更新）
        dist_to_server = np.linalg.norm(self.position - self.last_position_in_server_range) # 粗略估計
        if dist_to_server < SERVER_COMM_RANGE:
            self.is_in_server_range = True
        else:
            self.is_in_server_range = False


    def _select_node(self, all_robots):
        """
        根據演算法選擇下一個圖節點作為移動目標。
        (從 worker.py 搬移過來)
        """
        candidates = self.graph_generator.target_candidates
        
        if candidates is None or len(candidates) == 0:
            return self.position, 0, 0
        
        utilities = self.graph_generator.candidates_utility
        dists = np.linalg.norm(candidates - self.position, axis=1)
        
        # 1. 過濾掉 utility 為 0 的節點
        valid_mask = utilities > 0
        
        # 2. 過濾掉與**任何**其他機器人位置相同的節點
        for robot in all_robots:
            same_pos = np.all(candidates == robot.position, axis=1)
            valid_mask &= ~same_pos
        
        if not np.any(valid_mask):
            return self.position, 0, 0
        
        # 3. 建立有效節點子集
        valid_candidates = candidates[valid_mask]
        valid_utilities = utilities[valid_mask]
        valid_dists = dists[valid_mask]
        
        # 4. 計算分數並選出最佳
        λ = 1.0 # 來自 worker.py
        epsilon = 1e-6
        min_valid_dists = np.min(valid_dists) if len(valid_dists) > 0 else 0
        
        scores = λ * valid_utilities / (valid_dists + epsilon)
        best_idx_in_valid = np.argmax(scores)
        selected_coord = valid_candidates[best_idx_in_valid]
        
        # 5. 回推到原 coords 的索引
        original_indices = np.where(valid_mask)[0]
        original_idx = original_indices[best_idx_in_valid]
        
        return selected_coord, original_idx, min_valid_dists
        
    def _plan_local_path(self, target, avoid_pos=None):
        """
        回傳從當前機器人位置到 target 的完整節點路徑（list of coords）
        (從 env.py 搬移過來)
        """
        gen = self.graph_generator
        current = self.position
        
        # 暫時從圖中移除要避開的點
        if avoid_pos is not None:
            # 這部分邏輯比較複雜，先簡化
            pass 

        coords = gen.node_coords
        graph = gen.graph
        
        if coords is None or len(coords) == 0 or graph is None:
            return [current]
            
        # 用 A* 找到節點路徑
        _, route = gen.find_shortest_path(current, target, coords, graph)
        return route if route is not None else [current]