import numpy as np
from skimage.measure import block_reduce
from scipy.optimize import linear_sum_assignment

from parameter import *
from graph_generator import Graph_generator
from graph import Graph, a_star # 需要匯入 a_star

class Server():
    def __init__(self, start_position, real_map_size, resolution, k_size, plot=False):
        self.position = start_position
        self.global_map = np.ones(real_map_size) * 127
        self.downsampled_map = None
        self.comm_range = SERVER_COMM_RANGE
        self.all_robot_position = [] # 會由 robot 或 env 初始化
        self.robot_in_range = [] # 會由 robot 或 env 初始化
        
        self.graph_generator:Graph_generator = Graph_generator(map_size=real_map_size, sensor_range=SENSOR_RANGE, k_size=k_size, plot=plot)
        self.graph_generator.route_node.append(start_position)
        
        self.frontiers = []
        self.node_coords = []
        self.local_map_graph = []
        self.node_utility = []
        self.guidepost = []
        
        self.resolution = resolution # 需要解析度

    def update_and_assign_tasks(self, robot_list, real_map, find_frontier_func):
        """
        執行一步的伺服器決策：更新全局地圖、圖結構，並指派任務。
        取代舊 env.server_step()
        """
        
        # --- 1. 更新全局地圖與圖結構 ---
        self.downsampled_map = block_reduce(
            self.global_map.copy(),
            block_size=(self.resolution, self.resolution),
            func=np.min
        )
        new_frontiers = find_frontier_func(self.downsampled_map)
        
        node_coords, graph, node_utility, guidepost = self.graph_generator.update_graph(
            self.global_map,
            new_frontiers,
            self.frontiers,
            self.position,
            self.all_robot_position)
        
        self.node_coords = node_coords
        self.local_map_graph = graph
        self.node_utility = node_utility
        self.guidepost = guidepost
        self.frontiers = new_frontiers
        
        # --- 2. 篩選需要任務的機器人 ---
        robots_need_assignment = []
        robot_positions = []
        
        for i, in_range in enumerate(self.robot_in_range):
            if in_range and self.all_robot_position[i] is not None:
                robot = robot_list[i]
                if robot.needs_new_target(): # 使用 robot 的方法
                    robots_need_assignment.append(i)  # 機器人索引
                    robot_positions.append(self.all_robot_position[i])
        
        # --- 檢查探索是否完成 ---
        total_frontiers = len(self.frontiers)
        total_free = np.sum(real_map == 255)
        explored = np.sum(self.global_map == 255)
        coverage = explored / total_free if total_free > 0 else 0.0
        done = (total_frontiers == 0) or (coverage >= 0.95)
        
        if not robots_need_assignment:
            # print("Server: No robots need assignment")
            return done, coverage
        
        # --- 3. 準備候選目標 ---
        if len(self.graph_generator.target_candidates) < 1:
            # print("Server: No target candidates available")
            return done, coverage
        
        candidates = np.array([list(coord) for coord in self.graph_generator.target_candidates])
        utilities = np.array(self.graph_generator.candidates_utility)
        
        available_candidates, available_utilities = self._filter_occupied_targets(
            candidates, utilities, robot_list, robots_need_assignment
        )
        
        if len(available_candidates) == 0:
            # print("Server: No available targets after filtering")
            return done, coverage
        
        # --- 4. 準備匈牙利演算法 ---
        m = len(robots_need_assignment)  # 機器人數量
        k = len(available_candidates)    # 目標數量
        
        if k < m:
            sorted_indices = np.argsort(-available_utilities)
            extended_candidates = list(available_candidates)
            extended_utilities = list(available_utilities)
            
            while len(extended_candidates) < m:
                for idx in sorted_indices:
                    if len(extended_candidates) >= m:
                        break
                    extended_candidates.append(available_candidates[idx])
                    extended_utilities.append(available_utilities[idx])
            
            available_candidates = np.array(extended_candidates)
            available_utilities = np.array(extended_utilities)
            k = m
        
        # --- 5. 建立成本矩陣 ---
        cost_matrix = np.zeros((m, k))
        lambda_dist = 1.1  # 距離權重係數 (來自舊 env.py)
        
        for i, robot_pos in enumerate(robot_positions):
            for j, candidate in enumerate(available_candidates):
                distance = np.linalg.norm(np.array(robot_pos) - np.array(candidate))
                utility = available_utilities[j]
                cost_matrix[i, j] = -utility + lambda_dist * distance
        
        # --- 6. 執行匈牙利演算法 ---
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        except Exception as e:
            print(f"Server: Hungarian algorithm failed: {e}")
            return done, coverage
        
        # --- 7. 執行指派 ---
        for i, j in zip(row_indices, col_indices):
            robot_idx:int = robots_need_assignment[i]
            target = available_candidates[j]
            robot = robot_list[robot_idx]
            
            robot.target_pos = np.array(target)
            
            try:
                robot.planned_path = self._plan_global_path(robot.position, robot.target_pos)
                robot.target_given_by_server = True # 標記為伺服器指派
                
                if not robot.planned_path or len(robot.planned_path) == 0:
                    robot.target_pos = None
                    robot.target_given_by_server = False
                    
            except Exception as e:
                print(f"Server: Path planning failed for Robot {robot_idx}: {e}")
                robot.target_pos = None
                robot.target_given_by_server = False
        
        return done, coverage

    def _filter_occupied_targets(self, candidates, utilities, robot_list, requesting_robots):
        """
        過濾掉已被其他機器人佔據或指派的目標
        (從 env.py 搬移過來)
        """
        available_mask = np.ones(len(candidates), dtype=bool)
        
        for i, robot in enumerate(robot_list):
            if i in requesting_robots:
                continue
            
            if hasattr(robot, 'target_pos') and robot.target_pos is not None:
                distances = np.linalg.norm(candidates - robot.target_pos, axis=1)
                available_mask &= (distances > 20)
            
            distances_to_robot = np.linalg.norm(candidates - robot.position, axis=1)
            available_mask &= (distances_to_robot > 15)
            
            if hasattr(robot, 'planned_path') and len(robot.planned_path) > 0:
                for planned_pos in robot.planned_path[:3]:
                    planned_pos = np.array(planned_pos)
                    distances_to_planned = np.linalg.norm(candidates - planned_pos, axis=1)
                    available_mask &= (distances_to_planned > 10)
        
        return candidates[available_mask], utilities[available_mask]
        
    def _plan_global_path(self, current_pos, target_pos):
        """
        回傳從當前機器人位置到 target 的完整節點路徑（list of coords）
        (從 env.py 搬移過來)
        """
        gen = self.graph_generator
        current = current_pos
        target = target_pos
        
        coords = gen.node_coords
        graph = gen.graph

        if coords is None or len(coords) == 0 or graph is None:
            return [current]
            
        _, route = gen.find_shortest_path(current, target, coords, graph)
        return route if route is not None else [current]