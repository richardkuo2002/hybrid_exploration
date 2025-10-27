from skimage import io
import os
from skimage.measure import block_reduce
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
from scipy.optimize import linear_sum_assignment

from robot import Robot
from server import Server
from graph_generator import Graph_generator
from sensor import *
from parameter import *

class Env():
    def __init__(self, n_agent:int, k_size=20, map_index=0, plot=True):
        self.resolution = 4
        self.coverage = 0
        
        self.map_path = "DungeonMaps/train/easy"
        self.map_list = os.listdir(self.map_path)
        self.real_map, self.start_position = self.import_map(self.map_path + '/img_' + str(map_index) + '.png')
        self.real_map_size = np.shape(self.real_map)
        
        self.server = Server()
        self.server.global_map = np.ones(self.real_map_size) * 127
        self.server.position = self.start_position
        
        self.n_agent = n_agent
        self.robot_list = [Robot() for _ in range(self.n_agent)]
        for robot in self.robot_list:
            robot.local_map = np.ones(self.real_map_size) * 127
            robot.position = self.start_position
            robot.last_position_in_server_range = self.start_position
            self.server.all_robot_position.append(robot.position)
            self.server.robot_in_range.append(True)
            
            robot.local_map = self.update_robot_local_map(robot.position, robot.sensor_range, robot.local_map, self.real_map)
            robot.downsampled_map = block_reduce(robot.local_map.copy(), block_size=(self.resolution, self.resolution), func=np.min)
            robot.frontiers = self.find_frontier(robot.downsampled_map)
            robot.graph_generator = Graph_generator(map_size=self.real_map_size, sensor_range=robot.sensor_range, k_size=k_size, plot=plot)
            robot.graph_generator.route_node.append(self.start_position)
            
            node_coords, graph, node_utility, guidepost = robot.graph_generator.generate_graph(self.start_position, robot.local_map, robot.frontiers)
            robot.node_coords = node_coords
            robot.local_map_graph = graph
            robot.node_utility = node_utility
            robot.guidepost = guidepost
        
        
        merged = self.merge_maps([robot.local_map, self.server.global_map])
        robot.local_map[:] = merged
        self.server.global_map[:] = merged
        
        self.server.downsampled_map = block_reduce(self.server.global_map.copy(), block_size=(self.resolution, self.resolution), func=np.min)
        self.server.frontiers = self.find_frontier(self.server.downsampled_map)
        self.server.graph_generator = Graph_generator(map_size=self.real_map_size, sensor_range=robot.sensor_range, k_size=k_size, plot=plot)
        self.server.graph_generator.route_node.append(self.start_position)
        node_coords, graph, node_utility, guidepost = self.server.graph_generator.generate_graph(self.start_position, self.server.global_map, self.server.frontiers)
        self.server.node_coords = node_coords
        self.server.local_map_graph = graph
        self.server.node_utility = node_utility
        self.server.guidepost = guidepost
        
    
    def step(self, agent_id: int, step:int):
        """
        執行一次模擬更新，包含指定機器人的感測、本地圖與全域圖同步、節點圖更新，以及機器人間與機器人–伺服器的地圖合併。

        Args:
            agent_id (int): 要執行這一步驟的機器人索引。

        Returns:
            done (bool): True 表示所有前沿皆已探索完畢，False 表示仍有未探索區域。
            info (dict): 當前步驟相關資訊，包括各機器人位置、前沿數量與覆蓋率等。
        """
        # 1. 當前機器人感測並更新本地地圖
        robot = self.robot_list[agent_id]
        robot.local_map = self.update_robot_local_map(
            robot.position, robot.sensor_range, robot.local_map, self.real_map
        )

        # 2. 與其他機器人同步 local_map（基於通訊範圍）
        
        dist_to_server = np.linalg.norm(robot.position - self.server.position)
        for other_id, other in enumerate(self.robot_list):
            if other_id == agent_id:
                continue
            dist = np.linalg.norm(robot.position - other.position)
            if dist < ROBOT_COMM_RANGE:
                # 雙向合併
                merged = self.merge_maps([robot.local_map, other.local_map])
                robot.local_map[:] = merged
                other.local_map[:] = merged
                
                robot.out_range_step = min(robot.out_range_step, other.out_range_step)
                if (dist_to_server > SERVER_COMM_RANGE or robot.out_range_step > OUT_RANGE_STEP) and not robot.target_gived_by_server:
                    robot.planned_path = []

        # 3. 與伺服器同步 local_map
        
        if dist_to_server < SERVER_COMM_RANGE:
            merged = self.merge_maps([robot.local_map, self.server.global_map])
            robot.local_map[:] = merged
            self.server.global_map[:] = merged
            
            self.server.all_robot_position[agent_id] = robot.position
            robot.last_position_in_server_range = robot.position
            self.server.robot_in_range[agent_id] = True
            robot.out_range_step = 0
            if not robot.target_gived_by_server:
                robot.planned_path = []
        else:
            if not robot.target_gived_by_server:
                robot.out_range_step += 1
            self.server.robot_in_range[agent_id] = False


        # 4. 更新 frontiers
        robot.downsampled_map = block_reduce(
            robot.local_map.copy(),
            block_size=(self.resolution, self.resolution),
            func=np.min
        )
        new_frontiers = self.find_frontier(robot.downsampled_map)

        # 5. 更新節點圖（混合式探勘）
        node_coords, graph, node_utility, guidepost = robot.graph_generator.update_graph(
            robot.local_map,
            new_frontiers,
            robot.frontiers,
            robot.position
        )
        robot.node_coords = node_coords
        robot.local_map_graph = graph
        robot.node_utility = node_utility
        robot.guidepost = guidepost
        robot.frontiers = new_frontiers

        # # 6. 評估探索完成條件
        # total_frontiers = sum(len(r.frontiers) for r in self.robot_list)
        # total_free = np.sum(self.real_map == 255)
        # explored = np.sum(self.server.global_map == 255)
        # coverage = explored / total_free if total_free > 0 else 0.0
        # done = (total_frontiers == 0) or (coverage >= 0.95)

        # 7. 回傳狀態資訊
        info = {
            'agent_positions': [r.position for r in self.robot_list],
            'frontiers_remaining': [len(r.frontiers) for r in self.robot_list],
            # 'coverage_ratio': coverage,
            'step_count': getattr(self, 'step_count', 0) + 1
        }
        self.step_count = info['step_count']
        
        return info

    def server_step(self):
        """
        self.server.robot_in_range:list[bool]
        如果機器人沒有目標則
        len(robot.planned_path)<1
        
        self.server.all_robot_position
        self.server.graph_generator.target_candidates:list[tuple]
        self.server.graph_generator.candidates_utility
        
        分配完後
        robot.target_pos = 被server分配到的
        robot.planned_path = self.plan_global_path(i, robot.target_pos)
        """
        self.server.downsampled_map = block_reduce(
            self.server.global_map.copy(),
            block_size=(self.resolution, self.resolution),
            func=np.min
        )
        new_frontiers = self.find_frontier(self.server.downsampled_map)
        
        node_coords, graph, node_utility, guidepost = self.server.graph_generator.update_graph(
            self.server.global_map,
            new_frontiers,
            self.server.frontiers,
            self.server.position,
            self.server.all_robot_position)
        
        self.server.node_coords = node_coords
        self.server.local_map_graph = graph
        self.server.node_utility = node_utility
        self.server.guidepost = guidepost
        self.server.frontiers = new_frontiers
        
        # 1. 篩選出在通訊範圍內且需要指派的機器人
        robots_need_assignment = []
        robot_positions = []
        
        for i, in_range in enumerate(self.server.robot_in_range):
            if in_range and self.server.all_robot_position[i] is not None:
                robot = self.robot_list[i]
                # 檢查是否需要指派：planned_path 長度 < 1
                if len(robot.planned_path) < 1:
                    robots_need_assignment.append(i)  # 機器人索引
                    robot_positions.append(self.server.all_robot_position[i])
        
        # 2. 如果沒有需要指派的機器人，直接返回
        if not robots_need_assignment:
            # print("Server: No robots need assignment")
            total_frontiers = len(self.server.frontiers)
            total_free = np.sum(self.real_map == 255)
            explored = np.sum(self.server.global_map == 255)
            coverage = explored / total_free if total_free > 0 else 0.0
            done = (total_frontiers == 0) or (coverage >= 0.95)
            # print(coverage)
            self.coverage = coverage
            return done, coverage
        
        # print(f"Server: Found {len(robots_need_assignment)} robots needing assignment")
        
        # 3. 準備候選目標節點
        if len(self.server.graph_generator.target_candidates) < 1:
            # print("Server: No target candidates available")
            total_frontiers = len(self.server.frontiers)
            total_free = np.sum(self.real_map == 255)
            explored = np.sum(self.server.global_map == 255)
            coverage = explored / total_free if total_free > 0 else 0.0
            done = (total_frontiers == 0) or (coverage >= 0.95)
            self.coverage = coverage
            return done, coverage
        
        candidates = np.array([list(coord) for coord in self.server.graph_generator.target_candidates])
        utilities = np.array(self.server.graph_generator.candidates_utility)
        
        # 4. 過濾掉已被其他機器人指派的目標
        available_candidates, available_utilities = self._filter_occupied_targets(
            candidates, utilities, robots_need_assignment
        )
        
        if len(available_candidates) == 0:
            # print("Server: No available targets after filtering")
            total_frontiers = len(self.server.frontiers)
            total_free = np.sum(self.real_map == 255)
            explored = np.sum(self.server.global_map == 255)
            coverage = explored / total_free if total_free > 0 else 0.0
            done = (total_frontiers == 0) or (coverage >= 0.95)
            # print(coverage)
            self.coverage = coverage
            return done, coverage
        
        # 5. 準備匈牙利演算法的維度
        m = len(robots_need_assignment)  # 需要指派的機器人數量
        k = len(available_candidates)    # 可用候選目標數量
        
        # 如果候選數少於機器人數，擴展候選集合
        if k < m:
            # 複製最高效益的候選直到數量足夠
            sorted_indices = np.argsort(-available_utilities)
            extended_candidates = []
            extended_utilities = []
            
            while len(extended_candidates) < m:
                for idx in sorted_indices:
                    if len(extended_candidates) >= m:
                        break
                    extended_candidates.append(available_candidates[idx])
                    extended_utilities.append(available_utilities[idx])
            
            available_candidates = np.array(extended_candidates)
            available_utilities = np.array(extended_utilities)
            k = m
        
        # 6. 建立成本矩陣
        cost_matrix = np.zeros((m, k))
        lambda_dist = 1.1  # 距離權重係數
        
        for i, robot_pos in enumerate(robot_positions):
            for j, candidate in enumerate(available_candidates):
                # 計算歐氏距離
                distance = np.linalg.norm(np.array(robot_pos) - np.array(candidate))
                utility = available_utilities[j]
                # 成本 = -效益 + λ * 距離
                cost_matrix[i, j] = -utility + lambda_dist * distance
        
        # 7. 執行匈牙利演算法
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            # print(f"Server: Hungarian algorithm completed, assigning {len(row_indices)} targets")
        except Exception as e:
            print(f"Server: Hungarian algorithm failed: {e}")
            return
        
        # 8. 執行指派並規劃路徑
        successful_assignments = 0
        
        for i, j in zip(row_indices, col_indices):
            robot_idx:int = robots_need_assignment[i]
            target = available_candidates[j]
            robot = self.robot_list[robot_idx]
            
            # 設定目標位置
            robot.target_pos = np.array(target)
            robot.target_gived_by_server = True
            
            # 規劃全域路徑
            try:
                robot.planned_path = self.plan_global_path(robot_idx, robot.target_pos)
                
                if robot.planned_path and len(robot.planned_path) > 0:
                    successful_assignments += 1
                    utility = available_utilities[j]
                    distance = np.linalg.norm(robot_positions[i] - target)
                    # print(f"Server: Robot {robot_idx} → Target {target} (utility: {utility}, dist: {distance:.1f})")
                else:
                    # print(f"Server: Failed to plan path for Robot {robot_idx} to {target}")
                    robot.target_pos = None
                    
            except Exception as e:
                print(f"Server: Path planning failed for Robot {robot_idx}: {e}")
                robot.target_pos = None
        
        # print(f"Server: Successfully assigned {successful_assignments}/{len(robots_need_assignment)} robots")
        
        total_frontiers = len(self.server.frontiers)
        total_free = np.sum(self.real_map == 255)
        explored = np.sum(self.server.global_map == 255)
        coverage = explored / total_free if total_free > 0 else 0.0
        done = (total_frontiers == 0) or (coverage >= 0.95)
        self.coverage = coverage
        return done, coverage

    def _filter_occupied_targets(self, candidates, utilities, requesting_robots):
        """過濾掉已被其他機器人佔據或指派的目標"""
        available_mask = np.ones(len(candidates), dtype=bool)
        
        for i, robot in enumerate(self.robot_list):
            # 跳過正在請求指派的機器人
            if i in requesting_robots:
                continue
            
            # 過濾掉其他機器人的當前目標位置
            if hasattr(robot, 'target_pos') and robot.target_pos is not None:
                distances = np.linalg.norm(candidates - robot.target_pos, axis=1)
                available_mask &= (distances > 20)  # 避免指派到相同或太接近的目標
            
            # 過濾掉其他機器人的當前位置
            distances_to_robot = np.linalg.norm(candidates - robot.position, axis=1)
            available_mask &= (distances_to_robot > 15)  # 避免指派到其他機器人附近
            
            # 過濾掉其他機器人計劃路徑上的節點
            if hasattr(robot, 'planned_path') and len(robot.planned_path) > 0:
                for planned_pos in robot.planned_path[:3]:  # 只檢查前3步，避免過度限制
                    planned_pos = np.array(planned_pos)
                    distances_to_planned = np.linalg.norm(candidates - planned_pos, axis=1)
                    available_mask &= (distances_to_planned > 10)
        
        return candidates[available_mask], utilities[available_mask]
        

    def calculate_coverage_ratio(self):
        """計算地圖探索覆蓋率"""
        # 統計所有機器人已探索的區域
        explored_area = 0
        total_free_area = 0
        
        for robot in self.robot_list:
            # 計算已探索區域（像素值 = 255 的區域）
            explored_pixels = np.sum(robot.local_map == 255)
            explored_area += explored_pixels
        
        # 計算實際地圖的自由空間總面積
        total_free_pixels = np.sum(self.real_map == 255)
        
        return min(explored_area / total_free_pixels, 1.0) if total_free_pixels > 0 else 0.0

    def merge_maps(self, maps_to_merge):
        """ Merge map beliefs together"""
        merged_map = np.ones_like(self.real_map) * 127  # unknown
        for belief in maps_to_merge:
            merged_map[belief == 1] = 1   # Obstacle
            merged_map[belief == 255] = 255   # Free
        return merged_map
    
    def update_robot_local_map(self, robot_position, sensor_range, robot_local_map, real_map):
        """
        依據機器人位置與感測器範圍，更新機器人的本地地圖。

        Args:
            robot_position (array-like): 機器人當前座標 [x, y]。
            sensor_range (float or int): 機器人感測器的感知半徑（單位依據地圖比例）。
            robot_local_map (ndarray): 機器人原先的本地地圖（待更新）。
            real_map (ndarray): 實際完整地圖（含全部障礙物資訊）。

        Returns:
            robot_local_map (ndarray): 更新後的本地地圖資訊。
        """
        robot_local_map = sensor_work(robot_position, sensor_range, robot_local_map, real_map)
        return robot_local_map
    
    def find_frontier(self, downsampled_map):
        """
        在當前地圖（downsampled_map）中找出「探索邊界」（Frontiers），即判斷可探索區域的像素座標。

        Args:
            downsampled_map (ndarray): 當前下採樣的地圖陣列，像素值分別代表障礙物、自由空間、未知區域。

        Returns:
            frontiers (ndarray): 前沿邊界座標陣列（N,2），每個座標已乘以解析度 self.resolution。
                                代表從地圖中判斷出的所有可探索 frontier 點之(x, y)座標。
        """
        y_len = downsampled_map.shape[0]
        x_len = downsampled_map.shape[1]
        mapping = downsampled_map.copy()
        belief = downsampled_map.copy()
        # 0-1 unknown area map
        mapping = (mapping == 127) * 1
        mapping = np.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)
        fro_map = mapping[2:][:, 1:x_len + 1] + mapping[:y_len][:, 1:x_len + 1] + mapping[1:y_len + 1][:, 2:] + \
                  mapping[1:y_len + 1][:, :x_len] + mapping[:y_len][:, 2:] + mapping[2:][:, :x_len] + mapping[2:][:,
                                                                                                      2:] + \
                  mapping[:y_len][:, :x_len]
        ind_free = np.where(belief.ravel(order='F') == 255)[0]
        ind_fron_1 = np.where(1 < fro_map.ravel(order='F'))[0]
        ind_fron_2 = np.where(fro_map.ravel(order='F') < 8)[0]
        ind_fron = np.intersect1d(ind_fron_1, ind_fron_2)
        ind_to = np.intersect1d(ind_free, ind_fron)

        map_x = x_len
        map_y = y_len
        x = np.linspace(0, map_x - 1, map_x)
        y = np.linspace(0, map_y - 1, map_y)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T

        f = points[ind_to]
        f = f.astype(int)

        f = f * self.resolution

        return f
    
    def plan_local_path(self, robot_id:int, target):
        """
        回傳從當前機器人位置到 target 的完整節點路徑（list of coords）
        """
        gen = self.robot_list[robot_id].graph_generator
        current = self.robot_list[robot_id].position
        # node_coords, graph 都已存在
        coords = gen.node_coords
        graph = gen.graph
        # 用 A* 找到節點路徑
        _, route = gen.find_shortest_path(current, target, coords, graph)
        # route 包含 start 和 end，中間都是連續節點
        return route if route is not None else [current]
    
    def plan_local_path_again(self, robot_id:int, target, remove_position):
        """
        回傳從當前機器人位置到 target 的完整節點路徑（list of coords）
        """
        gen = self.robot_list[robot_id].graph_generator
        current = self.robot_list[robot_id].position
        # node_coords, graph 都已存在
        coords = gen.node_coords
        gen.node_clear(remove_position)
        graph = gen.graph
        # 用 A* 找到節點路徑
        _, route = gen.find_shortest_path(current, target, coords, graph)
        # route 包含 start 和 end，中間都是連續節點
        return route if route is not None else [current]
    
    def plan_global_path(self, robot_id:int, target):
        """
        回傳從當前機器人位置到 target 的完整節點路徑（list of coords）
        """
        gen = self.server.graph_generator
        current = self.robot_list[robot_id].position
        # node_coords, graph 都已存在
        coords = gen.node_coords
        graph = gen.graph
        # 用 A* 找到節點路徑
        _, route = gen.find_shortest_path(current, target, coords, graph)
        # route 包含 start 和 end，中間都是連續節點
        return route if route is not None else [current]

    
    def import_map(self, map_path):
        """
        匯入地圖檔案，並解析障礙物、自由空間與機器人當前位置。

        說明：
            地圖以像素數值區分：
            - 1：障礙物
            - 255：自由空間
            - 127：未探索區域
            - 208：機器人起始位置

        Args:
            map_path (str): 地圖檔案路徑（支援常見影像格式，如 PNG、JPG）。

        Returns:
            map (ndarray): 處理後的地圖陣列（障礙物為1，自由空間為255，未探索為127）。
            robot_location (ndarray): 機器人目前座標 [x, y]，若未偵測則可能回傳預設值（依影像）。
        """
        # try:
        map = (io.imread(map_path, 1)).astype(int)
        if np.all(map == 0):
            map = (io.imread(map_path, 1) * 255).astype(int)
        # except:
        #     new_map_index = self.map_dir + '/' + self.map_list[0]
        #     map = (io.imread(new_map_index, 1)).astype(int)
        #     print('could not read the map_path ({}), hence skipping it and using ({}).'.format(map_index, new_map_index))

        robot_location = np.nonzero(map == 208)
        robot_location = np.array([np.array(robot_location)[1, 127], np.array(robot_location)[0, 127]])
        map = (map > 150)
        map = map * 254 + 1
        return map, robot_location
    
    def plot_env(self, step):
        """Real 縱橫跨2x2；Robot1 的 local map 從第3列第1欄開始、每列2張、每兩列換行"""
        if not hasattr(self, 'fig') or self.fig is None:
            plt.ion()  # 啟用互動模式
            self.fig = plt.figure(figsize=(8, 10))
        else:
            plt.figure(self.fig.number)  # 切換到現有圖形
            plt.clf()  # 清除所有內容
        
        
        color_list = ["r", "g", "c", "m", "y", "k"]
        n_maps = self.n_agent
        rows_for_maps = 0 if n_maps == 0 else ((n_maps - 1) // 2 + 1)  # 每列2張
        base_rows = 2   # real 縱向佔2列
        gaps = rows_for_maps  # 每組之間空一列（美觀）
        total_rows = base_rows + rows_for_maps + gaps
        total_cols = 2

        # Real Map
        ax_real = plt.subplot2grid((total_rows, total_cols), (0, 0), rowspan=2, colspan=2)
        ax_real.plot(self.server.position[0], self.server.position[1], 
                    markersize=8, zorder=9999, marker="h", ls="-", 
                    c=color_list[-2], mec="black")
        
        circle = patches.Circle(
            (self.server.position[0], self.server.position[1]), 160,
            color=color_list[-2], alpha=0.3
        )
        ax_real.add_patch(circle)
        ax_real.imshow(self.real_map, cmap='gray')
        ax_real.set_title(f'Step: {step}, Real Map')
        ax_real.axis('off')

        # Global Map
        ax_env = plt.subplot2grid((total_rows, total_cols), (2, 0), rowspan=2, colspan=2)
        ax_env.plot(self.server.position[0], self.server.position[1], 
                markersize=8, zorder=9999, marker="h", ls="-", 
                c=color_list[-2], mec="black")
        ax_env.scatter(self.server.frontiers[:, 0], self.server.frontiers[:, 1], c='r', s=1)
        
        ax_env.imshow(self.real_map, cmap='gray')
        ax_env.imshow(self.server.global_map, cmap='gray', alpha=0.5)
        ax_env.set_title(f'Global Map, Coverage: {self.coverage}')
        ax_env.axis('off')

        # Robot local maps
        row_start = 4
        for i, robot in enumerate(self.robot_list):
            robot_marker_color = color_list[i % len(color_list)]
            row = row_start + int(i/2)
            
            if i % 2:
                ax = plt.subplot2grid((total_rows, total_cols), (row, 1), rowspan=1, colspan=1)
            else:
                ax = plt.subplot2grid((total_rows, total_cols), (row, 0), rowspan=1, colspan=1)
            
            # 繪製機器人位置和路徑
            ax.plot(robot.position[0], robot.position[1], markersize=6, 
                zorder=9999, marker="D", ls="-", c=robot_marker_color, mec="black")
            
            ax_real.plot(robot.position[0], robot.position[1], markersize=6, 
                        zorder=9999, marker="D", ls="-", c=robot_marker_color, mec="black")
            
            show_robot_in_server_alpha = 1 if self.server.robot_in_range[i] else 0.3
            ax_env.plot(self.server.all_robot_position[i][0], self.server.all_robot_position[i][1], 
                    markersize=6, zorder=9999, marker="D", ls="-", c=robot_marker_color, mec="black", alpha=show_robot_in_server_alpha)
            
            circle = patches.Circle(
                (robot.position[0], robot.position[1]),  # 中心座標
                80,                                      # 半徑
                color=robot_marker_color,                # 顏色可自訂
                alpha=0.1                                # 透明度，0~1
            )
            ax_real.add_patch(circle)
            ax.imshow(robot.local_map, cmap='gray')
            if np.array_equal(robot.target_pos, robot.last_position_in_server_range):
                ax.set_title(f'Robot{i+1} Local Map')
            else:
                ax.set_title(f'Robot{i+1}, out range step {robot.out_range_step}')
            ax.axis('off')
            
            ax.scatter(robot.frontiers[:, 0], robot.frontiers[:, 1], c='r', s=1)
            
            # 繪製路徑
            if hasattr(robot, 'movement_history') and len(robot.movement_history) > 1:
                history = np.array(robot.movement_history)
                ax_real.plot(history[:,0], history[:,1], '-', linewidth=2, 
                            alpha=0.7, color=robot_marker_color)
            
            if len(robot.planned_path) >= 1:
                planned_path_with_current = [robot.position.copy()] + robot.planned_path
                path = np.array(planned_path_with_current)
                if robot.target_gived_by_server:
                    ax_env.plot(path[:,0], path[:,1], '-', linewidth=2, color="k")
                    ax_env.plot(robot.target_pos[0], robot.target_pos[1], markersize=6, 
                        zorder=9999, marker="x", ls="-", c='k', mec="black")
                else:
                    ax.plot(path[:,0], path[:,1], '-', linewidth=2, color="k")
                    ax.plot(robot.target_pos[0], robot.target_pos[1], markersize=6, 
                        zorder=9999, marker="x", ls="-", c='k', mec="black")
            

        plt.draw()  # 重畫圖形
        plt.pause(0.001)  # 短暫暫停，讓圖形更新
        
    def plot_env_without_window(self, step):
        """不顯示視窗，直接保存影片幀"""
        
        # 初始化圖形（無GUI）
        if not hasattr(self, 'fig') or self.fig is None:
            plt.ioff()  # 關閉互動模式
            self.fig = plt.figure(figsize=(8, 10))
            
            # 初始化影片寫入器
            if not hasattr(self, 'frames_data'):
                self.frames_data = []
        else:
            plt.figure(self.fig.number)
            plt.clf()
        
        color_list = ["r", "g", "c", "m", "y", "k"]
        n_maps = self.n_agent
        rows_for_maps = 0 if n_maps == 0 else ((n_maps - 1) // 2 + 1)
        base_rows = 2
        gaps = rows_for_maps
        total_rows = base_rows + rows_for_maps + gaps
        total_cols = 2

        # Real Map
        ax_real = plt.subplot2grid((total_rows, total_cols), (0, 0), rowspan=2, colspan=2)
        ax_real.plot(self.server.position[0], self.server.position[1], 
                    markersize=8, zorder=9999, marker="h", ls="-", 
                    c=color_list[-2], mec="black")
        
        circle = patches.Circle(
            (self.server.position[0], self.server.position[1]), 160,
            color=color_list[-2], alpha=0.3
        )
        ax_real.add_patch(circle)
        ax_real.imshow(self.real_map, cmap='gray')
        ax_real.set_title(f'Step: {step}, Real Map')
        ax_real.axis('off')

        # Global Map
        ax_env = plt.subplot2grid((total_rows, total_cols), (2, 0), rowspan=2, colspan=2)
        ax_env.plot(self.server.position[0], self.server.position[1], 
                markersize=8, zorder=9999, marker="h", ls="-", 
                c=color_list[-2], mec="black")
        ax_env.scatter(self.server.frontiers[:, 0], self.server.frontiers[:, 1], c='r', s=1)
        
        ax_env.imshow(self.real_map, cmap='gray')
        ax_env.imshow(self.server.global_map, cmap='gray', alpha=0.5)
        ax_env.set_title(f'Global Map, Coverage: {self.coverage*100:6.2f}%')
        ax_env.axis('off')

        # Robot local maps
        row_start = 4
        for i, robot in enumerate(self.robot_list):
            robot_marker_color = color_list[i % len(color_list)]
            row = row_start + int(i/2)
            
            if i % 2:
                ax = plt.subplot2grid((total_rows, total_cols), (row, 1), rowspan=1, colspan=1)
            else:
                ax = plt.subplot2grid((total_rows, total_cols), (row, 0), rowspan=1, colspan=1)
            
            # 繪製機器人位置和路徑
            ax.plot(robot.position[0], robot.position[1], markersize=6, 
                zorder=9999, marker="D", ls="-", c=robot_marker_color, mec="black")
            
            ax_real.plot(robot.position[0], robot.position[1], markersize=6, 
                        zorder=9999, marker="D", ls="-", c=robot_marker_color, mec="black")
            
            show_robot_in_server_alpha = 1 if self.server.robot_in_range[i] else 0.3
            ax_env.plot(self.server.all_robot_position[i][0], self.server.all_robot_position[i][1], 
                    markersize=6, zorder=9999, marker="D", ls="-", c=robot_marker_color, mec="black", alpha=show_robot_in_server_alpha)
            
            circle = patches.Circle(
                (robot.position[0], robot.position[1]),  # 中心座標
                80,                                      # 半徑
                color=robot_marker_color,                # 顏色可自訂
                alpha=0.1                                # 透明度，0~1
            )
            ax_real.add_patch(circle)
            ax.imshow(robot.local_map, cmap='gray')
            # if np.array_equal(robot.target_pos, robot.last_position_in_server_range):
            #     ax.set_title(f'Robot{i+1} Local Map')
            # else:
            ax.set_title(f'Robot{i+1}, out {robot.out_range_step}, {int(robot.target_gived_by_server)}')
            # ax.set_title(f'Robot{i+1}')
            ax.axis('off')
            
            ax.scatter(robot.frontiers[:, 0], robot.frontiers[:, 1], c='r', s=1)
            
            # 繪製路徑
            if hasattr(robot, 'movement_history') and len(robot.movement_history) > 1:
                history = np.array(robot.movement_history)
                ax_real.plot(history[:,0], history[:,1], '-', linewidth=2, 
                            alpha=0.7, color=robot_marker_color)
            
            if len(robot.planned_path) >= 1:
                planned_path_with_current = [robot.position.copy()] + robot.planned_path
                path = np.array(planned_path_with_current)
                if robot.target_gived_by_server:
                    ax_env.plot(path[:,0], path[:,1], '-', linewidth=2, color="k")
                    ax_env.plot(robot.target_pos[0], robot.target_pos[1], markersize=6, 
                        zorder=9999, marker="x", ls="-", c='k', mec="black")
                else:
                    ax.plot(path[:,0], path[:,1], '-', linewidth=2, color="k")
                    ax.plot(robot.target_pos[0], robot.target_pos[1], markersize=6, 
                        zorder=9999, marker="x", ls="-", c='k', mec="black")

        plt.tight_layout()
        # 保存當前幀到記憶體
        self._save_frame_to_memory()

    def _save_frame_to_memory(self):
        """將當前圖形保存到記憶體中"""
        import io
        from PIL import Image
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        self.frames_data.append(np.array(image))
        buf.close()

    def save_video(self, filename="exploration_video.mp4", fps=5):
        """將所有幀保存為影片"""
        if not hasattr(self, 'frames_data') or not self.frames_data:
            print("No frames to save")
            return
        
        import cv2
        # 取得影片尺寸
        height, width = self.frames_data[0].shape[:2]
        
        # 創建影片寫入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        cnt = 0
        for frame in self.frames_data:
            cnt += 1
            # 轉換 RGB 到 BGR (OpenCV 格式)
            if len(frame.shape) == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            video_writer.write(frame_bgr)
        video_writer.release()
        print(f"Video saved as {filename}")
        
        # 清理記憶體
        self.frames_data = []