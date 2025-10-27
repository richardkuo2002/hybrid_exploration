from skimage import io
import os
from skimage.measure import block_reduce
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches

# 移除了 linear_sum_assignment，因為它被移到 server.py
# from scipy.optimize import linear_sum_assignment 

from robot import Robot
from server import Server
from graph_generator import Graph_generator
from sensor import *
from parameter import *

class Env():
    def __init__(self, n_agent:int, k_size=20, map_index=0, plot=True):
        self.resolution = 4
        
        self.map_path = "DungeonMaps/train/easy"
        self.map_list = os.listdir(self.map_path)
        self.map_list.sort(reverse=True)
        self.map_index = map_index % np.size(self.map_list)
        self.file_path = self.map_list[self.map_index]
        self.real_map, self.start_position = self.import_map(self.map_path + '/' + self.file_path)
        self.real_map_size = np.shape(self.real_map)
        
        # --- 初始化 Server ---
        self.server = Server(self.start_position, self.real_map_size, self.resolution, k_size, plot)
        self.server.global_map = np.ones(self.real_map_size) * 127
        
        # --- 初始化 Robots ---
        self.n_agent = n_agent
        self.robot_list:list[Robot] = []
        for i in range(self.n_agent):
            robot = Robot(self.start_position, self.real_map_size, self.resolution, k_size, plot)
            robot.robot_id = i # <--- 賦予 robot 一個 ID
            
            # 初始化機器人的 local map 和 graph
            robot.local_map = self.update_robot_local_map(robot.position, robot.sensor_range, robot.local_map, self.real_map)
            robot.downsampled_map = block_reduce(robot.local_map.copy(), block_size=(self.resolution, self.resolution), func=np.min)
            robot.frontiers = self.find_frontier(robot.downsampled_map)
            
            node_coords, graph, node_utility, guidepost = robot.graph_generator.generate_graph(self.start_position, robot.local_map, robot.frontiers)
            robot.node_coords = node_coords
            robot.local_map_graph = graph
            robot.node_utility = node_utility
            robot.guidepost = guidepost
            
            self.robot_list.append(robot)
            
            # 初始化 server 對 robot 的追蹤
            self.server.all_robot_position.append(robot.position)
            self.server.robot_in_range.append(True)
        
        # --- 第一次地圖合併 ---
        maps_to_merge = [robot.local_map for robot in self.robot_list] + [self.server.global_map]
        merged = self.merge_maps(maps_to_merge)
        
        for robot in self.robot_list:
            robot.local_map[:] = merged
        self.server.global_map[:] = merged
        
        # --- 第一次 server 圖更新 ---
        # 讓 server 在第一次執行時自己更新
        self.server.update_and_assign_tasks(self.robot_list, self.real_map, self.find_frontier)
        
        
    # <--- 修改點：移除了 step 函式 --- >
    
    # <--- 修改點：移除了 server_step 函式 --- >

    # <--- 修改點：移除了 _filter_occupied_targets 函式 --- >
        
    def calculate_coverage_ratio(self):
        """計算地圖探索覆蓋率 (基於 server 的全局地圖)"""
        explored_pixels = np.sum(self.server.global_map == 255)
        total_free_pixels = np.sum(self.real_map == 255)
        
        return min(explored_pixels / total_free_pixels, 1.0) if total_free_pixels > 0 else 0.0

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
        (這個函式保持在 Env 中，因為它代表了物理世界的感測)
        """
        robot_local_map = sensor_work(robot_position, sensor_range, robot_local_map, real_map)
        return robot_local_map
    
    def find_frontier(self, downsampled_map):
        """
        在當前地圖（downsampled_map）中找出「探索邊界」（Frontiers）
        (這個函式保持在 Env 中，作為一個地圖處理工具)
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
    
    # <--- 修改點：移除了 plan_local_path 函式 --- >
    
    # <--- 修改點：移除了 plan_global_path 函式 --- >
    
    def import_map(self, map_path):
        """
        匯入地圖檔案，並解析障礙物、自由空間與機器人當前位置。
        (保持不變)
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
    
    # ... plot_env 和 plot_env_without_window ...
    # (這兩個繪圖函式保持不變，它們需要 Env 中的所有物件)
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

        # Real
        ax_real = plt.subplot2grid((total_rows, total_cols), (0, 0), rowspan=2, colspan=2)
        ax_real.plot(self.server.position[0], self.server.position[1], markersize=8, zorder=9999, marker="h", ls="-", c=color_list[-2], mec="black")
        circle = patches.Circle(
            (self.server.position[0], self.server.position[1]),  # 中心座標
            160,                                                 # 半徑
            color=color_list[-2],                                # 顏色可自訂
            alpha=0.1                                            # 透明度，0~1
        )
        ax_real.add_patch(circle)
        ax_real.imshow(self.real_map, cmap='gray')
        ax_real.set_title(f'Step: {step}, Real Map')
        ax_real.axis('off')

        # Global Map 
        ax_env = plt.subplot2grid((total_rows, total_cols), (2, 0), rowspan=2, colspan=2)
        ax_env.plot(self.server.position[0], self.server.position[1], markersize=8, zorder=9999, marker="h", ls="-", c=color_list[-2], mec="black")
        ax_env.imshow(self.real_map, cmap='gray')
        ax_env.imshow(self.server.global_map, cmap='gray', alpha=0.5)
        ax_env.set_title('Global Map')
        ax_env.axis('off')
        
        
        
        # frontiers = getattr(self, "global_map_frontiers", None)

        # if frontiers is not None and len(frontiers) > 0:
        #     xs = frontiers[:, 0]
        #     ys = frontiers[:, 1]

        #     # 用紅點疊加（s=2 可視需要調整大小，zorder 提升圖層）
        #     ax_env.scatter(xs, ys, c='r', s=2, zorder=5)

        # Robot local maps：從第3列開始（row=2），每列最多2張，下一組往下跳一列間隔
        
        row_start = 4
        for i, robot in enumerate(self.robot_list):
            robot_marker_color = color_list[i % len(color_list)]
            row = row_start + int(i/2)  # 佔一列，下一列空白
            if i % 2:
                ax = plt.subplot2grid((total_rows, total_cols), (row, 1), rowspan=1, colspan=1)
            else:
                ax = plt.subplot2grid((total_rows, total_cols), (row, 0), rowspan=1, colspan=1)
            
            ax.plot(robot.position[0], robot.position[1], markersize=6, zorder=9999, marker="D", ls="-", c=robot_marker_color, mec="black")
            ax_real.plot(robot.position[0], robot.position[1], markersize=6, zorder=9999, marker="D", ls="-", c=robot_marker_color, mec="black")
            show_robot_in_server_alpha = 1 if self.server.robot_in_range[i] else 0.3
            ax_env.plot(self.server.all_robot_position[i][0], self.server.all_robot_position[i][1], 
                    markersize=6, zorder=9999, marker="D", ls="-", c=robot_marker_color, mec="black", alpha=show_robot_in_server_alpha)
            
            circle = patches.Circle(
                (robot.position[0], robot.position[1]),  # 中心座標
                80,                                                 # 半徑
                color=robot_marker_color,                                # 顏色可自訂
                alpha=0.1                                            # 透明度，0~1
            )
            ax_real.add_patch(circle)
            ax.imshow(robot.local_map, cmap='gray')
            ax.set_title(f'Robot{i+1} Local Map')
            ax.axis('off')
            
            ax.scatter(robot.node_coords[:, 0], robot.node_coords[:, 1], s=0.5, c='b')  # grid pattern
            ax.scatter(robot.frontiers[:, 0], robot.frontiers[:, 1], c='r', s=1)
            
            # 過去走過的路徑
            if hasattr(robot, 'movement_history') and len(robot.movement_history) > 1:
                history = np.array(robot.movement_history)
                ax_real.plot(history[:,0], history[:,1], '-', linewidth=2, 
                        alpha=0.7, color=robot_marker_color, 
                        label=f'Robot{i+1} history')
                
                # 在軌跡上標記一些點顯示方向
                # if len(history) > 5:
                #     step_size = max(1, len(history) // 10)
                #     for j in range(0, len(history), step_size):
                #         ax.plot(history[j,0], history[j,1], '-', 
                #                 markersize=3, color=robot_marker_color, alpha=0.5)
            
            # 預計路徑
            if  len(robot.planned_path) >= 1:
                # 先將目前位置加入路徑的起點
                planned_path_with_current = [robot.position.copy()] + robot.planned_path
                path = np.array(planned_path_with_current)
                ax.plot(path[:,0], path[:,1], '-', linewidth=2, 
                        color="k")
                ax.plot(robot.target_pos[0], robot.target_pos[1], markersize=6, zorder=9999, marker="x", ls="-", c='k', mec="black")
            # # 畫機器人當前位置
            # plt.scatter(robot.position[0], robot.position[1], 
            #             c=f'C{i}', edgecolors='k', zorder=5)
            

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
        ax_env.set_title('Global Map')
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
                80,                                                 # 半徑
                color=robot_marker_color,                                # 顏色可自訂
                alpha=0.1                                            # 透明度，0~1
            )
            ax_real.add_patch(circle)
            ax.imshow(robot.local_map, cmap='gray')
            ax.set_title(f'Robot{i+1} Local Map')
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
        
        for frame in self.frames_data:
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