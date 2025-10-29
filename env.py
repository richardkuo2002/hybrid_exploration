from skimage import io
import os
from skimage.measure import block_reduce
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
import logging # <--- 匯入 logging (如果 Env 本身需要 log)

from robot import Robot
from server import Server
from graph_generator import Graph_generator
from sensor import *
from parameter import *

logger = logging.getLogger(__name__) # <--- 獲取 logger

class Env():
    # <--- 移除 debug 參數 ---
    def __init__(self, n_agent:int, k_size=20, map_index=0, plot=True):
        self.resolution = 4
        # ... (讀取地圖等不變) ...
        # self.map_path = "DungeonMaps/train/easy"
        self.map_path = "map_train"
        self.map_list = os.listdir(self.map_path)
        self.map_list.sort(reverse=True)
        self.map_index = map_index % np.size(self.map_list)
        self.file_path = self.map_list[self.map_index]

        try: # <--- 增加檔案讀取錯誤處理 ---
             self.real_map, self.start_position = self.import_map(self.map_path + '/' + self.file_path)
        except Exception as e:
             logger.error(f"Failed to load map: {self.map_path + '/' + self.file_path}. Error: {e}")
             # 可以選擇退出或使用預設地圖
             raise # 重新拋出錯誤，讓程式停止

        self.real_map_size = np.shape(self.real_map)

        # --- 初始化 Server (移除 debug 參數) ---
        self.server = Server(self.start_position, self.real_map_size, self.resolution, k_size, plot)
        # self.server.global_map = np.ones(self.real_map_size) * 127 # Server 內部會初始化

        # --- 初始化 Robots (移除 debug 參數) ---
        self.n_agent = n_agent
        self.robot_list:list[Robot] = []
        for i in range(self.n_agent):
            robot = Robot(self.start_position, self.real_map_size, self.resolution, k_size, plot) # 移除 debug
            robot.robot_id = i

            try: # <--- 增加 robot 初始化錯誤處理 ---
                robot.local_map = self.update_robot_local_map(robot.position, robot.sensor_range, robot.local_map, self.real_map)
                robot.downsampled_map = block_reduce(robot.local_map.copy(), block_size=(self.resolution, self.resolution), func=np.min)
                robot.frontiers = self.find_frontier(robot.downsampled_map)

                # 確保 generate_graph 被呼叫
                if hasattr(robot, 'graph_generator') and robot.graph_generator is not None:
                     node_coords, graph, node_utility, guidepost = robot.graph_generator.generate_graph(self.start_position, robot.local_map, robot.frontiers)
                     robot.node_coords = node_coords
                     robot.local_map_graph = graph
                     robot.node_utility = node_utility
                     robot.guidepost = guidepost
                else:
                     logger.error(f"Robot {i} failed to initialize graph_generator.")
                     # 可能需要退出或採取其他措施

                self.robot_list.append(robot)

                self.server.all_robot_position.append(robot.position)
                self.server.robot_in_range.append(True)
            except Exception as e:
                 logger.error(f"Failed to initialize Robot {i}. Error: {e}")
                 # 可能需要跳過這個 robot 或退出

        # --- 第一次地圖合併 ---
        maps_to_merge = [robot.local_map for robot in self.robot_list] + [self.server.global_map]
        merged = self.merge_maps(maps_to_merge)

        for robot in self.robot_list: robot.local_map[:] = merged
        self.server.global_map[:] = merged

        # --- 第一次 server 圖更新 ---
        try: # <--- 增加錯誤處理 ---
            self.server.update_and_assign_tasks(self.robot_list, self.real_map, self.find_frontier)
        except Exception as e:
             logger.error(f"Initial server update failed. Error: {e}")
             raise


    # ... (其他函式 calculate_coverage_ratio, merge_maps, etc. 保持不變) ...
    def calculate_coverage_ratio(self):
        explored_pixels = np.sum(self.server.global_map == 255)
        total_free_pixels = np.sum(self.real_map == 255)
        return min(explored_pixels / total_free_pixels, 1.0) if total_free_pixels > 0 else 0.0

    def merge_maps(self, maps_to_merge):
        merged_map = np.ones_like(self.real_map) * 127
        for belief in maps_to_merge:
            merged_map[belief == 1] = 1
            merged_map[belief == 255] = 255
        return merged_map

    def update_robot_local_map(self, robot_position, sensor_range, robot_local_map, real_map):
        try: # <--- 增加 sensor_work 錯誤處理 ---
             robot_local_map = sensor_work(robot_position, sensor_range, robot_local_map, real_map)
        except Exception as e:
             logger.error(f"Error in sensor_work for position {robot_position}. Error: {e}")
             # 可以選擇返回原始地圖或拋出錯誤
        return robot_local_map

    def find_frontier(self, downsampled_map):
        # ... (保持不變) ...
        try: # <--- 增加 find_frontier 錯誤處理 ---
            y_len, x_len = downsampled_map.shape[:2] # 確保是 2D
            mapping = (downsampled_map == 127).astype(np.int8) # 使用 int8 節省記憶體
            mapping = np.pad(mapping, 1, 'constant', constant_values=0)
            fro_map = mapping[2:, 1:-1] + mapping[:-2, 1:-1] + mapping[1:-1, 2:] + \
                      mapping[1:-1, :-2] + mapping[:-2, 2:] + mapping[2:, :-2] + mapping[2:, 2:] + \
                      mapping[:-2, :-2]

            belief = downsampled_map # 直接使用，不 copy

            # 使用 boolean indexing 加速
            is_free = (belief == 255)
            is_frontier_neighbor = (fro_map > 0) & (fro_map < 8) # 鄰居有未知

            frontier_mask = is_free & is_frontier_neighbor
            ind_to = np.where(frontier_mask.ravel(order='F'))[0]

            if ind_to.size == 0:
                 return np.array([]).reshape(0, 2) # 返回空陣列

            # 簡化座標生成
            rows, cols = np.indices((y_len, x_len))
            points = np.stack([cols.ravel(order='F'), rows.ravel(order='F')], axis=-1)

            f = points[ind_to] * self.resolution # 直接乘以 resolution
            return f.astype(int) # 確保是整數
        except Exception as e:
            logger.error(f"Error in find_frontier. Input shape: {downsampled_map.shape}. Error: {e}")
            return np.array([]).reshape(0, 2) # 返回空陣列


    def import_map(self, map_path):
        # ... (保持不變, 但之前的修正要保留) ...
        map_img = io.imread(map_path, as_gray=True) # 使用 as_gray 讀取灰度圖
        # 閾值化 (假設白色 > 0.5 是自由空間)
        map_data = (map_img > 0.5).astype(np.uint8) # 1 = free, 0 = obstacle/unknown

        start_location = None
        # 尋找特殊顏色 (假設起始點像素值在 0.8 附近，白色是 1)
        start_points = np.where((map_img > 0.79) & (map_img < 0.81))

        if start_points[0].size > 0:
            mid_idx = start_points[0].size // 2
            start_location = np.array([start_points[1][mid_idx], start_points[0][mid_idx]]) # [x, y]
            map_data[start_points] = 1 # 將起始點設為 free
        else:
            logger.warning(f"Start position (pixel ~0.8) not found in {map_path}. Using default [100, 100].")
            start_location = np.array([100, 100])

        # 轉換為 project 使用的數值: 1=obstacle, 255=free, 127=unknown
        # 由於讀取的是二值圖，先假設沒有 unknown
        final_map = np.where(map_data == 0, 1, 255).astype(np.uint8)

        return final_map, start_location


    # ... plot_env 和 plot_env_without_window ...
    # (保持不變)
    def plot_env(self, step):
        if not hasattr(self, 'fig') or self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(8, 10))
            if not hasattr(self, 'frames_data'): self.frames_data = []
        else:
            plt.figure(self.fig.number); plt.clf()

        color_list = ["r", "g", "c", "m", "y", "k"]
        n_maps = self.n_agent
        rows_for_maps = 0 if n_maps == 0 else ((n_maps - 1) // 2 + 1)
        base_rows = 2; gaps = rows_for_maps
        total_rows = base_rows + rows_for_maps + gaps; total_cols = 2

        # Real Map
        ax_real = plt.subplot2grid((total_rows, total_cols), (0, 0), rowspan=2, colspan=2)
        ax_real.plot(self.server.position[0], self.server.position[1], markersize=8, zorder=999, marker="h", ls="-", c=color_list[-2], mec="black")
        circle = patches.Circle(self.server.position, 160, color=color_list[-2], alpha=0.1)
        ax_real.add_patch(circle)
        ax_real.imshow(self.real_map, cmap='gray')
        ax_real.set_title(f'Step: {step}, Real Map'); ax_real.axis('off')

        # Global Map
        ax_env = plt.subplot2grid((total_rows, total_cols), (2, 0), rowspan=2, colspan=2)
        ax_env.plot(self.server.position[0], self.server.position[1], markersize=8, zorder=999, marker="h", ls="-", c=color_list[-2], mec="black")
        ax_env.imshow(self.real_map, cmap='gray')
        ax_env.imshow(self.server.global_map, cmap='gray', alpha=0.5)
        ax_env.set_title('Global Map'); ax_env.axis('off')

        # Plot server frontiers if they exist
        if self.server.frontiers is not None and len(self.server.frontiers) > 0:
             ax_env.scatter(self.server.frontiers[:, 0], self.server.frontiers[:, 1], c='lime', s=1, zorder=5) # 用亮綠色

        # Robot local maps
        row_start = 4
        for i, robot in enumerate(self.robot_list):
            robot_marker_color = color_list[i % len(color_list)]
            row = row_start + (i // 2) # 修正 row 計算
            col = i % 2 # 修正 col 計算
            ax = plt.subplot2grid((total_rows, total_cols), (row, col), rowspan=1, colspan=1)

            # Robot Position on all maps
            if robot.position is not None:
                ax.plot(robot.position[0], robot.position[1], markersize=6, zorder=999, marker="D", ls="-", c=robot_marker_color, mec="black")
                ax_real.plot(robot.position[0], robot.position[1], markersize=6, zorder=999, marker="D", ls="-", c=robot_marker_color, mec="black")
                # Plot server's record of robot position
                if i < len(self.server.all_robot_position) and self.server.all_robot_position[i] is not None:
                     show_alpha = 1.0 if i < len(self.server.robot_in_range) and self.server.robot_in_range[i] else 0.3
                     ax_env.plot(self.server.all_robot_position[i][0], self.server.all_robot_position[i][1],
                            markersize=6, zorder=999, marker="D", ls="-", c=robot_marker_color, mec="black", alpha=show_alpha)

                # Robot Comm Range Circle on Real Map
                circle_comm = patches.Circle(robot.position, 80, color=robot_marker_color, alpha=0.1)
                ax_real.add_patch(circle_comm)

            # Local Map Display
            ax.imshow(robot.local_map, cmap='gray')
            ax.set_title(f'Robot{robot.robot_id+1} Local Map'); ax.axis('off') # 使用 robot_id

            # Plot local nodes and frontiers
            if robot.node_coords is not None and len(robot.node_coords) > 0:
                ax.scatter(robot.node_coords[:, 0], robot.node_coords[:, 1], s=0.5, c='blue', zorder=2)
            if robot.frontiers is not None and len(robot.frontiers) > 0:
                ax.scatter(robot.frontiers[:, 0], robot.frontiers[:, 1], c='red', s=1, zorder=3)

            # Plot history on Real Map
            if hasattr(robot, 'movement_history') and len(robot.movement_history) > 1:
                history = np.array(robot.movement_history)
                ax_real.plot(history[:,0], history[:,1], '-', linewidth=1, alpha=0.6, color=robot_marker_color)

            # Plot planned path on Local Map
            if robot.target_pos is not None and robot.planned_path and len(robot.planned_path) >= 1:
                try: # 加 try-except 保護繪圖
                     planned_path_with_current = [robot.position.copy()] + robot.planned_path
                     path = np.array(planned_path_with_current)
                     ax.plot(path[:,0], path[:,1], 'k--', linewidth=1, zorder=4) # 虛線
                     ax.plot(robot.target_pos[0], robot.target_pos[1], markersize=6, zorder=999, marker="x", ls="-", c='black', mec="black")
                except Exception as plot_err:
                     logger.warning(f"Plotting path failed for Robot {robot.robot_id}: {plot_err}")


        plt.tight_layout()
        self._save_frame_to_memory()
        plt.draw(); plt.pause(0.001)

    def plot_env_without_window(self, step):
        # ... (繪圖邏輯與 plot_env 幾乎相同，只是沒有 plt.ion(), plt.draw(), plt.pause()) ...
        if not hasattr(self, 'fig') or self.fig is None:
            plt.ioff(); self.fig = plt.figure(figsize=(8, 10))
            if not hasattr(self, 'frames_data'): self.frames_data = []
        else:
            plt.figure(self.fig.number); plt.clf()

        color_list = ["r", "g", "c", "m", "y", "k"]
        n_maps = self.n_agent
        rows_for_maps = 0 if n_maps == 0 else ((n_maps - 1) // 2 + 1)
        base_rows = 2; gaps = rows_for_maps
        total_rows = base_rows + rows_for_maps + gaps; total_cols = 2

        # Real Map
        ax_real = plt.subplot2grid((total_rows, total_cols), (0, 0), rowspan=2, colspan=2)
        ax_real.plot(self.server.position[0], self.server.position[1], markersize=8, zorder=999, marker="h", ls="-", c=color_list[-2], mec="black")
        circle = patches.Circle(self.server.position, 160, color=color_list[-2], alpha=0.1)
        ax_real.add_patch(circle)
        ax_real.imshow(self.real_map, cmap='gray')
        ax_real.set_title(f'Step: {step}, Real Map'); ax_real.axis('off')

        # Global Map
        ax_env = plt.subplot2grid((total_rows, total_cols), (2, 0), rowspan=2, colspan=2)
        ax_env.plot(self.server.position[0], self.server.position[1], markersize=8, zorder=999, marker="h", ls="-", c=color_list[-2], mec="black")
        ax_env.imshow(self.real_map, cmap='gray')
        ax_env.imshow(self.server.global_map, cmap='gray', alpha=0.5)
        ax_env.set_title('Global Map'); ax_env.axis('off')
        if self.server.frontiers is not None and len(self.server.frontiers) > 0:
             ax_env.scatter(self.server.frontiers[:, 0], self.server.frontiers[:, 1], c='lime', s=1, zorder=5)

        # Robot local maps
        row_start = 4
        for i, robot in enumerate(self.robot_list):
            robot_marker_color = color_list[i % len(color_list)]
            row = row_start + (i // 2); col = i % 2
            ax = plt.subplot2grid((total_rows, total_cols), (row, col), rowspan=1, colspan=1)
            if robot.position is not None:
                ax.plot(robot.position[0], robot.position[1], markersize=6, zorder=999, marker="D", ls="-", c=robot_marker_color, mec="black")
                ax_real.plot(robot.position[0], robot.position[1], markersize=6, zorder=999, marker="D", ls="-", c=robot_marker_color, mec="black")
                if i < len(self.server.all_robot_position) and self.server.all_robot_position[i] is not None:
                     show_alpha = 1.0 if i < len(self.server.robot_in_range) and self.server.robot_in_range[i] else 0.3
                     ax_env.plot(self.server.all_robot_position[i][0], self.server.all_robot_position[i][1],
                            markersize=6, zorder=999, marker="D", ls="-", c=robot_marker_color, mec="black", alpha=show_alpha)
                circle_comm = patches.Circle(robot.position, 80, color=robot_marker_color, alpha=0.1)
                ax_real.add_patch(circle_comm)
            ax.imshow(robot.local_map, cmap='gray')
            ax.set_title(f'Robot{robot.robot_id+1} Local Map'); ax.axis('off')
            if robot.node_coords is not None and len(robot.node_coords) > 0:
                ax.scatter(robot.node_coords[:, 0], robot.node_coords[:, 1], s=0.5, c='blue', zorder=2)
            if robot.frontiers is not None and len(robot.frontiers) > 0:
                ax.scatter(robot.frontiers[:, 0], robot.frontiers[:, 1], c='red', s=1, zorder=3)
            if hasattr(robot, 'movement_history') and len(robot.movement_history) > 1:
                history = np.array(robot.movement_history)
                ax_real.plot(history[:,0], history[:,1], '-', linewidth=1, alpha=0.6, color=robot_marker_color)
            if robot.target_pos is not None and robot.planned_path and len(robot.planned_path) >= 1:
                 try:
                     planned_path_with_current = [robot.position.copy()] + robot.planned_path
                     path = np.array(planned_path_with_current)
                     ax.plot(path[:,0], path[:,1], 'k--', linewidth=1, zorder=4)
                     ax.plot(robot.target_pos[0], robot.target_pos[1], markersize=6, zorder=999, marker="x", ls="-", c='black', mec="black")
                 except Exception as plot_err: pass #忽略繪圖錯誤

        plt.tight_layout()
        self._save_frame_to_memory()

    def _save_frame_to_memory(self):
        # ... (保持不變) ...
        import io
        from PIL import Image
        if not hasattr(self, 'frames_data'): self.frames_data = []
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        try: # 增加 PIL 錯誤處理
            image = Image.open(buf)
            self.frames_data.append(np.array(image))
        except Exception as e:
             logger.error(f"Error saving frame to memory: {e}")
        finally:
            buf.close()


    def save_video(self, filename="exploration_video.mp4", fps=5):
        # ... (保持不變) ...
        if not hasattr(self, 'frames_data') or not self.frames_data:
            logger.warning("No frames captured, skipping video save.")
            return
        import cv2
        height, width = self.frames_data[0].shape[:2]
        try: # 增加 OpenCV 錯誤處理
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            if not video_writer.isOpened():
                 logger.error(f"Failed to open video writer for {filename}")
                 return

            for frame in self.frames_data:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if len(frame.shape) == 3 else frame
                video_writer.write(frame_bgr)
            video_writer.release()
            logger.info(f"Video saved as {filename}")
        except Exception as e:
             logger.error(f"Error saving video {filename}: {e}")
        finally:
             self.frames_data = [] # 確保清空