from skimage import io
import os
from skimage.measure import block_reduce
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
import logging # <--- 確認 logging 已匯入

from robot import Robot
from server import Server
from graph_generator import Graph_generator
from sensor import *
from parameter import *

logger = logging.getLogger(__name__)

class Env():
    def __init__(self, n_agent:int, k_size=20, map_index=0, plot=True): # Removed debug
        self.resolution = 4
        self.map_path = "DungeonMaps/train/easy" # 確保路徑正確
        self.map_list = os.listdir(self.map_path)
        # --- 修改點：使用正向排序 ---
        self.map_list.sort() # 正向排序，讓 index 0 對應 img_0 或 img_1
        # --- ---
        # 排除非圖片檔案，例如 .DS_Store
        self.map_list = [f for f in self.map_list if f.lower().endswith('.png') or f.lower().endswith('.jpg')]
        if not self.map_list:
             logger.critical(f"No valid map files found in {self.map_path}. Exiting.")
             sys.exit(1) # 沒有地圖無法執行

        self.map_index = map_index % len(self.map_list) # 使用 len()
        self.file_path = self.map_list[self.map_index]
        logger.info(f"Using map index {self.map_index}: {self.file_path}")

        try:
             # --- 修改點：呼叫新的 import_map ---
             self.real_map, self.start_position = self.import_map_revised(os.path.join(self.map_path, self.file_path))
             # --- ---
        except Exception as e:
             logger.critical(f"Failed to load map: {os.path.join(self.map_path, self.file_path)}. Error: {e}", exc_info=True)
             raise

        self.real_map_size = np.shape(self.real_map)

        # --- 初始化 Server (移除 debug) ---
        self.server = Server(self.start_position, self.real_map_size, self.resolution, k_size, plot)
        # Server 內部會初始化 global_map

        # --- 初始化 Robots (移除 debug) ---
        self.n_agent = n_agent
        self.robot_list:list[Robot] = []
        # --- 修改點：確保 Server 的列表在 Robot 初始化前已創建 ---
        # （Server 的 __init__ 已經處理了）
        # --- ---
        for i in range(self.n_agent):
            # 傳遞 plot 給 Robot
            robot = Robot(self.start_position, self.real_map_size, self.resolution, k_size, plot=plot)
            robot.robot_id = i

            try:
                robot.local_map = self.update_robot_local_map(robot.position, robot.sensor_range, robot.local_map, self.real_map)
                robot.downsampled_map = block_reduce(robot.local_map.copy(), block_size=(self.resolution, self.resolution), func=np.min)
                robot.frontiers = self.find_frontier(robot.downsampled_map)

                if hasattr(robot, 'graph_generator') and robot.graph_generator is not None:
                     # 確保傳遞有效的 frontiers
                     valid_frontiers = robot.frontiers if robot.frontiers is not None else np.array([]).reshape(0,2)
                     node_coords, graph, node_utility, guidepost = robot.graph_generator.generate_graph(self.start_position, robot.local_map, valid_frontiers)
                     robot.node_coords = node_coords
                     robot.local_map_graph = graph
                     robot.node_utility = node_utility
                     robot.guidepost = guidepost
                else:
                     logger.error(f"Robot {i} failed graph_generator init.")

                self.robot_list.append(robot)

                # --- 修改點：檢查 Server 屬性是否存在 ---
                if hasattr(self.server, 'all_robot_position') and hasattr(self.server, 'robot_in_range'):
                    self.server.all_robot_position.append(robot.position)
                    self.server.robot_in_range.append(True)
                else:
                     logger.error("Server object missing required attributes during robot init!")
            except Exception as e:
                 logger.error(f"Failed init Robot {i}: {e}", exc_info=True)


        # --- 第一次地圖合併 ---
        maps_to_merge = [robot.local_map for robot in self.robot_list] + [self.server.global_map]
        merged = self.merge_maps(maps_to_merge)

        for robot in self.robot_list: robot.local_map[:] = merged
        self.server.global_map[:] = merged

        # --- 第一次 server 圖更新 ---
        try:
            # 確保 robot_list 非空
            if self.robot_list:
                self.server.update_and_assign_tasks(self.robot_list, self.real_map, self.find_frontier)
            else:
                 logger.warning("No robots initialized, skipping initial server update.")
        except Exception as e:
             logger.critical(f"Initial server update failed: {e}", exc_info=True)
             raise

    # ... (calculate_coverage_ratio, merge_maps, update_robot_local_map, find_frontier 不變) ...
    def calculate_coverage_ratio(self):
        # ... (同前) ...
        explored_pixels = np.sum(self.server.global_map == 255)
        total_free_pixels = np.sum(self.real_map == 255)
        return min(explored_pixels / total_free_pixels, 1.0) if total_free_pixels > 0 else 0.0

    def merge_maps(self, maps_to_merge):
        # ... (同前) ...
        merged_map = np.ones_like(self.real_map) * 127
        valid_maps = [m for m in maps_to_merge if isinstance(m, np.ndarray)] # 過濾 None
        if not valid_maps: return merged_map # 如果沒有有效 map，返回未知
        for belief in valid_maps:
            merged_map[belief == 1] = 1
            merged_map[belief == 255] = 255
        return merged_map

    def update_robot_local_map(self, robot_position, sensor_range, robot_local_map, real_map):
        # ... (同前) ...
        try:
             # 確保 sensor_work 返回的是 numpy array
             updated_map = sensor_work(robot_position, sensor_range, robot_local_map, real_map)
             if isinstance(updated_map, np.ndarray):
                  return updated_map
             else:
                  logger.error(f"sensor_work did not return a numpy array! Got: {type(updated_map)}")
                  return robot_local_map # 返回原始地圖
        except Exception as e:
             logger.error(f"Error in sensor_work @ {robot_position}: {e}", exc_info=True)
             return robot_local_map # 返回原始地圖

    def find_frontier(self, downsampled_map):
        # ... (同前) ...
        try:
            if downsampled_map is None or downsampled_map.ndim != 2:
                 logger.warning("Invalid downsampled_map in find_frontier.")
                 return np.array([]).reshape(0, 2)
            y_len, x_len = downsampled_map.shape[:2]
            mapping = (downsampled_map == 127).astype(np.int8)
            mapping = np.pad(mapping, 1, 'constant', constant_values=0)
            fro_map = mapping[2:, 1:-1] + mapping[:-2, 1:-1] + mapping[1:-1, 2:] + \
                      mapping[1:-1, :-2] + mapping[:-2, 2:] + mapping[2:, :-2] + mapping[2:, 2:] + \
                      mapping[:-2, :-2]
            belief = downsampled_map
            is_free = (belief == 255); is_frontier_neighbor = (fro_map > 0) & (fro_map < 8)
            frontier_mask = is_free & is_frontier_neighbor
            ind_to = np.where(frontier_mask.ravel(order='F'))[0]
            if ind_to.size == 0: return np.array([]).reshape(0, 2)
            rows, cols = np.indices((y_len, x_len))
            points = np.stack([cols.ravel(order='F'), rows.ravel(order='F')], axis=-1)
            f = points[ind_to] * self.resolution
            return f.astype(int)
        except Exception as e:
            logger.error(f"Error in find_frontier: {e}", exc_info=True)
            return np.array([]).reshape(0, 2)


    # <--- 修改點：使用新的 import_map_revised ---
    def import_map_revised(self, map_path):
        """
        更健壯的地圖讀取函式，基於原始邏輯 (像素值 > 150 free, 208 start)。
        """
        try:
            # 讀取原始像素值 (灰階)
            map_img_gray = io.imread(map_path, as_gray=True)

            # skimage 讀取的灰階圖可能是 float [0, 1] 或 int [0, 255]
            # 將 float 轉換為 int [0, 255]
            if map_img_gray.dtype == float:
                 map_img_int = (map_img_gray * 255).astype(np.uint8)
            else:
                 map_img_int = map_img_gray.astype(np.uint8)


            # 尋找起始點 (接近 208)
            start_points = np.where(map_img_int == 208) # 直接找 208
            start_location = None
            if start_points[0].size > 0:
                mid_idx = start_points[0].size // 2
                start_location = np.array([start_points[1][mid_idx], start_points[0][mid_idx]]) # [x, y]
                logger.info(f"Start position (208) found at {start_location} in {os.path.basename(map_path)}")
            else:
                logger.warning(f"Start position (pixel 208) not found in {os.path.basename(map_path)}. Using default [100, 100].")
                start_location = np.array([100, 100])

            # 轉換為 project 格式: 1=obstacle, 255=free
            # 閾值 > 150 為 free (包括 208)
            final_map = np.ones_like(map_img_int, dtype=np.uint8) * 1 # 預設為障礙物
            final_map[map_img_int > 150] = 255 # 大於 150 設為自由

            # 確保起始點是自由空間 (即使它不在 > 150 範圍內)
            if start_location is not None:
                 y, x = start_location[1], start_location[0]
                 if 0 <= y < final_map.shape[0] and 0 <= x < final_map.shape[1]:
                      final_map[y, x] = 255

            # 檢查自由空間比例
            free_ratio = np.sum(final_map == 255) / final_map.size
            logger.debug(f"Loaded map {os.path.basename(map_path)}. Shape: {final_map.shape}. Free ratio: {free_ratio:.2f}")
            if free_ratio < 0.01:
                 logger.warning(f"Map {os.path.basename(map_path)} has very little free space ({free_ratio*100:.1f}%).")

            return final_map, start_location

        except FileNotFoundError:
            logger.error(f"Map file not found: {map_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading/processing map {map_path}: {e}", exc_info=True)
            raise

    # 移除舊的 import_map
    # def import_map(self, map_path): ...

    # ... plot_env 和 plot_env_without_window ...
    # (保持不變)
    def plot_env(self, step):
        if not hasattr(self, 'fig') or self.fig is None:
            try: # 處理 MacOS Tkinter 後端問題
                plt.switch_backend('agg') # 先切換到非 GUI 後端
                plt.figure() # 創建一個 figure 實例
                plt.close() # 關閉它
                plt.switch_backend('tkagg') # 切換回 TkAgg (或其他 GUI 後端)
            except Exception as e:
                logger.warning(f"Failed to switch matplotlib backend: {e}")

            plt.ion(); self.fig = plt.figure(figsize=(8, 10))
            if not hasattr(self, 'frames_data'): self.frames_data = []
        else:
            plt.figure(self.fig.number); plt.clf()

        color_list = ["r", "g", "c", "m", "y", "k"]
        n_maps = self.n_agent
        rows_for_maps = 0 if n_maps == 0 else ((n_maps - 1) // 2 + 1)
        base_rows = 2; gaps = 1 # 減少 gap
        total_rows = base_rows + rows_for_maps + gaps; total_cols = 2

        # Real Map
        ax_real = plt.subplot2grid((total_rows, total_cols), (0, 0), rowspan=2, colspan=2)
        if self.server.position is not None: ax_real.plot(self.server.position[0], self.server.position[1], markersize=8, zorder=999, marker="h", ls="-", c=color_list[-2], mec="black")
        circle = patches.Circle(self.server.position, SERVER_COMM_RANGE, color=color_list[-2], alpha=0.1) # 使用參數
        ax_real.add_patch(circle)
        ax_real.imshow(self.real_map, cmap='gray')
        ax_real.set_title(f'Step: {step}, Real Map'); ax_real.axis('off')

        # Global Map
        ax_env = plt.subplot2grid((total_rows, total_cols), (2, 0), rowspan=2, colspan=2)
        if self.server.position is not None: ax_env.plot(self.server.position[0], self.server.position[1], markersize=8, zorder=999, marker="h", ls="-", c=color_list[-2], mec="black")
        ax_env.imshow(self.real_map, cmap='gray')
        ax_env.imshow(self.server.global_map, cmap='gray', alpha=0.5)
        ax_env.set_title('Global Map'); ax_env.axis('off')
        if self.server.frontiers is not None and len(self.server.frontiers) > 0: ax_env.scatter(self.server.frontiers[:, 0], self.server.frontiers[:, 1], c='lime', s=1, zorder=5)

        # Robot local maps
        row_start = 4 # 起始行
        for i, robot in enumerate(self.robot_list):
            robot_marker_color = color_list[i % len(color_list)]
            row = row_start + (i // 2)
            col = i % 2
            # 邊界檢查，避免 subplot index 超出
            if row >= total_rows: continue
            ax = plt.subplot2grid((total_rows, total_cols), (row, col), rowspan=1, colspan=1)

            if robot.position is not None:
                ax.plot(robot.position[0], robot.position[1], markersize=6, zorder=999, marker="D", ls="-", c=robot_marker_color, mec="black")
                ax_real.plot(robot.position[0], robot.position[1], markersize=6, zorder=999, marker="D", ls="-", c=robot_marker_color, mec="black")
                if i < len(self.server.all_robot_position) and self.server.all_robot_position[i] is not None:
                     show_alpha = 1.0 if i < len(self.server.robot_in_range) and self.server.robot_in_range[i] else 0.3
                     ax_env.plot(self.server.all_robot_position[i][0], self.server.all_robot_position[i][1], markersize=6, zorder=999, marker="D", ls="-", c=robot_marker_color, mec="black", alpha=show_alpha)
                circle_comm = patches.Circle(robot.position, ROBOT_COMM_RANGE, color=robot_marker_color, alpha=0.05) # 更淡
                ax_real.add_patch(circle_comm)

            ax.imshow(robot.local_map, cmap='gray')
            ax.set_title(f'Robot{robot.robot_id+1} Local Map'); ax.axis('off')

            if robot.node_coords is not None and len(robot.node_coords) > 0: ax.scatter(robot.node_coords[:, 0], robot.node_coords[:, 1], s=0.5, c='blue', zorder=2)
            if robot.frontiers is not None and len(robot.frontiers) > 0: ax.scatter(robot.frontiers[:, 0], robot.frontiers[:, 1], c='red', s=1, zorder=3)

            if hasattr(robot, 'movement_history') and len(robot.movement_history) > 1:
                history = np.array(robot.movement_history); ax_real.plot(history[:,0], history[:,1], '-', linewidth=1, alpha=0.6, color=robot_marker_color)

            if robot.target_pos is not None and robot.planned_path and len(robot.planned_path) >= 1:
                 try:
                     planned_path_with_current = [robot.position.copy()] + robot.planned_path
                     path = np.array(planned_path_with_current)
                     ax.plot(path[:,0], path[:,1], 'k--', linewidth=1, zorder=4)
                     ax.plot(robot.target_pos[0], robot.target_pos[1], markersize=6, zorder=999, marker="x", ls="-", c='black', mec="black")
                 except Exception as plot_err: logger.warning(f"Plot path failed R{robot.robot_id}: {plot_err}")

        plt.tight_layout()
        self._save_frame_to_memory()
        plt.draw(); plt.pause(0.001)

    def plot_env_without_window(self, step):
        # ... (繪圖邏輯與 plot_env 相同，只是沒有 ion/draw/pause) ...
        if not hasattr(self, 'fig') or self.fig is None:
            plt.ioff(); self.fig = plt.figure(figsize=(8, 10))
            if not hasattr(self, 'frames_data'): self.frames_data = []
        else: plt.figure(self.fig.number); plt.clf()

        color_list = ["r", "g", "c", "m", "y", "k"]; n_maps = self.n_agent
        rows_for_maps = 0 if n_maps == 0 else ((n_maps - 1) // 2 + 1)
        base_rows = 2; gaps = 1; total_rows = base_rows + rows_for_maps + gaps; total_cols = 2

        ax_real = plt.subplot2grid((total_rows, total_cols), (0, 0), rowspan=2, colspan=2)
        if self.server.position is not None: ax_real.plot(self.server.position[0], self.server.position[1], markersize=8, zorder=999, marker="h", ls="-", c=color_list[-2], mec="black")
        circle = patches.Circle(self.server.position, SERVER_COMM_RANGE, color=color_list[-2], alpha=0.1)
        ax_real.add_patch(circle); ax_real.imshow(self.real_map, cmap='gray')
        ax_real.set_title(f'Step: {step}, Real Map'); ax_real.axis('off')

        ax_env = plt.subplot2grid((total_rows, total_cols), (2, 0), rowspan=2, colspan=2)
        if self.server.position is not None: ax_env.plot(self.server.position[0], self.server.position[1], markersize=8, zorder=999, marker="h", ls="-", c=color_list[-2], mec="black")
        ax_env.imshow(self.real_map, cmap='gray'); ax_env.imshow(self.server.global_map, cmap='gray', alpha=0.5)
        ax_env.set_title('Global Map'); ax_env.axis('off')
        if self.server.frontiers is not None and len(self.server.frontiers) > 0: ax_env.scatter(self.server.frontiers[:, 0], self.server.frontiers[:, 1], c='lime', s=1, zorder=5)

        row_start = 4
        for i, robot in enumerate(self.robot_list):
            robot_marker_color = color_list[i % len(color_list)]; row = row_start + (i // 2); col = i % 2
            if row >= total_rows: continue
            ax = plt.subplot2grid((total_rows, total_cols), (row, col), rowspan=1, colspan=1)
            if robot.position is not None:
                ax.plot(robot.position[0], robot.position[1], markersize=6, zorder=999, marker="D", ls="-", c=robot_marker_color, mec="black")
                ax_real.plot(robot.position[0], robot.position[1], markersize=6, zorder=999, marker="D", ls="-", c=robot_marker_color, mec="black")
                if i < len(self.server.all_robot_position) and self.server.all_robot_position[i] is not None:
                     show_alpha = 1.0 if i < len(self.server.robot_in_range) and self.server.robot_in_range[i] else 0.3
                     ax_env.plot(self.server.all_robot_position[i][0], self.server.all_robot_position[i][1], markersize=6, zorder=999, marker="D", ls="-", c=robot_marker_color, mec="black", alpha=show_alpha)
                circle_comm = patches.Circle(robot.position, ROBOT_COMM_RANGE, color=robot_marker_color, alpha=0.05)
                ax_real.add_patch(circle_comm)
            ax.imshow(robot.local_map, cmap='gray')
            ax.set_title(f'Robot{robot.robot_id+1} Local Map'); ax.axis('off')
            if robot.node_coords is not None and len(robot.node_coords) > 0: ax.scatter(robot.node_coords[:, 0], robot.node_coords[:, 1], s=0.5, c='blue', zorder=2)
            if robot.frontiers is not None and len(robot.frontiers) > 0: ax.scatter(robot.frontiers[:, 0], robot.frontiers[:, 1], c='red', s=1, zorder=3)
            if hasattr(robot, 'movement_history') and len(robot.movement_history) > 1:
                history = np.array(robot.movement_history); ax_real.plot(history[:,0], history[:,1], '-', linewidth=1, alpha=0.6, color=robot_marker_color)
            if robot.target_pos is not None and robot.planned_path and len(robot.planned_path) >= 1:
                 try:
                     planned_path_with_current = [robot.position.copy()] + robot.planned_path
                     path = np.array(planned_path_with_current)
                     ax.plot(path[:,0], path[:,1], 'k--', linewidth=1, zorder=4)
                     ax.plot(robot.target_pos[0], robot.target_pos[1], markersize=6, zorder=999, marker="x", ls="-", c='black', mec="black")
                 except Exception as plot_err: pass

        plt.tight_layout()
        self._save_frame_to_memory()

    def _save_frame_to_memory(self):
        # ... (保持不變) ...
        import io; from PIL import Image
        if not hasattr(self, 'frames_data'): self.frames_data = []
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        try: image = Image.open(buf); self.frames_data.append(np.array(image))
        except Exception as e: logger.error(f"Save frame error: {e}")
        finally: buf.close()


    def save_video(self, filename="exploration_video.mp4", fps=5):
        # ... (保持不變) ...
        if not hasattr(self, 'frames_data') or not self.frames_data: logger.warning("No frames captured."); return
        import cv2
        height, width = self.frames_data[0].shape[:2]
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v'); video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            if not video_writer.isOpened(): logger.error(f"Failed to open {filename}"); return
            for frame in self.frames_data:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if len(frame.shape) == 3 else frame
                video_writer.write(frame_bgr)
            video_writer.release(); logger.info(f"Video saved: {filename}")
        except Exception as e: logger.error(f"Error saving video {filename}: {e}", exc_info=True)
        finally: self.frames_data = []