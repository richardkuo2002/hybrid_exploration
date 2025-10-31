from skimage import io
import os
from skimage.measure import block_reduce
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
import logging
import sys

from robot import Robot
from server import Server
from graph_generator import Graph_generator
from sensor import *
from parameter import *

logger = logging.getLogger(__name__)

class Env():
    def __init__(self, n_agent:int, k_size=20, map_index=0, plot=True):
        self.resolution = 4
        self.map_path = "DungeonMaps/train/easy"
        self.map_list = os.listdir(self.map_path)
        self.map_list.sort()
        self.map_list = [f for f in self.map_list if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.map_list:
             logger.critical(f"No valid map files found in {self.map_path}. Exiting.")
             sys.exit(1)

        self.map_index = map_index % len(self.map_list)
        self.file_path = self.map_list[self.map_index]
        logger.info(f"Using map index {self.map_index}: {self.file_path}")

        try:
             self.real_map, self.start_position = self.import_map_revised(os.path.join(self.map_path, self.file_path))
        except Exception as e:
             logger.critical(f"Failed loading map: {os.path.join(self.map_path, self.file_path)}. Error: {e}", exc_info=True)
             raise

        self.real_map_size = np.shape(self.real_map)
        self.server = Server(self.start_position, self.real_map_size, self.resolution, k_size, plot)
        self.n_agent = n_agent
        self.robot_list:list[Robot] = []
        self.server.all_robot_position = [None] * n_agent
        self.server.robot_in_range = [False] * n_agent

        for i in range(self.n_agent):
            robot = Robot(self.start_position, self.real_map_size, self.resolution, k_size, plot=plot)
            robot.robot_id = i
            try:
                robot.local_map = self.update_robot_local_map(robot.position, robot.sensor_range, robot.local_map, self.real_map)
                robot.downsampled_map = block_reduce(robot.local_map.copy(), block_size=(self.resolution, self.resolution), func=np.min)
                robot.frontiers = self.find_frontier(robot.downsampled_map)
                if hasattr(robot, 'graph_generator') and robot.graph_generator is not None:
                     valid_frontiers = robot.frontiers if robot.frontiers is not None else np.array([]).reshape(0,2)
                     node_coords, graph, node_utility, guidepost = robot.graph_generator.generate_graph(self.start_position, robot.local_map, valid_frontiers)
                     robot.node_coords = node_coords
                     robot.local_map_graph = graph
                     robot.node_utility = node_utility
                     robot.guidepost = guidepost
                else: logger.error(f"Robot {i} failed graph_generator init.")
                self.robot_list.append(robot)
                if hasattr(self.server, 'all_robot_position') and hasattr(self.server, 'robot_in_range') and i < len(self.server.all_robot_position):
                    self.server.all_robot_position[i] = robot.position
                    self.server.robot_in_range[i] = True
                else: logger.error("Server list attributes missing or index out of bounds during robot init!")
            except Exception as e:
                 logger.error(f"Failed init Robot {i}: {e}", exc_info=True)
        
        maps_to_merge = [robot.local_map for robot in self.robot_list] + [self.server.global_map]
        merged = self.merge_maps(maps_to_merge)
        for robot in self.robot_list: robot.local_map[:] = merged
        self.server.global_map[:] = merged

        try:
            if self.robot_list:
                self.server.update_and_assign_tasks(self.robot_list, self.real_map, self.find_frontier)
            else: logger.warning("No robots initialized, skip initial server update.")
        except Exception as e:
             logger.critical(f"Initial server update failed: {e}", exc_info=True); raise

    # ... (calculate_coverage_ratio, merge_maps, update_robot_local_map, find_frontier 不變) ...
    def calculate_coverage_ratio(self):
        explored_pixels = np.sum(self.server.global_map == 255); total_free_pixels = np.sum(self.real_map == 255)
        return min(explored_pixels / total_free_pixels, 1.0) if total_free_pixels > 0 else 0.0
    def merge_maps(self, maps_to_merge):
        merged_map = np.ones_like(self.real_map) * 127
        valid_maps = [m for m in maps_to_merge if isinstance(m, np.ndarray)]
        if not valid_maps: return merged_map
        for belief in valid_maps:
            merged_map[belief == 1] = 1; merged_map[belief == 255] = 255
        return merged_map
    def update_robot_local_map(self, robot_position, sensor_range, robot_local_map, real_map):
        try:
             updated_map = sensor_work(robot_position, sensor_range, robot_local_map, real_map)
             return updated_map if isinstance(updated_map, np.ndarray) else robot_local_map
        except Exception as e: logger.error(f"Error sensor_work @ {robot_position}: {e}", exc_info=True); return robot_local_map
    def find_frontier(self, downsampled_map):
        try:
            if downsampled_map is None or downsampled_map.ndim != 2: return np.array([]).reshape(0, 2)
            y_len, x_len = downsampled_map.shape[:2]; mapping = (downsampled_map == 127).astype(np.int8)
            mapping = np.pad(mapping, 1, 'constant', constant_values=0)
            fro_map = mapping[2:, 1:-1] + mapping[:-2, 1:-1] + mapping[1:-1, 2:] + mapping[1:-1, :-2] + \
                      mapping[:-2, 2:] + mapping[2:, :-2] + mapping[2:, 2:] + mapping[:-2, :-2]
            belief = downsampled_map; is_free = (belief == 255); is_frontier_neighbor = (fro_map > 0) & (fro_map < 8)
            frontier_mask = is_free & is_frontier_neighbor; ind_to = np.where(frontier_mask.ravel(order='F'))[0]
            if ind_to.size == 0: return np.array([]).reshape(0, 2)
            rows, cols = np.indices((y_len, x_len)); points = np.stack([cols.ravel(order='F'), rows.ravel(order='F')], axis=-1)
            f = points[ind_to] * self.resolution; return f.astype(int)
        except Exception as e: logger.error(f"Error find_frontier: {e}", exc_info=True); return np.array([]).reshape(0, 2)

    def import_map_revised(self, map_path):
        try:
            map_img_gray = io.imread(map_path, as_gray=True)
            if map_img_gray.dtype == float: map_img_int = (map_img_gray * 255).astype(np.uint8)
            else: map_img_int = map_img_gray.astype(np.uint8)
            start_points = np.where(map_img_int == 208); start_location = None
            if start_points[0].size > 0:
                mid_idx = start_points[0].size // 2
                start_location = np.array([start_points[1][mid_idx], start_points[0][mid_idx]])
                logger.info(f"Start pos (208) found at {start_location} in {os.path.basename(map_path)}")
            else:
                logger.warning(f"Start pos (208) not found in {os.path.basename(map_path)}. Using default [100, 100].")
                start_location = np.array([100, 100])
            final_map = np.ones_like(map_img_int, dtype=np.uint8) * 1
            final_map[map_img_int > 150] = 255
            if start_location is not None:
                 y, x = start_location[1], start_location[0]
                 if 0 <= y < final_map.shape[0] and 0 <= x < final_map.shape[1]: final_map[y, x] = 255
            free_ratio = np.sum(final_map == 255) / final_map.size
            logger.debug(f"Loaded {os.path.basename(map_path)}. Shape:{final_map.shape}. Free:{free_ratio:.2f}")
            if free_ratio < 0.01: logger.warning(f"Map {os.path.basename(map_path)} has very little free space ({free_ratio*100:.1f}%).")
            return final_map, start_location
        except FileNotFoundError: logger.error(f"Map not found: {map_path}"); raise
        except Exception as e: logger.error(f"Error loading map {map_path}: {e}", exc_info=True); raise


    def _get_plot_layout(self):
        n_maps = self.n_agent; base_rows_fixed = 4
        rows_for_maps = 0 if n_maps == 0 else ((n_maps - 1) // 2 + 1)
        total_rows = base_rows_fixed + rows_for_maps; total_cols = 2
        robot_row_start = base_rows_fixed
        return total_rows, total_cols, robot_row_start

    # <--- 修改點：在 plot_env 加入 save_frame 參數 ---
    def plot_env(self, step, save_frame=False):
        # --- ---
        if not hasattr(self, 'fig') or self.fig is None:
            try: plt.switch_backend('agg'); plt.figure(); plt.close(); plt.switch_backend('tkagg')
            except Exception as e: logger.warning(f"Failed backend switch: {e}")
            plt.ion(); self.fig = plt.figure(figsize=(8, 10))
            if not hasattr(self, 'frames_data'): self.frames_data = []
        else: plt.figure(self.fig.number); plt.clf()

        color_list = ["r", "g", "c", "m", "y", "k"]
        total_rows, total_cols, robot_row_start = self._get_plot_layout()

        # ... (Real Map 和 Global Map 繪製邏輯不變) ...
        # Real Map
        ax_real = plt.subplot2grid((total_rows, total_cols), (0, 0), rowspan=2, colspan=2)
        if self.server.position is not None: ax_real.plot(self.server.position[0], self.server.position[1], markersize=8, zorder=999, marker="h", ls="-", c=color_list[-2], mec="black")
        circle = patches.Circle(self.server.position, SERVER_COMM_RANGE, color=color_list[-2], alpha=0.1); ax_real.add_patch(circle)
        ax_real.imshow(self.real_map, cmap='gray'); ax_real.set_title(f'Step: {step}, Real Map'); ax_real.axis('off')
        # Global Map
        ax_env = plt.subplot2grid((total_rows, total_cols), (2, 0), rowspan=2, colspan=2)
        if self.server.position is not None: ax_env.plot(self.server.position[0], self.server.position[1], markersize=8, zorder=999, marker="h", ls="-", c=color_list[-2], mec="black")
        ax_env.imshow(self.real_map, cmap='gray'); ax_env.imshow(self.server.global_map, cmap='gray', alpha=0.5)
        ax_env.set_title('Global Map'); ax_env.axis('off')
        if self.server.frontiers is not None and len(self.server.frontiers) > 0: ax_env.scatter(self.server.frontiers[:, 0], self.server.frontiers[:, 1], c='lime', s=1, zorder=5)

        # ... (Robot local maps 繪製邏輯不變) ...
        for i, robot in enumerate(self.robot_list):
            robot_marker_color = color_list[i % len(color_list)]
            row = robot_row_start + (i // 2); col = i % 2
            if row >= total_rows: continue
            try:
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
                      except Exception as plot_err: logger.warning(f"Plot path failed R{robot.robot_id}: {plot_err}")
            except IndexError:
                 logger.error(f"Error creating subplot for Robot {i} at grid ({row}, {col}) with total_rows={total_rows}. Skipping.")
                 continue

        plt.tight_layout()
        # <--- 修改點：根據 save_frame 決定是否儲存 ---
        if save_frame:
            self._save_frame_to_memory()
        # --- ---
        plt.draw(); plt.pause(0.001)

    def plot_env_without_window(self, step):
        # ... (此函式不變，它總是儲存影格) ...
        if not hasattr(self, 'fig') or self.fig is None:
            plt.ioff(); self.fig = plt.figure(figsize=(8, 10))
            if not hasattr(self, 'frames_data'): self.frames_data = []
        else: plt.figure(self.fig.number); plt.clf()

        color_list = ["r", "g", "c", "m", "y", "k"]
        total_rows, total_cols, robot_row_start = self._get_plot_layout()

        # Real Map
        ax_real = plt.subplot2grid((total_rows, total_cols), (0, 0), rowspan=2, colspan=2)
        if self.server.position is not None: ax_real.plot(self.server.position[0], self.server.position[1], markersize=8, zorder=999, marker="h", ls="-", c=color_list[-2], mec="black")
        circle = patches.Circle(self.server.position, SERVER_COMM_RANGE, color=color_list[-2], alpha=0.1); ax_real.add_patch(circle)
        ax_real.imshow(self.real_map, cmap='gray'); ax_real.set_title(f'Step: {step}, Real Map'); ax_real.axis('off')
        # Global Map
        ax_env = plt.subplot2grid((total_rows, total_cols), (2, 0), rowspan=2, colspan=2)
        if self.server.position is not None: ax_env.plot(self.server.position[0], self.server.position[1], markersize=8, zorder=999, marker="h", ls="-", c=color_list[-2], mec="black")
        ax_env.imshow(self.real_map, cmap='gray'); ax_env.imshow(self.server.global_map, cmap='gray', alpha=0.5)
        ax_env.set_title('Global Map'); ax_env.axis('off')
        if self.server.frontiers is not None and len(self.server.frontiers) > 0: ax_env.scatter(self.server.frontiers[:, 0], self.server.frontiers[:, 1], c='lime', s=1, zorder=5)
        # Robot maps
        for i, robot in enumerate(self.robot_list):
            robot_marker_color = color_list[i % len(color_list)]; row = robot_row_start + (i // 2); col = i % 2
            if row >= total_rows: continue
            try:
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
            except IndexError:
                logger.error(f"Error subplot w/o window R{i} @ ({row},{col}), total_rows={total_rows}. Skip.")
                continue

        plt.tight_layout()
        self._save_frame_to_memory() # 這裡總是儲存

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