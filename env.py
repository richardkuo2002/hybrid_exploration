import io
import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import matplotlib.patches as patches
import matplotlib.pylab as plt
import numpy as np
import skimage.io as skimage_io
from PIL import Image
from skimage.measure import block_reduce

from graph_generator import Graph_generator
from parameter import *
from utils import check_collision, merge_maps_jit
from robot import Robot
from sensor import *
from server import Server

logger = logging.getLogger(__name__)


class Env:
    def __init__(
        self,
        n_agent: int,
        k_size: int = 20,
        map_index: int = 0,
        plot: bool = True,
        force_sync_debug: bool = False,
        graph_update_interval: Optional[int] = None,
        debug_mode: bool = False,
        map_type: str = "odd",
    ) -> None:
        """初始化環境 (讀取地圖、建立伺服器與機器人)。

        Args:
            n_agent (int): 機器人數量。
            k_size (int): graph generator 的 k。
            map_index (int): 選擇地圖的索引。
            plot (bool): 是否啟用繪圖。
            force_sync_debug (bool): 是否強制同步除錯。
            graph_update_interval (Optional[int]): Graph 更新間隔。
            debug_mode (bool): 是否啟用除錯模式。
            map_type (str): 地圖類型 ("odd" or "even")。

        Returns:
            None
        """
        self.resolution = 4
        # self.map_path = "DungeonMaps/train/easy/"
        # self.map_path = "maps/easy_even"
        if map_type == "even":
            self.map_path = "maps/easy_even"
        elif map_type == "odd":
            self.map_path = "maps/easy_odd"
        elif map_type == "ir2":
            self.map_path = "maps/ir2_maps/test/complex"
        else:
            self.map_path = "maps/easy_even"  # Default fallback
            
        self.map_list = os.listdir(self.map_path)
        self.map_list.sort()
        self.map_list = [
            f for f in self.map_list if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not self.map_list:
            logger.critical(f"No valid map files found in {self.map_path}. Exiting.")
            sys.exit(1)

        self.map_index = map_index % len(self.map_list)
        self.file_path = self.map_list[self.map_index]
        logger.info(f"Using map index {self.map_index}: {self.file_path}")

        try:
            self.real_map, self.start_position = self.import_map_revised(
                os.path.join(self.map_path, self.file_path)
            )
        except Exception as e:
            logger.critical(
                f"Failed loading map: {os.path.join(self.map_path, self.file_path)}. Error: {e}",
                exc_info=True,
            )
            raise

        self.real_map_size = np.shape(self.real_map)
        self.force_sync_debug = force_sync_debug
        self.server = Server(
            self.start_position,
            self.real_map_size,
            self.resolution,
            k_size,
            plot,
            force_sync_debug=self.force_sync_debug,
            graph_update_interval=graph_update_interval,
            debug_mode=debug_mode,
        )
        self.n_agent = n_agent
        self.robot_list: List[Robot] = []
        self.server.all_robot_position = [None] * n_agent
        self.server.robot_in_range = [False] * n_agent
        # Deterministic placement: cycle through top-left, top-right, bottom-right, bottom-left
        start_positions = []
        center = (
            self.start_position if self.start_position is not None else np.array([0, 0])
        )
        # offset distance from center (keep within server comm range and map bounds)
        offset_dist = int(min(SERVER_COMM_RANGE * 0.05, min(self.real_map.shape) // 4))
        # ordered offsets: TL, TR, BR, BL (x,y)
        offsets = [
            (-offset_dist, -offset_dist),
            (offset_dist, -offset_dist),
            (offset_dist, offset_dist),
            (-offset_dist, offset_dist),
        ]

        def find_nearest_free(
            xc: int, yc: int, max_search: int = 8
        ) -> Optional[np.ndarray]:
            # deterministic neighborhood search (increasing Manhattan radius)
            h, w = self.real_map.shape
            from parameter import PIXEL_FREE

            if 0 <= xc < w and 0 <= yc < h and self.real_map[yc, xc] == PIXEL_FREE:
                return np.array([xc, yc])
            for r in range(1, max_search + 1):
                for dx in range(-r, r + 1):
                    for dy in (-r, r):
                        nx, ny = xc + dx, yc + dy
                        if (
                            0 <= nx < w
                            and 0 <= ny < h
                            and self.real_map[ny, nx] == PIXEL_FREE
                        ):
                            return np.array([nx, ny])
                for dy in range(-r + 1, r):
                    for dx in (-r, r):
                        nx, ny = xc + dx, yc + dy
                        if (
                            0 <= nx < w
                            and 0 <= ny < h
                            and self.real_map[ny, nx] == PIXEL_FREE
                        ):
                            return np.array([nx, ny])
            return None

        for i in range(self.n_agent):
            off = offsets[i % len(offsets)]
            cand_x = int(round(center[0] + off[0]))
            cand_y = int(round(center[1] + off[1]))
            pos = find_nearest_free(cand_x, cand_y, max_search=8)
            if pos is None:
                pos = np.array(center)
            start_positions.append(pos)

        for i in range(self.n_agent):
            robot_start = start_positions[i]
            robot = Robot(
                robot_start,
                self.real_map_size,
                self.resolution,
                k_size,
                plot=plot,
                graph_update_interval=graph_update_interval,
                debug_mode=debug_mode,
            )
            robot.robot_id = i
            try:
                robot.local_map = self.update_robot_local_map(
                    robot.position, robot.sensor_range, robot.local_map, self.real_map
                )
                robot.downsampled_map = block_reduce(
                    robot.local_map.copy(),
                    block_size=(self.resolution, self.resolution),
                    func=np.min,
                )
                robot.frontiers = self.find_frontier(robot.downsampled_map)
                if (
                    hasattr(robot, "graph_generator")
                    and robot.graph_generator is not None
                ):
                    valid_frontiers = (
                        robot.frontiers
                        if robot.frontiers is not None
                        else np.array([]).reshape(0, 2)
                    )
                    node_coords, graph, node_utility, guidepost = (
                        robot.graph_generator.generate_graph(
                            robot_start, robot.local_map, valid_frontiers
                        )
                    )
                    robot.node_coords = node_coords
                    robot.local_map_graph = graph
                    robot.node_utility = node_utility
                    robot.guidepost = guidepost
                else:
                    logger.error(f"Robot {i} failed graph_generator init.")
                self.robot_list.append(robot)
                if (
                    hasattr(self.server, "all_robot_position")
                    and hasattr(self.server, "robot_in_range")
                    and i < len(self.server.all_robot_position)
                ):
                    self.server.all_robot_position[i] = robot.position
                    self.server.robot_in_range[i] = True
                else:
                    logger.error(
                        "Server list attributes missing or index out of bounds during robot init!"
                    )
            except Exception as e:
                logger.error(f"Failed init Robot {i}: {e}", exc_info=True)

        maps_to_merge = [robot.local_map for robot in self.robot_list] + [
            self.server.global_map
        ]
        merged = self.merge_maps(maps_to_merge)
        for robot in self.robot_list:
            robot.local_map[:] = merged
        self.server.global_map[:] = merged

        # 新增：影片串流與緩衝設定（避免 frames_data 無限增長）
        # 若系統有 cv2 則預設啟用串流（更省記憶體）
        try:
            import cv2  # 檢查是否可用

            self._video_streaming_available = True
        except Exception:
            self._video_streaming_available = False

        self._video_writer = None  # cv2.VideoWriter 物件（若串流）
        self._video_tmp_path = None  # 臨時影片檔路徑（串流時使用）
        self._frames_buffer_max = 300  # 若無 cv2，最多保留多少幀（可調）
        self.frames_data: List[bytes] = []  # 緩衝（可能為 numpy 或 bytes，視情況而定）
        self._frames_compressed = True  # 在緩衝模式下儲存壓縮 bytes 可降低記憶體

        try:
            if self.robot_list:
                self.server.update_and_assign_tasks(
                    self.robot_list, self.real_map, self.find_frontier
                )
            else:
                logger.warning("No robots initialized, skip initial server update.")
        except Exception as e:
            logger.critical(f"Initial server update failed: {e}", exc_info=True)
            raise

    def calculate_coverage_ratio(self) -> float:
        """計算目前覆蓋率（使用 server.global_map 與真實地圖）。

        Returns:
            float: 覆蓋率 (0-1)。
        """
        from parameter import PIXEL_FREE

        explored_pixels = np.sum(self.server.global_map == PIXEL_FREE)
        total_free_pixels = np.sum(self.real_map == PIXEL_FREE)
        return (
            min(explored_pixels / total_free_pixels, 1.0)
            if total_free_pixels > 0
            else 0.0
        )

    def merge_maps(self, maps_to_merge: List[np.ndarray]) -> np.ndarray:
        """合併多張信念地圖，優先考慮障礙與可通行。

        Args:
            maps_to_merge (List[np.ndarray]): 要合併的地圖清單。

        Returns:
            np.ndarray: 合併後的地圖。
        """
        from parameter import PIXEL_FREE, PIXEL_OCCUPIED, PIXEL_UNKNOWN

        # Filter valid maps
        valid_maps = [m for m in maps_to_merge if isinstance(m, np.ndarray)]
        
        # Initialize merged map with UNKNOWN
        merged_map = np.full_like(self.real_map, PIXEL_UNKNOWN)
        
        if not valid_maps:
            return merged_map

        # Use Numba-optimized merge
        # Numba needs a typed list or tuple of arrays. List is fine.
        # Ensure all maps are same shape as real_map (should be guaranteed by logic)
        
        # Call JIT function
        # Note: Numba might recompile if list length changes often, 
        # but for fixed agent count it should be stable.
        # Passing list of arrays to Numba JIT function works if they are same type/dim.
        
        # To be safe with Numba's typed list requirement for some versions:
        # We pass a simple list. If Numba complains, we might need TypedList.
        # But usually list of arrays is supported in recent Numba.
        
        merged_map = merge_maps_jit(
            valid_maps, 
            merged_map, 
            PIXEL_FREE, 
            PIXEL_OCCUPIED, 
            PIXEL_UNKNOWN
        )
        
        return merged_map

    def update_robot_local_map(
        self,
        robot_position: np.ndarray,
        sensor_range: int,
        robot_local_map: np.ndarray,
        real_map: np.ndarray,
    ) -> np.ndarray:
        """呼叫 sensor_work 更新機器人的 local map。

        Args:
            robot_position (np.ndarray): 機器人位置。
            sensor_range (int): 感測半徑。
            robot_local_map (np.ndarray): 當前 local map。
            real_map (np.ndarray): 真實地圖。

        Returns:
            np.ndarray: 更新後的 local map（若發生例外則回傳原本的 local_map）。
        """
        try:
            updated_map = sensor_work(
                robot_position, sensor_range, robot_local_map, real_map
            )
            return (
                updated_map if isinstance(updated_map, np.ndarray) else robot_local_map
            )
        except Exception as e:
            logger.error(f"Error sensor_work @ {robot_position}: {e}", exc_info=True)
            return robot_local_map

    def find_frontier(self, downsampled_map: np.ndarray) -> np.ndarray:
        """找出 downsampled_map 的 frontier 點。

        Args:
            downsampled_map (np.ndarray): 下採樣後的地圖陣列。

        Returns:
            np.ndarray: frontier 座標陣列 (N,2)，若沒有則為空陣列。
        """
        try:
            if downsampled_map is None or downsampled_map.ndim != 2:
                return np.array([]).reshape(0, 2)
            from parameter import PIXEL_UNKNOWN

            y_len, x_len = downsampled_map.shape[:2]
            mapping = (downsampled_map == PIXEL_UNKNOWN).astype(np.int8)
            mapping = np.pad(mapping, 1, "constant", constant_values=0)
            fro_map = (
                mapping[2:, 1:-1]
                + mapping[:-2, 1:-1]
                + mapping[1:-1, 2:]
                + mapping[1:-1, :-2]
                + mapping[:-2, 2:]
                + mapping[2:, :-2]
                + mapping[2:, 2:]
                + mapping[:-2, :-2]
            )
            from parameter import PIXEL_FREE
            from parameter import ENABLE_FRONTIER_CLUSTERING, MIN_FRONTIER_SIZE

            belief = downsampled_map
            is_free = belief == PIXEL_FREE
            is_frontier_neighbor = (fro_map > 0) & (fro_map < 8)
            frontier_mask = is_free & is_frontier_neighbor
            
            # --- Frontier Clustering Logic ---
            if ENABLE_FRONTIER_CLUSTERING:
                try:
                    from scipy.ndimage import label, center_of_mass
                    
                    # 1. Label connected components
                    # structure=np.ones((3,3)) allows diagonal connectivity
                    labeled_array, num_features = label(frontier_mask, structure=np.ones((3,3)))
                    
                    if num_features == 0:
                        return np.array([]).reshape(0, 2)
                    
                    centroids = []
                    # 2. Iterate through features
                    # Note: label indices start from 1
                    for i in range(1, num_features + 1):
                        # Get mask for this component
                        component_mask = (labeled_array == i)
                        component_size = np.sum(component_mask)
                        
                        # Filter small noise
                        if component_size < MIN_FRONTIER_SIZE:
                            continue
                            
                        # Calculate centroid
                    # center_of_mass returns (y, x) float coordinates
                    cy, cx = center_of_mass(component_mask)
                    centroids.append([cx, cy])
                    
                    if not centroids:
                        return np.array([]).reshape(0, 2)
                        
                    # Convert to numpy array and scale to real coordinates
                    # Note: centroids are in downsampled map coordinates
                    f_pixels = np.array(centroids)
                    f = f_pixels * self.resolution
                    return f.astype(int)
                    
                except ImportError:
                    logger.warning("scipy.ndimage not found. Falling back to raw frontiers.")
                except Exception as e:
                    logger.error(f"Error in frontier clustering: {e}. Falling back to raw.")

            # --- Fallback / Original Logic ---
            ind_to = np.where(frontier_mask.ravel(order="F"))[0]
            if ind_to.size == 0:
                return np.array([]).reshape(0, 2)
            rows, cols = np.indices((y_len, x_len))
            points = np.stack([cols.ravel(order="F"), rows.ravel(order="F")], axis=-1)
            f = points[ind_to] * self.resolution
            return f.astype(int)
            
        except Exception as e:
            logger.error(f"Error find_frontier: {e}", exc_info=True)
            return np.array([]).reshape(0, 2)

    def import_map_revised(
        self, map_path: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """讀取地圖檔案並回傳轉換後的地圖與起始點座標。

        Args:
            map_path (str): 地圖檔案路徑。

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: (final_map (ndarray), start_location (ndarray))
        """
        try:
            map_img_gray = skimage_io.imread(map_path, as_gray=True)
            if map_img_gray.dtype == float:
                map_img_int = (map_img_gray * 255).astype(np.uint8)
            else:
                map_img_int = map_img_gray.astype(np.uint8)
            from parameter import PIXEL_START

            start_points = np.where(map_img_int == PIXEL_START)
            start_location = None
            if start_points[0].size > 0:
                mid_idx = start_points[0].size // 2
                start_location = np.array(
                    [start_points[1][mid_idx], start_points[0][mid_idx]]
                )
                logger.info(
                    f"Start pos ({PIXEL_START}) found at {start_location} in {os.path.basename(map_path)}"
                )
            else:
                logger.warning(
                    f"Start pos ({PIXEL_START}) not found in {os.path.basename(map_path)}. Using default [100, 100]."
                )
                start_location = np.array([100, 100])
            from parameter import PIXEL_OCCUPIED

            final_map = np.ones_like(map_img_int, dtype=np.uint8) * PIXEL_OCCUPIED
            from parameter import MAP_THRESHOLD, PIXEL_FREE

            final_map[map_img_int > MAP_THRESHOLD] = PIXEL_FREE
            if start_location is not None:
                y, x = start_location[1], start_location[0]
                from parameter import PIXEL_FREE

                if 0 <= y < final_map.shape[0] and 0 <= x < final_map.shape[1]:
                    final_map[y, x] = PIXEL_FREE
            from parameter import PIXEL_FREE

            free_ratio = np.sum(final_map == PIXEL_FREE) / final_map.size
            logger.debug(
                f"Loaded {os.path.basename(map_path)}. Shape:{final_map.shape}. Free:{free_ratio:.2f}"
            )
            if free_ratio < 0.01:
                logger.warning(
                    f"Map {os.path.basename(map_path)} has very little free space ({free_ratio*100:.1f}%)."
                )
            return final_map, start_location
        except FileNotFoundError:
            logger.error(f"Map not found: {map_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading map {map_path}: {e}", exc_info=True)
            raise

    def _get_plot_layout(self) -> Tuple[int, int, int]:
        """計算繪圖格局 (內部)。Returns total_rows, total_cols, robot_row_start。"""
        n_maps = self.n_agent
        base_rows_fixed = 4
        rows_for_maps = 0 if n_maps == 0 else ((n_maps - 1) // 2 + 1)
        total_rows = base_rows_fixed + rows_for_maps
        total_cols = 2
        robot_row_start = base_rows_fixed
        return total_rows, total_cols, robot_row_start

    def plot_env(self, step: int, save_frame: bool = False) -> None:
        """繪製環境視覺化，並選擇性將影格存入記憶（供 later saving）。

        Args:
            step (int): 當前步數。
            save_frame (bool): 是否將此影格存入內存。

        Returns:
            None
        """
        if not hasattr(self, "fig") or self.fig is None:
            try:
                plt.switch_backend("agg")
                plt.figure()
                plt.close()
                plt.switch_backend("tkagg")
            except Exception as e:
                logger.warning(f"Failed backend switch: {e}")
            plt.ion()
            self.fig = plt.figure(figsize=(8, 10))
            if not hasattr(self, "frames_data"):
                self.frames_data = []
        else:
            plt.figure(self.fig.number)
            plt.clf()

        color_list = ["r", "g", "c", "m", "y", "k"]
        total_rows, total_cols, robot_row_start = self._get_plot_layout()

        # Real Map
        ax_real = plt.subplot2grid(
            (total_rows, total_cols), (0, 0), rowspan=2, colspan=2
        )
        if self.server.position is not None:
            ax_real.plot(
                self.server.position[0],
                self.server.position[1],
                markersize=8,
                zorder=999,
                marker="h",
                ls="-",
                c=color_list[-2],
                mec="black",
            )
        circle = patches.Circle(
            self.server.position, SERVER_COMM_RANGE, color=color_list[-2], alpha=0.1
        )
        ax_real.add_patch(circle)
        ax_real.imshow(self.real_map, cmap="gray")
        ax_real.set_title(f"Step: {step}, Real Map")
        ax_real.axis("off")
        # Global Map
        ax_env = plt.subplot2grid(
            (total_rows, total_cols), (2, 0), rowspan=2, colspan=2
        )
        if self.server.position is not None:
            ax_env.plot(
                self.server.position[0],
                self.server.position[1],
                markersize=8,
                zorder=999,
                marker="h",
                ls="-",
                c=color_list[-2],
                mec="black",
            )
        ax_env.imshow(self.real_map, cmap="gray")
        ax_env.imshow(self.server.global_map, cmap="gray", alpha=0.5)
        ax_env.set_title("Global Map")
        ax_env.axis("off")
        if self.server.frontiers is not None and len(self.server.frontiers) > 0:
            ax_env.scatter(
                self.server.frontiers[:, 0],
                self.server.frontiers[:, 1],
                c="lime",
                s=1,
                zorder=5,
            )

        for i, robot in enumerate(self.robot_list):
            robot_marker_color = color_list[i % len(color_list)]
            row = robot_row_start + (i // 2)
            col = i % 2
            if row >= total_rows:
                continue
            try:
                ax = plt.subplot2grid(
                    (total_rows, total_cols), (row, col), rowspan=1, colspan=1
                )
                if robot.position is not None:
                    ax.plot(
                        robot.position[0],
                        robot.position[1],
                        markersize=6,
                        zorder=999,
                        marker="D",
                        ls="-",
                        c=robot_marker_color,
                        mec="black",
                    )
                    ax_real.plot(
                        robot.position[0],
                        robot.position[1],
                        markersize=6,
                        zorder=999,
                        marker="D",
                        ls="-",
                        c=robot_marker_color,
                        mec="black",
                    )
                    if (
                        i < len(self.server.all_robot_position)
                        and self.server.all_robot_position[i] is not None
                    ):
                        show_alpha = (
                            1.0
                            if i < len(self.server.robot_in_range)
                            and self.server.robot_in_range[i]
                            else 0.3
                        )
                        ax_env.plot(
                            self.server.all_robot_position[i][0],
                            self.server.all_robot_position[i][1],
                            markersize=6,
                            zorder=999,
                            marker="D",
                            ls="-",
                            c=robot_marker_color,
                            mec="black",
                            alpha=show_alpha,
                        )
                    circle_comm = patches.Circle(
                        robot.position,
                        ROBOT_COMM_RANGE,
                        color=robot_marker_color,
                        alpha=0.05,
                    )
                    ax_real.add_patch(circle_comm)
                ax.imshow(robot.local_map, cmap="gray")
                ax.set_title(f"Robot{robot.robot_id+1} Local Map")
                ax.axis("off")
                if robot.node_coords is not None and len(robot.node_coords) > 0:
                    ax.scatter(
                        robot.node_coords[:, 0],
                        robot.node_coords[:, 1],
                        s=0.5,
                        c="blue",
                        zorder=2,
                    )
                if robot.frontiers is not None and len(robot.frontiers) > 0:
                    ax.scatter(
                        robot.frontiers[:, 0],
                        robot.frontiers[:, 1],
                        c="red",
                        s=1,
                        zorder=3,
                    )
                if (
                    hasattr(robot, "movement_history")
                    and len(robot.movement_history) > 1
                ):
                    history = np.array(robot.movement_history)
                    ax_real.plot(
                        history[:, 0],
                        history[:, 1],
                        "-",
                        linewidth=1,
                        alpha=0.6,
                        color=robot_marker_color,
                    )
                if (
                    robot.target_pos is not None
                    and robot.planned_path
                    and len(robot.planned_path) >= 1
                ):
                    try:
                        planned_path_with_current = [
                            robot.position.copy()
                        ] + robot.planned_path
                        path = np.array(planned_path_with_current)
                        ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1, zorder=4)
                        ax.plot(
                            robot.target_pos[0],
                            robot.target_pos[1],
                            markersize=6,
                            zorder=999,
                            marker="x",
                            ls="-",
                            c="black",
                            mec="black",
                        )
                    except Exception as plot_err:
                        logger.warning(
                            f"Plot path failed R{robot.robot_id}: {plot_err}"
                        )
            except IndexError:
                logger.error(
                    f"Error creating subplot for Robot {i} at grid ({row}, {col}) with total_rows={total_rows}. Skipping."
                )
                continue

        plt.tight_layout()
        if save_frame:
            self._save_frame_to_memory()
        plt.draw()
        plt.pause(0.001)

    def plot_env_without_window(self, step: int) -> None:
        """不開視窗直接繪製並儲存影格（always save frame）。

        Args:
            step (int): 當前步數。

        Returns:
            None
        """
        if not hasattr(self, "fig") or self.fig is None:
            plt.ioff()
            self.fig = plt.figure(figsize=(8, 10))
            if not hasattr(self, "frames_data"):
                self.frames_data = []
        else:
            plt.figure(self.fig.number)
            plt.clf()

        color_list = ["r", "g", "c", "m", "y", "k"]
        total_rows, total_cols, robot_row_start = self._get_plot_layout()

        # Real Map
        ax_real = plt.subplot2grid(
            (total_rows, total_cols), (0, 0), rowspan=2, colspan=2
        )
        if self.server.position is not None:
            ax_real.plot(
                self.server.position[0],
                self.server.position[1],
                markersize=8,
                zorder=999,
                marker="h",
                ls="-",
                c=color_list[-2],
                mec="black",
            )
        circle = patches.Circle(
            self.server.position, SERVER_COMM_RANGE, color=color_list[-2], alpha=0.1
        )
        ax_real.add_patch(circle)
        ax_real.imshow(self.real_map, cmap="gray")
        ax_real.set_title(f"Step: {step}, Real Map")
        ax_real.axis("off")
        # Global Map
        ax_env = plt.subplot2grid(
            (total_rows, total_cols), (2, 0), rowspan=2, colspan=2
        )
        if self.server.position is not None:
            ax_env.plot(
                self.server.position[0],
                self.server.position[1],
                markersize=8,
                zorder=999,
                marker="h",
                ls="-",
                c=color_list[-2],
                mec="black",
            )
        ax_env.imshow(self.real_map, cmap="gray")
        ax_env.imshow(self.server.global_map, cmap="gray", alpha=0.5)
        ax_env.set_title("Global Map")
        ax_env.axis("off")
        if self.server.frontiers is not None and len(self.server.frontiers) > 0:
            ax_env.scatter(
                self.server.frontiers[:, 0],
                self.server.frontiers[:, 1],
                c="lime",
                s=1,
                zorder=5,
            )
        # Robot maps
        for i, robot in enumerate(self.robot_list):
            robot_marker_color = color_list[i % len(color_list)]
            row = robot_row_start + (i // 2)
            col = i % 2
            if row >= total_rows:
                continue
            try:
                ax = plt.subplot2grid(
                    (total_rows, total_cols), (row, col), rowspan=1, colspan=1
                )
                if robot.position is not None:
                    ax.plot(
                        robot.position[0],
                        robot.position[1],
                        markersize=6,
                        zorder=999,
                        marker="D",
                        ls="-",
                        c=robot_marker_color,
                        mec="black",
                    )
                    ax_real.plot(
                        robot.position[0],
                        robot.position[1],
                        markersize=6,
                        zorder=999,
                        marker="D",
                        ls="-",
                        c=robot_marker_color,
                        mec="black",
                    )
                    if (
                        i < len(self.server.all_robot_position)
                        and self.server.all_robot_position[i] is not None
                    ):
                        show_alpha = (
                            1.0
                            if i < len(self.server.robot_in_range)
                            and self.server.robot_in_range[i]
                            else 0.3
                        )
                        ax_env.plot(
                            self.server.all_robot_position[i][0],
                            self.server.all_robot_position[i][1],
                            markersize=6,
                            zorder=999,
                            marker="D",
                            ls="-",
                            c=robot_marker_color,
                            mec="black",
                            alpha=show_alpha,
                        )
                    circle_comm = patches.Circle(
                        robot.position,
                        ROBOT_COMM_RANGE,
                        color=robot_marker_color,
                        alpha=0.05,
                    )
                    ax_real.add_patch(circle_comm)
                ax.imshow(robot.local_map, cmap="gray")
                ax.set_title(f"Robot{robot.robot_id+1} Local Map")
                ax.axis("off")
                if robot.node_coords is not None and len(robot.node_coords) > 0:
                    ax.scatter(
                        robot.node_coords[:, 0],
                        robot.node_coords[:, 1],
                        s=0.5,
                        c="blue",
                        zorder=2,
                    )
                if robot.frontiers is not None and len(robot.frontiers) > 0:
                    ax.scatter(
                        robot.frontiers[:, 0],
                        robot.frontiers[:, 1],
                        c="red",
                        s=1,
                        zorder=3,
                    )
                if (
                    hasattr(robot, "movement_history")
                    and len(robot.movement_history) > 1
                ):
                    history = np.array(robot.movement_history)
                    ax_real.plot(
                        history[:, 0],
                        history[:, 1],
                        "-",
                        linewidth=1,
                        alpha=0.6,
                        color=robot_marker_color,
                    )
                if (
                    robot.target_pos is not None
                    and robot.planned_path
                    and len(robot.planned_path) >= 1
                ):
                    try:
                        planned_path_with_current = [
                            robot.position.copy()
                        ] + robot.planned_path
                        path = np.array(planned_path_with_current)
                        ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1, zorder=4)
                        ax.plot(
                            robot.target_pos[0],
                            robot.target_pos[1],
                            markersize=6,
                            zorder=999,
                            marker="x",
                            ls="-",
                            c="black",
                            mec="black",
                        )
                    except Exception as plot_err:
                        pass
            except IndexError:
                logger.error(
                    f"Error subplot w/o window R{i} @ ({row},{col}), total_rows={total_rows}. Skip."
                )
                continue

        plt.tight_layout()
        self._save_frame_to_memory()  # 這裡總是儲存

    def _save_frame_to_memory(self) -> None:
        """將當前圖形保存到記憶體或直接寫入影片（減少記憶體佔用）。"""
        import io

        from PIL import Image

        # 取得目前畫布為 PIL Image
        # 取得目前畫布為 PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        try:
            image = Image.open(buf).convert("RGB")
            frame = np.array(image)  # RGB ndarray (H,W,3)
        except Exception as e:
            buf.close()
            logger.error(f"_save_frame_to_memory: failed to capture frame: {e}")
            return
        finally:
            buf.close()

        # 若支援 OpenCV 且啟用串流 -> 直接寫入 VideoWriter（不保留於記憶體）
        if getattr(self, "_video_streaming_available", False):
            try:
                import cv2

                # 初始化 video writer（第一次寫入時）
                if self._video_writer is None:
                    # 建臨時檔案
                    import tempfile

                    tmp = tempfile.NamedTemporaryFile(
                        prefix="hybrid_vid_", suffix=".mp4", delete=False
                    )
                    self._video_tmp_path = tmp.name
                    tmp.close()
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    # 使用 5 fps 預設（可改）
                    self._video_writer = cv2.VideoWriter(
                        self._video_tmp_path, fourcc, 5, (width, height)
                    )
                    if not self._video_writer.isOpened():
                        logger.error(
                            "_save_frame_to_memory: Failed to open VideoWriter, falling back to buffer."
                        )
                        self._video_writer = None
                if self._video_writer is not None:
                    # OpenCV 需 BGR
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self._video_writer.write(frame_bgr)
                    return
            except Exception as e:
                logger.warning(
                    f"_save_frame_to_memory: video streaming failed, fallback to buffer: {e}"
                )
                # 若串流失敗則退回到緩衝模式

        # 緩衝模式：儲存壓縮 JPEG bytes（比 numpy 陣列小），並限制緩衝大小
        try:
            from PIL import Image

            buf2 = io.BytesIO()
            image.save(buf2, format="JPEG", quality=70)  # 壓縮品質可調
            buf2.seek(0)
            img_bytes = buf2.read()
            buf2.close()
            self.frames_data.append(img_bytes)
            # 若超過上限，丟棄最舊的幀
            if len(self.frames_data) > self._frames_buffer_max:
                # 保留最新 N 幀
                excess = len(self.frames_data) - self._frames_buffer_max
                del self.frames_data[0:excess]
        except Exception as e:
            logger.error(f"_save_frame_to_memory: failed to compress/store frame: {e}")

    def save_video(self, filename: str = "exploration_video.mp4", fps: int = 5) -> None:
        """將所有幀匯出為影片。若使用串流已在臨時檔產生影片，則關閉並搬移檔案；否則由緩衝產生影片。"""
        # 若正在以 VideoWriter 串流
        if getattr(self, "_video_writer", None) is not None:
            try:
                import shutil

                import cv2

                self._video_writer.release()
                self._video_writer = None
                # 將臨時檔案移動到目標位置
                if self._video_tmp_path is not None:
                    shutil.move(self._video_tmp_path, filename)
                    logger.info(f"Saved streamed video to {filename}")
                    self._video_tmp_path = None
                    return
            except Exception as e:
                logger.error(
                    f"save_video: failed to finalize streamed video: {e}", exc_info=True
                )
                # 若失敗則嘗試使用緩衝（如果有）
        # 若沒有串流影片，使用緩衝 frames_data 產生影片
        if not hasattr(self, "frames_data") or not self.frames_data:
            logger.warning("No frames to save")
            return

        # 建立一個暫存的影像檔列表並用 OpenCV 寫入影片
        try:
            # 先將 bytes 重建為 numpy frames 並取得尺寸
            first_img = Image.open(io.BytesIO(self.frames_data[0])).convert("RGB")
            height, width = np.array(first_img).shape[:2]
            # Try avc1 (H.264) first for better compatibility, then mp4v
            try:
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                if not video_writer.isOpened():
                    logger.warning("Failed to open VideoWriter with avc1, falling back to mp4v")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            except Exception:
                 logger.warning("Exception with avc1, falling back to mp4v")
                 fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                 video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

            if not video_writer.isOpened():
                logger.error(f"save_video: Failed to open VideoWriter for {filename}")
                return
            for img_bytes in self.frames_data:
                arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            video_writer.release()
            logger.info(f"Video saved as {filename}")
        except Exception as e:
            logger.error(
                f"save_video: failed to write buffered frames: {e}", exc_info=True
            )
        finally:
            # 無論成功或失敗都釋放緩衝
            self.frames_data = []
