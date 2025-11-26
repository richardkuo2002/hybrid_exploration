import argparse
import copy
import datetime
import logging
import multiprocessing
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from env import Env
from parameter import *
from robot import Robot

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        global_step: int = 0,
        agent_num: int = 3,
        map_index: int = 0,
        plot: bool = False,
        save_video: bool = True,
        force_sync_debug: bool = False,
        graph_update_interval: Optional[int] = None,
    ) -> None:
        """建立 Worker 實例，初始化環境與機器人位置。

        Args:
            global_step (int): 全域步數初始值。
            agent_num (int): 機器人數量。
            map_index (int): 要使用的地圖索引。
            plot (bool): 是否啟用即時繪圖。
            save_video (bool): 是否儲存模擬影片。
            force_sync_debug (bool): 是否強制同步除錯。
            graph_update_interval (Optional[int]): 圖更新間隔。

        Returns:
            None
        """
        self.global_step = global_step
        self.agent_num = agent_num
        self.k_size = K_SIZE
        self.env = Env(
            self.agent_num,
            map_index=map_index,
            k_size=self.k_size,
            plot=plot,
            force_sync_debug=force_sync_debug,
            graph_update_interval=graph_update_interval,
            debug_mode=logger.getEffectiveLevel() == logging.DEBUG,
        )
        self.step_count = 0
        self.plot = plot
        self.save_video = save_video
        for i, robot in enumerate(self.env.robot_list):
            if (
                not hasattr(robot, "node_coords")
                or robot.node_coords is None
                or len(robot.node_coords) == 0
            ):
                logger.warning(f"Robot {i} has no node_coords at init.")
            else:
                iter_idx = min(i, len(robot.node_coords) - 1)
                robot_position = robot.node_coords[iter_idx]
                robot.position = robot_position
                if not hasattr(robot, "movement_history") or not robot.movement_history:
                    robot.movement_history = [robot.position.copy()]
                elif not np.array_equal(robot.position, robot.movement_history[-1]):
                    robot.movement_history.append(robot.position.copy())

    def run_episode(self, curr_episode: int = 0) -> Tuple[bool, int]:
        """執行一個 episode 的主迴圈。

        Args:
            curr_episode (int): 當前 episode 編號。

        Returns:
            Tuple[bool, int, Dict[str, Any]]:
                success (bool): 是否完成探索 (True=完成)。
                step (int): 結束時的步數或 timeout 步數。
                metrics (Dict[str, Any]): 包含 coverage, total_distance, replanning_count, map_merge_count 等。
        """
        # <--- 修改點：恢復 PLOT_INTERVAL ---
        PLOT_INTERVAL = 5
        output_dir = "videos"

        # ThreadPoolExecutor for parallel sensor updates
        # Use a reasonable number of threads (e.g., min(32, agent_num))
        max_workers = min(32, self.agent_num)
        
        # Metrics tracking
        total_distance = 0.0
        target_selection_count = 0 # Renamed from replanning_count
        collision_replan_count = 0 # New metric
        map_merge_count = 0
        
        # Track previous positions for distance calculation
        prev_positions = [robot.position.copy() if robot.position is not None else np.zeros(2) for robot in self.env.robot_list]

        for step in range(MAX_EPS_STEPS):
            msg_cnt = ""
            
            # === Phase 1: Parallel Sensor & Graph Update ===
            # Execute sense_and_update_graph in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, robot in enumerate(self.env.robot_list):
                    futures.append(
                        executor.submit(
                        robot.sense_and_update_graph,
                            self.env.real_map,
                            self.env.find_frontier,
                            self.env.robot_list,
                        )
                    )
                # Wait for all to complete and handle exceptions
                for i, future in enumerate(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error in Robot {i} parallel update step {step}: {e}", exc_info=True)

            # === Phase 2: Sequential Interaction & Decision ===
            for i, robot in enumerate(self.env.robot_list):
                try:
                    # Interaction & Merge (Sequential)
                    robot.interact_and_merge(
                        self.env.robot_list,
                        self.env.server,
                        self.env.merge_maps,
                    )
                    map_merge_count += 1 # Count merge operations (per robot per step)
                    
                    # Decision & Move (Sequential)
                    if robot.needs_new_target():
                        if not robot.is_in_server_range:
                            robot.decide_next_target(self.env.robot_list)
                            target_selection_count += 1 # Count target selections
                    robot.move_one_step(self.env.robot_list)
                    
                    # Calculate distance
                    if robot.position is not None:
                        dist = np.linalg.norm(robot.position - prev_positions[i])
                        total_distance += dist
                        prev_positions[i] = robot.position.copy()

                except Exception as e:
                    logger.error(f"Error in Robot {i} sequential step {step}: {e}", exc_info=True)
                
                num_targets = 0
                if (
                    hasattr(robot.graph_generator, "target_candidates")
                    and robot.graph_generator.target_candidates is not None
                ):
                    num_targets = len(robot.graph_generator.target_candidates)
                msg_cnt += f"| R{i} targets: {num_targets} "

            # Debug: 印出目前每台機器人的位置 (在 server 更新之前)
            positions_debug = []
            for i, robot in enumerate(self.env.robot_list):
                try:
                    if robot.position is None:
                        positions_debug.append(f"R{i}:None")
                    else:
                        positions_debug.append(f"R{i}:{robot.position.tolist()}")
                except Exception:
                    positions_debug.append(f"R{i}:ERR")
            logger.debug("[Debug] Robot Positions: %s", ", ".join(positions_debug))

            # === Update Server State & Merge Maps ===
            # Update robot positions and range status on server
            self.env.server.all_robot_position = [r.position for r in self.env.robot_list]
            self.env.server.robot_in_range = [r.is_in_server_range for r in self.env.robot_list]

            # Merge local maps from robots in range into global map
            maps_to_merge = [self.env.server.global_map]
            for r in self.env.robot_list:
                if r.is_in_server_range:
                    maps_to_merge.append(r.local_map)
            
            if len(maps_to_merge) > 1:
                merged_global = self.env.merge_maps(maps_to_merge)
                self.env.server.global_map[:] = merged_global

            # ... (階段二：伺服器集中調度) ...
            try:
                done, coverage = self.env.server.update_and_assign_tasks(
                    self.env.robot_list, self.env.real_map, self.env.find_frontier
                )
            except Exception as e:
                logger.error(f"Error in Server step {step}: {e}", exc_info=True)
                done = False
                coverage = self.env.calculate_coverage_ratio()  # 使用 env 的方法

            # <--- 修改點：新的繪圖/儲存邏輯 ---
            # === 階段三：繪圖與日誌 ===
            try:
                # Case 1: 啟用即時繪圖
                if self.plot:
                    # 只在間隔時更新繪圖 (為了效能)
                    if step % PLOT_INTERVAL == 0:
                        # 呼叫 plot_env, 並根據 save_video 決定是否儲存影格
                        self.env.plot_env(step, save_frame=self.save_video)

                # Case 2: 未啟用繪圖，但啟用儲存影片
                elif self.save_video:
                    # 每一步都儲存影格，以獲得流暢影片
                    self.env.plot_env_without_window(step)

            except Exception as e:
                logger.error(
                    f"Error plotting/saving frame at step {step}: {e}", exc_info=True
                )
            # --- ---

            msg = f"\rEP: {curr_episode} | Step {step:5d} | Coverage: {coverage * 100:6.2f}% "
            msg += msg_cnt
            sys.stdout.write(msg)
            sys.stdout.flush()

            # ... (if done: ... 區塊保持不變) ...
            if done:
                sys.stdout.write("\n")
                logger.info(
                    f"Episode {curr_episode} completed at step {step} with coverage {coverage*100:.2f}%"
                )
                if self.save_video:
                    os.makedirs(output_dir, exist_ok=True)
                    time_str = datetime.datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
                    map_idx = self.env.map_index
                    robot_count = self.agent_num
                    base_filename = f"{time_str}_map{map_idx}_robots{robot_count}.mp4"
                    filename = os.path.join(output_dir, base_filename)
                    self.env.save_video(filename)
                    filename = os.path.join(output_dir, base_filename)
                    self.env.save_video(filename)
                
                metrics = {
                    "coverage": coverage,
                    "total_distance": total_distance,
                    "replanning_count": replanning_count,
                    "map_merge_count": map_merge_count,
                }
                return True, step, metrics

            self.step_count = step

        # ... (timeout 區塊保持不變) ...
        sys.stdout.write("\n")
        logger.warning(
            f"Episode {curr_episode} timeout at step {step} with coverage {coverage*100:.2f}%"
        )
        if self.save_video:
            os.makedirs(output_dir, exist_ok=True)
            time_str = datetime.datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
            map_idx = self.env.map_index
            robot_count = self.agent_num
            base_filename = f"{time_str}_map{map_idx}_robots{robot_count}.mp4"
            filename = os.path.join(output_dir, base_filename)
            self.env.save_video(filename)
            filename = os.path.join(output_dir, base_filename)
            self.env.save_video(filename)
        
        metrics = {
            "coverage": coverage,
            "total_distance": total_distance,
            "coverage": coverage,
            "total_distance": total_distance,
            "target_selection_count": target_selection_count,
            "collision_replan_count": sum(r.collision_replan_count for r in self.env.robot_list),
            "map_merge_count": map_merge_count,
        }
        return False, step, metrics


if __name__ == "__main__":
    # ... (argparse 邏輯不變) ...
    parser = argparse.ArgumentParser(description="Run a single episode...")
    parser.add_argument(
        "--TEST_MAP_INDEX", type=int, default=1, help="Map index (default: 1)"
    )
    parser.add_argument(
        "--TEST_AGENT_NUM", type=int, default=3, help="Number of agents (default: 3)"
    )
    parser.add_argument("--plot", action="store_true", help="Enable real-time plotting")
    parser.add_argument(
        "--save_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save video (default: True)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug/info level logging (default: show only ERROR/CRITICAL)",
    )
    parser.add_argument(
        "--force_sync_debug",
        action="store_true",
        help="(debug) force server to sync candidates/frontiers to all robots each step",
    )
    parser.add_argument(
        "--graph-update-interval",
        "-g",
        type=int,
        default=None,
        help="Graph full rebuild interval override",
    )
    args = parser.parse_args()

    # Default: only show ERROR and CRITICAL to reduce log noise.
    # If --debug is provided, enable DEBUG so we can print per-step robot positions.
    log_level = logging.DEBUG if args.debug else logging.ERROR
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info(
        f"Starting worker with Map Index: {args.TEST_MAP_INDEX}, Agent Num: {args.TEST_AGENT_NUM}, Plot: {args.plot}, Save Video: {args.save_video}"
    )

    try:
        worker = Worker(
            global_step=0,
            agent_num=args.TEST_AGENT_NUM,
            map_index=args.TEST_MAP_INDEX,
            plot=args.plot,
            save_video=args.save_video,
            force_sync_debug=args.force_sync_debug,
            graph_update_interval=args.graph_update_interval,
        )
        success, steps, metrics = worker.run_episode(curr_episode=0)
        logger.info(f"Episode finished: {success} in {steps} steps. Metrics: {metrics}")
    except Exception as e:
        logger.critical(
            f"Simulation failed with unhandled exception: {e}", exc_info=True
        )
        sys.exit(1)
