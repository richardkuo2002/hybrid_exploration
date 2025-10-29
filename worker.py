import numpy as np
import copy
import sys
import random
import datetime
import argparse
import logging

from parameter import *
from robot import Robot
from env import Env

logger = logging.getLogger(__name__)

class Worker:
    def __init__(self, global_step=0, agent_num=3, map_index=0, plot=False, save_video=True):

        self.global_step = global_step
        self.agent_num = agent_num
        self.k_size = K_SIZE
        self.env = Env(self.agent_num, map_index=map_index, k_size=self.k_size, plot=plot)
        self.step_count = 0
        self.plot = plot
        self.save_video = save_video

        for i, robot in enumerate(self.env.robot_list):
            # robot.debug = self.debug # No longer needed
            if not hasattr(robot, 'node_coords') or robot.node_coords is None or len(robot.node_coords) == 0:
                 logger.warning(f"Robot {i} has no node_coords at init.")
            else:
                 iter_idx = min(i, len(robot.node_coords)-1 )
                 robot_position = robot.node_coords[iter_idx]
                 robot.position = robot_position
                 # Initialize movement history if needed
                 if not hasattr(robot, 'movement_history') or not robot.movement_history:
                     robot.movement_history = [robot.position.copy()]
                 elif not np.array_equal(robot.position, robot.movement_history[-1]):
                      robot.movement_history.append(robot.position.copy())

        # self.env.server.debug = self.debug # No longer needed


    def run_episode(self, curr_episode=0):
        PLOT_INTERVAL = 5
        for step in range(MAX_EPS_STEPS):

            # === 階段一：機器人自主行動 ===
            msg_cnt = ""
            for i, robot in enumerate(self.env.robot_list):
                try: # Add error handling for robot step
                    robot.update_local_awareness(
                        self.env.real_map, self.env.robot_list, self.env.server,
                        self.env.find_frontier, self.env.merge_maps
                    )
                    if robot.needs_new_target():
                        if not robot.is_in_server_range:
                            robot.decide_next_target(self.env.robot_list)
                    robot.move_one_step(self.env.robot_list)
                except Exception as e:
                     logger.error(f"Error in Robot {i} step {step}: {e}", exc_info=True)
                     # Decide how to handle: continue, stop robot, stop simulation?
                     # For now, log and continue

                num_targets = 0
                if hasattr(robot.graph_generator, 'target_candidates') and robot.graph_generator.target_candidates is not None:
                     num_targets = len(robot.graph_generator.target_candidates)
                msg_cnt += f"| R{i} targets: {num_targets} "

            # === 階段二：伺服器集中調度 ===
            try: # Add error handling for server step
                done, coverage = self.env.server.update_and_assign_tasks(
                    self.env.robot_list, self.env.real_map, self.env.find_frontier
                )
            except Exception as e:
                 logger.error(f"Error in Server step {step}: {e}", exc_info=True)
                 done = False # Assume not done if server fails
                 coverage = self.env.calculate_coverage_ratio() # Get current coverage


            # === 階段三：繪圖與日誌 ===
            if (self.plot or self.save_video) and (step % PLOT_INTERVAL == 0):
                try: # Add error handling for plotting
                    if self.plot:
                        self.env.plot_env(step)
                    elif self.save_video:
                        self.env.plot_env_without_window(step)
                except Exception as e:
                     logger.error(f"Error during plotting/saving frame at step {step}: {e}")


            msg = f"\rEP: {curr_episode} | Step {step:5d} | Coverage: {coverage * 100:6.2f}% "
            msg += msg_cnt
            sys.stdout.write(msg)
            sys.stdout.flush()

            if done:
                sys.stdout.write("\n")
                logger.info(f"Episode {curr_episode} completed at step {step} with coverage {coverage*100:.2f}%")
                if self.save_video:
                    # <--- 修改點：更改時間格式 --- >
                    time_str = datetime.datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
                    # --- ---
                    map_idx = self.env.map_index
                    robot_count = self.agent_num
                    filename = f"{time_str}_map{map_idx}_robots{robot_count}.mp4"
                    self.env.save_video(filename)
                return True, step

            self.step_count = step

        sys.stdout.write("\n")
        logger.warning(f"Episode {curr_episode} timeout at step {step} with coverage {coverage*100:.2f}%")
        if self.save_video:
            # <--- 修改點：更改時間格式 --- >
            time_str = datetime.datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
            # --- ---
            map_idx = self.env.map_index
            robot_count = self.agent_num
            filename = f"{time_str}_map{map_idx}_robots{robot_count}.mp4"
            self.env.save_video(filename)
        return False, step


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a single episode...")
    parser.add_argument('--TEST_MAP_INDEX', type=int, default=1, help='Map index (default: 1)')
    parser.add_argument('--TEST_AGENT_NUM', type=int, default=3, help='Number of agents (default: 3)')
    parser.add_argument('--plot', action='store_true', help='Enable real-time plotting')
    parser.add_argument('--save_video', action=argparse.BooleanOptionalAction, default=True, help='Save video (default: True)')
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG level logging (default: INFO)')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)-8s [%(name)s] %(message)s', datefmt='%H:%M:%S') # 加入時間戳

    logger.info(f"Starting worker with Map Index: {args.TEST_MAP_INDEX}, Agent Num: {args.TEST_AGENT_NUM}, Plot: {args.plot}, Save Video: {args.save_video}")

    try: # Add top-level error handling
        worker = Worker(
            global_step=0,
            agent_num=args.TEST_AGENT_NUM,
            map_index=args.TEST_MAP_INDEX,
            plot=args.plot,
            save_video=args.save_video,
        )
        success, steps = worker.run_episode(curr_episode=0)
        logger.info(f'Episode finished: {success} in {steps} steps.')
    except Exception as e:
         logger.critical(f"Simulation failed with unhandled exception: {e}", exc_info=True)
         sys.exit(1) # Exit with error code