import numpy as np
import copy
import sys
import random
import datetime
import argparse

from parameter import *
from robot import Robot
from env import Env

class Worker:
    # <--- 修改點：新增 debug 參數 ---
    def __init__(self, global_step=0, agent_num=3, map_index=0, plot=False, save_video=True, debug=False):

        self.global_step = global_step
        self.agent_num = agent_num
        self.k_size = K_SIZE
        # <--- 修改點：將 debug 傳給 Env (如果 Env 需要) 或 Robot/Server ---
        # Env 本身不需要 debug, 但 Robot/Server 需要
        self.env = Env(self.agent_num, map_index=map_index, k_size=self.k_size, plot=plot)
        self.step_count = 0
        self.plot = plot
        self.save_video = save_video
        self.debug = debug # <--- 儲存 debug 狀態

        # <--- 修改點：在初始化 Robot/Server 時傳入 debug 狀態 ---
        # (假設 Env 在 __init__ 中創建了 Robot/Server 實例)
        # 我們需要在 Env 的 __init__ 中修改以接收 debug 參數
        # 或者在這裡創建 Robot/Server 並傳遞給 Env (後者更好)
        # 為了簡化，我們先假設 Env 的 __init__ 會處理 debug 參數的傳遞
        # (需要同步修改 env.py)
        # ---> 更正：直接修改 Robot 和 Server 的 __init__ 並在這裡設置更簡單

        for i, robot in enumerate(self.env.robot_list):
            robot.debug = self.debug # 直接設置 Robot 的 debug 標誌
            if robot.node_coords is not None and len(robot.node_coords) > 0:
                iter = min(copy.deepcopy(i), len(robot.node_coords)-1 )
                robot_position = robot.node_coords[iter]
                robot.position = robot_position
                robot.movement_history.append(robot.position.copy())
            else:
                print(f"Warning: Robot {i} has no node_coords at init.")
        
        # 設置 Server 的 debug 標誌
        self.env.server.debug = self.debug


    def run_episode(self, curr_episode=0):
        """
        使用混合式策略執行一個完整探險回合 (重構後的主循環)。
        """
        for step in range(MAX_EPS_STEPS):

            # === 階段一：機器人自主行動 (感知、決策、移動) ===
            msg_cnt = ""
            for i, robot in enumerate(self.env.robot_list):

                robot.update_local_awareness(
                    self.env.real_map,
                    self.env.robot_list,
                    self.env.server,
                    self.env.find_frontier,
                    self.env.merge_maps
                )

                if robot.needs_new_target():
                    if not robot.is_in_server_range:
                        robot.decide_next_target(self.env.robot_list)

                robot.move_one_step(self.env.robot_list)

                # 更新狀態欄的目標數量 (不受 debug 影響)
                num_targets = len(robot.graph_generator.target_candidates) if hasattr(robot.graph_generator, 'target_candidates') and robot.graph_generator.target_candidates is not None else 0
                msg_cnt += f"| R{i} targets: {num_targets} "


            # === 階段二：伺服器集中調度 ===
            done, coverage = self.env.server.update_and_assign_tasks(
                self.env.robot_list,
                self.env.real_map,
                self.env.find_frontier
            )

            # === 階段三：繪圖與日誌 ===
            if self.plot:
                self.env.plot_env(step)
            # <--- 修改點：確保 plot_env_without_window 在 save_video 時被呼叫 ---
            elif self.save_video: # 只有在需要存影片但又不 plot 時才呼叫
                self.env.plot_env_without_window(step)

            # 更新狀態欄 (不受 debug 影響)
            msg = f"\rEP: {curr_episode} | Step {step:5d} | Coverage: {coverage * 100:6.2f}% "
            msg += msg_cnt
            sys.stdout.write(msg)
            sys.stdout.flush()

            if done:
                print(f"\n Episode {curr_episode} completed at step {step} with coverage {coverage*100:.2f}%")
                if self.save_video:
                    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    map_idx = self.env.map_index
                    robot_count = self.agent_num
                    filename = f"{time_str}_map{map_idx}_robots{robot_count}.mp4"
                    self.env.save_video(filename)
                return True, step

            self.step_count = step

        print(f"\n Episode {curr_episode} timeout at step {step} with coverage {coverage*100:.2f}%")
        if self.save_video:
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            map_idx = self.env.map_index
            robot_count = self.agent_num
            filename = f"{time_str}_map{map_idx}_robots{robot_count}.mp4"
            self.env.save_video(filename)
        return False, step


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run a single episode of the hybrid exploration simulation.")
    parser.add_argument('--TEST_MAP_INDEX', type=int, default=1, help='Index of the map to use for the test (default: 2)')
    parser.add_argument('--TEST_AGENT_NUM', type=int, default=3, help='Number of agents to use for the test (default: 3)')
    parser.add_argument('--plot', action='store_true', help='Enable real-time plotting (will also save video if --no_save_video is not used)')
    parser.add_argument('--save_video', action=argparse.BooleanOptionalAction, default=True, help='Save the video at the end (default: True). Use --no_save_video to disable.')
    # <--- 修改點：新增 debug 參數 ---
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug logging to console (default: False)')
    # --- ---

    args = parser.parse_args()

    # <--- 修改點：更新 print 訊息 ---
    print(f"Starting worker with Map Index: {args.TEST_MAP_INDEX}, Agent Num: {args.TEST_AGENT_NUM}, Plot: {args.plot}, Save Video: {args.save_video}, Debug: {args.debug}")

    worker = Worker(
        global_step=0,
        agent_num=args.TEST_AGENT_NUM,
        map_index=args.TEST_MAP_INDEX,
        plot=args.plot,
        save_video=args.save_video,
        debug=args.debug # <--- 傳入 debug 參數
    )

    success, steps = worker.run_episode(curr_episode=0)
    print(f'Episode finished: {success} in {steps} steps.')