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
    def __init__(self, global_step=0, agent_num=3, map_index=0, plot=False, save_video=True):
        
        self.global_step = global_step
        self.agent_num = agent_num
        self.k_size = K_SIZE
        self.env = Env(self.agent_num, map_index=map_index, k_size=self.k_size, plot=plot)
        self.step_count = 0
        self.plot = plot
        self.save_video = save_video
        
        for i, robot in enumerate(self.env.robot_list):
            if robot.node_coords is not None and len(robot.node_coords) > 0:
                iter = min(copy.deepcopy(i), len(robot.node_coords)-1 )
                robot_position = robot.node_coords[iter]
                robot.position = robot_position
                robot.movement_history.append(robot.position.copy())
            else:
                print(f"Warning: Robot {i} has no node_coords at init.")


    def run_episode(self, curr_episode=0):
        """
        使用混合式策略執行一個完整探險回合 (重構後的主循環)。
        """
        for step in range(MAX_EPS_STEPS):
            
            # === 階段一：機器人自主行動 (感知、決策、移動) ===
            msg_cnt = ""
            for i, robot in enumerate(self.env.robot_list):
                
                # 1. 機器人感知與更新地圖、圖結構
                robot.update_local_awareness(
                    self.env.real_map,
                    self.env.robot_list,
                    self.env.server,
                    self.env.find_frontier,  # 傳遞 env 的工具函式
                    self.env.merge_maps      # 傳遞 env 的工具函式
                )
                
                # 2. 機器人決策 (如果需要新目標)
                if robot.needs_new_target():
                    if not robot.is_in_server_range:
                        # 只有在伺服器範圍外才自主決策
                        robot.decide_next_target(self.env.robot_list)
                    # (如果在伺服器範圍內，它會等待 Server 指派)
                
                # 3. 機器人移動
                robot.move_one_step(self.env.robot_list)

                msg_cnt += f"| R{i} targets: {len(robot.graph_generator.target_candidates)} "

            # === 階段二：伺服器集中調度 ===
            done, coverage = self.env.server.update_and_assign_tasks(
                self.env.robot_list,
                self.env.real_map,
                self.env.find_frontier
            )
            
            # === 階段三：繪圖與日誌 ===
            # <--- 修改點：移除 PLOT_INTERVAL，恢復每幀繪圖 --- >
            if self.plot:
                self.env.plot_env(step)
            else:
                self.env.plot_env_without_window(step)
            
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
    parser.add_argument('--TEST_MAP_INDEX', type=int, default=2, help='Index of the map to use for the test (default: 2)')
    parser.add_argument('--TEST_AGENT_NUM', type=int, default=3, help='Number of agents to use for the test (default: 3)')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=False, help='Enable real-time plotting (will also save video if --no_save_video is not used)')
    parser.add_argument('--save_video', action=argparse.BooleanOptionalAction, default=True, help='Save the video at the end (default: True). Use --no_save_video to disable.')

    args = parser.parse_args()

    print(f"Starting worker with Map Index: {args.TEST_MAP_INDEX}, Agent Num: {args.TEST_AGENT_NUM}, Plot: {args.plot}, Save Video: {args.save_video}")
    
    worker = Worker(
        global_step=0, 
        agent_num=args.TEST_AGENT_NUM, 
        map_index=args.TEST_MAP_INDEX, 
        plot=args.plot,
        save_video=args.save_video
    )
    
    success, steps = worker.run_episode(curr_episode=0)
    print(f'Episode finished: {success} in {steps} steps.')