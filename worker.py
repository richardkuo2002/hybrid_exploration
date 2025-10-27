import numpy as np
import copy
import sys
import random

from parameter import *
from robot import Robot
from env import Env

class Worker:
    def __init__(self, global_step=0, agent_num=3, map_index=0, plot=False):
        
        self.global_step = global_step
        self.agent_num = agent_num
        self.k_size = K_SIZE
        self.env = Env(self.agent_num, map_index=map_index)
        self.step_count = 0
        self.plot = plot
        
        for i, robot in enumerate(self.env.robot_list):
            iter = min(copy.deepcopy(i), len(robot.node_coords)-1 )    # In case idx out of bounds
            robot_position = robot.node_coords[iter]
            robot.position = robot_position
            # if hasattr(robot, 'movement_history'):
            #     robot.movement_history.append(robot.position.copy())

    def run_episode(self, curr_episode=0):
        """
        使用混合式策略執行一個完整探險回合，
        每一步驟呼叫 env.step() 同步地圖與圖結構，並選擇動作移動機器人。
        """
        for step in range(MAX_EPS_STEPS):
            # print(f"=====Step {step}=====")
            for i, robot in enumerate(self.env.robot_list):
                
                # 3. 呼叫混合式 step：本地更新 + 與其他機器人/伺服器同步
                info = self.env.step(i, step)
            
            done, coverage = self.env.server_step()
            
            msg_cnt = ""
            
            for i, robot in enumerate(self.env.robot_list):
                # 1. 檢查機器人是否有剩餘路徑，若無則重新規劃
                if len(robot.planned_path) < 1:
                    # 選擇目標節點
                    # if robot.out_range_step > OUT_RANGE_STEP and not robot.target_given_by_server:
                    #     target_pos = robot.last_position_in_server_range
                    # else:
                    target_pos, action_idx, min_valid_dists = self.select_node(i)
                    if min_valid_dists > (robot.sensor_range * 1.5) and robot.out_range_step:
                        target_pos = robot.last_position_in_server_range
                    robot.target_pos = target_pos
                    robot.target_given_by_server = False
                    
                    # 規劃完整路徑（從當前位置到目標節點）
                    full_path = self.env.plan_local_path(i, target_pos)
                    robot.planned_path = full_path if full_path else [robot.position]
                    # print(f"Robot {i} planned new path with {robot.planned_path} steps")
                # 2. 從路徑中取出下一步，只移動一個節點
                if len(robot.planned_path) >= 1:
                    next_step = robot.planned_path[0]  # 移除並取得第一個節點
                    next_step_equal_robot_position = False
                    if np.array_equal(next_step, robot.position):
                        next_step = robot.planned_path[1]  # 移除並取得第一個節點
                        next_step_equal_robot_position = True
                    is_blocked = False
                    for other_id, other_robot in enumerate(self.env.robot_list):
                        if other_id != i:  # 排除自己
                            if np.linalg.norm(next_step - other_robot.position) < 1:
                                is_blocked = True
                                robot.stay_count += 1   # 計待在原地多久
                                if robot.stay_count > 2 and i > other_id: # 待在原地兩次且對方id在前面，則重新規劃路徑
                                    full_path = self.env.plan_local_path_again(i, robot.target_pos, next_step)
                                    robot.planned_path = full_path if full_path else [robot.position]
                                    next_step = robot.planned_path[0]  # 移除並取得第一個節點
                                    next_step_equal_robot_position = False
                                    if np.array_equal(next_step, robot.position):
                                        next_step = robot.planned_path[1]  # 移除並取得第一個節點
                                        next_step_equal_robot_position = True
                                    is_blocked = False
                                break
                    if not is_blocked:
                        robot.planned_path.pop(0)
                        if next_step_equal_robot_position:
                            robot.planned_path.pop(0)
                        robot.position = np.array(next_step)
                        dist_to_server = np.linalg.norm(robot.position - self.env.server.position)
                        
                        if dist_to_server < SERVER_COMM_RANGE:
                            self.env.server.all_robot_position[i] = robot.position
                            robot.last_position_in_server_range = robot.position
                            self.env.server.robot_in_range[i] = True
                            robot.out_range_step = 0
                            if not robot.target_given_by_server:
                                robot.planned_path = []
                        else:
                            if not robot.target_given_by_server:
                                robot.out_range_step += 1
                            self.env.server.robot_in_range[i] = False
                        
                        if hasattr(robot, 'movement_history'):
                            robot.movement_history.append(robot.position.copy())
                        # print(f"Robot {i} moved to {robot.position}, palanned path with{robot.planned_path} steps")
                    # else:
                        # print(f"Robot {i} waiting at {robot.position}, path preserved")
                else:
                    print(f"Robot {i} reached destination or no valid path")
                
                msg_cnt += f"| Robot {i}'s target candidates num: {len(robot.graph_generator.target_candidates)} "

            # 每個完整循環後可視化
            # self.env.plot_env(step)
            
            # self.env.plot_env_without_window(step)
            
            
            msg = f"\rEP: {curr_episode} | Step {step:5d} | Coverage: {coverage * 100:6.2f}% "
            msg += msg_cnt
            sys.stdout.write(msg)
            sys.stdout.flush()
            
            if done:
                print(f"\n Episode {curr_episode} completed at step {step}")
                print(info)
                self.env.save_video(f"episode_{curr_episode}.mp4")
                return True, step
            self.step_count = step
            
        print(f"\n Episode {curr_episode} not completed at step {step}")
        print(info)
        self.env.save_video(f"episode_{curr_episode}.mp4")
        return False, step

    def select_node(self, robot_id: int):
        """
        根據演算法選擇下一個圖節點作為移動目標。
        過濾掉 utility 為 0，以及被其他機器人佔據的節點。
        """
        robot = self.env.robot_list[robot_id]
        coords = robot.node_coords          # (M,2)
        candidates = robot.graph_generator.target_candidates
        
        if candidates is None or len(candidates) == 0:
            # print(f"Warning: No available nodes for robot {robot_id}, using current position")
            return robot.position, 0, 0
        # utilities = robot.node_utility      # (M,)
        utilities = robot.graph_generator.candidates_utility
        dists = np.linalg.norm(candidates - robot.position, axis=1)
        # 1. 過濾掉 utility 為 0 的節點
        valid_mask = utilities > 0
        
        # 2. 過濾掉與**任何**其他機器人位置相同的節點
        for robot in self.env.robot_list:
            # if other_id != robot_id:
            # 找出 coords 中恰好等於 other_robot.position 的索引
            same_pos = np.all(candidates == robot.position, axis=1)
            valid_mask &= ~same_pos  # 將這些位置標為 False
        
        # 如果沒有任何合法節點
        if not np.any(valid_mask):
            # print(f"Warning: No valid nodes for robot {robot_id} after filtering")
            return robot.position, 0, 0
        
        # 3. 基於 valid_mask 建立有效節點子集
        valid_candidates = candidates[valid_mask]
        valid_utilities = utilities[valid_mask]
        valid_dists = dists[valid_mask]
        
        # 4. 計算分數並選出最佳
        λ = 1
        min_valid_dists = min(valid_dists)
        scores = λ * valid_utilities /  valid_dists
        best_idx_in_valid = np.argmax(scores)
        selected_coord = valid_candidates[best_idx_in_valid]
        
        # 5. 回推到原 coords 的索引
        original_indices = np.where(valid_mask)[0]
        original_idx = original_indices[best_idx_in_valid]
        
        # print(f"Robot {robot_id+1}'s target score {scores[best_idx_in_valid]}(utility:{valid_utilities[original_idx]})")
        
        return selected_coord, original_idx, min_valid_dists




if __name__ == '__main__':
    # random_index = random.randint(0, 10000)
    
    worker = Worker(global_step=0, agent_num=3, map_index=7)
    success = worker.run_episode(curr_episode=0)
    print('Episode finished:', success)
