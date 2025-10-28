import sys
from parameter import *
import numpy as np
from numba import jit
from utils import check_collision

# <--- 修改點：將 JIT 函式移到類別外部 ---

@jit(nopython=True)
def _initialize_observable_frontiers_jit_internal(coords_param, frontiers_param, robot_map_param, utility_calc_range):
    """ (Standalone JIT Function) Calculates observable frontier indices. """
    observable_indices = [] 
    # 安全檢查 frontiers_param
    if frontiers_param.shape[0] == 0:
        return observable_indices

    utility_calc_range_sq = utility_calc_range**2 # 預先計算平方

    for i in range(len(frontiers_param)):
        point = frontiers_param[i]
        # 手動計算距離平方
        dist_sq = (point[0] - coords_param[0])**2 + (point[1] - coords_param[1])**2
        
        if dist_sq < utility_calc_range_sq: # 使用平方比較
            collision = check_collision(coords_param, point, robot_map_param)
            if not collision:
                 observable_indices.append(i) # 儲存索引
                 
    return observable_indices

@jit(nopython=True)
def _update_observable_frontiers_jit_internal(coords_param, 
                                             new_frontiers_param, robot_map_param,
                                             utility_calc_range):
    """ (Standalone JIT Function) Calculates newly observable frontier indices from new_frontiers. """
    newly_added_indices = [] 
    # 安全檢查 new_frontiers_param
    if new_frontiers_param.shape[0] == 0:
        return newly_added_indices
        
    utility_calc_range_sq = utility_calc_range**2 # 預先計算平方

    if len(new_frontiers_param) > 0:
        for i in range(len(new_frontiers_param)):
            point = new_frontiers_param[i]
            dist_sq = (point[0] - coords_param[0])**2 + (point[1] - coords_param[1])**2
            if dist_sq < utility_calc_range_sq: # 使用平方比較
                collision = check_collision(coords_param, point, robot_map_param)
                if not collision:
                    newly_added_indices.append(i)
                    
    return newly_added_indices

# --- ---

class Node():
    def __init__(self, coords, frontiers, robot_map):
        self.coords = coords
        self.observable_frontiers_list = [] 
        self.sensor_range = SENSOR_RANGE 
        self._initialize_observable_frontiers_jit(frontiers, robot_map) # <--- 現在呼叫的是外部 JIT
        self.utility = self.get_node_utility()
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False
            
    # <--- 修改點：移除內部的 JIT 函式定義 ---

    # Wrapper 函式現在呼叫外部的 JIT 函式
    def _initialize_observable_frontiers_jit(self, frontiers, robot_map):
        if frontiers is None or len(frontiers) == 0:
            self.observable_frontiers_list = []
            return
            
        # <--- 修改點：直接呼叫外部函式 ---
        observable_indices = _initialize_observable_frontiers_jit_internal(
            self.coords, frontiers, robot_map, UTILITY_CALC_RANGE
        )
        self.observable_frontiers_list = [tuple(frontiers[i]) for i in observable_indices]


    def get_node_utility(self):
        return len(self.observable_frontiers_list)

    # <--- 修改點：移除內部的 JIT 函式定義 ---

    # Wrapper 函式現在呼叫外部的 JIT 函式
    def update_observable_frontiers(self, observed_frontiers_set, new_frontiers, robot_map):
        if len(observed_frontiers_set) > 0:
            self.observable_frontiers_list = [
                obs_tuple for obs_tuple in self.observable_frontiers_list 
                if obs_tuple not in observed_frontiers_set
            ]

        if new_frontiers is None or len(new_frontiers) == 0:
             newly_added_indices = []
        else:
            # <--- 修改點：直接呼叫外部函式 ---
             # (注意：這裡不需要傳 current_observable_indices 和 observed_indices 了)
            newly_added_indices = _update_observable_frontiers_jit_internal(
                self.coords, 
                new_frontiers, # 只傳入 new_frontiers
                robot_map, 
                UTILITY_CALC_RANGE
            )
        
        current_tuples = set(self.observable_frontiers_list)
        for idx in newly_added_indices:
            if idx < len(new_frontiers):
                 new_tuple = tuple(new_frontiers[idx])
                 if new_tuple not in current_tuples:
                     self.observable_frontiers_list.append(new_tuple)
                     current_tuples.add(new_tuple)

        self.utility = self.get_node_utility()
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def reset_observable_frontiers(self, new_frontiers, robot_map):
        self._initialize_observable_frontiers_jit(new_frontiers, robot_map) # Wrapper 不變
        
        self.utility = self.get_node_utility()
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def set_visited(self):
        self.observable_frontiers_list = []
        self.utility = 0
        self.zero_utility_node = True

    def frontiers_within_utility_calc_range(self, frontiers):
        # (保持不變)
        dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
        return len(dist_list[dist_list < GLOBAL_NODES_TO_FRONTIER_AVOID_SPARSE_RAD]) > 0