import sys
from typing import List, Optional, Set, Tuple

import numpy as np
from numba import jit

from parameter import *
from utils import check_collision

# <--- 修改點：將 JIT 函式移到類別外部 ---


@jit(nopython=True)
def _initialize_observable_frontiers_jit_internal(
    coords_param: np.ndarray,
    frontiers_param: np.ndarray,
    robot_map_param: np.ndarray,
    utility_calc_range: float,
) -> List[int]:
    """計算從 coords 可觀察到的 frontier 索引（JIT 優化）。

    Args:
        coords_param (np.ndarray): 節點座標。
        frontiers_param (np.ndarray): frontier 陣列 (N,2)。
        robot_map_param (np.ndarray): 地圖資料 (y,x)。
        utility_calc_range (float): 可觀察半徑。

    Returns:
        List[int]: 可觀察到的 frontier 索引列表（整數索引）。
    """
    observable_indices = []
    # 安全檢查 frontiers_param
    if frontiers_param.shape[0] == 0:
        return observable_indices

    utility_calc_range_sq = utility_calc_range**2  # 預先計算平方

    for i in range(len(frontiers_param)):
        point = frontiers_param[i]
        # 手動計算距離平方
        dist_sq = (point[0] - coords_param[0]) ** 2 + (point[1] - coords_param[1]) ** 2

        if dist_sq < utility_calc_range_sq:  # 使用平方比較
            collision = check_collision(coords_param, point, robot_map_param)
            if not collision:
                observable_indices.append(i)  # 儲存索引

    return observable_indices


@jit(nopython=True)
def _update_observable_frontiers_jit_internal(
    coords_param: np.ndarray,
    new_frontiers_param: np.ndarray,
    robot_map_param: np.ndarray,
    utility_calc_range: float,
) -> List[int]:
    """計算從 coords 對 new_frontiers 中新可觀察到的 frontier 索引（JIT 優化）。

    Args:
        coords_param (np.ndarray): 節點座標。
        new_frontiers_param (np.ndarray): 新的 frontier 陣列 (M,2)。
        robot_map_param (np.ndarray): 地圖資料 (y,x)。
        utility_calc_range (float): 可觀察半徑。

    Returns:
        List[int]: newly added frontier 的索引列表（相對於 new_frontiers_param）。
    """
    newly_added_indices = []
    # 安全檢查 new_frontiers_param
    if new_frontiers_param.shape[0] == 0:
        return newly_added_indices

    utility_calc_range_sq = utility_calc_range**2  # 預先計算平方

    if len(new_frontiers_param) > 0:
        for i in range(len(new_frontiers_param)):
            point = new_frontiers_param[i]
            dist_sq = (point[0] - coords_param[0]) ** 2 + (
                point[1] - coords_param[1]
            ) ** 2
            if dist_sq < utility_calc_range_sq:  # 使用平方比較
                collision = check_collision(coords_param, point, robot_map_param)
                if not collision:
                    newly_added_indices.append(i)

    return newly_added_indices


# --- ---


class Node:
    def __init__(
        self, coords: np.ndarray, frontiers: np.ndarray, robot_map: np.ndarray
    ) -> None:
        """建立 Node，並初始化可觀察的 frontier 列表與效用。

        Args:
            coords (np.ndarray): 節點座標。
            frontiers (np.ndarray): 當前 frontier 陣列 (N,2)。
            robot_map (np.ndarray): 當前機器人觀察到的地圖 (y,x)。

        Returns:
            None
        """
        self.coords = coords
        self.observable_frontiers_list: List[Tuple[int, int]] = []
        self.sensor_range = SENSOR_RANGE
        self._initialize_observable_frontiers_jit(
            frontiers, robot_map
        )  # <--- 現在呼叫的是外部 JIT
        self.utility = self.get_node_utility()
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    # <--- 修改點：移除內部的 JIT 函式定義 ---

    # Wrapper 函式現在呼叫外部的 JIT 函式
    def _initialize_observable_frontiers_jit(
        self, frontiers: np.ndarray, robot_map: np.ndarray
    ) -> None:
        """Wrapper：初始化 observable_frontiers_list。

        Args:
            frontiers (np.ndarray): frontier 陣列 (N,2)。
            robot_map (np.ndarray): 地圖資料 (y,x)。

        Returns:
            None
        """
        if frontiers is None or len(frontiers) == 0:
            self.observable_frontiers_list = []
            return

        # <--- 修改點：直接呼叫外部函式 ---
        observable_indices = _initialize_observable_frontiers_jit_internal(
            self.coords, frontiers, robot_map, UTILITY_CALC_RANGE
        )
        self.observable_frontiers_list = [
            tuple(frontiers[i]) for i in observable_indices
        ]

    def get_node_utility(self) -> int:
        """計算節點效用（可觀察 frontier 數量）。

        Returns:
            int: utility 值（整數）。
        """
        return len(self.observable_frontiers_list)

    # <--- 修改點：移除內部的 JIT 函式定義 ---

    # Wrapper 函式現在呼叫外部的 JIT 函式
    def update_observable_frontiers(
        self,
        observed_frontiers_set: Set[Tuple[int, int]],
        new_frontiers: np.ndarray,
        robot_map: np.ndarray,
    ) -> None:
        """更新 observable_frontiers（移除已被觀察到的，並加入 new_frontiers 中新可見的）。

        Args:
            observed_frontiers_set (Set[Tuple[int, int]]): 已被觀察到的 frontier 的 tuple set。
            new_frontiers (np.ndarray): 新的 frontier 陣列 (M,2)。
            robot_map (np.ndarray): 地圖資料 (y,x)。

        Returns:
            None
        """
        if len(observed_frontiers_set) > 0:
            self.observable_frontiers_list = [
                obs_tuple
                for obs_tuple in self.observable_frontiers_list
                if obs_tuple not in observed_frontiers_set
            ]

        if new_frontiers is None or len(new_frontiers) == 0:
            newly_added_indices = []
        else:
            # <--- 修改點：直接呼叫外部函式 ---
            # (注意：這裡不需要傳 current_observable_indices 和 observed_indices 了)
            newly_added_indices = _update_observable_frontiers_jit_internal(
                self.coords,
                new_frontiers,  # 只傳入 new_frontiers
                robot_map,
                UTILITY_CALC_RANGE,
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

    def reset_observable_frontiers(
        self, new_frontiers: np.ndarray, robot_map: np.ndarray
    ) -> None:
        """用新的 frontier 重置可觀察清單並更新效用。

        Args:
            new_frontiers (np.ndarray): 新的 frontier 陣列 (N,2)。
            robot_map (np.ndarray): 地圖資料 (y,x)。

        Returns:
            None
        """
        self._initialize_observable_frontiers_jit(
            new_frontiers, robot_map
        )  # Wrapper 不變

        self.utility = self.get_node_utility()
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def set_visited(self) -> None:
        """標記節點為已拜訪：清空 observable list 並將效用設為 0。

        Returns:
            None
        """
        self.observable_frontiers_list = []
        self.utility = 0
        self.zero_utility_node = True

    def frontiers_within_utility_calc_range(self, frontiers: np.ndarray) -> bool:
        """檢查是否有 frontiers 在預設避免稀疏半徑內（使用 numpy 計算距離）。

        Args:
            frontiers (np.ndarray): frontier 陣列 (N,2)。

        Returns:
            bool: 若存在符合條件的 frontier 回傳 True，否則 False。
        """
        # (保持不變)
        dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
        return len(dist_list[dist_list < GLOBAL_NODES_TO_FRONTIER_AVOID_SPARSE_RAD]) > 0
