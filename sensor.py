#######################################################################
# Name: sensor.py
# Simulate the sensor model of Lidar.
#######################################################################

import numpy as np
import copy
from numba import jit # <--- 1. 匯入 jit
# 假設 collision_check 已經在 utils.py 中被 @jit 修飾
from utils import check_collision # 確保從 utils 匯入的是 JIT 版本

# 注意：因為 sensor_work 呼叫了 check_collision，
# 且 check_collision 被 JIT 編譯，
# 所以 sensor_work 也必須被 JIT 編譯才能高效呼叫。
@jit(nopython=True) # <--- 2. 加上 JIT 裝飾器
def sensor_work(robot_position, sensor_range, robot_local_map, real_map):
    """ Expands explored region on map (JIT Compiled) """
    # 角度增量需要是常數，或者傳入
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0.0 # 使用浮點數
    x0 = robot_position[0]
    y0 = robot_position[1]
    
    # 預先計算 map 邊界，避免在迴圈內重複計算
    map_height, map_width = real_map.shape
    
    # Numba 不支援直接修改傳入的 numpy array，但通常可以運作。
    # 如果遇到問題，可能需要回傳一個新的 map。
    # 暫時假設可以直接修改 robot_local_map。
    
    local_map_copy = robot_local_map.copy() # Numba 中建議操作 copy

    while sensor_angle < 2 * np.pi:
        # 使用 np.cos/sin，numba 支援
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        
        # 呼叫 JIT 編譯過的 check_collision
        # 注意：check_collision 現在應該回傳修改後的 map
        local_map_copy = _sensor_collision_check_wrapper(x0, y0, x1, y1, real_map, local_map_copy, map_height, map_width)

        sensor_angle += sensor_angle_inc
        
    return local_map_copy # 回傳修改後的 map copy

# 因為 JIT 函式不能直接修改傳入的 array 引數來回傳結果，
# 我們需要一個 wrapper 來處理 check_collision 的回傳值。
# 原本 utils.check_collision 應修改為不直接修改傳入的 map，而是回傳碰撞點或路徑點。
# 但為了簡化，我們先假設 utils.check_collision 被修改成類似以下形式
# (或者創建一個新的輔助函式)

# 這裡我們先創建一個模擬的 JIT 內部輔助函式，它執行類似 check_collision 的邏輯
# 但直接修改傳入的 local_map_copy
@jit(nopython=True)
def _sensor_collision_check_wrapper(x0_f, y0_f, x1_f, y1_f, real_map, local_map_copy, map_height, map_width):
    """ Internal JIT helper for line checking, modifies local_map_copy """
    # Numba 需要整數索引
    x0, y0 = int(round(x0_f)), int(round(y0_f))
    x1, y1 = int(round(x1_f)), int(round(y1_f))

    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    
    # 邊界檢查初始點
    if not (0 <= x < map_width and 0 <= y < map_height):
         return local_map_copy # 起點無效

    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    # Numba 不支援 sys.maxsize, 用一個大數
    max_steps = 10000 
    steps = 0

    while steps < max_steps:
        steps += 1
        # 邊界檢查
        if not (0 <= x < map_width and 0 <= y < map_height):
            break

        k = real_map[y, x] # 使用標準索引

        # 更新 local map (只更新 free 和 obstacle)
        if k == 1 or k == 255: # 假設 1=obs, 255=free
             local_map_copy[y, x] = k

        # 碰到障礙物停止
        if k == 1:
            break
        # 碰到未知區域也可能需要停止，取決於感測器模型
        # if k == 127:
        #    break

        # 到達終點
        # (因為是射線，不需要檢查是否精確到達 x1, y1)
        # if x == x1 and y == y1:
        #    break
            
        # Bresenham 步進
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
            
        # 檢查是否超出 sensor_range (近似)
        # Numba 不支援 np.linalg.norm, 手動計算
        current_dist_sq = (x - x0)**2 + (y - y0)**2
        # sensor_range_sq = sensor_range**2 # 在外部計算傳入會更好
        # if current_dist_sq > sensor_range_sq:
        #    break # 超出範圍

    return local_map_copy

# 移除舊的 collision_check 函式 (已移到 utils.py)