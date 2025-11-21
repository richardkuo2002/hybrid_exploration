import numpy as np
from numba import jit # <--- 1. 匯入 numba
from parameter import PIXEL_OCCUPIED, PIXEL_UNKNOWN

@jit(nopython=True) # <--- 2. 加上 JIT 裝飾器
def check_collision(start: np.ndarray, end: np.ndarray, robot_map: np.ndarray) -> bool:
    """檢查從 start 到 end 的線段是否與地圖中的障礙或未知區域發生碰撞 (Bresenham)。

    Args:
        start (np.ndarray): 起點座標 [x, y]（整數或可轉為整數）。
        end (np.ndarray): 終點座標 [x, y]（整數或可轉為整數）。
        robot_map (np.ndarray): 表示地圖的 2D numpy 陣列 (y, x)。

    Returns:
        bool: 若路徑上有碰撞（遇到障礙或未知）則回傳 True，否則 False。
    """
    collision = False
    map = robot_map 

    x0 = start[0]
    y0 = start[1]
    x1 = end[0]
    y1 = end[1]
    # ... (函式內容完全不變) ...
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        # .item() 在 nopython=True 中可能不支持，改成標準索引
        k = map[int(y), int(x)] # <--- 3. 將 .item() 改成標準索引
        if x == x1 and y == y1:
            break
        if k == PIXEL_OCCUPIED:
            collision = True
            break
        if k == PIXEL_UNKNOWN:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return collision