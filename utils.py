import numpy as np

def check_collision(start, end, robot_map):
    """
    檢查兩點間路徑在地圖上是否有碰撞（障礙物），以 Bresenham 演算法沿線偵測。

    Args:
        start (array-like): 起點座標 [x, y]。
        end (array-like): 終點座標 [x, y]。
        robot_map (ndarray): 機器人本地地圖（陣列），其中 1 或 127 代表障礙物，其他代表可通行。

    Returns:
        collision (bool): True 代表路徑有碰撞（遇到障礙物），False 代表路徑暢通。
    """
    collision = False
    map = robot_map 

    x0 = start[0]
    y0 = start[1]
    x1 = end[0]
    y1 = end[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == 1:
            collision = True
            break
        if k == 127:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return collision