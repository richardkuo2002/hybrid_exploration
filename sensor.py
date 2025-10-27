#######################################################################
# Name: sensor.py
# Simulate the sensor model of Lidar.
#######################################################################

import numpy as np
import copy

def collision_check(x0, y0, x1, y1, real_map, robot_local_map):
    """ Checks if line is blocked by obstacle """
    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    collision_flag = 0
    max_collision = 10

    while 0 <= x < real_map.shape[1] and 0 <= y < real_map.shape[0]:
        k = real_map.item(y, x)
        if k == 1 and collision_flag < max_collision:
            collision_flag += 1
            if collision_flag >= max_collision:
                break

        if k !=1 and collision_flag > 0:
            break

        if x == x1 and y == y1:
            break

        robot_local_map[int(y), int(x)] = k

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return robot_local_map


def sensor_work(robot_position, sensor_range, robot_local_map, real_map):
    """ Expands explored region on map """
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0
    x0 = robot_position[0]
    y0 = robot_position[1]
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        robot_local_map = collision_check(x0, y0, x1, y1, real_map, robot_local_map)
        sensor_angle += sensor_angle_inc
    return robot_local_map
