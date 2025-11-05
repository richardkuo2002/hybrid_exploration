import os
import sys
sys.path.insert(0, os.getcwd())

from env import Env
from parameter import *
import numpy as np

if __name__ == '__main__':
    E = Env(n_agent=1, k_size=K_SIZE, map_index=0, plot=False)
    server_vals = np.unique(E.server.global_map)
    robot_vals = np.unique(E.robot_list[0].local_map)
    print('server unique values:', ','.join(map(str, server_vals.tolist())))
    print('robot unique values:', ','.join(map(str, robot_vals.tolist())))
