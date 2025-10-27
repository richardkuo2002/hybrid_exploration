from parameter import *
from graph_generator import Graph_generator

class Robot():
    def __init__(self):
        self.position = None
        self.local_map = None
        self.downsampled_map = None
        self.sensor_range = SENSOR_RANGE
        self.frontiers = []
        self.node_coords = []
        self.local_map_graph = []
        self.node_utility = []
        self.guidepost = []
        self.planned_path = []
        self.target_pos = None
        self.movement_history = []
        self.graph_generator:Graph_generator = None
        self.target_gived_by_server = False
        self.last_position_in_server_range = None
        self.out_range_step = 0
        self.stay_count = 0