from parameter import *
from graph_generator import Graph_generator

class Server():
    def __init__(self):
        self.position = None
        self.global_map = None
        self.downsampled_map = None
        self.comm_range = SERVER_COMM_RANGE
        self.all_robot_position = []
        self.robot_in_range = []
        self.graph_generator:Graph_generator = None
        self.frontiers = []
        self.node_coords = []
        self.local_map_graph = []
        self.node_utility = []
        self.guidepost = []
    