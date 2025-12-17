
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import heapq
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Set, Dict

from graph_generator import Graph_generator
from parameter import *

# Constants for Visualization
VIS_SCALE = 1  # Map is 640x640, might need scaling? No, 640 is fine for video.
COLOR_FREE = (255, 255, 255)
COLOR_OCCUPIED = (0, 0, 0)
COLOR_START = (0, 255, 0)      # Green
COLOR_GOAL = (0, 0, 255)       # Blue
COLOR_PATH = (255, 0, 0)       # Red
COLOR_VISITED_GRID = (200, 200, 255) # Light Red/Pink for Grid Visited
COLOR_VISITED_GRAPH = (255, 200, 200) # Light Blue for Graph Visited (BGR)
# Note: CV2 uses BGR

def load_map_cv2(map_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads map and returns BGR image and binary grid."""
    img = cv2.imread(map_path)
    if img is None:
        raise FileNotFoundError(f"Map not found: {map_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold
    # Free space is usually light, obstacles dark.
    # Check Env logic: PIXEL_FREE=255
    _, binary_map = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find start position (optional, usually center or heuristic)
    # Using generic start 
    start_pos = np.array([320, 320])
    
    return img, binary_map, start_pos

class VideoRecorder:
    def __init__(self, filename, width, height, fps=30):
        self.filename = filename
        self.width = width
        self.height = height
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filename, self.fourcc, fps, (width, height))
        
    def write(self, frame):
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)
        
    def release(self):
        self.writer.release()

# --- Grid A* with Visualization Hooks ---
def grid_a_star_vis(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], 
                   bg_image: np.ndarray, recorder: VideoRecorder, label="Grid A*"):
    
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    
    visited = set()
    visited.add(start)
    
    vis_img = bg_image.copy()
    
    # Draw Start/Goal
    cv2.circle(vis_img, start, 5, (0, 255, 0), -1) # BGR: Green
    cv2.circle(vis_img, goal, 5, (0, 0, 255), -1)  # BGR: Red
    
    steps = 0
    # Frame skip for speed
    SKIP_FRAMES = 50 
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while open_set:
        steps += 1
        current = heapq.heappop(open_set)[1]
        
        # Visualize "Current"
        # Mark visited on image
        vis_img[current[1], current[0]] = (0, 255, 255) # Yellow for visiting
        
        if steps % SKIP_FRAMES == 0:
            # Refresh text
            display_img = vis_img.copy()
            cv2.putText(display_img, f"{label}: Steps {steps}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            recorder.write(display_img)
        
        if current == goal:
            # Reconstruct path
            path = []
            curr = current
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(start)
            
            # Draw Path
            for p in path:
                 cv2.circle(vis_img, p, 2, (255, 0, 0), -1) # Blue Path
            
            # Final frames
            for _ in range(30):
                display_img = vis_img.copy()
                cv2.putText(display_img, f"{label}: Found! Len {len(path)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                recorder.write(display_img)
            return True, vis_img
        
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if 0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows:
                if grid[neighbor[1], neighbor[0]] == 255: # Free
                    dist = 1.414 if dx != 0 and dy != 0 else 1.0
                    tentative_g_score = g_score[current] + dist
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal))
                        heapq.heappush(open_set, (f_score, neighbor))
                        
                        if neighbor not in visited:
                            visited.add(neighbor)
                            vis_img[neighbor[1], neighbor[0]] = (200, 200, 200) # Grey visited

    return False, vis_img

# --- Graph A* Visualization ---
def graph_a_star_vis(gg: Graph_generator, node_coords, start_node_idx, goal_node_idx, 
                     bg_image: np.ndarray, recorder: VideoRecorder, label="Graph A*"):
    
    # Standard A* on graph
    # gg.graph is Dict[tuple, Dict]
    # We need to map index back to coords for drawing
    
    vis_img = bg_image.copy()
    
    # Draw Graph Edges (Background)
    # Too dense? draw only nodes
    for node in node_coords:
        pt = (int(node[0]), int(node[1]))
        cv2.circle(vis_img, pt, 2, (100, 100, 100), -1)
        
    start_pt = (int(node_coords[start_node_idx][0]), int(node_coords[start_node_idx][1]))
    goal_pt = (int(node_coords[goal_node_idx][0]), int(node_coords[goal_node_idx][1]))
    
    cv2.circle(vis_img, start_pt, 8, (0, 255, 0), -1)
    cv2.circle(vis_img, goal_pt, 8, (0, 0, 255), -1)
    
    # A*
    open_set = []
    # (f, curr_idx)
    heapq.heappush(open_set, (0, start_node_idx))
    
    came_from = {}
    g_score = {start_node_idx: 0}
    
    steps = 0
    # Graph is huge?
    
    start_coords = node_coords[start_node_idx]
    goal_coords = node_coords[goal_node_idx]

    while open_set:
        steps += 1
        curr_score, curr_idx = heapq.heappop(open_set)
        curr_coords = node_coords[curr_idx]
        curr_pt = (int(curr_coords[0]), int(curr_coords[1]))
        
        # Draw visited
        cv2.circle(vis_img, curr_pt, 4, (0, 255, 255), -1)
        
        # Link to parent for viz
        if curr_idx in came_from:
            parent_idx = came_from[curr_idx]
            parent_pt = (int(node_coords[parent_idx][0]), int(node_coords[parent_idx][1]))
            cv2.line(vis_img, parent_pt, curr_pt, (0, 100, 255), 1)
            
        display_img = vis_img.copy()
        cv2.putText(display_img, f"{label}: Steps {steps}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        recorder.write(display_img)
        
        if curr_idx == goal_node_idx:
            # Found
             # Reconstruct path
            path_nodes = []
            curr = curr_idx
            while curr in came_from:
                path_nodes.append(curr)
                curr = came_from[curr]
            path_nodes.append(start_node_idx)
            
            # Draw Path
            for i in range(len(path_nodes)-1):
                p1 = node_coords[path_nodes[i]]
                p2 = node_coords[path_nodes[i+1]]
                pt1 = (int(p1[0]), int(p1[1]))
                pt2 = (int(p2[0]), int(p2[1]))
                cv2.line(vis_img, pt1, pt2, (255, 0, 0), 3)
            
            for _ in range(30):
                display_img = vis_img.copy()
                cv2.putText(display_img, f"{label}: Found! Len {len(path_nodes)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                recorder.write(display_img)
            return True, vis_img
        
        # Neighbors
        curr_tuple = tuple(curr_coords)
        if curr_tuple in gg.graph:
            for neighbor_tuple, edge_data in gg.graph[curr_tuple].items():
                # Find neighbor index
                # Optimization: use dict mapping
                # But here we loop for simplicity or assume index access
                # gg.graph stores Edge objects.
                # Assuming node_coords aligns? 
                # This part is tricky if we don't have index map.
                # Let's simple scan or precompute.
                pass 
                
        # Actually gg.find_shortest_path logic uses:
        # neighbors = self.graph[current_node_tuple]
        # But we need INDICES for A* priority queue usually, or just store tuples.
        # Let's switch to using Tuples in heap to avoid index lookup, simpler.
    
    return False, vis_img

def graph_a_star_vis_tuples(gg: Graph_generator, start_coords, goal_coords, 
                     bg_image: np.ndarray, recorder: VideoRecorder, label="Graph A*"):
    
    vis_img = bg_image.copy()

    # Draw ALL Graph Edges (Background) - Static
    # This visualizes the "Pre-built" graph structure
    for u, u_adj in gg.graph.edges.items():
        u_pt = (int(u[0]), int(u[1]))
        cv2.circle(vis_img, u_pt, 2, (80, 80, 80), -1) # Dark gray nodes
        for v, edge in u_adj.items():
            v_pt = (int(v[0]), int(v[1]))
            cv2.line(vis_img, u_pt, v_pt, (60, 60, 60), 1) # Dark gray edges

    # Draw Start/Goal
    start_pt = (int(start_coords[0]), int(start_coords[1]))
    goal_pt = (int(goal_coords[0]), int(goal_coords[1]))
    
    cv2.circle(vis_img, start_pt, 8, (0, 255, 0), -1)
    cv2.circle(vis_img, goal_pt, 8, (0, 0, 255), -1)
    
    # Burn explicit "Graph Structure" frames before searching
    display_img = vis_img.copy()
    cv2.putText(display_img, f"{label}: Structure Built ({len(gg.graph.nodes)} nodes)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for _ in range(20): recorder.write(display_img)

    # Start A*
    open_set = []
    heapq.heappush(open_set, (0, tuple(start_coords)))
    came_from = {}
    g_score = {tuple(start_coords): 0}
    
    steps = 0
    
    while open_set:
        steps += 1
        curr_score, current = heapq.heappop(open_set) # current is tuple
        curr_pt = (int(current[0]), int(current[1]))
        
        # Draw Visited (Yellow)
        cv2.circle(vis_img, curr_pt, 3, (0, 255, 255), -1)
        if current in came_from:
            parent = came_from[current]
            parent_pt = (int(parent[0]), int(parent[1]))
            # Trace active search path (Red/Orange)
            cv2.line(vis_img, parent_pt, curr_pt, (0, 165, 255), 1) 

        display_img = vis_img.copy()
        cv2.putText(display_img, f"{label}: Searching... Visited {steps}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        recorder.write(display_img)
        
        if np.linalg.norm(np.array(current) - np.array(goal_coords)) < 1e-3: # Approx match
             # Path
            path = []
            curr = current
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(tuple(start_coords))
            
            for i in range(len(path)-1):
                pt1 = (int(path[i][0]), int(path[i][1]))
                pt2 = (int(path[i+1][0]), int(path[i+1][1]))
                cv2.line(vis_img, pt1, pt2, (255, 0, 0), 3)
                
            for _ in range(30):
                recorder.write(vis_img)
            return True, vis_img
        
        # Neighbors
        if current in gg.graph.edges:
            for neighbor, edge in gg.graph.edges[current].items():
                tentative_g_score = g_score[current] + edge.length
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal_coords))
                    heapq.heappush(open_set, (f, neighbor))

    return False, vis_img

def run_vis():
    map_path = "/home/oem/hybrid_exploration/maps/easy_even/img_100.png" # Using a valid map file
    if not os.path.exists(map_path):
        print("Map not found")
        return

    # Load
    bg_img, binary_map, start_pos = load_map_cv2(map_path)
    h, w = binary_map.shape
    
    # 1. Grid A*
    # Choose two points far apart
    start = (20, 20)
    goal = (w-20, h-20)
    # Ensure they are free
    # choose start/goal within bounds
    margin = 20
    while binary_map[start[1], start[0]] != 255: 
        start = (random.randint(margin, w-margin), random.randint(margin, h-margin))
    while binary_map[goal[1], goal[0]] != 255: 
        goal = (random.randint(margin, w-margin), random.randint(margin, h-margin))
    
    print(f"Start: {start}, Goal: {goal}")
    
    # Setup Video
    rec_grid = VideoRecorder("comparison_grid.mp4", w, h, fps=60) # Fast FPS
    
    print("Running Grid A*...")
    t0 = time.time()
    grid_a_star_vis(binary_map, start, goal, bg_img, rec_grid)
    print(f"Grid Done: {time.time()-t0:.2f}s")
    rec_grid.release()
    
    # 2. Graph A*
    print("Building Graph...")
    gg = Graph_generator(map_size=(h, w), k_size=20, sensor_range=80, plot=False)
    # We need to construct the graph manually or use generate_graph
    # generate_graph uses start_position to build connected component?
    # Or takes robot_map?
    # gg.generate_graph(start, robot_map, None)
    # Note: start arg in generate_graph is robot position.
    
    # We use the FULL map for graph generation but pass Start/Goal to trigger Pruning
    # Hybrid System only keeps nodes relevant to Frontiers (Goal) and Robot (Start)
    # This simulates the "Active Graph Pruning" efficiency.
    frontiers = np.array([goal])
    node_coords, edges, _, _ = gg.generate_graph(np.array(start), binary_map, frontiers)
    print(f"Graph built: {len(node_coords)} nodes (Pruned)")
    
    # Find nearest nodes to start/goal
    def find_nearest(pt, nodes):
        dists = np.linalg.norm(nodes - np.array(pt), axis=1)
        return nodes[np.argmin(dists)]
    
    start_node = find_nearest(start, node_coords)
    goal_node = find_nearest(goal, node_coords)
    
    rec_graph = VideoRecorder("comparison_graph.mp4", w, h, fps=10) # Slower FPS as fewer steps
    print("Running Graph A*...")
    t0 = time.time()
    graph_a_star_vis_tuples(gg, start_node, goal_node, bg_img, rec_graph)
    print(f"Graph Done: {time.time()-t0:.2f}s")
    rec_graph.release()
    
    # Merge Videos? 
    # Or just leave as two files. User can view comparison.
    # Actually side-by-side on html/markdown is fine.

if __name__ == "__main__":
    run_vis()
