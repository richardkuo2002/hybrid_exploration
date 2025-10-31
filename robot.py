import numpy as np
from collections import deque
import logging

from parameter import *
from graph_generator import Graph_generator
from sensor import sensor_work
from skimage.measure import block_reduce
import sys
from graph import Graph, Edge # <--- 確保匯入 Graph 和 Edge

logger = logging.getLogger(__name__)

class Robot():
    def __init__(self, start_position, real_map_size, resolution, k_size, plot=False): # Removed debug
        self.position = start_position
        self.local_map = np.ones(real_map_size) * 127
        self.downsampled_map = None
        self.sensor_range = SENSOR_RANGE
        self.frontiers = []
        self.node_coords = None
        self.local_map_graph = None # This will store graph.edges dict
        self.node_utility = None
        self.guidepost = None
        self.planned_path = []
        self.target_pos = None
        self.movement_history = [start_position.copy()]
        self.graph_generator:Graph_generator = Graph_generator(map_size=real_map_size, sensor_range=self.sensor_range, k_size=k_size, plot=plot)
        if start_position is not None:
             self.graph_generator.route_node.append(start_position)

        self.target_gived_by_server = False
        self.last_position_in_server_range = start_position
        self.out_range_step = 0
        self.stay_count = 0

        self.resolution = resolution
        self.is_in_server_range = True

        self.info_gain_history = deque(maxlen=INFO_GAIN_HISTORY_LEN)
        self.current_info_gain = 0
        self.last_explored_area = 0

        self.graph_update_counter = 0

        self.robot_id = -1
        self.is_returning = False
        self.return_replan_attempts = 0
        self.return_fail_cooldown = 0
        # self.debug removed


    def update_local_awareness(self, real_map, all_robots, server, find_frontier_func, merge_maps_func):
        """
        執行一步的感知和地圖更新。
        """
        # --- 1. 感測與地圖合併 ---
        self.local_map = sensor_work(self.position, self.sensor_range, self.local_map, real_map)
        dist_to_server = np.linalg.norm(self.position - server.position)
        was_in_range = self.is_in_server_range
        self.is_in_server_range = dist_to_server < SERVER_COMM_RANGE

        if self.is_in_server_range and not was_in_range:
             self.is_returning = False; self.return_replan_attempts = 0; self.return_fail_cooldown = 0
             logger.debug(f"Robot {self.robot_id} re-entered server range. State reset.")

        # --- 機器人交互 ---
        for other_robot in all_robots:
            if other_robot is self: continue
            
            dist = np.linalg.norm(self.position - other_robot.position)
            
            if dist < ROBOT_COMM_RANGE:
                # 1. Standard Map Merge
                merged = merge_maps_func([self.local_map, other_robot.local_map])
                self.local_map[:] = merged
                other_robot.local_map[:] = merged
                
                # --- 2. 修改點：機會主義任務交接 (Task Handoff Logic) ---
                
                # Case 1: 我 (self) 正在返回 (A)，遇到 剛出發的 B (other_robot)
                i_am_returning = self.is_returning
                other_has_server_task = other_robot.target_gived_by_server and not other_robot.is_returning
                
                # Case 2: 我 (self) 剛出發 (B)，遇到 正在返回的 A (other_robot)
                i_have_server_task = self.target_gived_by_server and not self.is_returning
                other_is_returning = other_robot.is_returning
                
                if i_am_returning and other_has_server_task:
                    logger.info(f"[R{self.robot_id} Handoff] I am returning, taking task from R{other_robot.robot_id}. R{other_robot.robot_id} is now returning.")
                    
                    # 我 (A) 接收 B 的任務
                    self.target_pos = other_robot.target_pos
                    self.planned_path = other_robot.planned_path
                    self.target_gived_by_server = True # 標記為伺服器任務
                    self.is_returning = False; self.return_replan_attempts = 0; self.return_fail_cooldown = 0
                    
                    # B 接收我的返回任務 (作為數據中繼)
                    other_robot.target_pos = self.last_position_in_server_range # B 的新目標是伺服器
                    other_robot.planned_path = [] # 強制 B 在下一步重新規劃返回路徑
                    other_robot.is_returning = True # B 現在是返回狀態
                    other_robot.target_gived_by_server = False # B 不再是執行伺服器任務
                    other_robot.return_replan_attempts = 0 # 重置 B 的計數器

                elif i_have_server_task and other_is_returning:
                    logger.info(f"[R{self.robot_id} Handoff] I am departing, giving task to R{other_robot.robot_id}. I am now returning.")
                    
                    # 儲存我 (B) 的原始任務
                    my_original_target = self.target_pos
                    my_original_path = self.planned_path
                    
                    # 我 (B) 接收 A 的返回任務
                    self.target_pos = other_robot.last_position_in_server_range
                    self.planned_path = [] # 強制我重新規劃返回路徑
                    self.is_returning = True
                    self.target_gived_by_server = False
                    self.return_replan_attempts = 0
                    
                    # A 接收我 (B) 的原始任務
                    other_robot.target_pos = my_original_target
                    other_robot.planned_path = my_original_path
                    other_robot.target_gived_by_server = True
                    other_robot.is_returning = False; other_robot.return_replan_attempts = 0; other_robot.return_fail_cooldown = 0

                # Case 3: 其他情況 (例如兩個自主探索者相遇)
                elif not self.is_in_server_range and not self.is_returning and not self.target_gived_by_server:
                     # 雙方都清空路徑，以便基於合併後的新地圖重新決策
                     self.planned_path = []
                
                # (如果 other_robot 也是 Case 3，它自己的 update_local_awareness 會清空它的路徑)
                # --- 任務交接結束 ---

        # --- 1c. 與伺服器同步 (在交互之後) ---
        if self.is_in_server_range:
             merged = merge_maps_func([self.local_map, server.global_map])
             self.local_map[:] = merged; server.global_map[:] = merged
             if 0 <= self.robot_id < len(server.all_robot_position):
                 server.all_robot_position[self.robot_id] = self.position
                 server.robot_in_range[self.robot_id] = True
             self.last_position_in_server_range = self.position
             self.out_range_step = 0
             if not self.target_gived_by_server: self.planned_path = []
        else:
            # <--- 修正: 只有在非伺服器任務且非返回時才計數 ---
            if not self.target_gived_by_server and not self.is_returning:
                 self.out_range_step += 1
            if 0 <= self.robot_id < len(server.robot_in_range):
                server.robot_in_range[self.robot_id] = False

        # --- 2. 計算地圖增益 ---
        current_explored_area = np.sum(self.local_map == 255)
        self.current_info_gain = current_explored_area - self.last_explored_area
        self.last_explored_area = current_explored_area
        if not self.is_in_server_range and not self.target_gived_by_server and not self.is_returning:
             self.info_gain_history.append(self.current_info_gain)
        elif self.is_in_server_range:
             self.info_gain_history.clear()

        # --- 3. 更新 Frontiers ---
        self.downsampled_map = block_reduce(self.local_map.copy(), block_size=(self.resolution, self.resolution), func=np.min)
        new_frontiers = find_frontier_func(self.downsampled_map)

        # --- 4. 更新節點圖 (間歇性) ---
        self.graph_update_counter += 1
        if self.graph_update_counter % GRAPH_UPDATE_INTERVAL == 0:
            logger.debug(f"[R{self.robot_id} Awareness] Rebuilding graph structure...")
            try:
                node_coords, graph_edges, node_utility, guidepost = self.graph_generator.rebuild_graph_structure(
                    self.local_map, new_frontiers, self.frontiers, self.position
                )
                self.node_coords = node_coords
                self.local_map_graph = graph_edges
                self.node_utility = node_utility
                self.guidepost = guidepost
            except Exception as e:
                 logger.error(f"Robot {self.robot_id} failed rebuild_graph_structure: {e}", exc_info=True)
        else:
            logger.debug(f"[R{self.robot_id} Awareness] Updating node utilities...")
            try:
                node_utility, guidepost = self.graph_generator.update_node_utilities(
                    self.local_map, new_frontiers, self.frontiers
                )
                self.node_utility = node_utility
                self.guidepost = guidepost
                if hasattr(self.graph_generator, 'graph') and hasattr(self.graph_generator.graph, 'edges'):
                     self.local_map_graph = self.graph_generator.graph.edges
            except Exception as e:
                 logger.error(f"Robot {self.robot_id} failed update_node_utilities: {e}", exc_info=True)

        self.frontiers = new_frontiers

    # ... (needs_new_target, decide_next_target, move_one_step, _select_node 保持不變) ...
    def needs_new_target(self):
         return len(self.planned_path) < 1

    def decide_next_target(self, all_robots):
        MAX_RETURN_REPLAN_ATTEMPTS = 2
        RETURN_FAIL_COOLDOWN_STEPS = 10
        logger.debug(f"[R{self.robot_id} Decide] Pos:{self.position}, InRange:{self.is_in_server_range}, OutSteps:{self.out_range_step}, Return:{self.is_returning}, Cooldown:{self.return_fail_cooldown}")
        if self.is_returning:
            if len(self.planned_path) < 1:
                if self.return_replan_attempts < MAX_RETURN_REPLAN_ATTEMPTS:
                    logger.debug(f"[R{self.robot_id} Decide] Return path empty. Replan attempt {self.return_replan_attempts + 1}.")
                    self.target_pos = self.last_position_in_server_range
                    self.planned_path = self._plan_local_path(self.target_pos)
                    self.return_replan_attempts += 1
                    if not self.planned_path or (len(self.planned_path) == 1 and np.array_equal(self.planned_path[0], self.position)):
                         logger.debug(f"[R{self.robot_id} Decide] Replan attempt {self.return_replan_attempts} failed.")
                         self.planned_path = [self.position]
                else:
                    logger.warning(f"[R{self.robot_id} Decide] Failed return plan after {MAX_RETURN_REPLAN_ATTEMPTS} attempts. Aborting.")
                    self.is_returning = False; self.return_replan_attempts = 0; self.return_fail_cooldown = RETURN_FAIL_COOLDOWN_STEPS
            else:
                 logger.debug(f"[R{self.robot_id} Decide] Continue return path."); return
        if self.return_fail_cooldown > 0:
            logger.debug(f"[R{self.robot_id} Decide] In cooldown ({self.return_fail_cooldown} left). Force local.")
            self.return_fail_cooldown -= 1
            target_pos_local, _, _ = self._select_node(all_robots); self.target_pos = target_pos_local
            self.planned_path = self._plan_local_path(self.target_pos)
            if not self.planned_path: self.planned_path = [self.position]
            self.is_returning = False; self.target_gived_by_server = False; return
        target_pos_local, _, min_valid_dists = self._select_node(all_robots)
        max_local_utility = 0
        if hasattr(self.graph_generator, 'candidates_utility') and self.graph_generator.candidates_utility is not None and len(self.graph_generator.candidates_utility) > 0:
             max_local_utility = np.max(self.graph_generator.candidates_utility)
        recent_info_gain = np.sum(self.info_gain_history)
        is_stagnated = (len(self.info_gain_history) == INFO_GAIN_HISTORY_LEN) and (recent_info_gain < MIN_INFO_GAIN_THRESHOLD)
        is_depleted = max_local_utility < LOCAL_UTILITY_THRESHOLD
        is_timeout = self.out_range_step > OUT_RANGE_STEP
        should_return_criteria = (is_stagnated and is_depleted) or is_timeout
        local_target_too_far = min_valid_dists > (self.sensor_range * 1.5) and self.out_range_step > 5
        logger.debug(f"[R{self.robot_id} Decide] Checks: Stag={is_stagnated}({recent_info_gain}<{MIN_INFO_GAIN_THRESHOLD}), Dep={is_depleted}({max_local_utility}<{LOCAL_UTILITY_THRESHOLD}), Time={is_timeout}({self.out_range_step}>{OUT_RANGE_STEP}), Far={local_target_too_far}({min_valid_dists:.1f})")
        decision_reason = "Local Explore"
        if should_return_criteria or local_target_too_far:
            target_pos = self.last_position_in_server_range
            planned_return_path = self._plan_local_path(target_pos)
            if planned_return_path and not (len(planned_return_path) == 1 and np.array_equal(planned_return_path[0], self.position)):
                self.planned_path = planned_return_path; self.is_returning = True; self.return_replan_attempts = 0; self.return_fail_cooldown = 0
                decision_reason = "Return (Criteria)" if should_return_criteria else "Return (Too Far)"
                logger.debug(f"[R{self.robot_id} Decide] START RETURN ({decision_reason}). PathLen={len(self.planned_path)}")
            else:
                 logger.warning(f"[R{self.robot_id} Decide] Wanted return ({'Criteria' if should_return_criteria else 'TooFar'}), but path failed. Trigger cooldown & local.")
                 self.return_fail_cooldown = RETURN_FAIL_COOLDOWN_STEPS; target_pos = target_pos_local
                 self.planned_path = self._plan_local_path(self.target_pos)
                 if not self.planned_path: self.planned_path = [self.position]
                 self.is_returning = False; decision_reason = "Local (Return Failed)"
        else:
            target_pos = target_pos_local; self.is_returning = False; self.return_fail_cooldown = 0
            self.planned_path = self._plan_local_path(self.target_pos)
            if not self.planned_path: self.planned_path = [self.position]
            decision_reason = "Local Explore"
        self.target_pos = target_pos; self.target_gived_by_server = False
        path_len = len(self.planned_path) if self.planned_path else 0
        logger.debug(f"[R{self.robot_id} Decide] Final Target: {self.target_pos}, Reason: {decision_reason}, PathLen: {path_len}")

    def move_one_step(self, all_robots):
        logger.debug(f"[R{self.robot_id} Move] Pos:{self.position}, Target:{self.target_pos}, Return:{self.is_returning}, PathLen:{len(self.planned_path)}")
        if len(self.planned_path) < 1: logger.debug(f"[R{self.robot_id} Move] No path."); return
        next_step = self.planned_path[0]
        if np.array_equal(next_step, self.position):
            if len(self.planned_path) > 1:
                self.planned_path.pop(0); next_step = self.planned_path[0]
                logger.debug(f"[R{self.robot_id} Move] Path start=curr, popped. Next:{next_step}")
            else:
                logger.debug(f"[R{self.robot_id} Move] Reached/Stuck @ {self.position}")
                self.planned_path = []; return
        is_blocked = False; blocker_id = -1
        for other_robot in all_robots:
            if other_robot is self: continue
            if other_robot.position is not None:
                dist_to_other = np.linalg.norm(next_step - other_robot.position)
                if dist_to_other < 1.5: is_blocked = True; blocker_id = other_robot.robot_id; break
        if is_blocked:
            logger.debug(f"[R{self.robot_id} Move] Blocked by R{blocker_id} @ {all_robots[blocker_id].position}. Next:{next_step}. Stay:{self.stay_count}")
            self.stay_count += 1; MAX_STAY_COUNT = 5
            if self.stay_count > MAX_STAY_COUNT:
                logger.debug(f"[R{self.robot_id} Move] Waited too long, try back step.")
                if len(self.movement_history) > 1:
                    previous_pos = self.movement_history[-2]; can_move_back = True
                    for r in all_robots:
                         if r is not self and np.linalg.norm(previous_pos - r.position) < 1.5: can_move_back = False; break
                    if can_move_back:
                         self.position = np.array(previous_pos); self.planned_path = []; self.stay_count = 0
                         logger.debug(f"[R{self.robot_id} Move] Moved back to {self.position}. Path cleared.")
                         if not self.movement_history or not np.array_equal(self.position, self.movement_history[-1]):
                              self.movement_history.append(self.position.copy())
                    else: logger.debug(f"[R{self.robot_id} Move] Cannot move back.")
            elif self.stay_count > 2 and self.robot_id > blocker_id and not self.is_returning:
                logger.debug(f"[R{self.robot_id} Move] Yielding to R{blocker_id} & replanning.")
                new_path = self._plan_local_path(self.target_pos, avoid_pos=next_step)
                if new_path and not (len(new_path)==1 and np.array_equal(new_path[0], self.position)): self.planned_path = new_path
        else:
            logger.debug(f"[R{self.robot_id} Move] Moving to {next_step}")
            self.stay_count = 0; popped_step = None
            if len(self.planned_path) > 0:
                popped_step = self.planned_path.pop(0)
                if np.array_equal(popped_step, self.position) and len(self.planned_path) > 0:
                    next_step = self.planned_path.pop(0)
                elif popped_step is not None: next_step = popped_step
            if next_step is not None and not np.array_equal(next_step, self.position):
                self.position = np.array(next_step)
                if not self.movement_history or not np.array_equal(self.position, self.movement_history[-1]):
                    self.movement_history.append(self.position.copy())
        dist_to_server = np.linalg.norm(self.position - self.last_position_in_server_range)
        if dist_to_server < SERVER_COMM_RANGE: self.is_in_server_range = True
        else: self.is_in_server_range = False

    def _select_node(self, all_robots):
        if not hasattr(self.graph_generator, 'target_candidates') or self.graph_generator.target_candidates is None:
             logger.debug(f"[R{self.robot_id} SelectNode] graph_generator.target_candidates is None!"); return self.position, 0, 0
        candidates = self.graph_generator.target_candidates; num_candidates = len(candidates)
        if num_candidates == 0: logger.debug(f"[R{self.robot_id} SelectNode] No candidates."); return self.position, 0, 0
        if not hasattr(self.graph_generator, 'candidates_utility') or self.graph_generator.candidates_utility is None or len(self.graph_generator.candidates_utility) != num_candidates:
            logger.error(f"[R{self.robot_id} SelectNode] Mismatch/missing candidates_utility!"); return self.position, 0, 0
        utilities = self.graph_generator.candidates_utility
        dists = np.linalg.norm(candidates - self.position, axis=1)
        valid_mask = utilities > 0
        for robot in all_robots:
             if robot.position is not None: same_pos = np.all(candidates == robot.position, axis=1); valid_mask &= ~same_pos
             if robot.target_pos is not None and not robot.is_in_server_range:
                 dist_to_other_target = np.linalg.norm(candidates - robot.target_pos, axis=1)
                 too_close_to_target = dist_to_other_target < 20; valid_mask &= ~too_close_to_target
        num_valid_after_filter = np.sum(valid_mask)
        if not np.any(valid_mask): logger.debug(f"[R{self.robot_id} SelectNode] No valid after filter."); return self.position, 0, 0
        valid_candidates = candidates[valid_mask]; valid_utilities = utilities[valid_mask]; valid_dists = dists[valid_mask]
        if len(valid_dists) == 0: logger.debug(f"[R{self.robot_id} SelectNode] Valid dists empty."); return self.position, 0, 0
        λ = 1.0; epsilon = 1e-6
        min_valid_dists = np.min(valid_dists) if len(valid_dists) > 0 else 0
        scores = λ * valid_utilities / (valid_dists + epsilon); best_idx_in_valid = np.argmax(scores)
        selected_coord = valid_candidates[best_idx_in_valid]; original_indices = np.where(valid_mask)[0]
        if best_idx_in_valid >= len(original_indices): logger.error(f"[R{self.robot_id} SelectNode] Index error!"); return self.position, 0, 0
        original_idx = original_indices[best_idx_in_valid]
        logger.debug(f"[R{self.robot_id} SelectNode] Cands:{num_candidates}, Valid:{num_valid_after_filter} -> Sel:{selected_coord}, Score:{scores[best_idx_in_valid]:.2f}, Util:{valid_utilities[best_idx_in_valid]}, Dist:{valid_dists[best_idx_in_valid]:.1f}")
        return selected_coord, original_idx, min_valid_dists

    def _plan_local_path(self, target, avoid_pos=None):
        gen = self.graph_generator; current = self.position
        if avoid_pos is not None: pass
        coords = self.node_coords
        graph_edges = self.local_map_graph # 這是 dict

        if coords is None or len(coords) == 0 or graph_edges is None or not isinstance(graph_edges, dict):
             logger.warning(f"[R{self.robot_id} PlanLocal] Invalid graph/nodes for target {target}. Coords:{coords is not None}, Graph:{graph_edges is not None}")
             return [current]

        graph_obj = Graph()
        if coords is not None:
             try:
                 for node_coord in coords: graph_obj.add_node(tuple(node_coord))
             except TypeError:
                  logger.error(f"[R{self.robot_id} PlanLocal] Coords not iterable? Type: {type(coords)}")
                  return [current]
        if isinstance(graph_edges, dict):
             for from_node_tuple, edges_dict in graph_edges.items():
                 if from_node_tuple not in graph_obj.nodes: graph_obj.add_node(from_node_tuple)
                 if isinstance(edges_dict, dict):
                      for to_node_tuple, edge_obj in edges_dict.items():
                          if to_node_tuple not in graph_obj.nodes: graph_obj.add_node(to_node_tuple)
                          if isinstance(edge_obj, Edge): graph_obj.add_edge(from_node_tuple, to_node_tuple, edge_obj.length)
                          else: logger.warning(f"Unexpected edge type: {type(edge_obj)}")

        dist, route = gen.find_shortest_path(current, target, coords, graph_obj)
        path_len = len(route) if route is not None else 0
        logger.debug(f"[R{self.robot_id} PlanLocal] Target:{target}. A* Result:{'Success' if route is not None else 'Failed!'}, Dist:{dist:.1f}, PathLen:{path_len}")
        if route is None or dist >= 1e5: return [current]
        if len(route) > 1 and np.array_equal(route[0], current): route.pop(0)
        if not route or (len(route) == 1 and np.array_equal(route[0], current)): return [current]
        return route