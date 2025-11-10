import numpy as np
from collections import deque
import logging
import copy

from parameter import *
from graph_generator import Graph_generator
from sensor import sensor_work
from skimage.measure import block_reduce
import sys
from graph import Graph, Edge  # <--- 確保匯入 Graph 和 Edge

logger = logging.getLogger(__name__)


class Robot():
    def __init__(self, start_position, real_map_size, resolution, k_size, plot=False):
        """初始化 Robot。

        Args:
            start_position (array-like[2]): 機器人起始位置。
            real_map_size (tuple): 地圖大小 (height, width)。
            resolution (int): 下採樣比例。
            k_size (int): Graph generator 的 k。
            plot (bool): 是否啟用繪圖。

        Returns:
            None
        """
        self.position = start_position
        self.local_map = np.ones(real_map_size) * 127
        self.downsampled_map = None
        self.sensor_range = SENSOR_RANGE
        self.frontiers = []
        self.node_coords = None
        self.local_map_graph = None  # This will store graph.edges dict
        self.node_utility = None
        self.guidepost = None
        self.planned_path = []
        self.target_pos = None
        self.movement_history = [start_position.copy()]
        self.graph_generator: Graph_generator = Graph_generator(
            map_size=real_map_size, sensor_range=self.sensor_range, k_size=k_size, plot=plot
        )
        if start_position is not None:
            self.graph_generator.route_node.append(start_position)

        # Create an initial local graph so robots have candidates even when out of server range
        # Use rebuild_graph_structure once at init to ensure Node objects (nodes_list) are created
        try:
            empty_frontiers = np.array([]).reshape(0, 2)
            node_coords, graph_edges, node_utility, guidepost = self.graph_generator.rebuild_graph_structure(
                self.local_map, empty_frontiers, None, self.position
            )
            self.node_coords = node_coords
            self.local_map_graph = graph_edges
            self.node_utility = node_utility
            self.guidepost = guidepost
        except Exception:
            logger.debug(f"Robot init: initial rebuild_graph_structure failed for R{getattr(self,'robot_id',-1)}", exc_info=True)

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
        # 新增 handoff 冷卻計數器
        self.handoff_cooldown = 0
        # self.debug removed

    def update_local_awareness(self, real_map, all_robots, server, find_frontier_func, merge_maps_func):
        """執行感知、地圖合併、機器人交互與節點/效用更新（一步）。

        Args:
            real_map (ndarray): 真實地圖 (y,x)。
            all_robots (list[Robot]): 所有機器人清單。
            server (Server): 伺服器物件。
            find_frontier_func (callable): 用於尋找 frontier 的函式。
            merge_maps_func (callable): 用於合併地圖的函式。

        Returns:
            None
        """
        # --- 在每步一開始遞減 handoff_cooldown ---
        if self.handoff_cooldown > 0:
            self.handoff_cooldown -= 1

        # --- 1. 感測與地圖合併 ---
        self.local_map = sensor_work(self.position, self.sensor_range, self.local_map, real_map)
        dist_to_server = np.linalg.norm(self.position - server.position)
        was_in_range = self.is_in_server_range
        self.is_in_server_range = dist_to_server < SERVER_COMM_RANGE

        if self.is_in_server_range and not was_in_range:
            self.is_returning = False
            self.return_replan_attempts = 0
            self.return_fail_cooldown = 0
            logger.debug(f"Robot {self.robot_id} re-entered server range. State reset.")

        # --- 機器人交互 ---
        for other_robot in all_robots:
            if other_robot is self:
                continue

            dist = np.linalg.norm(self.position - other_robot.position)

            if dist < ROBOT_COMM_RANGE:
                # 1. Standard Map Merge
                merged = merge_maps_func([self.local_map, other_robot.local_map])
                self.local_map[:] = merged
                other_robot.local_map[:] = merged

                # 若任一方仍在 handoff 冷卻期，跳過交接判斷（但保留地圖合併）
                if self.handoff_cooldown > 0 or getattr(other_robot, 'handoff_cooldown', 0) > 0:
                    continue

                # --- 2. 修改點：機會主義任務交接 (Task Handoff Logic) ---
                i_am_returning = self.is_returning
                other_has_server_task = other_robot.target_gived_by_server and not other_robot.is_returning
                i_have_server_task = self.target_gived_by_server and not self.is_returning
                other_is_returning = other_robot.is_returning

                # 當自己正在回去時，無論對方是否有 server 任務，都改成讓對方回去，自己接手對方原本要做的事
                if i_am_returning and not other_is_returning:
                    logger.info(f"[R{self.robot_id} Handoff] I am returning; swap roles with R{other_robot.robot_id}. R{other_robot.robot_id} will return; I will take its task.")
                    # self 接手 other 的原始任務（深拷貝以避免共享參考）
                    try:
                        # 只複製目標位置，讓接手方自己重規劃路徑以避免使用過時的 planned_path
                        self.target_pos = copy.deepcopy(other_robot.target_pos)
                        self.planned_path = []
                        # preserve whether the task was server-given for the receiver
                        self.target_gived_by_server = bool(getattr(other_robot, 'target_gived_by_server', False))
                    except Exception:
                        logger.exception(f"[R{self.robot_id} Handoff] Failed to copy task from R{other_robot.robot_id}")
                    # self 停止返回（因為要去執行對方的任務）
                    self.is_returning = False
                    self.return_replan_attempts = 0
                    self.return_fail_cooldown = 0

                    # other 改為回到其 last_position_in_server_range
                    try:
                        other_robot.target_pos = copy.deepcopy(other_robot.last_position_in_server_range)
                    except Exception:
                        other_robot.target_pos = copy.deepcopy(getattr(other_robot, 'last_position_in_server_range', None))
                    other_robot.planned_path = []
                    other_robot.is_returning = True
                    other_robot.target_gived_by_server = False
                    other_robot.return_replan_attempts = 0

                    # 設定冷卻，避免立即反向交接
                    self.handoff_cooldown = HANDOFF_COOLDOWN
                    other_robot.handoff_cooldown = HANDOFF_COOLDOWN

                elif i_have_server_task and other_is_returning:
                    logger.info(f"[R{self.robot_id} Handoff] I am departing, giving task to R{other_robot.robot_id}. I am now returning.")
                    # 深拷貝原始任務，避免後續修改造成共享
                    my_original_target = copy.deepcopy(self.target_pos)
                    # 只複製目標位置即可，讓接手方自行 replan
                    # 我 (B) 接收 A 的返回任務
                    self.target_pos = copy.deepcopy(other_robot.last_position_in_server_range)
                    self.planned_path = []
                    self.is_returning = True
                    self.target_gived_by_server = False
                    self.return_replan_attempts = 0
                    # A 接收我 (B) 的原始任務（拷貝）
                    other_robot.target_pos = my_original_target
                    other_robot.planned_path = []
                    other_robot.target_gived_by_server = True
                    other_robot.is_returning = False
                    other_robot.return_replan_attempts = 0
                    other_robot.return_fail_cooldown = 0

                    # 設定冷卻，避免立即反向交接
                    self.handoff_cooldown = HANDOFF_COOLDOWN
                    other_robot.handoff_cooldown = HANDOFF_COOLDOWN

                # Case 3: 其他情況 (例如兩個自主探索者相遇)
                elif not self.is_in_server_range and not self.is_returning and not self.target_gived_by_server:
                    self.planned_path = []

                # --- 任務交接結束 ---

        # --- 1c. 與伺服器同步 (在交互之後) ---
        if self.is_in_server_range:
            merged = merge_maps_func([self.local_map, server.global_map])
            self.local_map[:] = merged
            server.global_map[:] = merged
            if 0 <= self.robot_id < len(server.all_robot_position):
                server.all_robot_position[self.robot_id] = self.position
                server.robot_in_range[self.robot_id] = True
            self.last_position_in_server_range = self.position
            self.out_range_step = 0
            if not self.target_gived_by_server:
                self.planned_path = []
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
        # 首先嘗試每步做輕量級的 node utility 更新，並把 graph.edges 回寫到 local_map_graph
        logger.debug(f"[R{self.robot_id} Awareness] Attempting lightweight update_node_utilities...")
        try:
            node_utility, guidepost = self.graph_generator.update_node_utilities(
                self.local_map, new_frontiers, self.frontiers, caller=f'robot-{getattr(self, "robot_id", -1)}'
            )
            self.node_utility = node_utility
            self.guidepost = guidepost
            if hasattr(self.graph_generator, 'graph') and hasattr(self.graph_generator.graph, 'edges'):
                # 如果 generator 持有 graph.edges，就把它同步回 robot 端，供 local path planning 使用
                try:
                    if isinstance(self.graph_generator.graph.edges, dict) and len(self.graph_generator.graph.edges) > 0:
                        self.local_map_graph = self.graph_generator.graph.edges
                    else:
                        # 若 graph.edges 空，保留現狀，稍後判斷是否需要重建
                        pass
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Robot {self.robot_id} failed update_node_utilities: {e}", exc_info=True)

        # Fallback: if robot is in server range and graph_generator produced no candidates,
        # but server has candidates, copy them (deepcopy) to robot to allow selection/assignment.
        try:
            if self.is_in_server_range and hasattr(server, 'graph_generator') and hasattr(server.graph_generator, 'target_candidates'):
                own_cands = getattr(self.graph_generator, 'target_candidates', None)
                server_cands = getattr(server.graph_generator, 'target_candidates', None)
                if (own_cands is None or (isinstance(own_cands, (list, tuple, np.ndarray)) and len(own_cands) == 0)) and server_cands is not None and len(server_cands) > 0:
                    # Only emit extra diagnostics when mismatch occurs (server has candidates, robot has none)
                    try:
                        server_frontiers = getattr(server, 'frontiers', None)
                        robot_frontiers = new_frontiers if 'new_frontiers' in locals() else getattr(self, 'frontiers', None)
                        srv_f_len = len(server_frontiers) if server_frontiers is not None else 0
                        rbt_f_len = len(robot_frontiers) if robot_frontiers is not None else 0
                        srv_sample = server_frontiers[:3].tolist() if server_frontiers is not None and len(server_frontiers) > 0 else []
                        rbt_sample = robot_frontiers[:3].tolist() if robot_frontiers is not None and len(robot_frontiers) > 0 else []
                        srv_map_shape = getattr(server, 'global_map', None).shape if getattr(server, 'global_map', None) is not None else None
                        rbt_map_shape = getattr(self, 'local_map', None).shape if getattr(self, 'local_map', None) is not None else None
                        # concise fingerprint: sum of coords (stable small-int) to help quick diffing
                        try:
                            srv_f_fp = int(np.sum(server_frontiers)) if server_frontiers is not None and len(server_frontiers) > 0 else None
                        except Exception:
                            srv_f_fp = None
                        try:
                            rbt_f_fp = int(np.sum(robot_frontiers)) if robot_frontiers is not None and len(robot_frontiers) > 0 else None
                        except Exception:
                            rbt_f_fp = None
                        logger.warning(f"[R{self.robot_id} Awareness] MISMATCH DIAG: server_cands={len(server_cands)} vs robot_cands=0 | srv_frontiers_len={srv_f_len} srv_sample={srv_sample} srv_fp={srv_f_fp} | rbt_frontiers_len={rbt_f_len} rbt_sample={rbt_sample} rbt_fp={rbt_f_fp} | srv_map_shape={srv_map_shape} rbt_map_shape={rbt_map_shape}")
                    except Exception:
                        logger.exception(f"[R{self.robot_id} Awareness] Error while logging mismatch diagnostics.")
                    try:
                        # copy server-side candidate info into robot
                        self.graph_generator.target_candidates = copy.deepcopy(server.graph_generator.target_candidates)
                        self.graph_generator.candidates_utility = copy.deepcopy(server.graph_generator.candidates_utility)
                        self.node_utility = copy.deepcopy(server.graph_generator.node_utility)
                        self.node_coords = copy.deepcopy(server.graph_generator.node_coords)
                        self.local_map_graph = copy.deepcopy(server.local_map_graph) if hasattr(server, 'local_map_graph') else copy.deepcopy(server.graph_generator.graph.edges)
                        logger.debug(f"[R{self.robot_id} Awareness] Fallback: copied {len(self.graph_generator.target_candidates)} candidates from server.")
                    except Exception:
                        logger.exception(f"[R{self.robot_id} Awareness] Fallback copy from server failed.")
                    # Focused warning if robot's lightweight update produced zero candidates while server has candidates
                    try:
                        logger.warning(f"[R{self.robot_id} Awareness] MISMATCH: robot candidates=0 but server candidates={len(server_cands)}. Fallback applied.")
                    except Exception:
                        pass
        except Exception:
            logger.exception(f"[R{self.robot_id} Awareness] Error in fallback candidate sync.")

        # 若 local_map_graph 缺失或為空，則在間隔到達時執行一次重建（避免每步重建造成高昂代價）
        need_local_rebuild = False
        try:
            if self.local_map_graph is None:
                need_local_rebuild = True
            elif isinstance(self.local_map_graph, dict) and len(self.local_map_graph) == 0:
                need_local_rebuild = True
        except Exception:
            need_local_rebuild = True

        if need_local_rebuild and (self.graph_update_counter % GRAPH_UPDATE_INTERVAL == 0):
            logger.debug(f"[R{self.robot_id} Awareness] local_map_graph missing/empty -> performing local rebuild...")
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

        self.frontiers = new_frontiers

    # ... (needs_new_target, decide_next_target, move_one_step, _select_node 保持不變) ...
    def needs_new_target(self):
        """檢查是否需要新目標（路徑為空）。 Returns bool。"""
        return len(self.planned_path) < 1

    def decide_next_target(self, all_robots):
        """決策下一個目標（含返回、冷卻、重新規劃邏輯）。

        Args:
            all_robots (list[Robot]): 其他機器人清單。

        Returns:
            None
        """
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
                    self.is_returning = False
                    self.return_replan_attempts = 0
                    self.return_fail_cooldown = RETURN_FAIL_COOLDOWN_STEPS
            else:
                logger.debug(f"[R{self.robot_id} Decide] Continue return path.")
                return
        if self.return_fail_cooldown > 0:
            logger.debug(f"[R{self.robot_id} Decide] In cooldown ({self.return_fail_cooldown} left). Force local.")
            self.return_fail_cooldown -= 1
            target_pos_local, _, _ = self._select_node(all_robots)
            self.target_pos = target_pos_local
            self.planned_path = self._plan_local_path(self.target_pos)
            if not self.planned_path:
                self.planned_path = [self.position]
            self.is_returning = False
            self.target_gived_by_server = False
            return
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
                self.planned_path = planned_return_path
                self.is_returning = True
                self.return_replan_attempts = 0
                self.return_fail_cooldown = 0
                decision_reason = "Return (Criteria)" if should_return_criteria else "Return (Too Far)"
                logger.debug(f"[R{self.robot_id} Decide] START RETURN ({decision_reason}). PathLen={len(self.planned_path)}")
            else:
                logger.warning(f"[R{self.robot_id} Decide] Wanted return ({'Criteria' if should_return_criteria else 'TooFar'}), but path failed. Trigger cooldown & local.")
                self.return_fail_cooldown = RETURN_FAIL_COOLDOWN_STEPS
                target_pos = target_pos_local
                self.planned_path = self._plan_local_path(self.target_pos)
                if not self.planned_path:
                    self.planned_path = [self.position]
                self.is_returning = False
                decision_reason = "Local (Return Failed)"
        else:
            target_pos = target_pos_local
            self.is_returning = False
            self.return_fail_cooldown = 0
            self.planned_path = self._plan_local_path(self.target_pos)
            if not self.planned_path:
                self.planned_path = [self.position]
            decision_reason = "Local Explore"
        self.target_pos = target_pos
        self.target_gived_by_server = False
        path_len = len(self.planned_path) if self.planned_path else 0
        logger.debug(f"[R{self.robot_id} Decide] Final Target: {self.target_pos}, Reason: {decision_reason}, PathLen: {path_len}")

    def move_one_step(self, all_robots):
        """執行單步移動：檢查阻擋、退讓、重新規劃或前進。

        Args:
            all_robots (list[Robot]): 其他機器人清單。

        Returns:
            None
        """
        logger.debug(f"[R{self.robot_id} Move] Pos:{self.position}, Target:{self.target_pos}, Return:{self.is_returning}, PathLen:{len(self.planned_path)}")
        if len(self.planned_path) < 1:
            logger.debug(f"[R{self.robot_id} Move] No path.")
            return
        next_step = self.planned_path[0]
        if np.array_equal(next_step, self.position):
            if len(self.planned_path) > 1:
                self.planned_path.pop(0)
                next_step = self.planned_path[0]
                logger.debug(f"[R{self.robot_id} Move] Path start=curr, popped. Next:{next_step}")
            else:
                logger.debug(f"[R{self.robot_id} Move] Reached/Stuck @ {self.position}")
                self.planned_path = []
                return
        is_blocked = False
        blocker_id = -1
        for other_robot in all_robots:
            if other_robot is self:
                continue
            if other_robot.position is not None:
                dist_to_other = np.linalg.norm(next_step - other_robot.position)
                if dist_to_other < 1.5:
                    is_blocked = True
                    blocker_id = other_robot.robot_id
                    break
        if is_blocked:
            logger.debug(f"[R{self.robot_id} Move] Blocked by R{blocker_id} @ {all_robots[blocker_id].position}. Next:{next_step}. Stay:{self.stay_count}")
            self.stay_count += 1
            MAX_STAY_COUNT = 5
            if self.stay_count > MAX_STAY_COUNT:
                logger.debug(f"[R{self.robot_id} Move] Waited too long, try back step.")
                if len(self.movement_history) > 1:
                    previous_pos = self.movement_history[-2]
                    can_move_back = True
                    for r in all_robots:
                        if r is not self and np.linalg.norm(previous_pos - r.position) < 1.5:
                            can_move_back = False
                            break
                    if can_move_back:
                        self.position = np.array(previous_pos)
                        self.planned_path = []
                        self.stay_count = 0
                        logger.debug(f"[R{self.robot_id} Move] Moved back to {self.position}. Path cleared.")
                        if not self.movement_history or not np.array_equal(self.position, self.movement_history[-1]):
                            self.movement_history.append(self.position.copy())
                    else:
                        logger.debug(f"[R{self.robot_id} Move] Cannot move back.")
            elif self.stay_count > 2 and self.robot_id > blocker_id and not self.is_returning:
                logger.debug(f"[R{self.robot_id} Move] Yielding to R{blocker_id} & replanning.")
                new_path = self._plan_local_path(self.target_pos, avoid_pos=next_step)
                if new_path and not (len(new_path) == 1 and np.array_equal(new_path[0], self.position)):
                    self.planned_path = new_path
        else:
            logger.debug(f"[R{self.robot_id} Move] Moving to {next_step}")
            self.stay_count = 0
            popped_step = None
            if len(self.planned_path) > 0:
                popped_step = self.planned_path.pop(0)
                if np.array_equal(popped_step, self.position) and len(self.planned_path) > 0:
                    next_step = self.planned_path.pop(0)
                elif popped_step is not None:
                    next_step = popped_step
            if next_step is not None and not np.array_equal(next_step, self.position):
                self.position = np.array(next_step)
                if not self.movement_history or not np.array_equal(self.position, self.movement_history[-1]):
                    self.movement_history.append(self.position.copy())
        dist_to_server = np.linalg.norm(self.position - self.last_position_in_server_range)
        if dist_to_server < SERVER_COMM_RANGE:
            self.is_in_server_range = True
        else:
            self.is_in_server_range = False

    def _select_node(self, all_robots):
        """選擇最佳 local 候選節點。

        Args:
            all_robots (list[Robot]): 其他機器人清單。

        Returns:
            tuple: (selected_coord (ndarray), original_idx (int), min_valid_dists (float))
        """
        if not hasattr(self.graph_generator, 'target_candidates') or self.graph_generator.target_candidates is None:
            logger.debug(f"[R{self.robot_id} SelectNode] graph_generator.target_candidates is None!")
            # Diagnostic: report local graph status
            try:
                gcoords_len = len(self.node_coords) if self.node_coords is not None else 0
            except Exception:
                gcoords_len = -1
            try:
                local_graph_empty = isinstance(self.local_map_graph, dict) and len(self.local_map_graph) == 0
            except Exception:
                local_graph_empty = True
            logger.debug(f"[Diag robot R{self.robot_id}] node_coords_len={gcoords_len} local_map_graph_empty={local_graph_empty}")
            return self.position, 0, 0
        candidates = self.graph_generator.target_candidates
        num_candidates = len(candidates)
        if num_candidates == 0:
            logger.debug(f"[R{self.robot_id} SelectNode] No candidates.")
            # Diagnostic: echo graph_generator state
            try:
                gcoords_len = len(self.graph_generator.node_coords) if hasattr(self.graph_generator, 'node_coords') and self.graph_generator.node_coords is not None else 0
            except Exception:
                gcoords_len = -1
            try:
                targ_len = len(self.graph_generator.target_candidates) if hasattr(self.graph_generator, 'target_candidates') and self.graph_generator.target_candidates is not None else 0
            except Exception:
                targ_len = -1
            try:
                util_len = len(self.graph_generator.candidates_utility) if hasattr(self.graph_generator, 'candidates_utility') and self.graph_generator.candidates_utility is not None else 0
            except Exception:
                util_len = -1
            logger.debug(f"[Diag robot R{self.robot_id}] gen_node_coords_len={gcoords_len} gen_target_candidates_len={targ_len} gen_candidates_utility_len={util_len}")
            return self.position, 0, 0
        if not hasattr(self.graph_generator, 'candidates_utility') or self.graph_generator.candidates_utility is None or len(self.graph_generator.candidates_utility) != num_candidates:
            logger.error(f"[R{self.robot_id} SelectNode] Mismatch/missing candidates_utility!")
            return self.position, 0, 0
        utilities = self.graph_generator.candidates_utility
        dists = np.linalg.norm(candidates - self.position, axis=1)
        valid_mask = utilities > 0
        for robot in all_robots:
            if robot.position is not None:
                same_pos = np.all(candidates == robot.position, axis=1)
                valid_mask &= ~same_pos
            if robot.target_pos is not None and not robot.is_in_server_range:
                dist_to_other_target = np.linalg.norm(candidates - robot.target_pos, axis=1)
                too_close_to_target = dist_to_other_target < 20
                valid_mask &= ~too_close_to_target
        num_valid_after_filter = np.sum(valid_mask)
        if not np.any(valid_mask):
            logger.debug(f"[R{self.robot_id} SelectNode] No valid after filter.")
            return self.position, 0, 0
        valid_candidates = candidates[valid_mask]
        valid_utilities = utilities[valid_mask]
        valid_dists = dists[valid_mask]
        if len(valid_dists) == 0:
            logger.debug(f"[R{self.robot_id} SelectNode] Valid dists empty.")
            return self.position, 0, 0
        # 新選點策略：在有效候選集合內對 utility 與 distance 做正規化，
        # 再以權重合併：score = w_u * util_norm + (1-w_u) * (1 - dist_norm)
        # 保留 epsilon 保護分母
        epsilon = 1e-6
        min_valid_dists = np.min(valid_dists) if len(valid_dists) > 0 else 0
        # 正規化 utility 與 distance，避免尺度不一致造成偏差
        max_util = np.max(valid_utilities) if len(valid_utilities) > 0 else 0
        min_util = np.min(valid_utilities) if len(valid_utilities) > 0 else 0
        max_dist = np.max(valid_dists) if len(valid_dists) > 0 else 0
        min_dist = np.min(valid_dists) if len(valid_dists) > 0 else 0
        # util_norm in [0,1], dist_norm in [0,1]
        if max_util - min_util <= 0:
            util_norm = np.ones_like(valid_utilities)
        else:
            util_norm = (valid_utilities - min_util) / (max_util - min_util + epsilon)
        if max_dist - min_dist <= 0:
            dist_norm = np.zeros_like(valid_dists)
        else:
            dist_norm = (valid_dists - min_dist) / (max_dist - min_dist + epsilon)
        # weight from parameter
        try:
            from parameter import SELECTION_UTILITY_WEIGHT, SELECTION_USE_PATH_COST, SELECTION_PATH_COST_TOPK
            w_u = float(SELECTION_UTILITY_WEIGHT)
        except Exception:
            w_u = 0.7
        # base score: 越大越好（util high, dist small）
        scores = w_u * util_norm + (1.0 - w_u) * (1.0 - dist_norm)
        # 若啟用 path-cost 模式，可對 top-K candidate 使用更精確的路徑成本 (A*)，並用 path-cost 替代 dist_norm
        try:
            if SELECTION_USE_PATH_COST:
                # 先以 scores 排序取 top-K 再用 A* 計算實際路徑距離作替換
                topk = int(SELECTION_PATH_COST_TOPK) if SELECTION_PATH_COST_TOPK is not None else 5
                topk = max(1, min(topk, len(valid_candidates)))
                topk_indices = np.argsort(-scores)[:topk]
                path_costs = np.zeros(len(topk_indices))
                for ii, vi in enumerate(topk_indices):
                    cand = valid_candidates[vi]
                    # 使用 graph_generator 的 find_shortest_path
                    try:
                        dist, route = self.graph_generator.find_shortest_path(self.position, cand, self.node_coords, None)
                        path_costs[ii] = dist if route is not None else 1e6
                    except Exception:
                        path_costs[ii] = 1e6
                # normalize path costs and replace corresponding scores
                if np.max(path_costs) - np.min(path_costs) > 0:
                    pc_norm = (path_costs - np.min(path_costs)) / (np.max(path_costs) - np.min(path_costs) + epsilon)
                else:
                    pc_norm = np.zeros_like(path_costs)
                for idx_local, pcn in zip(topk_indices, pc_norm):
                    # 用 1 - pcn 作為距離部分的替代（距離小 => 高分）
                    scores[idx_local] = w_u * util_norm[idx_local] + (1.0 - w_u) * (1.0 - pcn)
        except Exception:
            # 若計算路徑成本失敗，回退到原始 scores
            pass
        best_idx_in_valid = np.argmax(scores)
        selected_coord = valid_candidates[best_idx_in_valid]
        original_indices = np.where(valid_mask)[0]
        if best_idx_in_valid >= len(original_indices):
            logger.error(f"[R{self.robot_id} SelectNode] Index error!")
            return self.position, 0, 0
        original_idx = original_indices[best_idx_in_valid]
        logger.debug(f"[R{self.robot_id} SelectNode] Cands:{num_candidates}, Valid:{num_valid_after_filter} -> Sel:{selected_coord}, Score:{scores[best_idx_in_valid]:.2f}, Util:{valid_utilities[best_idx_in_valid]}, Dist:{valid_dists[best_idx_in_valid]:.1f}")
        return selected_coord, original_idx, min_valid_dists

    def _plan_local_path(self, target, avoid_pos=None):
        """使用本地圖(graph)計算到 target 的路徑，若失敗回傳 [current]。

        Args:
            target (array-like[2]): 目標座標。
            avoid_pos (array-like[2], optional): 要避免的位置（可為 None）。

        Returns:
            list: 路徑座標列表（至少包含 current position）。
        """
        gen = self.graph_generator
        current = self.position
        if avoid_pos is not None:
            pass
        coords = self.node_coords
        graph_edges = self.local_map_graph  # 這是 dict

        if coords is None or len(coords) == 0 or graph_edges is None or not isinstance(graph_edges, dict):
            logger.warning(f"[R{self.robot_id} PlanLocal] Invalid graph/nodes for target {target}. Coords:{coords is not None}, Graph:{graph_edges is not None}")
            return [current]

        graph_obj = Graph()
        if coords is not None:
            try:
                for node_coord in coords:
                    graph_obj.add_node(tuple(node_coord))
            except TypeError:
                logger.error(f"[R{self.robot_id} PlanLocal] Coords not iterable? Type: {type(coords)}")
                return [current]
        if isinstance(graph_edges, dict):
            for from_node_tuple, edges_dict in graph_edges.items():
                if from_node_tuple not in graph_obj.nodes:
                    graph_obj.add_node(from_node_tuple)
                if isinstance(edges_dict, dict):
                    for to_node_tuple, edge_obj in edges_dict.items():
                        if to_node_tuple not in graph_obj.nodes:
                            graph_obj.add_node(to_node_tuple)
                        if isinstance(edge_obj, Edge):
                            graph_obj.add_edge(from_node_tuple, to_node_tuple, edge_obj.length)
                        else:
                            logger.warning(f"Unexpected edge type: {type(edge_obj)}")

        dist, route = gen.find_shortest_path(current, target, coords, graph_obj)
        path_len = len(route) if route is not None else 0
        logger.debug(f"[R{self.robot_id} PlanLocal] Target:{target}. A* Result:{'Success' if route is not None else 'Failed!'}, Dist:{dist:.1f}, PathLen:{path_len}")
        if route is None or dist >= 1e5:
            return [current]
        if len(route) > 1 and np.array_equal(route[0], current):
            route.pop(0)
        if not route or (len(route) == 1 and np.array_equal(route[0], current)):
            return [current]
        return route