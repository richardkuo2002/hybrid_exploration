import math

SENSOR_RANGE = 80  # 80
SERVER_COMM_RANGE = 160  # 160
ROBOT_COMM_RANGE = 80  # 80
DISTANCE_WEIGHT = 1
OUT_RANGE_STEP = 20

MAX_EPS_STEPS = 196
K_SIZE = 30

# --- 修改點：加回圖更新間隔 ---
GRAPH_UPDATE_INTERVAL = 2  # 每 2 步重建圖結構（原本為 1，改為 2 以減少重建頻率）
# -----------------------------

# --- 智慧會合機制參數 ---
INFO_GAIN_HISTORY_LEN = 20
MIN_INFO_GAIN_THRESHOLD = 50
LOCAL_UTILITY_THRESHOLD = 10
# ---------------------------------

# 新增：交接冷卻（避免立即反向交接）
HANDOFF_COOLDOWN = 3  # 單位：步數

# 新增：是否啟用交接機制 (Handoff)
ENABLE_HANDOFF = True

NUM_DENSE_COORDS_WIDTH = 50
CUR_AGENT_KNN_RAD = 80
GLOBAL_GRAPH_KNN_RAD = 160
UTILITY_CALC_RANGE = 40
GLOBAL_NODES_TO_FRONTIER_AVOID_SPARSE_RAD = 120
# 當感測到連續 obstacle 時，達到此值才視為真正阻斷（obstacle thickness）
OBSTACLE_THICKNESS = 10
# 當 frontier 變化數量超過此門檻時，才觸發完整 rebuild（預設較保守）
FRONTIER_REBUILD_THRESHOLD = 50

# --- Pixel semantics ---
PIXEL_FREE = 255
PIXEL_UNKNOWN = 127
PIXEL_OCCUPIED = 1
PIXEL_START = 208
MAP_THRESHOLD = 150

# 選點策略參數：在選點時對 utility 與 distance 做正規化並以權重合併
# 0..1 的值，越靠近 1 表示越偏好效用（utility），越靠近 0 表示越偏好距離
SELECTION_UTILITY_WEIGHT = 0.7
# 若啟用則在選點時對前 N 個 Euclid 最佳候選使用 A* 取得真實路徑代價
SELECTION_USE_PATH_COST = False
# 當 SELECTION_USE_PATH_COST 為 True 時，先以 Euclid 排序並取 top K 再跑 A*
SELECTION_PATH_COST_TOPK = 5

# --- Sensor Parameters ---
SENSOR_ANGLE_INC = 0.5 / 180 * math.pi
MAX_SENSOR_STEPS = 10000

# --- Graph Generation Parameters ---
LARGE_DISTANCE = 1e5
VISITED_DIST_THRESHOLD = 10

# --- New Constants for Refactoring ---
COVERAGE_THRESHOLD = 0.95
ROBOT_SEPARATION_DIST = 15
TARGET_SEPARATION_DIST = 20
PLANNED_PATH_SEPARATION_DIST = 10
COLLISION_DIST = 1.5
MAX_STAY_COUNT = 5
RETURN_FAIL_COOLDOWN_STEPS = 10
MAX_RETURN_REPLAN_ATTEMPTS = 2

# --- Graph Pruning Parameters ---
ENABLE_GRAPH_PRUNING = True
PRUNING_KEEPALIVE_RADIUS = 120  # Always keep nodes within this radius of robots

# --- Coordination Parameters ---
ENABLE_SEQUENTIAL_ASSIGNMENT = True
SEQUENTIAL_REPULSION_RADIUS = 200  # Radius to penalize nearby targets after assignment
SEQUENTIAL_REPULSION_STRENGTH = 0.8 # Factor to reduce utility (e.g., utility *= (1 - 0.8))

# --- Frontier Clustering Parameters ---
ENABLE_FRONTIER_CLUSTERING = True
MIN_FRONTIER_SIZE = 5  # Ignore tiny frontier fragments
