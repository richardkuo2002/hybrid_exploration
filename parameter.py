SENSOR_RANGE=80
SERVER_COMM_RANGE = 160
ROBOT_COMM_RANGE = 80
DISTANCE_WEIGHT = 1
OUT_RANGE_STEP = 20

MAX_EPS_STEPS=196
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

NUM_DENSE_COORDS_WIDTH=50
CUR_AGENT_KNN_RAD=80
GLOBAL_GRAPH_KNN_RAD=160
UTILITY_CALC_RANGE=40
GLOBAL_NODES_TO_FRONTIER_AVOID_SPARSE_RAD=120
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