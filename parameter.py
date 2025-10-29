SENSOR_RANGE=80
SERVER_COMM_RANGE = 160
ROBOT_COMM_RANGE = 80
DISTANCE_WEIGHT = 1
OUT_RANGE_STEP = 20 # 超時返回的步數閾值

MAX_EPS_STEPS=196 # 單一回合最大步數
K_SIZE = 30 # K 鄰近的 K 值

# --- 移除：效能優化參數 ---
# GRAPH_UPDATE_INTERVAL = 10 # 不再需要
# --------------------------

# --- 智慧會合機制參數 ---
INFO_GAIN_HISTORY_LEN = 20  # 追蹤增益的步數窗口
MIN_INFO_GAIN_THRESHOLD = 50  # 增益停滯閾值 (窗口內總增益)
LOCAL_UTILITY_THRESHOLD = 10  # 本地機會枯竭閾值 (單點最大效益)
# ---------------------------------

NUM_DENSE_COORDS_WIDTH=50 # 圖節點採樣密度
CUR_AGENT_KNN_RAD=80
GLOBAL_GRAPH_KNN_RAD=160

UTILITY_CALC_RANGE=40 # 計算節點效益時考慮的前沿點範圍

GLOBAL_NODES_TO_FRONTIER_AVOID_SPARSE_RAD=120