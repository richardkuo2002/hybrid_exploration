# 混合式架構於多機器人協同探索 (Hybrid Architecture for Multi-Robot Exploration)

本專案旨在實現一個結合集中式與分散式（去中心化）策略的混合式多機器人探索系統。系統包含一個中央伺服器和多個自主機器人，利用「無碰撞圖」結構進行高效的地圖資訊處理與路徑規劃。

## 核心架構

本系統的架構設計如下：

### 1. 混合式架構

* **機器人設計 (Decentralized)：**
    * **即時感測：** 機器人即時建構並更新局部地圖。
    * **自主探索與規劃：** 當機器人不在伺服器通訊範圍內時（`out_range_step > OUT_RANGE_STEP`），能自主進行領域探索及路徑規劃（使用本地圖的 `plan_local_path`）。
    * **相互溝通：** 當其他機器人在通訊範圍内時（`dist < ROBOT_COMM_RANGE`），會合併彼此的局部資訊並同步。

* **伺服器設計 (Centralized)：**
    * **地圖收集與合併：** 伺服器持續接收通訊範圍內（`dist_to_server < SERVER_COMM_RANGE`）所有機器人的局部地圖和狀態，並整合成全域地圖。
    * **任務分配：** 伺服器在 `server_step` 中，根據全域地圖及機器人分布，使用匈牙利演算法（`scipy.optimize.linear_sum_assignment`）即時指派各機器人新的探索區域，並規劃全域路徑（`plan_global_path`）。

### 2. 地圖資訊處理：無碰撞圖 (Collision-Free Graph)

為了克服傳統網格地圖（Grid Map）規劃效率低落的問題，本研究採用圖論結構（Graph-based）進行地圖表達與規劃：

* **節點建立：** 在自由空間中均勻選取採樣點作為圖節點。
* **邊建立：** 節點之間透過 K-Nearest Neighbors 尋找鄰居，並使用 `check_collision` 確保邊的建立是無碰撞的。
* **節點資訊：** 每個節點（`Node`）會儲存該位置的探索價值（`utility`），此價值基於該節點可觀測到的前沿點（Frontiers）數量，作為伺服器指派任務的依據。

## 安裝指南 (Getting Started)

### 1. 複製儲存庫
```bash
git clone [https://github.com/richardkuo2002/hybrid_exploration.git](https://github.com/richardkuo2002/hybrid_exploration.git)
cd hybrid_exploration
```

### 2. 下載地圖資料 (重要)
專案結構應如下所示：
```bash
hybrid_exploration/
├── DungeonMaps/
│   ├── train/
│   │   └── easy/
│   │       └── (地圖檔案 .png)
│   └── ...
├── env.py
├── worker.py
├── driver.py
├── requirements.txt
└── ...
```
### 3. 安裝依賴套件
建議使用虛擬環境。本專案的依賴套件皆列於 `requirements.txt`。
```bash
pip install -r requirements.txt
```
## 如何運行 (Usage)
### 執行單次模擬
可以透過執行 `worker.py` 來進行單次模擬測試。此腳本會初始化一個 `Worker` 並運行一個 episode。
```bash
python worker.py
```
模擬過程將會產生 `episode_0.mp4` 影片檔。
### 執行批次實驗
若要進行多次實驗並收集統計數據，請執行 `driver.py`。
```bash
python driver.py
```
此腳本會運行 `N_RUNS` 次實驗，並將結果（完成步數、成功率、使用地圖等）儲存到 `results.csv` 中，同時繪製箱形圖（Boxplot）。

##專案檔案結構
- `driver.py`: 批次實驗的主執行檔。

- `worker.py`: 定義 `Worker` 類別，負責單次模擬的完整流程（episode loop）。

- `env.py`: 模擬環境核心。處理機器人移動、地圖更新、感測、以及機器人之間和伺服器之間的通訊與地圖合併。

- `server.py`: 定義 `Server` 類別，儲存全域地圖與狀態。

- `robot.py`: 定義 `Robot` 類別，儲存本地地圖與狀態。

- `graph_generator.py`: 核心演算法之一，負責建立、更新和管理無碰撞圖結構。

- `node.py`: 定義圖中的 `Node` 類別，並計算其探索效益（utility）。

- `graph.py`: 圖論的基礎類別（`Graph`, `Edge`）與 A* 尋路演算法。

- `sensor.py`: 模擬雷射雷達（Lidar）的感測器模型。

- `parameter.py`: 儲存所有模擬用的超參數（如感測範圍、通訊範圍等）。