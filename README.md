# 混合式多機器人探索 (Hybrid Multi-Robot Exploration)

本專案實作一套混合式架構的多機器人探索系統，結合集中式伺服器與分散式（自主）機器人策略。 核心功能包括：局部感測與地圖建構、圖結構（collision-free graph）路徑規劃、伺服器任務分配（匈牙利演算法）、以及多機器人之間的任務移轉與地圖合併。

快速導覽

* Env / Worker：模擬環境與單次實驗執行。
* Robot / Sensor：機器人行為與感測模擬（簡化 Lidar）。
* Server：全域地圖管理與任務分配。
* Graph_generator / Node / Graph：以圖為基礎的節點生成與 A\* 規劃。
* parameter.py：主要超參數。

必要資料與目錄結構 請確保專案內含地圖資料（範例路徑）：

```bash
hybrid_exploration/
├── DungeonMaps/
│   └── train/
│       └── easy/
│           └── <\*.png, \*.jpg ...>
├── env.py
├── worker.py
├── requirements.txt
└── ...
```

安裝

1. 建議使用虛擬環境：

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

2. 安裝相依套件：

```bash
pip install -r requirements.txt
```

若要從 GitHub 取得原始碼（示範）：

```bash
git clone https://github.com/richardkuo2002/hybrid_exploration.git
cd hybrid_exploration
```

如何執行（單次模擬）

* 以預設參數執行：

```bash
python worker.py
```

* 常用參數：
  * --TEST\_MAP\_INDEX N ：指定地圖索引（整數）。
  * --TEST\_AGENT\_NUM N ：機器人數量。
  * --plot ：啟用即時繪圖視窗（會較耗資源）。
  * --no-save\_video ：不儲存影片（預設會儲存）。
  * --debug ：啟用詳細日誌。

範例（指定地圖與機器人數）：

```bash
python worker.py --TEST_MAP_INDEX 1 --TEST_AGENT_NUM 3 --no-save_video
```

影片輸出 若啟用儲存影片，預設會在專案下產生 `videos/` 目錄，輸出檔名會包含時間、地圖索引與機器人數。

常見問題與注意事項

* 地圖格式：程式目前以灰階影像讀取，並將特定像素值 (e.g., 208) 當作起始點標記；請確認 DungeonMaps 內的地圖格式符合預期。
* numba / JIT：部分感測與碰撞檢查函式使用 numba 加速，若環境不支援 numba，可先移除或使用對應的純 Python 版本。
* 匯入錯誤：若模組互相引用產生循環匯入（ImportError），可檢查執行順序或將部分匯入移至函式內部以延遲加載。

開發者備註

* 若要調整探索行為，請檢查 `parameter.py` 中的參數（例如 SENSOR\_RANGE、GRAPH\_UPDATE\_INTERVAL、K\_SIZE 等）。
* 想要收集多次實驗結果，可撰寫或改良 `driver.py` 進行批次執行。

聯絡 如需協助或回報問題，請在專案的 issue 中描述執行環境、錯誤日誌與重現步驟。