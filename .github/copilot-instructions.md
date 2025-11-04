## 簡短說明（給 AI 編碼代理）

這個 repo 實作混合式多機器人探索模擬：集中式 Server 做任務分配，分散式 Robot 做本地探索與地圖合併。下列內容集中列出能讓 AI 代理立即上手、修改或擴充專案時最常需要的知識點。

## 重要檔案（先看這些）
- `env.py`：環境、地圖 I/O、視覺化與影片緩衝。  
- `worker.py`：實驗啟動器（CLI 參數在此）。  
- `robot.py`：機器人行為、local_map、任務移轉邏輯（handoff）。  
- `server.py`：global_map、效用計算、匈牙利指派（linear_sum_assignment）。  
- `graph_generator.py` / `graph.py` / `node.py`：節點/邊生成與 A* 路徑規劃。

## 關鍵設計與資料流（必讀）
- Local vs Global maps：每個 Robot 有 `Robot.local_map`，透過 `Env.merge_maps` 與 `Robot.update_local_awareness` 與其他機器人/Server 同步。  
- Graph 兩種表現：
	- 計算/搜尋使用 `Graph` 物件（`graph.py`）。
	- Server 與序列化使用 dict-based `graph.edges`（來自 `Graph_generator.rebuild_graph_structure`）。
 轉換範例：參考 `Server._plan_global_path`（把 dict 轉回 Graph，保留 Edge.length）。
- 設計權衡：圖結構重建成本高，專案廣泛採「每 N 步完整重建（GRAPH_UPDATE_INTERVAL）＋平常用較輕量更新」的策略。新增週期性工作時請沿用此模式。

## 常見數值語意（容易出錯）
- 地圖像素值：1=obstacle、127=unknown、255=free。大量程式會基於這些常數做判斷（例如 `Env.import_map_revised`、`Env.find_frontier`、`Server.calculate_coverage_ratio`）。

## CLI / 開發工作流程（實用命令）
1. 建議：建立虛擬環境並安裝套件：
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. 單次模擬（範例）：
```bash
python worker.py --TEST_MAP_INDEX 1 --TEST_AGENT_NUM 2 --no-save_video --debug
```
3. headless 或 CI：加 `--no-save_video` 或在程式中設定 `plot=False`。

## 整合點與外部相依
- 影像 I/O：使用 `skimage`；若有 `cv2` 可加速影片輸出，否則 fallback 為記憶體 JPEG。  
- 指派演算法：`scipy.optimize.linear_sum_assignment`（修改代價矩陣時注意 shape）。

## 變更風險 & 檢查點（實務檢核）
- 修改圖生成或路徑規劃：同時檢查 `graph_generator.find_shortest_path` 與 `Server._plan_global_path` 的節點編號與邊權一致性。  
- 修改地圖值或語意（例如把 255 改為其他）：會影響多處程式，請先確認所有比較條件。  
- 修改任務移轉（handoff）或 cooldown：檢查 `robot.py` 中的 cooldown 常數與 `parameter.py`；調整可能改變整體任務分配行為。

## 可執行的微型任務（給 AI 的範例）
- 新增一個 smoke-test：載入 `DungeonMaps/train/easy/` 的地圖，執行一輪 `Worker`（`plot=False`），檢查沒有例外並回傳 coverage 值在 [0,1]。  
- 若需將 `graph.edges`（dict）轉為 `Graph`：直接複用 `Server._plan_global_path` 的轉換步驟（create Graph, add nodes, add edges using Edge.length）。

## 何時回報人類
- 更動地圖語意（像素意義）或加入非 Python 相依時請先詢問（會影響 CI 與資料）。

## 變更紀錄 & 反饋
我已將原始指引精簡並保留專案關鍵模式；若有想要補充的實例或要更細的「模組間資料流圖」，告訴我我會加入。  

---
請檢視這份精簡版本，有沒有重要情境或檔案我漏掉？想要我把其中某段擴成更詳細的「修改 checklist」或加入快速範例程式碼嗎？
