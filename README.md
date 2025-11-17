# 混合式多機器人探索 (Hybrid Multi-Robot Exploration)

本專案實作混合式多機器人探索系統，結合集中式伺服器與分散式機器人策略。核心功能包括局部感測與地圖建構、圖結構路徑規劃、伺服器任務分配，以及多機器人間的任務移轉與地圖合併。

## 主要元件

- `env.py`：環境初始化、地圖讀取、視覺化與影格緩衝/影片輸出（`Env` 類）。
- `robot.py`：機器人行為、局部地圖與規劃。
- `server.py`：全域地圖管理、任務分配與協調。
- `graph_generator.py`、`graph.py`、`node.py`：節點生成與路徑規劃模組。
- `sensor.py`：感測模擬（可用 numba/JIT 加速的實作）。
- `utils.py`：通用工具（碰撞檢查等）。
- `parameter.py`：各項超參數（感測範圍、圖更新間隔等）。
- `worker.py` / `driver.py`：單次實驗或批次控制程式。

## 新增與重要說明（依程式碼）

- 預設地圖路徑：`Env.__init__` 目前預設 `self.map_path = "maps/easy_even"`；啟動時會掃描該目錄並過濾影像檔。若無有效地圖檔案，程式會記錄錯誤並退出。
- 影片儲存機制：`Env` 新增串流 / 緩衝設計（見 `Env._save_frame_to_memory` 與 `Env.save_video`）：
  - 若系統可用 `cv2`（OpenCV），程式會嘗試以 `cv2.VideoWriter` 串流寫入臨時 mp4 檔，減少記憶體使用。完成後會將臨時檔搬到目標位置。
  - 若不可用或串流失敗，會在記憶體中以 JPEG bytes 緩衝（`Env.frames_data`），並由 `save_video()` 在結束時合成影片。緩衝上限由 `Env._frames_buffer_max` 控制（預設 300）。
- 繪圖：提供有視窗即時繪圖 (`plot_env(step, save_frame=False)`) 與 headless（無視窗）模式 (`plot_env_without_window`)；可透過 `save_frame=True` 或 headless 模式儲存每步影格。
- numba/JIT：感測 (`sensor.py`) 與某些工具 (`utils.py`) 支援 numba 加速；若環境不支援 numba，可移除裝飾器以使用純 Python 實作。

## 安裝

建議使用 conda 建立隔離環境並安裝依賴（範例使用環境名稱 `hybrid`）：

```bash
conda create -n hybrid python=3.10 -y
conda activate hybrid
pip install -r requirements.txt
```

若希望啟用影片串流以節省記憶體，請安裝 OpenCV（在 conda 下推薦使用 pip 的輪子或 conda-forge）：

```bash
conda activate hybrid
pip install opencv-python          # 或: conda install -c conda-forge opencv
```

備註：若你偏好使用 `venv`，原本的 `python -m venv .venv` + `source .venv/bin/activate` 方式仍然有效。

## 如何執行（單次模擬）

預設執行：

```bash
python worker.py
```

- 常用參數（視 `worker.py` / `driver.py` 支援）：
- `--TEST_MAP_INDEX N`：指定要選取的地圖索引（從 `Env.map_path` 中列出檔案）。
- `--TEST_AGENT_NUM N`：機器人數量。
- `--plot`：啟用即時繪圖視窗（預設可能為啟用）。
- `--no-save_video`：不儲存影片輸出。
- `--debug`：啟用詳細日誌（增加 debug 輸出）。

注意：預設情況下程式只會顯示 ERROR / CRITICAL 訊息以減少日誌噪音；如果想看到 INFO 與 WARNING，請加上 `--debug`。若需要更低層級的 DEBUG 訊息，可進一步修改 `worker.py` 或在執行環境中設定（參見開發者注意事項）。

範例：

```bash
python worker.py --TEST_MAP_INDEX 1 --TEST_AGENT_NUM 3 --no-save_video
```

## 影片輸出與緩衝行為

- 若安裝 `opencv-python` 並可成功開啟 `cv2.VideoWriter`，程式會在執行期間直接串流寫入臨時 mp4 檔（通常較省記憶體），並在 `Env.save_video()` 被呼叫時將該臨時檔移至最終檔名。
- 若 `cv2` 不可用或串流失敗，則會以 JPEG bytes 緩衝在 `Env.frames_data`，執行 `save_video()` 時再合成影片。為避免無限制記憶體成長，可調整 `Env._frames_buffer_max`。

## 開發者注意事項

- 地圖格式：程式以灰階影像讀入，並把像素值 208（若存在）視為起始點；若未找到會使用預設起點 `[100, 100]`。
- 若遇到循環匯入（ImportError），請嘗試將部分匯入延後到函式內或調整模組間依賴。
- 若要停用 numba，加速裝飾器可在 `sensor.py` 與 `utils.py` 中移除。
- 主要可調參數請在 `parameter.py` 中修改（例如 SENSOR_RANGE、GRAPH_UPDATE_INTERVAL、K_SIZE、_frames_buffer_max 等）。

### 日誌（Logging）

- 預設：為了減少終端噪音，程式預設只顯示 ERROR 與 CRITICAL 等級的日誌。
- 如需看到 INFO 與 WARNING：在啟動時加上 `--debug`（例如 `python worker.py --debug`）。
- 如需顯示 DEBUG 級別訊息，可修改 `worker.py` 中的 logging 設定或手動在執行前設定環境變數（例如 `WORKER_DEBUG=1`，視實作而定）。

這可幫助在常規測試時只關注錯誤，而在除錯時取得更完整的運行資訊。

## 檔案位置速查

- 主控制 / 執行：`worker.py`, `driver.py`
- 環境/視覺化：`env.py`
- 機器人邏輯：`robot.py`
- 伺服器：`server.py`
- 圖結構：`graph_generator.py`, `graph.py`, `node.py`
- 感測：`sensor.py`
- 工具：`utils.py`
- 參數：`parameter.py`

## 批次測試

若要批次執行多張地圖或多次實驗，請參考 `driver.py` 中的範例/註解，或修改 `worker.py` 以接受外部參數與輸出結果到指定目錄。