import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 若不想跳出視窗，使用無GUI後端
import matplotlib.pyplot as plt

from worker import Worker

def run_batch(n_runs=100, map_max_index=1000, agent_min=3, agent_max=5, seed=None):
    """
    跑多次實驗並回傳每次 finished_ep 列表
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    finished_eps = []
    successes = []
    agent_used = []
    map_indices = []
    durations = []

    t0 = time.perf_counter()

    for i in range(n_runs):
        if (i)%2:
            continue
        agent_num = random.randint(agent_min, agent_max)  # 3~5（含上下界）
        map_index = random.randint(0, map_max_index)   # 0~1000（含上下界）
        map_index = i+1  # 固定測試用
        worker = Worker(global_step=0, agent_num=agent_num, map_index=map_index)
        t_start = time.perf_counter()
        success, finished_ep = worker.run_episode(curr_episode=i+1)
        t_end = time.perf_counter()

        dur = t_end - t_start
        
        finished_eps.append(finished_ep)
        successes.append(bool(success))
        agent_used.append(agent_num)
        map_indices.append(map_index)
        durations.append(dur)

        print(f"[{i+1}/{n_runs}] agents={agent_num}, map={map_index} -> success={success}, finished_ep={finished_ep}, time={dur:.3f}s")
    
    save_and_viz_results(finished_eps, successes, agent_used, map_indices, durations, csv_path="results1.csv")

    return finished_eps, successes, agent_used, map_indices, durations

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 無GUI環境可保留
import matplotlib.pyplot as plt

def save_and_viz_results(finished_eps, successes, agent_used, map_indices, durations, csv_path="results1.csv"):
    """
    將每次實驗指標存成 CSV，並計算按 map_index 奇偶分組的摘要統計：
      - finished_ep（中位數、平均）
      - duration（中位數、平均）
      - duration_per_ep（中位數、平均）
    回傳資料表與摘要表；箱形圖建議在互動式環境繪製。
    """
    # 若 finished_ep 是列表，取最後一個代表最終回合
    fin_scalar = [
        (fe[-1] if isinstance(fe, (list, tuple, np.ndarray)) and len(fe) > 0
         else (fe if isinstance(fe, (int, float)) else np.nan))
        for fe in finished_eps
    ]

    df = pd.DataFrame({
        'finished_ep': fin_scalar,
        'success': successes,
        'agent_used': agent_used,
        'map_index': map_indices,
        'duration': durations,
    })

    df['parity'] = np.where(df['map_index'] % 2 == 0, 'even', 'odd')
    df['duration_per_ep'] = df['duration'] / df['finished_ep']

    # 輸出 CSV
    df.to_csv(csv_path, index=False)

    # 摘要統計
    summary = df.groupby('parity').agg({
        'finished_ep': ['count', 'median', 'mean'],
        'duration': ['median', 'mean'],
        'duration_per_ep': ['median', 'mean']
    })

    return df, summary

def plot_boxplots_finished_duration(
    finished_eps,
    durations,
    output_png="finished_duration_boxplots.png",
    title="Finished EP, Duration, and Duration per EP",
    skip_zero_finished=True,
    show_mean=True
):
    """
    畫出三個箱形圖：
    1) finished_ep
    2) duration（秒）
    3) duration_per_ep = duration / finished_ep（若 finished_ep==0 依設定跳過）

    Args:
        finished_eps (Sequence[Number]): 每次 run 的 finished_ep。
        durations (Sequence[Number]): 每次 run 的耗時（秒）。
        output_png (str): 輸出檔名。
        title (str): 圖標題。
        skip_zero_finished (bool): 是否忽略 finished_ep==0 的樣本於第三個箱形圖。
        show_mean (bool): 箱形圖是否顯示平均值。
    """
    # 轉為 ndarray 並做長度/數值檢查
    fe = np.asarray(finished_eps, dtype=float)
    du = np.asarray(durations, dtype=float)

    if fe.size == 0 or du.size == 0 or fe.size != du.size:
        print("plot_boxplots_finished_duration: invalid inputs, skip.")
        return

    # 構造第三組資料：duration per ep
    if skip_zero_finished:
        mask = (fe > 0) & np.isfinite(du) & np.isfinite(fe)
        du_per_ep = du[mask] / fe[mask] if mask.any() else np.array([])
    else:
        # 對 0 做安全處理：以 NaN 代表不可除
        with np.errstate(divide='ignore', invalid='ignore'):
            du_per_ep = du / fe
            du_per_ep = du_per_ep[np.isfinite(du_per_ep)]

    # 準備要畫的資料與標籤
    data_list = [fe, du, du_per_ep]
    labels = ["finished_ep", "duration(s)", "duration/ep(s)"]

    # 過濾空資料組，避免空箱形圖報錯
    valid_data = []
    valid_labels = []
    for d, lab in zip(data_list, labels):
        if isinstance(d, np.ndarray) and d.size > 0 and np.isfinite(d).any():
            valid_data.append(d)
            valid_labels.append(lab)

    if len(valid_data) == 0:
        print("plot_boxplots_finished_duration: no valid data to plot, skip.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(
        valid_data,
        vert=True,
        patch_artist=True,
        showmeans=show_mean,
        meanline=True
    )

    # 美化外觀
    colors = ['#73a9ff', '#9fdc6c', '#f7a072']
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=colors[i % len(colors)], alpha=0.6, edgecolor='#444')
    for median in bp['medians']:
        median.set(color='#d62728', linewidth=2.0)
    if show_mean and 'means' in bp:
        for mean in bp['means']:
            mean.set(color='#2ca02c', linewidth=2.0)

    ax.set_title(title)
    ax.set_ylabel("value")
    ax.set_xticks(range(1, len(valid_labels) + 1))
    ax.set_xticklabels(valid_labels, rotation=0)

    # 輔助線與版面
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Boxplot saved to {output_png}")

import pandas as pd

def save_results_csv(finished_eps, successes, agent_used, map_indices, durations, filename="results.csv"):
    df = pd.DataFrame({
        "finished_ep": finished_eps,
        "success": successes,
        "agent_num": agent_used,
        "map_index": map_indices,
        "duration_s": durations
    })
    df.to_csv(filename, index=False)
    print(f"Saved CSV to {filename}")


if __name__ == '__main__':
    # 參數可自行調整
    N_RUNS = 100
    MAP_MAX_INDEX = 10000
    AGENT_MIN = 3
    AGENT_MAX = 3
    SEED = 42  # 可設 None

    finished_eps, successes, agent_used, map_indices, durations = run_batch(
        n_runs=N_RUNS,
        map_max_index=MAP_MAX_INDEX,
        agent_min=AGENT_MIN,
        agent_max=AGENT_MAX,
        seed=SEED
    )
    
    save_results_csv(finished_eps, successes, agent_used, map_indices, durations)
    
    # 印出簡單統計
    arr = np.array(finished_eps)
    print(f"finished_ep stats -> count={arr.size}, mean={arr.mean():.2f}, median={np.median(arr):.2f}, std={arr.std(ddof=1):.2f}, min={arr.min()}, max={arr.max()}")

    # 畫箱形圖
    plot_boxplots_finished_duration(finished_eps, durations,  output_png="finished_ep_boxplot.png", title=f"Finished Episodes over {N_RUNS} runs")
