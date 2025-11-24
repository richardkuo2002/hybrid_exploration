import random
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # 若不想跳出視窗，使用無GUI後端
import argparse
import logging
import multiprocessing
import datetime
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt

from worker import Worker


def run_single_experiment(
    args_tuple: Tuple[int, int, int, Optional[int]],
) -> Dict[str, Any]:
    """
    執行單次實驗的頂層函式，供 multiprocessing 調用。
    Args:
        args_tuple (tuple): (run_index, agent_num, map_index, graph_update_interval)
    Returns:
        dict: 實驗結果
    """
    run_index, agent_num, map_index, graph_update_interval = args_tuple

    # 每個 process 需有獨立的隨機狀態，mulitprocessing 會複製父 process
    # 但為了保險起見，在子 process 端明確地用 run_index 對其設定 deterministic seed
    seed_val = ((run_index + 1) * 12345) % (2**32)
    np.random.seed(seed_val)
    random.seed((run_index + 1) * 67890 % (2**32))

    worker = Worker(
        global_step=0,
        agent_num=agent_num,
        map_index=map_index,
        save_video=False,
        graph_update_interval=graph_update_interval,
    )

    t_start = time.perf_counter()
    success, finished_ep, metrics = worker.run_episode(curr_episode=run_index + 1)
    t_end = time.perf_counter()

    dur = t_end - t_start

    # 處理 finished_ep 格式
    finished_ep_scalar: Union[float, int] = np.nan
    if isinstance(finished_ep, (list, tuple, np.ndarray)) and len(finished_ep) > 0:
        finished_ep_scalar = finished_ep[-1]
    elif isinstance(finished_ep, (int, float)):
        finished_ep_scalar = finished_ep
    else:
        finished_ep_scalar = np.nan

    # Extract map ID from filename (e.g., "img_even_123.png" -> 123)
    try:
        import re
        map_filename = worker.env.file_path
        match = re.search(r"(\d+)", map_filename)
        if match:
            real_map_index = int(match.group(1))
        else:
            real_map_index = map_index # Fallback
    except Exception:
        real_map_index = map_index

    return {
        "run_index": run_index,
        "agent_num": agent_num,
        "map_index": real_map_index,
        "success": bool(success),
        "finished_ep": finished_ep_scalar,
        "duration": dur,
        "coverage": metrics.get("coverage", 0.0),
        "total_distance": metrics.get("total_distance", 0.0),
        "replanning_count": metrics.get("replanning_count", 0),
        "map_merge_count": metrics.get("map_merge_count", 0),
    }


def run_batch(
    n_runs: int = 100,
    map_max_index: int = 1000,
    agent_min: int = 3,
    agent_max: int = 5,
    seed: Optional[int] = None,
    graph_update_interval: Optional[int] = None,
    jobs: Optional[int] = None,
    same_maps: bool = False,
    output_dir: str = "results",
) -> Tuple[List[Any], List[bool], List[int], List[int], List[float]]:
    """
    跑多次實驗並回傳每次 finished_ep 列表 (平行化版本)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 準備實驗參數列表
    tasks = []
    if same_maps:
        # same_maps 模式：n_runs 代表地圖數量
        # 對每張地圖，跑遍所有 agent 數量 (agent_min ~ agent_max)
        print(f"Mode: Same Maps. Generating {n_runs} maps, each tested with agents {agent_min} to {agent_max}.")
        run_idx = 0
        for map_i in range(n_runs):
            # 隨機選一張圖 (或是循序，這裡維持隨機但固定種子)
            map_index = random.randint(0, map_max_index)
            for agent_num in range(agent_min, agent_max + 1):
                tasks.append((run_idx, agent_num, map_index, graph_update_interval))
                run_idx += 1
        # 更新總 runs 數以便顯示進度
        n_runs = len(tasks)
    else:
        # 原始模式：每次 run 隨機選 agent_num 和 map
        for i in range(n_runs):
            agent_num = random.randint(agent_min, agent_max)
            map_index = random.randint(0, map_max_index)
            tasks.append((i, agent_num, map_index, graph_update_interval))

    finished_eps: List[Any] = []
    successes: List[bool] = []
    agent_used: List[int] = []
    map_indices: List[int] = []
    durations: List[float] = []
    coverages: List[float] = []
    total_distances: List[float] = []
    replanning_counts: List[int] = []
    map_merge_counts: List[int] = []

    # 偵測 CPU 核心數，默認為 cpu_count - 1，但可透過 jobs 參數覆蓋
    max_possible_proc = multiprocessing.cpu_count()
    default_num_proc = max(1, max_possible_proc - 1)
    if jobs is None or jobs <= 0:
        num_processes = min(default_num_proc, n_runs) if n_runs > 0 else default_num_proc
    else:
        # 允許使用者指定到整台機器的核心數上限
        num_processes = min(max(1, jobs), max_possible_proc, n_runs)
    print(
        f"Starting batch run with {n_runs} episodes using {num_processes} processes..."
    )

    t0 = time.perf_counter()

    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 使用 imap_unordered 可以即時取得完成的結果，適合顯示進度
            # 若需要順序，可最後再根據 run_index 排序，或改用 map
            results = []
            for i, res in enumerate(pool.imap_unordered(run_single_experiment, tasks)):
                results.append(res)
                print(
                    f"[{i+1}/{n_runs}] agents={res['agent_num']}, map={res['map_index']} -> success={res['success']}, finished_ep={res['finished_ep']}, time={res['duration']:.3f}s"
                )

    except KeyboardInterrupt:
        print("Batch run interrupted. Terminating pool...")
        try:
            if "pool" in locals():
                pool.terminate()
        except Exception:
            pass
        raise

    # 整理結果 (依 run_index 排序回原本順序，雖然統計上沒差，但為了對齊)
    results.sort(key=lambda x: x["run_index"])

    for res in results:
        finished_eps.append(res["finished_ep"])
        successes.append(res["success"])
        agent_used.append(res["agent_num"])
        map_indices.append(res["map_index"])
        durations.append(res["duration"])
        coverages.append(res["coverage"])
        total_distances.append(res["total_distance"])
        replanning_counts.append(res["replanning_count"])
        map_merge_counts.append(res["map_merge_count"])

    total_time = time.perf_counter() - t0
    print(f"Batch run completed in {total_time:.2f}s")

    # Generate dynamic filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    g_str = f"_g{graph_update_interval}" if graph_update_interval is not None else ""
    filename_base = f"results_runs{n_runs}_agents{agent_min}-{agent_max}{g_str}_{timestamp}"
    
    # Create a dedicated subdirectory for this run
    run_output_dir = os.path.join(output_dir, filename_base)
    os.makedirs(run_output_dir, exist_ok=True)
    
    csv_path = os.path.join(run_output_dir, f"{filename_base}.csv")
    png_path = os.path.join(run_output_dir, f"{filename_base}.png")

    save_and_viz_results(
        finished_eps,
        successes,
        agent_used,
        map_indices,
        durations,
        coverages,
        total_distances,
        replanning_counts,
        map_merge_counts,
        csv_path=csv_path,
        png_path=png_path,
    )

    return finished_eps, successes, agent_used, map_indices, durations


def save_and_viz_results(
    finished_eps: List[Any],
    successes: List[bool],
    agent_used: List[int],
    map_indices: List[int],
    durations: List[float],
    coverages: List[float],
    total_distances: List[float],
    replanning_counts: List[int],
    map_merge_counts: List[int],
    csv_path: str = "results.csv",
    png_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    將每次實驗指標存成 CSV，並計算摘要統計：
      - finished_ep（中位數、平均）
      - duration（中位數、平均）
      - duration_per_ep（中位數、平均）
    回傳資料表與摘要表；箱形圖建議在互動式環境繪製。
    """
    # finished_eps 應為 scalar list（run_batch 已轉換）。若不是，上面方法也能容忍處理。
    df = pd.DataFrame(
        {
            "finished_ep": finished_eps,
            "success": successes,
            "agent_num": agent_used,
            "map_index": map_indices,
            "duration": durations,
            "coverage": coverages,
            "total_distance": total_distances,
            "replanning_count": replanning_counts,
            "map_merge_count": map_merge_counts,
        }
    )

    # df["parity"] = np.where(df["map_index"] % 2 == 0, "even", "odd")
    # 避免除以 0（或 NaN），以 NaN 代表不可計算
    df["duration_per_ep"] = df["duration"] / df["finished_ep"].replace({0: np.nan})

    # 輸出 CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # 輸出 CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # 畫進階圖表
    if png_path:
        # png_path 這裡只當作 base name 的參考，實際檔名會由 plot_grouped_results 決定
        # 我們從 png_path 提取目錄與檔名 base
        out_dir = os.path.dirname(png_path)
        base = os.path.splitext(os.path.basename(png_path))[0]
        plot_grouped_results(df, out_dir, base)

    # 摘要統計 (不再分 parity)
    summary = df.agg(
        {
            "finished_ep": ["count", "median", "mean"],
            "duration": ["median", "mean"],
            "duration_per_ep": ["median", "mean"],
            "coverage": ["mean", "std"],
            "total_distance": ["mean", "std"],
            "replanning_count": ["mean", "std"],
        }
    )

    print("\n=== Summary Statistics ===")
    print(summary)
    print("==========================\n")

    return df, summary


def plot_grouped_results(
    df: pd.DataFrame,
    output_dir: str,
    filename_base: str,
) -> None:
    """
    繪製分組視覺化圖表：
    1. Grouped Boxplots (依 agent_num 分組)
    2. Trend Lines (依 agent_num 平均值 + 標準差)
    """
    # 確保 agent_num 是整數以便排序與繪圖
    if "agent_num" not in df.columns:
        print("plot_grouped_results: 'agent_num' column missing, skipping.")
        return

    # 準備要畫的指標
    metrics = [
        ("finished_ep", "Finished Steps"),
        ("duration", "Duration (s)"),
        ("coverage", "Coverage"),
        ("total_distance", "Total Distance"),
        ("replanning_count", "Replanning Count"),
        ("map_merge_count", "Map Merge Count"),
    ]

    # 1. Grouped Boxplots
    # 針對每個指標畫一張圖，X軸為 agent_num
    for col, title in metrics:
        if col not in df.columns:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 準備數據：list of arrays
        agent_nums = sorted(df["agent_num"].unique())
        data = []
        labels = []
        
        for n in agent_nums:
            subset = df[df["agent_num"] == n][col].dropna()
            if len(subset) > 0:
                data.append(subset.values)
                labels.append(str(n))
        
        if not data:
            plt.close(fig)
            continue
            
        bp = ax.boxplot(
            data, tick_labels=labels, patch_artist=True, showmeans=True, meanline=True
        )
        
        # 美化
        colors = ["#73a9ff", "#9fdc6c", "#f7a072", "#e6c229", "#d11149"]
        for i, box in enumerate(bp["boxes"]):
            box.set(facecolor=colors[i % len(colors)], alpha=0.6)
            
        ax.set_title(f"{title} by Agent Num")
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        
        out_path = os.path.join(output_dir, f"{filename_base}_boxplot_{col}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved boxplot: {out_path}")

    # 2. Trend Lines (Summary Plot)
    # 將所有指標的趨勢畫在同一張大圖 (Subplots)
    # 計算平均與標準差
    summary = df.groupby("agent_num").agg(["mean", "std"])
    
    valid_metrics = [m for m in metrics if m[0] in df.columns]
    n_metrics = len(valid_metrics)
    if n_metrics == 0:
        return

    cols = 2
    rows = (n_metrics + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()
    
    agent_nums = summary.index.values
    
    for i, (col, title) in enumerate(valid_metrics):
        ax = axes[i]
        mean_val = summary[col]["mean"]
        std_val = summary[col]["std"].fillna(0)
        
        ax.errorbar(
            agent_nums, 
            mean_val, 
            yerr=std_val, 
            fmt='-o', 
            capsize=5, 
            linewidth=2, 
            markersize=6,
            label="Mean ± Std"
        )
        
        ax.set_title(f"{title} Trend")
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xticks(agent_nums)
        
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{filename_base}_trends.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trend plot: {out_path}")


def save_results_csv(
    finished_eps: List[Any],
    successes: List[bool],
    agent_used: List[int],
    map_indices: List[int],
    durations: List[float],
    filename: str = "results.csv",
) -> None:
    df = pd.DataFrame(
        {
            "finished_ep": finished_eps,
            "success": successes,
            "agent_num": agent_used,
            "map_index": map_indices,
            "duration_s": durations,
        }
    )
    df.to_csv(filename, index=False)
    print(f"Saved CSV to {filename}")


if __name__ == "__main__":
    # 參數可自行調整
    parser = argparse.ArgumentParser(description="Batch runner for hybrid_exploration")
    parser.add_argument(
        "--graph-update-interval",
        "-g",
        type=int,
        default=None,
        help="Graph full rebuild interval (override parameter.GRAPH_UPDATE_INTERVAL)",
    )
    parser.add_argument(
        "--n-runs", type=int, default=100, help="Number of runs for batch"
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=None,
        help="Number of parallel worker processes to use (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--map-max-index", type=int, default=10000, help="Maximum map index"
    )
    parser.add_argument(
        "--agent-min", type=int, default=3, help="Minimum number of agents"
    )
    parser.add_argument(
        "--agent-max", type=int, default=3, help="Maximum number of agents"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--same-maps",
        action="store_true",
        help="If set, n-runs becomes number of maps, and each map is tested with all agent counts (min to max).",
    )
    args = parser.parse_args()

    N_RUNS = args.n_runs
    MAP_MAX_INDEX = 10000
    AGENT_MIN = 3
    AGENT_MAX = 3
    SEED = 42  # 可設 None'
    logging.getLogger().setLevel(logging.ERROR)

    finished_eps, successes, agent_used, map_indices, durations = run_batch(
        n_runs=N_RUNS,
        map_max_index=(
            args.map_max_index if hasattr(args, "map_max_index") else MAP_MAX_INDEX
        ),
        agent_min=args.agent_min if hasattr(args, "agent_min") else AGENT_MIN,
        agent_max=args.agent_max if hasattr(args, "agent_max") else AGENT_MAX,
        seed=args.seed if hasattr(args, "seed") else SEED,
        graph_update_interval=args.graph_update_interval,
        jobs=args.jobs,
        same_maps=args.same_maps,
        output_dir=args.output_dir,
    )

    # save_results_csv is redundant now as run_batch calls save_and_viz_results
    # save_results_csv(finished_eps, successes, agent_used, map_indices, durations)

    # 印出簡單統計
    arr = np.array(finished_eps, dtype=float)
    if arr.size > 0 and np.isfinite(arr).any():
        finite = arr[np.isfinite(arr)]
        print(
            f"finished_ep stats -> count={finite.size}, mean={finite.mean():.2f}, median={np.median(finite):.2f}, std={finite.std(ddof=1):.2f}, min={finite.min()}, max={finite.max()}"
        )
    else:
        print("No valid finished_ep data to show stats.")

    # 畫箱形圖
    # plot_boxplots_finished_duration(finished_eps, durations,  output_png="finished_ep_boxplot.png", title=f"Finished Episodes over {N_RUNS} runs")
