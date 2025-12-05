
import multiprocessing
import time
import os
import datetime
import pandas as pd
import numpy as np
from driver import run_single_experiment, save_and_viz_results

def run_benchmark():
    # Configuration matching IR2 Custom (Even) benchmark
    N_RUNS = 20
    AGENT_NUM = 5
    MAP_TYPE = "even"
    OUTPUT_DIR = "results_comparison_even"
    
    # Tasks: Run indices 0-19.
    # IR2 loops global_step from 0 to N.
    # Hybrid's run_single_experiment uses map_index to pick from sorted map list.
    tasks = []
    for i in range(N_RUNS):
        # args_tuple: (run_index, agent_num, map_index, graph_update_interval, map_type)
        tasks.append((i, AGENT_NUM, i, None, MAP_TYPE))
        
    print(f"Starting Hybrid vs IR2 Benchmark (Even): {N_RUNS} runs, {AGENT_NUM} agents, map_type={MAP_TYPE}")
    
    num_processes = min(multiprocessing.cpu_count() - 1, N_RUNS)
    results = []
    
    t0 = time.perf_counter()
    with multiprocessing.Pool(processes=num_processes) as pool:
        for i, res in enumerate(pool.imap_unordered(run_single_experiment, tasks)):
            results.append(res)
            print(f"[{i+1}/{N_RUNS}] map={res['map_index']} -> success={res['success']}, finished_ep={res['finished_ep']}, time={res['duration']:.3f}s")
            
    total_time = time.perf_counter() - t0
    print(f"Benchmark completed in {total_time:.2f}s")
    
    # Sort results by run_index
    results.sort(key=lambda x: x["run_index"])
    
    # Extract data for saving
    finished_eps = [res["finished_ep"] for res in results]
    successes = [res["success"] for res in results]
    agent_used = [res["agent_num"] for res in results]
    map_indices = [res["map_index"] for res in results]
    durations = [res["duration"] for res in results]
    coverages = [res["coverage"] for res in results]
    total_distances = [res["total_distance"] for res in results]
    target_selection_counts = [res["target_selection_count"] for res in results]
    collision_replan_counts = [res["collision_replan_count"] for res in results]
    map_merge_counts = [res["map_merge_count"] for res in results]
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"hybrid_vs_ir2_even_{timestamp}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"{filename_base}.csv")
    
    save_and_viz_results(
        finished_eps,
        successes,
        agent_used,
        map_indices,
        durations,
        coverages,
        total_distances,
        target_selection_counts,
        collision_replan_counts,
        map_merge_counts,
        csv_path=csv_path
    )

if __name__ == "__main__":
    run_benchmark()
