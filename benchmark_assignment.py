
import subprocess
import pandas as pd
import random
import time
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# Benchmark Configuration
NUM_EPISODES = 20
AGENT_NUM = 4
TIMEOUT_SECONDS = 300  # 5 minutes per run safeguard
RESULTS_FILE = f"assignment_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv"
PYTHON_EXEC = "/home/oem/miniconda3/envs/hybrid/bin/python"
WORKER_SCRIPT = "worker.py"
MAX_PARALLEL_WORKERS = 6  # 24 cores / 4 agents ~ 6 workers. Conservative: 6.

# Modes to compare
MODES = {
    "Hungarian": ["--no-enable-sequential", "--no-save_video"], 
    "Sequential": ["--enable-sequential", "--no-save_video"]
}

def parse_metrics(output_str):
    metrics = {
        "success": False,
        "steps": 195, 
        "total_distance": 0.0,
        "coverage": 0.0
    }
    
    match_finish = re.search(r"Episode finished: (True|False) in (\d+) steps", output_str)
    if match_finish:
        metrics["success"] = (match_finish.group(1) == "True")
        metrics["steps"] = int(match_finish.group(2))
        
    match_timeout = re.search(r"timeout at step (\d+) with coverage ([\d\.]+)%", output_str)
    if match_timeout:
        metrics["steps"] = int(match_timeout.group(1))
        
    match_metrics = re.search(r"Metrics: ({.*})", output_str)
    if match_metrics:
        try:
            d = eval(match_metrics.group(1))
            metrics["total_distance"] = d.get("total_distance", 0.0)
            metrics["coverage"] = d.get("coverage", 0.0)
        except:
            pass
            
    return metrics

def run_single_simulation(job_args):
    """
    Run a single simulation (one map, one mode).
    job_args: (run_id, map_idx, mode_name, mode_flags_list)
    """
    run_id, map_idx, mode_name, mode_flags = job_args
    
    cmd = [
        PYTHON_EXEC, WORKER_SCRIPT,
        "--TEST_MAP_INDEX", str(map_idx),
        "--TEST_AGENT_NUM", str(AGENT_NUM),
        "--enable-handoff", # Implicitly True as BooleanOptional, but good to be explicit if using the flag logic
    ] + mode_flags
    
    start_time = time.time()
    result_data = {
        "run_id": run_id,
        "map_index": map_idx,
        "mode": mode_name,
        "success": False,
        "steps": 195,
        "total_distance": -1,
        "coverage": 0,
        "duration": 0
    }
    
    try:
        # Use capture_output=True to keep main terminal clean, 
        # but we lose real-time visibility of this specific subprocess.
        # Given parallel execution, mixed stdout is messy anyway.
        proc_res = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=TIMEOUT_SECONDS
        )
        duration = time.time() - start_time
        result_data["duration"] = duration
        
        parsed = parse_metrics(proc_res.stdout + proc_res.stderr)
        result_data.update(parsed)
        
        # Return checking
        log_snippet = (proc_res.stdout + proc_res.stderr)[-200:].replace('\n', ' ')
        return result_data, log_snippet
        
    except subprocess.TimeoutExpired:
        result_data["duration"] = TIMEOUT_SECONDS
        return result_data, "Timeout"
    except Exception as e:
        return result_data, f"Error: {e}"

def run_benchmark():
    map_indices = [random.randint(0, 5000) for _ in range(NUM_EPISODES)]
    
    print(f"Starting PARALLEL Benchmark: {NUM_EPISODES} Episodes, {AGENT_NUM} Agents, {MAX_PARALLEL_WORKERS} Parallel Workers")
    print(f"Map Indices: {map_indices}\n")
    
    # Prepare jobs
    jobs = []
    for i, map_idx in enumerate(map_indices):
        for mode_name, mode_args in MODES.items():
            jobs.append((i, map_idx, mode_name, mode_args))
            
    results = []
    
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        future_to_job = {executor.submit(run_single_simulation, job): job for job in jobs}
        
        total_jobs = len(jobs)
        completed = 0
        
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            completed += 1
            run_id, map_idx, mode_name, _ = job
            
            try:
                res_data, log_snippet = future.result()
                results.append(res_data)
                print(f"[{completed}/{total_jobs}] Ep {run_id+1} Map {map_idx} {mode_name}: Steps={res_data['steps']}, Dist={res_data['total_distance']:.1f}, Succ={res_data['success']}")
            except Exception as exc:
                print(f"Job {job} generated an exception: {exc}")

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nBenchmark Complete. Saved to {RESULTS_FILE}")
    
    try:
        summary = df.groupby("mode")[["success", "steps", "total_distance"]].mean()
        print("\nSummary:")
        print(summary)
    except Exception as e:
        print(f"Summary Error: {e}")
        print(df)

if __name__ == "__main__":
    run_benchmark()
