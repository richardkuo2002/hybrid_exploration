
import subprocess
import os
import re
import pandas as pd
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration ---
NUM_EPISODES = 10
AGENT_NUM = 3
TIMEOUT_SECONDS = 300
HYBRID_RESULTS_FILE = "results_hybrid.csv"
IR2_RESULTS_FILE = "results_ir2.csv"
FINAL_COMPARISON_FILE = "comparison_hybrid_ir2.csv"
MAX_WORKERS = 4 # Parallel workers for Hybrid

def parse_hybrid_output(output_str):
    metrics = {
        "success": False,
        "steps": 195, 
        "coverage": 0.0,
        "total_distance": 0.0
    }
    
    match_finish = re.search(r"Episode finished: (True|False) in (\d+) steps", output_str)
    if match_finish:
        metrics["success"] = (match_finish.group(1) == "True")
        metrics["steps"] = int(match_finish.group(2))
        
    match_timeout = re.search(r"timeout at step (\d+)", output_str)
    if match_timeout:
        metrics["steps"] = int(match_timeout.group(1))

    # Try to find coverage in metrics dict or timeout line
    match_metrics = re.search(r"Metrics: ({.*})", output_str)
    if match_metrics:
        try:
            d = eval(match_metrics.group(1))
            metrics["coverage"] = d.get("coverage", 0.0)
            metrics["total_distance"] = d.get("total_distance", 0.0)
        except:
            pass
    
    # Fallback for coverage if not in metrics
    if metrics["coverage"] == 0.0:
         match_cov = re.search(r"coverage ([\d\.]+)%", output_str)
         if match_cov:
             metrics["coverage"] = float(match_cov.group(1)) / 100.0

    return metrics

def run_single_hybrid(map_idx):
    cmd = [
        "python3", "worker.py",
        "--TEST_MAP_INDEX", str(map_idx),
        "--TEST_AGENT_NUM", str(AGENT_NUM),
        "--no-save_video",
        "--plot", # To disable plot (plot arg enables it). Wait, checking logic again.
                  # worker.py: parser.add_argument("--plot", action="store_true")
                  # So passing --plot ENABLEs it.
                  # If I want to DISABLE, I should NOT pass --plot.
                  # Correct logic: Remove --plot from cmd to disable.
        "--no-enable-handoff", # Using default True, but passing --no-enable-handoff DISABLES it?
                               # Wait, boolean optional action: --enable-handoff, --no-enable-handoff.
                               # Env default is True.
                               # User wants Hybrid comparison. Hybrid usually implies ENABLED features.
                               # So I should remove --no-enable-handoff to keep it ENABLED.
    ]
    # Remove --plot to disable plotting
    # Remove --no-enable-handoff to uses default (Enabled)
    
    # Correct CMD for Hybrid (Enabled)
    cmd = [
        "python3", "worker.py",
        "--TEST_MAP_INDEX", str(map_idx),
        "--TEST_AGENT_NUM", str(AGENT_NUM),
        "--no-save_video",
        "--enable-handoff",
        "--enable-sequential"
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        metrics = parse_hybrid_output(res.stdout + res.stderr)
        metrics["map_index"] = map_idx
        metrics["method"] = "Hybrid"
        return metrics
    except subprocess.TimeoutExpired:
        return {
            "map_index": map_idx, "method": "Hybrid", "success": False, "steps": 195, "coverage": 0.0, "total_distance": 0.0, "note": "Timeout"
        }
    except Exception as e:
        return {
            "map_index": map_idx, "method": "Hybrid", "success": False, "steps": 195, "coverage": 0.0, "total_distance": 0.0, "note": str(e)
        }

def run_hybrid_benchmarks():
    print(f"--- Running Hybrid Benchmark ({NUM_EPISODES} Maps) in Parallel ({MAX_WORKERS} workers) ---")
    results = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_map = {executor.submit(run_single_hybrid, i): i for i in range(NUM_EPISODES)}
        
        for future in as_completed(future_to_map):
            map_idx = future_to_map[future]
            try:
                metrics = future.result()
                results.append(metrics)
                note = metrics.get("note", "")
                succ_str = "Succ" if metrics['success'] else "Fail"
                print(f"  Hybrid Map {map_idx}: Steps={metrics['steps']}, Cov={metrics['coverage']*100:.1f}%, {succ_str} {note}")
            except Exception as exc:
                print(f"  Hybrid Map {map_idx} generated an exception: {exc}")

    df = pd.DataFrame(results)
    df.to_csv(HYBRID_RESULTS_FILE, index=False)
    print(f"Hybrid results saved to {HYBRID_RESULTS_FILE}\n")
    return df

def run_ir2_benchmarks():
    print(f"--- Running IR2 Benchmark ({NUM_EPISODES} Maps) ---")
    # IR2 runs batch via test_driver.py based on NUM_TEST in test_parameter.py
    # We already set NUM_TEST=10 and NUM_META_AGENT=4 there.
    
    cwd = os.path.join(os.getcwd(), "ir2_benchmark")
    cmd = ["python3", "test_driver.py"]
    
    try:
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS * NUM_EPISODES)
        if res.returncode != 0:
            print("IR2 Failed/Errors:")
            print(res.stderr[-1000:])
            print(res.stdout[-1000:])
        else:
            print("IR2 Completed.")
            
        # Find the generated CSV
        log_dir = os.path.join(cwd, "mar_inference", "test_results", "log")
        list_of_files = glob.glob(os.path.join(log_dir, 'data_*.csv'))
        if not list_of_files:
            print("No IR2 CSV found!")
            return pd.DataFrame()
            
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Found IR2 results: {latest_file}")
        
        df = pd.read_csv(latest_file)
        # Rename columns to match Hybrid
        # IR2 cols: eps, num_robots, max_dist, steps, explored, success, connectivity
        df = df.rename(columns={
            "eps": "map_index",
            "steps": "steps",
            "explored": "coverage",
            "max_dist": "total_distance",
            "success": "success"
        })
        df["method"] = "IR2"
        
        df.to_csv(IR2_RESULTS_FILE, index=False)
        return df
        
    except Exception as e:
        print(f"IR2 Execution Error: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df_hybrid = run_hybrid_benchmarks()
    df_ir2 = run_ir2_benchmarks()
    
    if not df_hybrid.empty and not df_ir2.empty:
        # Combine
        cols = ["method", "map_index", "success", "steps", "coverage", "total_distance"]
        
        # Ensure columns exist
        for col in cols:
            if col not in df_hybrid.columns: df_hybrid[col] = None
            if col not in df_ir2.columns: df_ir2[col] = None
            
        combined = pd.concat([df_hybrid[cols], df_ir2[cols]], ignore_index=True)
        
        combined.to_csv(FINAL_COMPARISON_FILE, index=False)
        
        print("\n--- Final Comparison Summary ---")
        summary = combined.groupby("method")[["success", "steps", "coverage", "total_distance"]].mean()
        print(summary)
        print(f"\nDetailed results saved to {FINAL_COMPARISON_FILE}")
    else:
        print("One or both benchmarks failed to produce data.")
