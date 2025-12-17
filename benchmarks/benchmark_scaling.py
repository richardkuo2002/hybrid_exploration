import argparse
import subprocess
import time
import pandas as pd
import random
import os
import sys

# Assume this script is in benchmarks/
# Root is one level up
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
WORKER_PATH = os.path.join(ROOT_DIR, "worker.py")


def run_benchmark(num_runs=20, map_index=1):
    agent_counts = [3, 4, 5]
    results = []

    print(f"Starting Benchmark: {num_runs} runs per agent count (3, 4, 5)")
    print(f"Map Index: {map_index}")
    
    for n_agents in agent_counts:
        print(f"\n--- Testing with {n_agents} Agents ---")
        param_success_count = 0
        steps_history = []
        
        for i in range(num_runs):
            print(f"Run {i+1}/{num_runs} (Agents: {n_agents})...", end="", flush=True)
            
            # Construct command
            # python worker.py --TEST_MAP_INDEX <map_index> --TEST_AGENT_NUM <n_agents> --no-save_video
            cmd = [
                "/home/oem/miniconda3/envs/hybrid/bin/python", WORKER_PATH,
                "--TEST_MAP_INDEX", str(map_index),
                "--TEST_AGENT_NUM", str(n_agents),
                "--no-save_video"
            ]
            
            try:
                # Run worker and capture output to parse steps/success
                # We need to parse the LAST line of output or use a specific format
                # worker.py logs: "Episode finished: True in 123 steps"
                # IMPORTANT: Cwd must be ROOT_DIR so worker can find maps/
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=ROOT_DIR) # 5 min timeout
                
                output = result.stderr # Logging usually goes to stderr in worker.py configuration
                # Also check stdout just in case
                full_output = output + "\n" + result.stdout
                
                success = False
                steps = 9999
                
                # Parse output for "Episode finished: True/False in X steps"
                import re
                match = re.search(r"Episode finished: (True|False) in (\d+) steps", full_output)
                if match:
                    success_str = match.group(1)
                    steps = int(match.group(2))
                    success = (success_str == "True")
                else:
                    print(" [Parse Fail]", end="")
                    
                if success:
                    param_success_count += 1
                    steps_history.append(steps)
                    print(f" Success ({steps} steps)")
                else:
                    print(f" Failed/Timeout ({steps} steps)")
                    
                results.append({
                    "agents": n_agents,
                    "run_id": i,
                    "map_index": map_index,
                    "success": success,
                    "steps": steps
                })
                
            except subprocess.TimeoutExpired:
                print(" [Timeout]")
                results.append({
                    "agents": n_agents,
                    "run_id": i,
                    "map_index": map_index,
                    "success": False,
                    "steps": -1
                })
            except Exception as e:
                print(f" [Error: {e}]")

    # Final Report
    df = pd.DataFrame(results)
    
    print("\n=== Benchmark Summary ===")
    summary = df.groupby("agents").agg(
        success_rate=("success", "mean"),
        avg_steps=("steps", lambda x: x[x > 0].mean()), # Filter out -1 or failed steps if desired
        min_steps=("steps", lambda x: x[x > 0].min()),
        max_steps=("steps", lambda x: x[x > 0].max())
    )
    print(summary)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmark_scaling_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nDetailed results saved to {csv_filename}")
    
    # Generate Markdown Report
    report_filename = f"report_scaling_benchmark_{timestamp}.md"
    with open(report_filename, "w") as f:
        f.write(f"# Scaling Benchmark Report\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Map Index: {map_index}\n\n")
        f.write("## Summary\n")
        f.write(summary.to_markdown())
        f.write("\n\n## Detailed Log\n")
        f.write(df.to_markdown())
    print(f"Report saved to {report_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_index", type=int, default=-1, help="Specific map index to use. -1 for random (fixed for all runs).")
    parser.add_argument("--runs", type=int, default=20, help="Number of runs per configuration")
    args = parser.parse_args()
    
    selected_map = args.map_index
    if selected_map == -1:
        selected_map = random.randint(0, 1000) # Pick one map for the whole batch
        print(f"Randomly selected Map Index: {selected_map}")
        
    run_benchmark(num_runs=args.runs, map_index=selected_map)
