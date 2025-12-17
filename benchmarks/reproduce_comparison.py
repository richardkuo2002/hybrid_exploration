import argparse
import subprocess
import time
import pandas as pd
import random
import os
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

def run_benchmark_comparison(num_runs=10, map_index=1, agent_num=3):
    """
    Run comparison between Standard (Handoff OFF) and Hybrid (Handoff ON).
    """
    modes = [
        {"name": "Standard (OFF)", "flag": "--no-enable-handoff"},
        {"name": "Hybrid (ON)", "flag": "--enable-handoff"}
    ]
    
    results = []
    
    logger.info(f"Starting Comparison Benchmark: {num_runs} runs each.")
    logger.info(f"Map Index: {map_index}, Agents: {agent_num}")
    
    for i in range(num_runs):
        logger.info(f"\n--- Run {i+1}/{num_runs} ---")
        
        for mode in modes:
            logger.info(f"Testing {mode['name']}...")
            
            cmd = [
                "/home/oem/miniconda3/envs/hybrid/bin/python", "worker.py",
                "--TEST_MAP_INDEX", str(map_index),
                "--TEST_AGENT_NUM", str(agent_num),
                "--no-save_video",
                "--debug",
                mode["flag"]
            ]
            
            start_time = time.time()
            try:
                # Run worker
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                duration = time.time() - start_time
                
                full_output = result.stderr + "\n" + result.stdout
                
                # Parse metrics
                import re
                success = False
                steps = 9999
                total_dist = 0.0
                
                # Look for standard log lines
                match = re.search(r"Episode finished: (True|False) in (\d+) steps. Metrics: (\{.*\})", full_output)
                if match:
                    success = (match.group(1) == "True")
                    steps = int(match.group(2))
                    metrics_str = match.group(3)
                    try:
                        metrics = eval(metrics_str)
                        total_dist = metrics.get('total_distance', 0.0)
                    except:
                        pass
                    logger.info(f" DONE. Steps: {steps}, Dist: {total_dist:.1f}, Success: {success}")
                else:
                    logger.info(f" FAILED parse. Output tail: {full_output[-200:]}")
                    
                results.append({
                    "run_id": i,
                    "mode": mode["name"],
                    "success": success,
                    "steps": steps,
                    "total_distance": total_dist,
                    "duration": duration
                })
                
            except subprocess.TimeoutExpired:
                logger.info(" TIMEOUT")
                results.append({
                    "run_id": i,
                    "mode": mode["name"],
                    "success": False,
                    "steps": -1,
                    "total_distance": -1,
                    "duration": 300
                })
            except Exception as e:
                logger.error(f" ERROR: {e}")

    # Summary
    df = pd.DataFrame(results)
    print("\n=== Comparison Summary ===")
    if not df.empty:
        summary = df.groupby("mode").agg(
            success_rate=("success", "mean"),
            avg_steps=("steps", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
            avg_dist=("total_distance", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
            avg_time=("duration", "mean")
        )
        print(summary)
    
    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"reproduction_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_index", type=int, default=10, help="Map index (default: 10)")
    parser.add_argument("--runs", type=int, default=5, help="Runs per mode")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents")
    args = parser.parse_args()
    
    run_benchmark_comparison(num_runs=args.runs, map_index=args.map_index, agent_num=args.agents)
