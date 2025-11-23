import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from driver import plot_grouped_results

def merge_results(input_dir: str, output_dir: str, output_filename: str = "merged_results"):
    """
    Merges all CSV files in input_dir and generates visualizations.
    """
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files. Merging...")
    
    df_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
            print(f"Loaded {f} ({len(df)} rows)")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not df_list:
        print("No valid data loaded.")
        return

    merged_df = pd.concat(df_list, ignore_index=True)
    
    os.makedirs(output_dir, exist_ok=True)
    csv_out_path = os.path.join(output_dir, f"{output_filename}.csv")
    merged_df.to_csv(csv_out_path, index=False)
    print(f"Saved merged results to {csv_out_path}")

    # Generate plots
    print("Generating plots...")
    plot_grouped_results(merged_df, output_dir, output_filename)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge result CSVs and plot")
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Directory containing CSV files")
    parser.add_argument("--output-dir", "-o", type=str, default="merged_results", help="Directory to save merged results")
    parser.add_argument("--filename", "-f", type=str, default="merged_results", help="Base filename for output")
    
    args = parser.parse_args()
    
    merge_results(args.input_dir, args.output_dir, args.filename)
