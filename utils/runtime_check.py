#!/usr/bin/env python3
"""
Quick script to check how many runs have been completed
"""

import os
import re
import numpy as np
import argparse

def quick_check(version='bnlearn_50rep_test_reproducibility', dataset='asia', method='abapc', expected_runs=50):
    """Quick check of experiment progress"""
    
    # Try different possible log file locations - prioritize main log file
    log_files = [
        f"results/log_{version}.log",  # Main log file with all run metrics
        f"results/{method}_{version}_{dataset}/log.log",  # Execution log (last run only)
        "results/log.log",
        f"{method}_{version}_{dataset}/log.log"
    ]
    
    log_file = None
    for log_path in log_files:
        if os.path.exists(log_path):
            log_file = log_path
            print(f"Using log file: {log_file}")
            break
    
    if log_file is None:
        print(f"Log files not found. Tried:")
        for log_path in log_files:
            print(f"  {log_path}")
        print("Experiments may not have started.")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Extract elapsed times and count completed runs
    elapsed_times = []
    completed_runs = 0
    
    # Count unique runs by looking at every other entry (since each run has DAG and CPDAG metrics)
    asia_abapc_lines = []
    for line in lines:
        if "'dataset': 'asia'" in line and "'model': 'ABAPC (Ours)'" in line and 'elapsed' in line:
            asia_abapc_lines.append(line)
    
    # Each run produces 2 entries (DAG and CPDAG), so count every other one
    completed_runs = len(asia_abapc_lines) // 2
    
    # Extract elapsed times from every other line (DAG metrics only)
    for i in range(0, len(asia_abapc_lines), 2):
        line = asia_abapc_lines[i]
        match = re.search(r"'elapsed':\s*([\d.]+)", line)
        if match:
            elapsed_times.append(float(match.group(1)))
    
    # Also check for "Time taken for ABAPC" format as backup
    for line in lines:
        if 'Time taken for ABAPC:' in line:
            # Extract elapsed time from "Time taken for ABAPC" format
            match = re.search(r'Time taken for ABAPC:\s*([\d.]+)s', line)
            if match:
                elapsed_times.append(float(match.group(1)))
    
    # Check if we have stored results that show aggregated statistics
    stored_results_file = f"results/stored_results_{version}.npy"
    if os.path.exists(stored_results_file):
        try:
            results = np.load(stored_results_file, allow_pickle=True)
            # Look for the specific dataset and method
            for row in results:
                if len(row) >= 2 and row[0] == dataset and method.upper() in row[1]:
                    if len(row) >= 4:
                        elapsed_mean = row[2]
                        elapsed_std = row[3]
                        print(f"📊 Stored aggregated results found:")
                        print(f"  Average time per run: {elapsed_mean:.2f} ± {elapsed_std:.2f}s")
                        print(f"  Note: This shows aggregated statistics, but we cannot")
                        print(f"        determine exactly how many individual runs completed")
                        print(f"        (the script only saves aggregated results)")
                        return
        except:
            pass
    
    # If no aggregated results, check for progress indicators
    print(f"📈 Progress check from execution log:")
    print(f"  Completed runs detected: {completed_runs}/{expected_runs} ({completed_runs/expected_runs*100:.1f}%)")
    
    if completed_runs == 0:
        print(f"  Status: Experiments may not have started or are in progress")
        print(f"  Check the execution log for activity")
    elif completed_runs < expected_runs:
        print(f"  Status: Experiments in progress")
        print(f"  Remaining runs: {expected_runs - completed_runs}")
        
        # Try to estimate progress from log activity
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Count various progress indicators
            skeleton_count = len(re.findall(r'Skeleton discovery done', content))
            running_count = len(re.findall(r'Running CausalABA', content))
            loading_count = len(re.findall(r'Loading facts from', content))
            
            print(f"  Progress indicators found:")
            print(f"    Skeleton discovery: {skeleton_count}")
            print(f"    CausalABA runs: {running_count}")
            print(f"    Facts loading: {loading_count}")
            
            if skeleton_count > 0:
                print(f"  Note: Skeleton discovery indicates algorithm is running")
            if running_count > 0:
                print(f"  Note: CausalABA execution detected")
    else:
        print(f"  Status: All {expected_runs} runs appear to be completed!")
    
    if elapsed_times:
        print(f"\nRuntime Statistics:")
        print(f"  Average: {np.mean(elapsed_times):.2f}s")
        print(f"  Min:     {np.min(elapsed_times):.2f}s")
        print(f"  Max:     {np.max(elapsed_times):.2f}s")
        print(f"  Std Dev: {np.std(elapsed_times):.2f}s")
        
        # Estimate remaining time
        remaining_runs = expected_runs - completed_runs
        avg_time = np.mean(elapsed_times)
        estimated_remaining = avg_time * remaining_runs
        print(f"\nTime Estimates:")
        print(f"  Estimated remaining: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f} minutes)")
    
    if completed_runs > 0:
        # Get last few log entries to show recent activity
        recent_entries = []
        for line in reversed(lines):
            if 'Time taken for ABAPC:' in line:
                # Extract key info from the log line
                match = re.search(r'Time taken for ABAPC:\s*([\d.]+)s', line)
                elapsed = match.group(1) if match else "N/A"
                recent_entries.append(f"Run {len(recent_entries)+1}: {elapsed}s")
                if len(recent_entries) >= 3:
                    break
        
        print(f"\nRecent runs:")
        for entry in reversed(recent_entries):
            print(f"  {entry}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check experiment progress')
    parser.add_argument('--version', '-v', type=str, default='bnlearn_50rep_test_reproducibility',
                       help='Experiment version/scenario (default: bnlearn_50rep_test_reproducibility)')
    parser.add_argument('--dataset', '-d', type=str, default='asia',
                       help='Dataset name (default: asia)')
    parser.add_argument('--method', '-m', type=str, default='abapc',
                       help='Method name (default: abapc)')
    parser.add_argument('--runs', '-r', type=int, default=50,
                       help='Expected number of runs (default: 50)')
    
    args = parser.parse_args()
    quick_check(args.version, args.dataset, args.method, args.runs)
