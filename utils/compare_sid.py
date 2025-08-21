#!/usr/bin/env python3
"""
Compare SID performance between bnlearn_50rep and bnlearn_50rep_test_reproducibility scenarios
"""

import os
import re
import numpy as np
import argparse

def extract_sid_from_log(log_file, dataset='asia', method='ABAPC (Ours)'):
    """Extract SID values from log file"""
    if not os.path.exists(log_file):
        return None, None
    
    dag_sids = []
    cpdag_sids = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    for line in lines:
        if f"'dataset': '{dataset}'" in line and f"'model': '{method}'" in line:
            # Extract SID value - look for the full pattern
            sid_match = re.search(r"'sid':\s*([^}]+)", line)
            if sid_match:
                sid_str = sid_match.group(1).strip()
                # Check if it's a tuple (CPDAG) or single value (DAG)
                if sid_str.startswith('(') and sid_str.endswith(')'):
                    # CPDAG SID - extract both values
                    cpdag_match = re.search(r'\(([^,]+),\s*([^)]+)\)', sid_str)
                    if cpdag_match:
                        try:
                            low = float(cpdag_match.group(1).replace('np.float64(', '').replace(')', ''))
                            high = float(cpdag_match.group(2).replace('np.float64(', '').replace(')', ''))
                            cpdag_sids.append((low, high))
                        except:
                            pass
                else:
                    # DAG SID - single value
                    try:
                        dag_sid = float(sid_str.replace('np.float64(', '').replace(')', ''))
                        dag_sids.append(dag_sid)
                    except:
                        pass
    
    return dag_sids, cpdag_sids

def compare_sid_performance(version1='bnlearn_50rep', version2='bnlearn_50rep_test_reproducibility', 
                          dataset='asia', method='abapc'):
    """Compare SID performance between two scenarios"""
    
    print("=== SID Performance Comparison ===")
    print()
    
    # Check first scenario (from stored results)
    print(f"📊 {version1} scenario:")
    results_file_1 = f"results/stored_results_{version1}.npy"
    results_cpdag_file_1 = f"results/stored_results_{version1}_cpdag.npy"
    
    if os.path.exists(results_file_1):
        try:
            results = np.load(results_file_1, allow_pickle=True)
            for row in results:
                if len(row) >= 2 and row[0] == dataset and method.upper() in row[1]:
                    if len(row) >= 22:  # SID mean is at index 20, SID std at index 21
                        sid_mean = row[20]  # sid_mean
                        sid_std = row[21]   # sid_std
                        print(f"  DAG SID (aggregated): {sid_mean} ± {sid_std}")
        except Exception as e:
            print(f"  Error loading DAG results: {e}")
    
    if os.path.exists(results_cpdag_file_1):
        try:
            results = np.load(results_cpdag_file_1, allow_pickle=True)
            for row in results:
                if len(row) >= 2 and row[0] == dataset and method.upper() in row[1]:
                    if len(row) >= 24:  # SID low and high are the last two columns
                        sid_low = row[-4]  # sid_low_mean
                        sid_high = row[-2]  # sid_high_mean
                        print(f"  CPDAG SID (aggregated): {sid_low} - {sid_high}")
        except Exception as e:
            print(f"  Error loading CPDAG results: {e}")
    
    print()
    
    # Check second scenario (from log file)
    print(f"📊 {version2} scenario:")
    log_file_2 = f"results/log_{version2}.log"
    
    dag_sids, cpdag_sids = extract_sid_from_log(log_file_2, dataset, f"{method.upper()} (Ours)")
    
    if dag_sids:
        dag_sids = [s for s in dag_sids if s is not None]
        if dag_sids:
            print(f"  DAG SID (individual runs):")
            print(f"    Mean: {np.mean(dag_sids):.2f}")
            print(f"    Std:  {np.std(dag_sids):.2f}")
            print(f"    Min:  {np.min(dag_sids):.2f}")
            print(f"    Max:  {np.max(dag_sids):.2f}")
            print(f"    Values: {dag_sids[:5]}..." if len(dag_sids) > 5 else f"    Values: {dag_sids}")
    
    if cpdag_sids:
        cpdag_sids = [s for s in cpdag_sids if s is not None]
        if cpdag_sids:
            low_values = [s[0] for s in cpdag_sids]
            high_values = [s[1] for s in cpdag_sids]
            print(f"  CPDAG SID (individual runs):")
            print(f"    Low range -  Mean: {np.mean(low_values):.2f}, Std: {np.std(low_values):.2f}")
            print(f"    High range - Mean: {np.mean(high_values):.2f}, Std: {np.std(high_values):.2f}")
            print(f"    Sample ranges: {cpdag_sids[:3]}..." if len(cpdag_sids) > 3 else f"    Sample ranges: {cpdag_sids}")
    
    print()
    print("=== Conclusion ===")
    
    # Get the DAG SID values for comparison
    dag_sid_1 = None
    cpdag_low_1 = None
    cpdag_high_1 = None
    
    if os.path.exists(results_file_1):
        try:
            results = np.load(results_file_1, allow_pickle=True)
            for row in results:
                if len(row) >= 2 and row[0] == dataset and method.upper() in row[1]:
                    if len(row) >= 22:
                        dag_sid_1 = row[20]  # sid_mean
        except:
            pass
    
    if os.path.exists(results_cpdag_file_1):
        try:
            results = np.load(results_cpdag_file_1, allow_pickle=True)
            for row in results:
                if len(row) >= 2 and row[0] == dataset and method.upper() in row[1]:
                    if len(row) >= 24:
                        cpdag_low_1 = row[-4]  # sid_low_mean
                        cpdag_high_1 = row[-2]  # sid_high_mean
        except:
            pass
    
    # Compare DAG SID
    if dag_sids and dag_sid_1 is not None:
        dag_mean_2 = np.mean(dag_sids)
        if abs(dag_mean_2 - dag_sid_1) < 0.01:
            print("✅ DAG SID values are the same!")
        else:
            print(f"❌ DAG SID values differ: {dag_mean_2:.2f} vs {dag_sid_1:.2f}")
    
    # Compare CPDAG SID
    if cpdag_sids and cpdag_low_1 is not None and cpdag_high_1 is not None:
        low_values = [s[0] for s in cpdag_sids]
        high_values = [s[1] for s in cpdag_sids]
        low_mean_2 = np.mean(low_values)
        high_mean_2 = np.mean(high_values)
        if abs(low_mean_2 - cpdag_low_1) < 0.01 and abs(high_mean_2 - cpdag_high_1) < 0.01:
            print("✅ CPDAG SID values are the same!")
        else:
            print(f"❌ CPDAG SID values differ: {low_mean_2:.2f}-{high_mean_2:.2f} vs {cpdag_low_1:.2f}-{cpdag_high_1:.2f}")
    
    print()
    print("Note: Lower SID values indicate better performance (closer to true graph)")
    print("CPDAG SID is typically reported as a range (low - high)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare SID performance between experiment scenarios')
    parser.add_argument('--version1', '-v1', type=str, default='bnlearn_50rep',
                       help='First experiment version (default: bnlearn_50rep)')
    parser.add_argument('--version2', '-v2', type=str, default='bnlearn_50rep_test_reproducibility',
                       help='Second experiment version (default: bnlearn_50rep_test_reproducibility)')
    parser.add_argument('--dataset', '-d', type=str, default='asia',
                       help='Dataset name (default: asia)')
    parser.add_argument('--method', '-m', type=str, default='abapc',
                       help='Method name (default: abapc)')
    
    args = parser.parse_args()
    compare_sid_performance(args.version1, args.version2, args.dataset, args.method)
