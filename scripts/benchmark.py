#!/usr/bin/env python3
"""
benchmark.py — Parse timing CSV and generate performance charts.

Reads results/timings.csv and produces:
  1. Execution Time vs Matrix Size (grouped bar chart)
  2. Speedup vs CPU Baseline (line chart)
  3. GFLOPS Comparison (bar chart)
  4. MPI Strong Scaling (speedup vs process count)
  5. MPI Weak Scaling (time vs problem size)
  6. Cost Efficiency Analysis (time × $/hr)

Charts are saved to results/charts/
"""

import os
import sys
import csv
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    print("[ERROR] matplotlib and numpy required. Install: pip install matplotlib numpy")
    sys.exit(1)

# Paths
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
CSV_FILE = os.path.join(RESULTS_DIR, 'timings.csv')
CHART_DIR = os.path.join(RESULTS_DIR, 'charts')

# Cloud cost assumptions ($/hr)
COST_PER_HOUR = {
    'CPU Sequential': 0.0,       # Local, free
    'OpenMP CPU Parallel': 0.0,  # Local, free
    'CUDA Naïve': 0.0,           # Colab, free
    'CUDA Tiled (Shared Memory)': 0.0,  # Colab, free
    'OpenMP GPU Offload': 0.0,   # Colab, free
    'mpi-1': 0.085,   # 1x c5.large
    'mpi-2': 0.170,   # 2x c5.large
    'mpi-4': 0.340,   # 4x c5.large
}

# Color palette
COLORS = {
    'CPU Sequential': '#6366f1',
    'OpenMP CPU Parallel': '#8b5cf6',
    'CUDA Naïve': '#ef4444',
    'CUDA Tiled (Shared Memory)': '#f97316',
    'OpenMP GPU Offload': '#22c55e',
    'mpi-1': '#3b82f6',
    'mpi-2': '#06b6d4',
    'mpi-4': '#14b8a6',
}


def load_csv(path):
    """Load timing CSV into structured dict."""
    data = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(path):
        print(f"[ERROR] CSV file not found: {path}")
        sys.exit(1)

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row['mode'].strip()
            size = int(row['matrix_size'])
            data[mode][size] = {
                'time': float(row['exec_time_ms']),
                'gflops': float(row['gflops']),
                'verified': row['verified'].strip(),
            }
    return data


def setup_plot_style():
    """Apply consistent dark-themed plot styling."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.figsize': (12, 7),
        'figure.dpi': 150,
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'legend.fontsize': 10,
        'figure.facecolor': '#1a1a2e',
        'axes.facecolor': '#16213e',
        'axes.edgecolor': '#e2e8f0',
        'text.color': '#e2e8f0',
        'xtick.color': '#e2e8f0',
        'ytick.color': '#e2e8f0',
    })


def chart_exec_time(data, sizes):
    """Chart 1: Execution Time vs Matrix Size."""
    fig, ax = plt.subplots()
    modes = [m for m in data if not m.startswith('mpi-')]
    x = np.arange(len(sizes))
    width = 0.8 / max(len(modes), 1)

    for i, mode in enumerate(modes):
        times = [data[mode].get(s, {}).get('time', 0) for s in sizes]
        color = COLORS.get(mode, '#888888')
        ax.bar(x + i * width, times, width, label=mode, color=color, alpha=0.85)

    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Execution Time vs Matrix Size')
    ax.set_xticks(x + width * len(modes) / 2)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(loc='upper left')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, '1_exec_time.png'), bbox_inches='tight')
    plt.close()
    print("[CHART] 1_exec_time.png")


def chart_speedup(data, sizes):
    """Chart 2: Speedup vs CPU Baseline."""
    fig, ax = plt.subplots()
    cpu_key = 'CPU Sequential'
    if cpu_key not in data:
        print("[SKIP] No CPU baseline data for speedup chart")
        return

    modes = [m for m in data if m != cpu_key and not m.startswith('mpi-')]
    for mode in modes:
        speedups = []
        for s in sizes:
            cpu_t = data[cpu_key].get(s, {}).get('time', 1)
            mode_t = data[mode].get(s, {}).get('time', 0)
            speedups.append(cpu_t / mode_t if mode_t > 0 else 0)
        color = COLORS.get(mode, '#888888')
        ax.plot(sizes, speedups, 'o-', label=mode, color=color, linewidth=2, markersize=8)

    ax.axhline(y=1.0, color='#ef4444', linestyle='--', alpha=0.5, label='CPU Baseline (1x)')
    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Speedup (×)')
    ax.set_title('Speedup vs CPU Sequential Baseline')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, '2_speedup.png'), bbox_inches='tight')
    plt.close()
    print("[CHART] 2_speedup.png")


def chart_gflops(data, sizes):
    """Chart 3: GFLOPS Comparison."""
    fig, ax = plt.subplots()
    modes = [m for m in data if not m.startswith('mpi-')]
    x = np.arange(len(sizes))
    width = 0.8 / max(len(modes), 1)

    for i, mode in enumerate(modes):
        gf = [data[mode].get(s, {}).get('gflops', 0) for s in sizes]
        color = COLORS.get(mode, '#888888')
        ax.bar(x + i * width, gf, width, label=mode, color=color, alpha=0.85)

    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('GFLOPS')
    ax.set_title('Computational Throughput (GFLOPS)')
    ax.set_xticks(x + width * len(modes) / 2)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, '3_gflops.png'), bbox_inches='tight')
    plt.close()
    print("[CHART] 3_gflops.png")


def chart_mpi_strong_scaling(data, sizes):
    """Chart 4: MPI Strong Scaling (fixed problem, vary processes)."""
    fig, ax = plt.subplots()
    mpi_modes = sorted([m for m in data if m.startswith('mpi-')])
    if not mpi_modes:
        print("[SKIP] No MPI data for strong scaling chart")
        return

    processes = [int(m.split('-')[1]) for m in mpi_modes]

    for size in sizes:
        base_time = data.get(mpi_modes[0], {}).get(size, {}).get('time', 0)
        if base_time == 0:
            continue
        speedups = []
        for mode in mpi_modes:
            t = data[mode].get(size, {}).get('time', 0)
            speedups.append(base_time / t if t > 0 else 0)
        ax.plot(processes, speedups, 'o-', label=f'N={size}', linewidth=2, markersize=8)

    # Ideal scaling line
    if processes:
        ax.plot(processes, processes, '--', color='#ef4444', alpha=0.5, label='Ideal')

    ax.set_xlabel('Number of MPI Processes')
    ax.set_ylabel('Speedup (×)')
    ax.set_title('MPI Strong Scaling')
    ax.set_xticks(processes)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, '4_mpi_strong_scaling.png'), bbox_inches='tight')
    plt.close()
    print("[CHART] 4_mpi_strong_scaling.png")


def chart_mpi_weak_scaling(data, sizes):
    """Chart 5: MPI time vs problem size for each process count."""
    fig, ax = plt.subplots()
    mpi_modes = sorted([m for m in data if m.startswith('mpi-')])
    if not mpi_modes:
        print("[SKIP] No MPI data for weak scaling chart")
        return

    for mode in mpi_modes:
        times = [data[mode].get(s, {}).get('time', 0) for s in sizes]
        color = COLORS.get(mode, '#888888')
        ax.plot(sizes, times, 'o-', label=mode, color=color, linewidth=2, markersize=8)

    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('MPI Execution Time vs Problem Size')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, '5_mpi_weak_scaling.png'), bbox_inches='tight')
    plt.close()
    print("[CHART] 5_mpi_weak_scaling.png")


def chart_cost_efficiency(data, sizes):
    """Chart 6: Cost Efficiency (estimated cost per run)."""
    fig, ax = plt.subplots()
    modes = list(data.keys())
    x = np.arange(len(sizes))
    width = 0.8 / max(len(modes), 1)

    for i, mode in enumerate(modes):
        costs = []
        rate = COST_PER_HOUR.get(mode, 0.0)
        for s in sizes:
            t = data[mode].get(s, {}).get('time', 0)
            cost = (t / 1000.0 / 3600.0) * rate  # ms → hours → $
            costs.append(cost * 1000)  # Show in milli-dollars for readability
        color = COLORS.get(mode, '#888888')
        ax.bar(x + i * width, costs, width, label=mode, color=color, alpha=0.85)

    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Estimated Cost (milli-$)')
    ax.set_title('Cost Efficiency (lower is better)')
    ax.set_xticks(x + width * len(modes) / 2)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, '6_cost_efficiency.png'), bbox_inches='tight')
    plt.close()
    print("[CHART] 6_cost_efficiency.png")


def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    setup_plot_style()

    print("╔══════════════════════════════════════════╗")
    print("║   Generating Performance Charts           ║")
    print("╚══════════════════════════════════════════╝")
    print(f"[INPUT] {CSV_FILE}")
    print(f"[OUTPUT] {CHART_DIR}/")
    print("")

    data = load_csv(CSV_FILE)
    expected_sizes = {256, 512, 1024, 2048}
    sizes = sorted(set(
        s for mode_data in data.values()
        for s in mode_data.keys()
        if s in expected_sizes
    ))

    if not sizes:
        print("[ERROR] No benchmark data found in CSV")
        sys.exit(1)

    chart_exec_time(data, sizes)
    chart_speedup(data, sizes)
    chart_gflops(data, sizes)
    chart_mpi_strong_scaling(data, sizes)
    chart_mpi_weak_scaling(data, sizes)
    chart_cost_efficiency(data, sizes)

    print("")
    print(f"[DONE] 6 charts saved to {CHART_DIR}/")


if __name__ == '__main__':
    main()
