#!/bin/bash
# ==============================================================================
# benchmark.sh — Run all available implementations and collect timing data
#
# Outputs results to results/timings.csv
# Each line: mode,matrix_size,exec_time_ms,gflops,verified
# ==============================================================================

set -e

RESULTS_DIR="results"
CSV_FILE="$RESULTS_DIR/timings.csv"
BUILD_DIR="build"

# Matrix sizes to benchmark
SIZES=(256 512 1024 2048)

# Edge-case sizes for correctness testing
EDGE_SIZES=(1 15 17 31 33 255 257)

mkdir -p "$RESULTS_DIR"

echo "mode,matrix_size,exec_time_ms,gflops,verified" > "$CSV_FILE"

echo "╔══════════════════════════════════════════╗"
echo "║   Matrix Multiplication Benchmark Suite   ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ------------------------------------------------------------------
# Helper: run a single benchmark and extract CSV line
# ------------------------------------------------------------------
run_bench() {
    local binary="$1"
    local mode="$2"
    local size="$3"
    local extra_args="$4"

    echo "[RUN] $mode | Size: ${size}×${size}"

    # Run and capture output
    output=$($binary --mode "$mode" --size "$size" $extra_args 2>&1) || true

    # Extract CSV line (our programs output "CSV,..." lines)
    csv_line=$(echo "$output" | grep "^CSV," | tail -1)

    if [ -n "$csv_line" ]; then
        # Remove "CSV," prefix and append to file
        echo "${csv_line#CSV,}" >> "$CSV_FILE"
        echo "  → $(echo "${csv_line#CSV,}" | cut -d',' -f3-5)"
    else
        echo "  → [SKIPPED or FAILED]"
    fi
}

# ------------------------------------------------------------------
# CPU Sequential + OpenMP CPU
# ------------------------------------------------------------------
if [ -f "$BUILD_DIR/matmul_cpu" ]; then
    echo ""
    echo "=== CPU Sequential ==="
    for size in "${SIZES[@]}"; do
        run_bench "$BUILD_DIR/matmul_cpu" "cpu" "$size"
    done

    echo ""
    echo "=== OpenMP CPU Parallel ==="
    for size in "${SIZES[@]}"; do
        run_bench "$BUILD_DIR/matmul_cpu" "openmp-cpu" "$size"
    done

    echo ""
    echo "=== Edge Case Correctness (CPU) ==="
    for size in "${EDGE_SIZES[@]}"; do
        run_bench "$BUILD_DIR/matmul_cpu" "cpu" "$size"
    done
else
    echo "[SKIP] CPU build not found. Run: make cpu"
fi

# ------------------------------------------------------------------
# CUDA Naïve + Tiled
# ------------------------------------------------------------------
if [ -f "$BUILD_DIR/matmul_cuda" ]; then
    echo ""
    echo "=== CUDA Naïve ==="
    for size in "${SIZES[@]}"; do
        run_bench "$BUILD_DIR/matmul_cuda" "cuda-naive" "$size"
    done

    echo ""
    echo "=== CUDA Tiled (Shared Memory) ==="
    for size in "${SIZES[@]}"; do
        run_bench "$BUILD_DIR/matmul_cuda" "cuda-tiled" "$size"
    done

    echo ""
    echo "=== Edge Case Correctness (CUDA) ==="
    for size in "${EDGE_SIZES[@]}"; do
        run_bench "$BUILD_DIR/matmul_cuda" "cuda-naive" "$size"
        run_bench "$BUILD_DIR/matmul_cuda" "cuda-tiled" "$size"
    done
else
    echo "[SKIP] CUDA build not found. Run on Colab or: make cuda"
fi

# ------------------------------------------------------------------
# MPI (run locally with mpirun)
# ------------------------------------------------------------------
if [ -f "$BUILD_DIR/matmul_mpi" ] && command -v mpirun &>/dev/null; then
    echo ""
    echo "=== MPI Distributed ==="
    for np in 1 2 4; do
        for size in "${SIZES[@]}"; do
            echo "[RUN] MPI (np=$np) | Size: ${size}×${size}"
            output=$(mpirun --allow-run-as-root -np "$np" "$BUILD_DIR/matmul_mpi" "$size" 2>&1) || true
            csv_line=$(echo "$output" | grep "^CSV," | tail -1)
            if [ -n "$csv_line" ]; then
                echo "${csv_line#CSV,}" >> "$CSV_FILE"
                echo "  → $(echo "${csv_line#CSV,}" | cut -d',' -f3-5)"
            fi
        done
    done
else
    echo "[SKIP] MPI build or mpirun not found. Run: make mpi"
fi

echo ""
echo "════════════════════════════════════════════"
echo "  Results saved to: $CSV_FILE"
echo "  Run: python3 scripts/benchmark.py to generate charts"
echo "════════════════════════════════════════════"
