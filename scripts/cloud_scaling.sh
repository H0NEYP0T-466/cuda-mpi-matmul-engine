#!/bin/bash
# ==============================================================================
# cloud_scaling.sh — AWS EC2 Cloud Scaling Experiment
#
# Run MPI matmul across EC2 c5.large instances to demonstrate
# how performance changes with resource allocation.
#
# Prerequisites:
#   - EC2 instances already provisioned with setup_ec2.sh
#   - OpenMPI installed on all instances
#   - SSH keys configured for passwordless access
#   - matmul_mpi binary deployed to all instances
# ==============================================================================

set -e

RESULTS_DIR="results"
CSV_FILE="$RESULTS_DIR/scaling.csv"
HOSTFILE="cloud_hostfile"
BINARY="/home/ubuntu/matmul_mpi"

# Matrix sizes to test
SIZES=(512 1024 2048)

# Process counts to test
PROCS=(1 2 4)

# Cost per hour per c5.large instance
COST_PER_HOUR=0.085

mkdir -p "$RESULTS_DIR"

echo "mode,matrix_size,num_procs,exec_time_ms,gflops,speedup,efficiency,est_cost_usd" > "$CSV_FILE"

echo "╔══════════════════════════════════════════╗"
echo "║   AWS EC2 Cloud Scaling Experiment        ║"
echo "║   Instance type: c5.large ($COST_PER_HOUR/hr) ║"
echo "╚══════════════════════════════════════════╝"
echo ""

for size in "${SIZES[@]}"; do
    # Get baseline (1 process) time for speedup calculation
    baseline_time=0

    for np in "${PROCS[@]}"; do
        echo "[RUN] Size: ${size}×${size} | Processes: $np"

        # Run MPI across cloud instances
        output=$(mpirun --hostfile "$HOSTFILE" \
                        --allow-run-as-root \
                        -np "$np" \
                        "$BINARY" "$size" 2>&1) || true

        # Extract CSV line from output
        csv_line=$(echo "$output" | grep "^CSV," | tail -1)

        if [ -n "$csv_line" ]; then
            time_ms=$(echo "${csv_line#CSV,}" | cut -d',' -f3)

            # Set baseline for speedup calc
            if [ "$np" -eq 1 ]; then
                baseline_time=$time_ms
            fi

            # Calculate metrics
            speedup=$(echo "scale=2; $baseline_time / $time_ms" | bc -l 2>/dev/null || echo "1.0")
            efficiency=$(echo "scale=2; $speedup / $np * 100" | bc -l 2>/dev/null || echo "100.0")
            time_hours=$(echo "scale=8; $time_ms / 1000 / 3600" | bc -l 2>/dev/null || echo "0")
            cost=$(echo "scale=8; $time_hours * $COST_PER_HOUR * $np" | bc -l 2>/dev/null || echo "0")
            gflops=$(echo "${csv_line#CSV,}" | cut -d',' -f4)

            echo "  → Time: ${time_ms}ms | Speedup: ${speedup}x | Efficiency: ${efficiency}% | Cost: \$${cost}"
            echo "cloud-mpi,$size,$np,$time_ms,$gflops,$speedup,$efficiency,$cost" >> "$CSV_FILE"
        else
            echo "  → [FAILED]"
        fi
    done
    echo ""
done

echo "════════════════════════════════════════════"
echo "  Scaling results saved to: $CSV_FILE"
echo "════════════════════════════════════════════"
