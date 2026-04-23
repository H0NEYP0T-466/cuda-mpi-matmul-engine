#!/bin/bash
# run_mpi_cloud.sh — Run MPI matmul across AWS EC2 instances
# Requires: cloud_hostfile with EC2 instance hostnames/IPs
HOSTFILE="cloud_hostfile"
BINARY="/home/ubuntu/matmul_mpi"
for np in 1 2 4; do
  for size in 512 1024 2048; do
    echo "=== np=$np, size=$size ==="
    mpirun --hostfile "$HOSTFILE" -np "$np" "$BINARY" "$size"
  done
done
