# Distributed Matrix Multiplication Framework

A cloud-enabled framework comparing CPU sequential, CUDA (naïve + tiled), OpenMP (GPU + CPU), and MPI distributed matrix multiplication implementations.

## Architecture

```
src/
├── core/          # Matrix utils (seed=42), timer
├── cpu/           # Sequential baseline
├── cuda/          # Naïve + tiled shared-memory kernels
├── openmp/        # GPU offload + CPU parallel (separate, no fallback)
├── mpi/           # Row-decomposition distributed matmul
└── main.c         # Unified CLI (explicit mode selection)
```

## Deployment Strategy

| Environment | Purpose |
|---|---|
| Local Docker | MPI cluster simulation (4 static nodes) |
| Google Colab | CUDA GPU experiments (free T4) |
| AWS EC2 (c5.large) | Cloud scaling demonstration |

## Prerequisites

- **Local:** GCC, Make, Docker, Docker Compose
- **Colab:** Free account (GPU runtime)
- **AWS:** AWS CLI, c5.large instances

## Build

```bash
# CPU only (sequential + OpenMP CPU)
make cpu

# MPI distributed
make mpi

# CUDA (on Colab or GPU machine)
make cuda

# OpenMP GPU offload
make openmp-gpu

# All available targets
make all
```

## Usage

```bash
# Explicit mode selection — no auto-detection
./build/matmul_cpu --mode cpu --size medium
./build/matmul_cpu --mode openmp-cpu --size 1024
./build/matmul_cuda --mode cuda-naive --size large
./build/matmul_cuda --mode cuda-tiled --size xlarge

# Manual input for correctness testing
./build/matmul_cpu --mode cpu --size 4 --manual

# MPI (separate binary)
mpirun -np 4 ./build/matmul_mpi 512
```

### Size Presets
| Preset | Dimensions |
|---|---|
| small | 256×256 |
| medium | 512×512 |
| large | 1024×1024 |
| xlarge | 2048×2048 |

## Docker MPI Cluster

```bash
cd docker
docker-compose up --build
```

Runs MPI across 4 static containers with pre-configured hostfile.

## Benchmarking

```bash
# Run all benchmarks
bash scripts/benchmark.sh

# Generate charts
python3 scripts/benchmark.py
```

## Design Decisions

1. **Deterministic matrices:** `srand(42)` everywhere — same size = same matrix
2. **Explicit modes:** No auto-detection or fallback — missing hardware = clear error
3. **Static Docker cluster:** Fixed compose + hostfile, no dynamic SSH setup
4. **Separate OpenMP targets:** GPU and CPU are different compile targets
5. **Row-wise MPI decomposition:** Minimizes communication (broadcast B, scatter A rows)
