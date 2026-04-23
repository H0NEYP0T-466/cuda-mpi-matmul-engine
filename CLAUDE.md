# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Distributed Matrix Multiplication Framework** - A cloud-enabled framework comparing CPU sequential, CUDA (naïve + tiled), OpenMP (GPU + CPU), and MPI distributed matrix multiplication implementations.

## Build System

### Makefile Targets

- `make cpu` - CPU-only build (sequential + OpenMP CPU)
- `make cuda` - CUDA build (naïve + tiled + CPU) [requires nvcc]
- `make openmp-gpu` - OpenMP GPU offload build [requires nvc/gcc-offload]
- `make mpi` - MPI distributed build [requires mpicc]
- `make all` - Build all available targets
- `make clean` - Remove build artifacts

### Build Structure

```
src/
├── core/          # Matrix utils (seed=42), timer
├── cpu/           # Sequential baseline
├── cuda/          # Naïve + tiled shared-memory kernels
├── openmp/        # GPU offload + CPU parallel (separate, no fallback)
├── mpi/           # Row-decomposition distributed matmul
└── main.c         # Unified CLI (explicit mode selection)
```

## Architecture Key Points

1. **Deterministic matrices**: `srand(42)` everywhere — same size = same matrix across Local, Docker, Colab, and AWS
2. **Explicit modes**: No auto-detection or fallback — missing hardware = clear error
3. **Separate OpenMP targets**: GPU and CPU are different compile targets
4. **Row-wise MPI decomposition**: Minimizes communication (broadcast B, scatter A rows)
5. **Unified CLI**: All execution modes explicitly selected via `--mode` flag

## Common Development Tasks

### Building and Testing

```bash
# Build specific implementations
make cpu
make cuda
make mpi

# Run benchmarks
bash scripts/benchmark.sh

# Generate performance charts
python3 scripts/benchmark.py

# Test specific matrix sizes
./build/matmul_cpu --mode cpu --size medium
./build/matmul_cuda --mode cuda-tiled --size large

# Manual input for small matrices (≤8×8)
./build/matmul_cpu --mode cpu --size 4 --manual

# MPI execution
mpirun -np 4 ./build/matmul_mpi 512
```

### Docker MPI Cluster

```bash
cd docker
docker-compose up --build
```

### Size Presets

| Preset | Dimensions |
|--------|------------|
| small  | 256×256    |
| medium | 512×512    |
| large  | 1024×1024  |
| xlarge | 2048×2048  |

## Code Structure

### Core Components

- **matrix.h/c**: Matrix allocation, deterministic initialization, verification
- **timer.h/c**: Timing utilities and performance calculations
- **sequential.h/c**: CPU baseline implementation

### Implementation Variants

- **CUDA**: `naive.cu` and `tiled.cu` with wrapper headers
- **OpenMP**: `offload_gpu.c` and `offload_cpu.c` as separate targets
- **MPI**: `distributed.c` with separate entry point

### Main Entry Point

`main.c` provides unified CLI with explicit mode selection:
- Modes: cpu, cuda-naive, cuda-tiled, openmp-gpu, openmp-cpu
- No automatic hardware detection or silent fallback

## Key Design Decisions

1. **VERIFY_TOLERANCE**: 1e-3f for floating-point comparison across implementations
2. **MATRIX_SEED**: Fixed at 42 for reproducible matrix generation
3. **Row-major storage**: Flat float arrays for all matrices
4. **Separate compilation**: Each implementation compiles to separate binary
5. **CPU reference**: Always computed for verification of other implementations

## Development Workflow

1. **Add new implementation**: Create source file in appropriate directory, add forward declaration in main.c, update Makefile
2. **Performance testing**: Run benchmark.sh to collect timing data
3. **Correctness verification**: All implementations verified against CPU reference
4. **Edge case testing**: Test with sizes 1, 15, 17, 31, 33, 255, 257

## Deployment Environments

| Environment | Purpose |
|-------------|---------|
| Local Docker | MPI cluster simulation (4 static nodes) |
| Google Colab | CUDA GPU experiments (free T4) |
| AWS EC2 (c5.large) | Cloud scaling demonstration |

## File Organization Principles

- Small, focused files (typically < 400 lines)
- Clear separation of concerns by implementation type
- Header files for public interfaces
- Consistent naming conventions (e.g., `*_matmul` functions)

## Important Constants

- `MATRIX_SEED`: 42 (deterministic matrix generation)
- `VERIFY_TOLERANCE`: 1e-3f (floating-point comparison)
- Size presets defined in main.c parse_size() function

## Error Handling

- All implementations return -1.0 on failure
- Clear error messages for missing hardware/compilation flags
- Memory allocation failures handled with explicit messages
- Verification failures print first mismatched element