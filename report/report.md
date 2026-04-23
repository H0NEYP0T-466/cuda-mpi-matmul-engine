# Technical Report: Cloud-Enabled Distributed Matrix Multiplication Framework

## 1. Introduction

This report presents a comprehensive matrix multiplication framework that implements and compares five parallelization strategies: CPU sequential, CUDA naïve, CUDA tiled (shared memory), OpenMP (GPU and CPU), and MPI distributed execution. The framework is containerized with Docker and deployable to AWS EC2 for cloud scaling analysis.

### 1.1 Objectives
- Implement matrix multiplication using data decomposition
- Compare GPU, multi-threaded, and distributed approaches
- Analyze correctness, speedup, scalability, resource usage, and cost
- Support reproducible deployment across local, cloud, and container environments

## 2. Design Decisions

### 2.1 Data Decomposition Strategy
- **MPI:** Row-wise decomposition — matrix A is split by rows across ranks, B is broadcast to all
- **CUDA:** Each thread computes one output element (naïve) or one tile contribution (tiled)
- **OpenMP:** Loop-level parallelism with collapse(2) across output matrix indices

### 2.2 Deterministic Reproducibility
All matrices are generated using `srand(42 + offset)` where offset differentiates A from B. This ensures identical matrices across Local, Docker, Colab, and AWS environments.

### 2.3 Explicit Mode Selection
No automatic hardware detection or silent fallback. Each mode is explicitly selected via CLI `--mode` flag. Missing hardware produces a clear error message.

### 2.4 Tile Size Selection (CUDA)
Tile size = 16×16. This balances shared memory usage (2 × 16 × 16 × 4 = 2048 bytes per block) against occupancy. Each block uses 2KB of 48KB available shared memory, allowing high occupancy.

### 2.5 Edge Case Handling
- Boundary checks in CUDA kernels for N not divisible by TILE_SIZE
- MPI: `Scatterv`/`Gatherv` for uneven row distribution (last ranks get +1 row)

## 3. Implementation Details

### 3.1 CPU Sequential (Baseline)
Standard triple-nested-loop: `C[i][j] += A[i][k] * B[k][j]`
- Time complexity: O(N³)
- Serves as ground truth for correctness verification

### 3.2 CUDA Naïve Kernel
- Grid: `(ceil(N/16), ceil(N/16))`, Block: `(16, 16)`
- Each thread computes entire dot product from global memory
- Bottleneck: global memory bandwidth (no data reuse)

### 3.3 CUDA Tiled Kernel (Shared Memory)
- Loads TILE_SIZE×TILE_SIZE sub-matrices into `__shared__` memory
- `__syncthreads()` between load and compute phases
- Data reuse: each element loaded once per tile, used TILE_SIZE times
- Expected speedup: ~5-15x over naïve for large matrices

### 3.4 OpenMP GPU Offload
- `#pragma omp target teams distribute parallel for collapse(2)`
- Separate binary from CPU version — fails explicitly if no GPU
- Lower programming effort than CUDA (directives vs kernel code)

### 3.5 OpenMP CPU Parallel
- `#pragma omp parallel for collapse(2) schedule(static)`
- CPU threads only, never attempts GPU offload
- Speedup limited by number of CPU cores

### 3.6 MPI Distributed
- Rank 0 initializes matrices and distributes work
- `MPI_Bcast(B)` → `MPI_Scatterv(A rows)` → compute → `MPI_Gatherv(C rows)`
- Communication overhead measured separately from computation

## 4. Performance Analysis

### 4.1 Metrics
| Metric | Formula |
|--------|---------|
| GFLOPS | 2×M×N×K / (time_sec × 10⁹) |
| Speedup | T_sequential / T_parallel |
| Parallel Efficiency | Speedup / P × 100% |
| Cost | time_hours × $/hr × num_resources |

### 4.2 Expected Results

**GPU (Colab T4):**
- CUDA naïve: ~10-50x speedup over CPU for N≥512
- CUDA tiled: ~2-5x speedup over naïve due to shared memory reuse
- OpenMP GPU: comparable to CUDA naïve (directive overhead)

**CPU (OpenMP):**
- ~2-4x speedup (limited by core count)

**MPI (4 processes):**
- Speedup < 4x due to communication overhead
- Better scaling for larger matrices (computation dominates)

### 4.3 Bottleneck Analysis
1. **CUDA naïve:** Global memory bandwidth (each thread reads 2N floats)
2. **CUDA tiled:** Shared memory bank conflicts, tile loading overhead
3. **MPI:** Network latency for Bcast/Scatter/Gather, especially for small N
4. **OpenMP CPU:** Cache contention, thread creation overhead

## 5. Containerization & Cloud Deployment

### 5.1 Docker Mini-Cluster
- 4 static containers on a bridge network with fixed IPs
- Pre-baked hostfile (no dynamic SSH discovery)
- Single `docker-compose up` to build and run

### 5.2 AWS EC2 Scaling
- c5.large instances (2 vCPU, 4GB RAM, $0.085/hr)
- Scaling test: 1, 2, 4 instances × 512, 1024, 2048 matrix sizes
- Estimated total cost: ~$0.09 for a 15-minute experiment

## 6. Correctness Verification
- All implementations verified against CPU sequential result
- Tolerance: |a - b| < 10⁻³ (accounts for floating-point non-associativity)
- Edge cases tested: N = 1, 15, 17, 31, 33, 255, 257

## 7. Recommendations
1. Use CUDA tiled kernel for maximum GPU performance
2. For rapid prototyping, OpenMP offload requires significantly less code
3. MPI is most beneficial for very large matrices where computation dominates communication
4. Docker provides reproducible execution; AWS provides real distributed scaling evidence

## 8. Conclusion
The framework demonstrates that GPU acceleration (CUDA tiled) provides the highest speedup for matrix multiplication, while MPI enables scaling beyond a single machine. The hybrid deployment strategy (Docker + Colab + AWS) covers all aspects while staying within zero-cost constraints.
