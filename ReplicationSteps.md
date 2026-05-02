# 📖 Complete Project Understanding Guide

# Table of Contents
1. [Project Overview](#1-project-overview)
2. [File-by-File Breakdown](#2-file-by-file-breakdown)
3. [Local Setup (Windows)](#3-local-setup-windows)
4. [Running on Local CPU](#4-running-on-local-cpu)
5. [Docker Setup & MPI Cluster](#5-docker-setup--mpi-cluster)
6. [Google Colab Setup (GPU)](#6-google-colab-setup-gpu)
7. [AWS EC2 Cloud Scaling](#7-aws-ec2-cloud-scaling)
8. [Generating Charts & Visuals](#8-generating-charts--visuals)
9. [Understanding the Output](#9-understanding-the-output)
10. [Troubleshooting](#10-troubleshooting)

---

# 1. Project Overview

## What This Project Does
This is a matrix multiplication framework that multiplies two matrices (A × B = C) using **5 different methods** and compares their performance:

| # | Method | Where It Runs | What It Tests |
|---|--------|--------------|---------------|
| 1 | CPU Sequential | Your PC / Docker / Colab | Baseline — single thread, no parallelism |
| 2 | CUDA Naïve | Google Colab (T4 GPU) | Basic GPU parallelism — 1 thread per element |
| 3 | CUDA Tiled | Google Colab (T4 GPU) | Optimized GPU — shared memory tiles |
| 4 | OpenMP CPU | Your PC / Docker | Multi-threaded CPU parallelism |
| 5 | MPI Distributed | Docker cluster / AWS EC2 | Multi-machine distributed computing |

## How It Works (Simple Explanation)

**Matrix multiplication** C = A × B means:
```
C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j] + ... + A[i][N-1]*B[N-1][j]
```
For an N×N matrix, this requires 2×N³ floating-point operations. For N=1024, that's ~2 billion operations.

**Why parallelize?**
- CPU does operations one-by-one (slow)
- GPU has thousands of cores (fast for parallel work)
- MPI splits the work across multiple machines

## Folder Structure Explained

```
CUDA-MPI MatMul Engine/
│
├── src/                          ← ALL SOURCE CODE
│   ├── core/                     ← Shared utilities used by everything
│   │   ├── matrix.h              ← Header: function declarations for matrix ops
│   │   ├── matrix.c              ← Code: allocate, initialize, verify, print matrices
│   │   ├── timer.h               ← Header: timing function declarations
│   │   └── timer.c               ← Code: measure execution time in milliseconds
│   │
│   ├── cpu/                      ← CPU implementation
│   │   ├── sequential.h          ← Header: CPU matmul function declaration
│   │   └── sequential.c          ← Code: triple-nested-loop multiplication
│   │
│   ├── cuda/                     ← GPU implementations (run on Colab)
│   │   ├── cuda_wrapper.h        ← Header: CUDA function declarations
│   │   ├── naive.cu              ← Code: basic GPU kernel (1 thread = 1 element)
│   │   └── tiled.cu              ← Code: optimized GPU kernel (shared memory)
│   │
│   ├── openmp/                   ← OpenMP implementations
│   │   ├── offload_gpu.c         ← Code: OpenMP GPU version (FAILS if no GPU)
│   │   └── offload_cpu.c         ← Code: OpenMP CPU threads version
│   │
│   ├── mpi/                      ← MPI distributed implementation
│   │   └── distributed.c         ← Code: split matrix across multiple machines
│   │
│   └── main.c                    ← MAIN PROGRAM — the CLI you run
│
├── docker/                       ← Docker files for MPI cluster
│   ├── Dockerfile                ← Recipe to build the container image
│   ├── docker-compose.yml        ← Defines 4 containers as MPI nodes
│   └── hostfile                  ← List of MPI nodes (node0, node1, node2, node3)
│
├── colab/                        ← Google Colab notebook
│   └── matmul_gpu.ipynb          ← Self-contained notebook for GPU experiments
│
├── scripts/                      ← Automation scripts
│   ├── benchmark.sh              ← Runs ALL implementations and collects timing data
│   ├── benchmark.py              ← Reads timing data and generates 6 charts
│   └── cloud_scaling.sh          ← Runs MPI scaling test on AWS
│
├── aws/                          ← AWS cloud experiment files
│   ├── setup_ec2.sh              ← Guide to set up EC2 instances
│   └── run_mpi_cloud.sh          ← Runs MPI across cloud instances
│
├── results/                      ← OUTPUT — generated when you run benchmarks
│   ├── timings.csv               ← Raw timing data (auto-generated)
│   └── charts/                   ← PNG chart images (auto-generated)
│
├── report/                       ← Final report
│   └── report.md                 ← Technical report with analysis
│
├── Makefile                      ← Build commands (like a recipe book)
└── README.md                     ← Quick-start guide
```

---

# 2. File-by-File Breakdown

## src/core/matrix.h & matrix.c — Matrix Utilities

**What it does:** Everything related to creating and managing matrices.

**Key functions:**
```c
// Allocate memory for a matrix (rows × cols)
float* matrix_alloc(int rows, int cols);

// Fill matrix with deterministic random numbers (seed=42)
// seed_offset=0 for matrix A, seed_offset=1 for matrix B
// This ensures A and B are DIFFERENT but REPRODUCIBLE
matrix_init_deterministic(A, 1024, 1024, 0);  // Always same A
matrix_init_deterministic(B, 1024, 1024, 1);  // Always same B

// Compare two matrices element-by-element (tolerance: 0.001)
// Returns 1 if PASS, 0 if FAIL
int ok = matrix_verify(expected, actual, rows, cols);

// Print small matrices to screen
matrix_print(A, 4, 4, "Matrix A");
```

**Why seed=42?** So that running the same size on your PC, Docker, Colab, or AWS all produce the EXACT same matrix. This proves correctness across platforms.

---

## src/core/timer.h & timer.c — Timing

**What it does:** Measures how long code takes to run.

```c
double start = timer_get_ms();   // Get current time in milliseconds
// ... do work ...
double end = timer_get_ms();
double elapsed = end - start;    // Time in ms

// Calculate GFLOPS (billions of operations per second)
double gflops = timer_calc_gflops(N, N, N, elapsed);

// Print formatted results box
timer_print_result("CPU Sequential", 1024, elapsed, gflops, 1, 1.0);
```

**Cross-platform:** Uses `clock_gettime` on Linux/Docker/Colab, `QueryPerformanceCounter` on Windows.

---

## src/cpu/sequential.h & sequential.c — CPU Baseline

**What it does:** The simplest matrix multiplication — one thread, three nested loops.

```c
for i = 0 to M:
    for j = 0 to N:
        for k = 0 to K:
            C[i][j] += A[i][k] * B[k][j]
```

**Why it matters:** This is the BASELINE. Everything else is compared against this. If CPU takes 5000ms and CUDA takes 50ms, the speedup is 100x.

---

## src/cuda/naive.cu — CUDA Naïve Kernel

**What it does:** Each GPU thread computes ONE element of the output matrix C.

```
GPU has thousands of threads running simultaneously:
Thread (0,0) → computes C[0][0]
Thread (0,1) → computes C[0][1]
Thread (1,0) → computes C[1][0]
... all at the same time!
```

**Grid/Block configuration:**
```
Block size: 16×16 = 256 threads per block
Grid size: ceil(N/16) × ceil(N/16) blocks
For N=1024: 64×64 = 4096 blocks × 256 threads = 1,048,576 threads!
```

**Boundary check:** If N is not a multiple of 16 (e.g., N=33), extra threads just do nothing:
```c
if (row < N && col < N) { /* compute */ }
```

**Bottleneck:** Every thread reads from slow global memory (no data reuse).

---

## src/cuda/tiled.cu — CUDA Tiled Kernel (Shared Memory)

**What it does:** Same result as naïve, but MUCH faster because it uses shared memory.

**How tiling works:**
```
Instead of each thread reading ALL of row A and column B from slow global memory:

1. Load a 16×16 "tile" of A and B into fast shared memory
2. __syncthreads() — wait for all threads to finish loading
3. Compute partial results using the fast shared memory
4. __syncthreads() — wait before loading next tile
5. Repeat for all tiles
6. Write final result to global memory
```

**Why it's faster:** Shared memory is ~100x faster than global memory. Each element loaded from global memory is reused 16 times from shared memory.

**Edge cases:** When N is not divisible by 16, out-of-bounds loads return 0.0f:
```c
tile_A[ty][tx] = (row < N && col < K) ? A[row*K + col] : 0.0f;
```

---

## src/openmp/offload_gpu.c — OpenMP GPU (Separate, No Fallback)

**What it does:** Uses compiler directives to run on GPU — much less code than CUDA.

```c
#pragma omp target teams distribute parallel for collapse(2)
for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) { /* compute C[i][j] */ }
```

**CRITICAL:** If no GPU is found, it prints an error and EXITS. It does NOT silently fall back to CPU:
```c
if (omp_get_num_devices() == 0) {
    fprintf(stderr, "[ERROR] No GPU found. Exiting.\n");
    return -1.0;  // FAIL, don't pretend it worked
}
```

---

## src/openmp/offload_cpu.c — OpenMP CPU (Threads Only)

**What it does:** Uses CPU threads to parallelize the loops. Never touches GPU.

```c
#pragma omp parallel for collapse(2) schedule(static)
for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) { /* compute */ }
```

On a 4-core CPU, this runs ~4x faster than sequential.

---

## src/mpi/distributed.c — MPI Distributed

**What it does:** Splits the work across multiple machines/containers.

**Data decomposition (row-wise):**
```
Matrix A (1024×1024) split across 4 MPI processes:
  Process 0: rows 0-255    of A  →  computes rows 0-255    of C
  Process 1: rows 256-511  of A  →  computes rows 256-511  of C
  Process 2: rows 512-767  of A  →  computes rows 512-767  of C
  Process 3: rows 768-1023 of A  →  computes rows 768-1023 of C

Matrix B: broadcast to ALL processes (everyone needs full B)
```

**MPI Communication flow:**
```
Step 1: Rank 0 creates A and B (seed=42)
Step 2: MPI_Bcast(B)       → Send full B to everyone
Step 3: MPI_Scatterv(A)    → Send each rank its rows of A
Step 4: Each rank computes  → local_C = local_A × B
Step 5: MPI_Gatherv(C#)     → Collect all rows of C back to rank 0
Step 6: Rank 0 verifies    → Compare against CPU result
```

**Handles uneven splits:** If N=10 and P=3: ranks get 4, 3, 3 rows (not crash).

---

## src/main.c — The Main Program

**What it does:** The unified entry point. You tell it WHAT to run via flags.

**Compile-time guards:** Features are enabled/disabled at compile time:
```c
#ifdef ENABLE_CUDA         // Only when compiled with: make cuda
#ifdef ENABLE_OPENMP_GPU   // Only when compiled with: make openmp-gpu
#ifdef ENABLE_OPENMP_CPU   // Only when compiled with: make cpu
```

If you try `--mode cuda-naive` but compiled without CUDA:
```
[ERROR] CUDA support not compiled. Rebuild with ENABLE_CUDA=1.
```

---

## Makefile — Build System

**What it does:** Compiles the source code into runnable programs.

| Command | What It Builds | Output Binary |
|---------|---------------|---------------|
| `make cpu` | CPU sequential + OpenMP CPU | `build/matmul_cpu` |
| `make cuda` | CUDA naïve + tiled + CPU | `build/matmul_cuda` |
| `make openmp-gpu` | OpenMP GPU offload | `build/matmul_openmp_gpu` |
| `make mpi` | MPI distributed | `build/matmul_mpi` |
| `make all` | Everything available | All of above |
| `make clean` | Delete all builds | (removes build/) |

---

# 3. Local Setup (Windows)

## Step 1: Install WSL2 (Windows Subsystem for Linux)

You need Linux tools (gcc, make, mpi) which don't run natively on Windows.

```powershell
# Open PowerShell as Administrator
wsl --install

# Restart your computer when prompted
# After restart, Ubuntu will open and ask for username/password
# Choose any username and password you'll remember
```

## Step 2: Install Build Tools in WSL

```bash
# Open Ubuntu (WSL) terminal
sudo apt update
sudo apt install -y build-essential gcc g++ make

# Verify
gcc --version
# Should show: gcc (Ubuntu ...) 11.x or higher
```

## Step 3: Install OpenMPI (for MPI mode)

```bash
sudo apt install -y openmpi-bin libopenmpi-dev

# Verify
mpirun --version
# Should show: mpirun (Open MPI) 4.x
```

## Step 4: Install Python + matplotlib (for charts)

```bash
sudo apt install -y python3 python3-pip
pip3 install matplotlib numpy
||
sudo apt install python3-matplotlib python3-numpy

# Verify
python3 -c "import matplotlib; print('OK')"
```

## Step 5: Navigate to Project

```bash
# Your Windows X: drive is accessible in WSL at /mnt/x
cd "/mnt/host/x/file/Projects/CUDA-MPI MatMul Engine"

# Verify you can see the files
ls src/
# Should show: core  cpu  cuda  main.c  mpi  openmp
```

---

# 4. Running on Local CPU

## Build the CPU version

```bash
cd "/mnt/host/x/file/Projects/CUDA-MPI MatMul Engine"
make cpu
```

Expected output:
```
[BUILD] build/matmul_cpu — CPU sequential + OpenMP CPU
```

## Run CPU Sequential

```bash
# Small matrix (256×256)
./build/matmul_cpu --mode cpu --size small

# Medium matrix (512×512)
./build/matmul_cpu --mode cpu --size medium

# Large matrix (1024×1024)
./build/matmul_cpu --mode cpu --size large

# Custom size
./build/matmul_cpu --mode cpu --size 300
```

Expected output:
```
[MODE] CPU Sequential
[MATRIX] 512×512 (seed=42)
[INFO] Computing CPU reference for verification...
[CPU REF] 312.456 ms
╔══════════════════════════════════════════╗
║  CPU Sequential                          ║
╠══════════════════════════════════════════╣
║  Matrix Size : 512 × 512                ║
║  Time        :    312.456 ms            ║
║  GFLOPS      :     0.8590               ║
║  Speedup     :       1.00x              ║
║  Verified    : PASS ✓                   ║
╚══════════════════════════════════════════╝
```

## Run OpenMP CPU (Multi-threaded)

```bash
# Set number of threads (use your CPU core count)
export OMP_NUM_THREADS=4

./build/matmul_cpu --mode openmp-cpu --size medium
```

## Run Manual Input Mode (Correctness Check)

```bash
./build/matmul_cpu --mode cpu --size 3 --manual
```

It will prompt you:
```
Enter elements for Matrix A (3×3), row by row:
  Row 0 (3 values): 1 2 3
  Row 1 (3 values): 4 5 6
  Row 2 (3 values): 7 8 9
Enter elements for Matrix B (3×3), row by row:
  Row 0 (3 values): 9 8 7
  Row 1 (3 values): 6 5 4
  Row 2 (3 values): 3 2 1
```

## Run MPI Locally (without Docker)

```bash
# Build MPI version
make mpi

# Run with 1 process
mpirun -np 1 ./build/matmul_mpi 512

# Run with 2 processes
mpirun -np 2 ./build/matmul_mpi 512

# Run with 4 processes
mpirun --allow-run-as-root -np 4 ./build/matmul_mpi 512
||
mpirun --allow-run-as-root --oversubscribe -np 4 ./build/matmul_mpi 512

```

---

# 5. Docker Setup & MPI Cluster

## Step 1: Install Docker Desktop on Windows

1. Go to: https://www.docker.com/products/docker-desktop/
2. Download "Docker Desktop for Windows"
3. Run the installer
4. **IMPORTANT:** During install, check "Use WSL 2 instead of Hyper-V"
5. Restart your computer
6. Open Docker Desktop — wait for it to say "Docker is running"

## Step 2: Verify Docker Installation

```powershell
# In PowerShell or WSL terminal
docker --version
# Should show: Docker version 24.x or higher

docker-compose --version
# Should show: Docker Compose version v2.x
```

## Step 3: Build and Run the MPI Cluster

```bash
# Navigate to project in WSL
cd "/mnt/x/file/Projects/CUDA-MPI MatMul Engine"

# Build and start all 4 containers
cd docker
docker-compose up --build
(want to rereun use this instead)
docker-compose up
```

**What happens behind the scenes:**
```
1. Docker reads the Dockerfile
2. Creates an Ubuntu 22.04 container image with:
   - GCC compiler
   - OpenMPI
   - SSH server
   - Your compiled source code
3. docker-compose creates 4 copies of this container:
   - mpi_node0 (172.28.0.10) — MASTER: runs mpirun
   - mpi_node1 (172.28.0.11) — WORKER
   - mpi_node2 (172.28.0.12) — WORKER
   - mpi_node3 (172.28.0.13) — WORKER
4. node0 waits 3 seconds for workers to start SSH
5. node0 runs: mpirun --hostfile hostfile -np 4 ./matmul_mpi 512
6. The 512×512 matrix is split across all 4 nodes
7. Results are collected and verified on node0
```

Expected output:
```
mpi_node0  | === Running MPI across 4 static nodes ===
mpi_node0  | [MPI] Matrix size: 512×512
mpi_node0  | [MPI] Number of processes: 4
mpi_node0  | ╔══════════════════════════════════════════╗
mpi_node0  | ║  MPI Distributed (4 processes)           ║
mpi_node0  | ║  Total Time    :     89.234 ms           ║
mpi_node0  | ║  Compute Time  :     78.123 ms           ║
mpi_node0  | ║  Comm Time     :     11.111 ms           ║
mpi_node0  | ║  Speedup vs CPU:       3.50x             ║
mpi_node0  | ║  Efficiency    :      87.50%             ║
mpi_node0  | ║  Verified      : PASS ✓                  ║
mpi_node0  | ╚══════════════════════════════════════════╝
mpi_node0  | === MPI run complete ===
```

## Step 4: Run with Different Matrix Sizes

Edit `docker/docker-compose.yml`, change `512` to your desired size:

```yaml
# Find this line in the node0 command:
/app/build/matmul_mpi 512
# Change to:
/app/build/matmul_mpi 1024
```

Then rebuild and run:
```bash
docker-compose down
docker-compose up --build
```

## Step 5: Stop and Clean Up

```bash
# Stop containers
docker-compose down

# Remove all images (free disk space)
docker system prune -a
```

---

# 6. Google Colab Setup (GPU)

## Step 1: Open Colab

1. Go to: https://colab.research.google.com/
2. Sign in with your Google account

## Step 2: Upload the Notebook

1. Click **File → Upload notebook**
2. Navigate to: `CUDA-MPI MatMul Engine/colab/matmul_gpu.ipynb`
3. Upload it

## Step 3: Enable GPU Runtime

1. Click **Runtime → Change runtime type**
2. Select **T4 GPU** from the dropdown
3. Click **Save**

## Step 4: Run All Cells

Click **Runtime → Run all** (or Ctrl+F9)

**What each section does:**

| Section | What It Does |
|---------|-------------|
| Cell 1 | Checks GPU is available (`nvidia-smi`) |
| Cell 2-3 | Writes `matrix.h` and `matrix.c` to Colab filesystem |
| Cell 4-5 | Writes and compiles CPU sequential baseline |
| Cell 6-7 | Writes and compiles CUDA naïve kernel, runs benchmarks |
| Cell 8-9 | Writes and compiles CUDA tiled kernel, runs benchmarks |
| Cell 10 | Tests edge cases (N=1, 15, 17, 33, 255, 257) |
| Cell 11 | Runs all implementations for sizes 256, 512, 1024, 2048 |
| Cell 12 | Generates 3 charts: Exec Time, Speedup, GFLOPS |
| Cell 13 | Downloads `gpu_timings.csv` and `gpu_benchmark_charts.png` |

## Step 5: Download Results

After Cell 13 runs, your browser will download:
- `gpu_timings.csv` — raw timing data
- `gpu_benchmark_charts.png` — performance comparison charts

Copy these to `results/` folder in your project.

---

# 7. AWS EC2 Cloud Scaling

## Step 1: Create AWS Account

1. Go to: https://aws.amazon.com/
2. Create a free account (needs credit card but won't charge for free tier)

## Step 2: Launch EC2 Instances

```bash
# Install AWS CLI (in WSL)
sudo apt install -y awscli

# Configure with your credentials
aws configure
# Enter: Access Key ID, Secret Key, Region (us-east-1), Output (json)

# Launch 4 c5.large instances
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type c5.large \
  --count 4 \
  --key-name matmul-keypair \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=matmul-node}]'
```

## Step 3: Install MPI on Each Instance

```bash
# SSH into each instance
ssh -i matmul-keypair.pem ubuntu@<INSTANCE_PUBLIC_IP>

# Install dependencies
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev
```

## Step 4: Deploy and Run

```bash
# Copy binary to each instance
scp -i matmul-keypair.pem build/matmul_mpi ubuntu@<IP>:~/

# Create hostfile with instance private IPs
echo "10.0.1.10 slots=1" > cloud_hostfile
echo "10.0.1.11 slots=1" >> cloud_hostfile
echo "10.0.1.12 slots=1" >> cloud_hostfile
echo "10.0.1.13 slots=1" >> cloud_hostfile

# Run scaling experiment
bash scripts/cloud_scaling.sh
```

## Step 5: TERMINATE Instances (Avoid Charges!)

```bash
aws ec2 terminate-instances --instance-ids i-xxx i-yyy i-zzz i-www
```

**Cost estimate:** c5.large = $0.085/hr × 4 instances × 0.25hr = **~$0.09 total**

---

# 8. Generating Charts & Visuals

## Method 1: Run Full Benchmark Suite (Linux/WSL)

```bash
cd "/mnt/x/file/Projects/CUDA-MPI MatMul Engine"

# Build everything available locally
make cpu
make mpi

# Run all benchmarks (creates results/timings.csv)
bash scripts/benchmark.sh

# Generate 6 charts (creates results/charts/*.png)
python3 scripts/benchmark.py
```

**Charts generated:**

| File | What It Shows |
|------|--------------|
| `1_exec_time.png` | Bar chart — execution time vs matrix size for all implementations |
| `2_speedup.png` | Line chart — how much faster each method is vs CPU |
| `3_gflops.png` | Bar chart — computational throughput (higher = better) |
| `4_mpi_strong_scaling.png` | Line chart — MPI speedup vs number of processes |
| `5_mpi_weak_scaling.png` | Line chart — MPI time vs problem size |
| `6_cost_efficiency.png` | Bar chart — estimated cloud cost per run |

## Method 2: Combine Colab + Local Results

1. Run Colab notebook → download `gpu_timings.csv`
2. Run local benchmarks → generates `results/timings.csv`
3. Merge both CSVs:

```bash
# Append Colab GPU results to local results
tail -n +2 gpu_timings.csv >> results/timings.csv

# Regenerate charts with combined data
python3 scripts/benchmark.py
```

---

# 9. Understanding the Output

## CSV Format

Every run outputs a CSV line: `mode,matrix_size,exec_time_ms,gflops,verified`

```
CPU Sequential,256,45.320,0.7400,PASS
CPU Sequential,512,312.456,0.8590,PASS
CUDA_Naive,512,5.230,51.2800,PASS
CUDA_Tiled,512,1.890,141.8400,PASS
mpi-4,512,89.234,3.0020,PASS
```

## Key Metrics Explained

| Metric         | What It Means                                          | Formula                        |
| -------------- | ------------------------------------------------------ | ------------------------------ |
| **Time (ms)**  | How long the computation took                          | End - Start                    |
| **GFLOPS**     | Billions of operations per second (higher = better)    | 2×N³ / (time_sec × 10⁹)        |
| **Speedup**    | How many times faster vs CPU                           | CPU_time / This_time           |
| **Efficiency** | How well resources are utilized (MPI only)             | Speedup / num_processes × 100% |
| **Verified**   | PASS if result matches CPU reference (tolerance 0.001) | element-wise comparison        |

## What "PASS ✓" Means

Every implementation computes the same multiplication. The result is compared element-by-element against the CPU sequential result. If all elements match within 0.001, it's PASS.

Why tolerance 0.001 and not exact? Because floating-point math is not associative:
```
(a + b) + c ≠ a + (b + c)  in floating-point
```
GPU and MPI add numbers in different order than CPU, causing tiny differences.

---

# 10. Troubleshooting

## "make: command not found"
```bash
sudo apt install build-essential
```

## "mpirun: command not found"
```bash
sudo apt install openmpi-bin libopenmpi-dev
```

## "Docker: command not found"
Install Docker Desktop from https://docker.com and ensure WSL2 integration is enabled.

## "No module named matplotlib"
```bash
pip3 install matplotlib numpy
```

## CUDA errors on local machine
CUDA only runs on Colab. Don't try `make cuda` locally — use the Colab notebook.

## Docker "permission denied"
```bash
sudo usermod -aG docker $USER
# Then log out and log back in
```

## MPI "all ports busy" or connection refused
```bash
# In Docker, ensure all containers are running:
docker-compose ps
# All 4 nodes should show "Up"
```

## WSL can't see X: drive
```bash
# Windows drives are mounted at /mnt/
ls /mnt/x/
# If not visible, ensure drive is connected and WSL is restarted
```
