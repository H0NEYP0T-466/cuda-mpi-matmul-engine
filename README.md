<!-- markdownlint-disable MD033 MD041 -->

<p align="center">

  <!-- Core -->
  ![GitHub License](https://img.shields.io/github/license/H0NEYP0T-466/cuda-mpi-matmul-engine?style=for-the-badge&color=brightgreen)
  ![GitHub Stars](https://img.shields.io/github/stars/H0NEYP0T-466/cuda-mpi-matmul-engine?style=for-the-badge&color=yellow)
  ![GitHub Forks](https://img.shields.io/github/forks/H0NEYP0T-466/cuda-mpi-matmul-engine?style=for-the-badge&color=blue)
  ![GitHub Issues](https://img.shields.io/github/issues/H0NEYP0T-466/cuda-mpi-matmul-engine?style=for-the-badge&color=red)
  ![GitHub Pull Requests](https://img.shields.io/github/issues-pr/H0NEYP0T-466/cuda-mpi-matmul-engine?style=for-the-badge&color=orange)
  ![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge)

  <!-- Activity -->
  ![Last Commit](https://img.shields.io/github/last-commit/H0NEYP0T-466/cuda-mpi-matmul-engine?style=for-the-badge&color=purple)
  ![Commit Activity](https://img.shields.io/github/commit-activity/m/H0NEYP0T-466/cuda-mpi-matmul-engine?style=for-the-badge&color=teal)
  ![Repo Size](https://img.shields.io/github/repo-size/H0NEYP0T-466/cuda-mpi-matmul-engine?style=for-the-badge&color=blueviolet)
  ![Code Size](https://img.shields.io/github/languages/code-size/H0NEYP0T-466/cuda-mpi-matmul-engine?style=for-the-badge&color=indigo)

  <!-- Languages -->
  ![Top Language](https://img.shields.io/github/languages/top/H0NEYP0T-466/cuda-mpi-matmul-engine?style=for-the-badge&color=critical)
  ![Languages Count](https://img.shields.io/github/languages/count/H0NEYP0T-466/cuda-mpi-matmul-engine?style=for-the-badge&color=success)

  <!-- Community -->
  ![Documentation](https://img.shields.io/badge/Docs-Available-green?style=for-the-badge&logo=readthedocs&logoColor=white)
  ![Open Source Love](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red?style=for-the-badge)

</p>

# 🔢 Distributed Matrix Multiplication Engine

A cloud-enabled high-performance computing framework that implements, benchmarks, and compares **five parallelization strategies** for matrix multiplication: CPU sequential, CUDA naïve, CUDA tiled (shared memory), OpenMP (GPU + CPU), and MPI distributed execution. Designed for reproducible research across local, Docker, Colab, and AWS EC2 environments.

---

## 🔗 Quick Links

| Link | Description |
|------|-------------|
| 📖 [Documentation](#-table-of-contents) | Full project documentation |
| 📊 [Results & Charts](#-performance-charts) | Benchmark visualizations |
| 🐛 [Issues](https://github.com/H0NEYP0T-466/cuda-mpi-matmul-engine/issues) | Report bugs or request features |
| 🤝 [Contributing](CONTRIBUTING.md) | How to contribute |

---

## 📑 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Dependencies](#-dependencies)
- [Installation](#-installation)
- [Usage](#-usage)
- [Folder Structure](#-folder-structure)
- [Performance Charts](#-performance-charts)
- [Benchmarking](#-benchmarking)
- [Docker MPI Cluster](#-docker-mpi-cluster)
- [Design Decisions](#-design-decisions)
- [Contributing](#-contributing)
- [License](#-license)
- [Security](#-security)
- [Code of Conduct](#-code-of-conduct)

---

## ✨ Features

- **🚀 Five Execution Modes** — CPU sequential, CUDA naïve, CUDA tiled (shared memory), OpenMP GPU offload, OpenMP CPU parallel, and MPI distributed
- **📊 Automated Benchmarking** — Single-script benchmark suite with 6 auto-generated performance charts
- **🐳 Docker MPI Cluster** — 4-node static MPI cluster via Docker Compose, ready with `docker-compose up`
- **☁️ Cloud-Ready** — Deploy to AWS EC2 c5.large instances for real distributed scaling experiments
- **🔬 Deterministic Reproducibility** — Fixed seed (`srand(42)`) ensures identical matrices across every environment
- **✅ Correctness Verification** — All implementations verified against CPU reference with `1e-3` tolerance
- **🧪 Edge Case Testing** — Tested with non-square-friendly sizes: 1, 15, 17, 31, 33, 255, 257
- **📈 Cost Analysis** — Built-in cost efficiency charting for AWS cloud runs ($/hr modeling)
- **🎓 Google Colab Support** — Pre-built notebook for free T4 GPU experiments
- **🔒 Explicit Modes** — No auto-detection or silent fallback; missing hardware = clear error message

---

## 🛠 Tech Stack

### Languages
![C](https://img.shields.io/badge/C-%2300599C.svg?style=for-the-badge&logo=c&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-%2376B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Shell](https://img.shields.io/badge/Shell-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white)
![Make](https://img.shields.io/badge/Make-%2300599C.svg?style=for-the-badge&logo=gnu&logoColor=white)

### Frameworks & Libraries
![OpenMP](https://img.shields.io/badge/OpenMP-%2300599C.svg?style=for-the-badge&logo=openmp&logoColor=white)
![MPI](https://img.shields.io/badge/OpenMPI-%2300599C.svg?style=for-the-badge&logo=mpi&logoColor=white)
![OpenMPI](https://img.shields.io/badge/OpenMPI-%230052C8.svg?style=for-the-badge&logo=mpi&logoColor=white)

### DevOps / CI / Tools
![Docker](https://img.shields.io/badge/Docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Docker Compose](https://img.shields.io/badge/Docker%20Compose-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Makefile](https://img.shields.io/badge/Makefile-%2300599C.svg?style=for-the-badge&logo=gnu&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-%23008FBA.svg?style=for-the-badge&logo=cmake&logoColor=white)

### Cloud / Hosting
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)

---

## 📦 Dependencies & Packages

### System Dependencies
| Dependency | Version | Purpose |
|------------|---------|---------|
| `gcc` | ≥ 9.0 | C compiler for CPU, OpenMP, MPI builds |
| `nvcc` | ≥ 11.0 | NVIDIA CUDA compiler (CUDA builds) |
| `mpicc` | ≥ 3.1 | MPI C compiler (distributed builds) |
| `make` | ≥ 4.0 | Build automation |
| `docker` | ≥ 20.10 | Containerization for MPI cluster |
| `docker-compose` | ≥ 1.29 | Multi-container orchestration |

### Python Dependencies (Benchmarking & Charting)

<details>
<summary><b>📦 Runtime Dependencies</b></summary>

| Package | Version | Purpose |
|---------|---------|---------|
| ![matplotlib](https://img.shields.io/pypi/v/matplotlib?style=for-the-badge&label=matplotlib) | ≥ 3.5 | Performance chart generation |
| ![numpy](https://img.shields.io/pypi/v/numpy?style=for-the-badge&label=numpy) | ≥ 1.21 | Numerical data processing |

</details>

Install Python dependencies:
```bash
pip install matplotlib numpy
```

---

## 🚀 Installation

### Prerequisites

- **Linux / WSL / macOS:** GCC, Make
- **CUDA support:** NVIDIA GPU + `nvcc` (CUDA Toolkit ≥ 11.0)
- **MPI support:** OpenMPI (`mpicc`) — `sudo apt install openmpi-bin libopenmpi-dev`
- **Docker:** Docker Engine + Docker Compose (for MPI cluster)
- **Python:** Python 3.8+ with `matplotlib` and `numpy` (for benchmark charts)

### Clone & Build

```bash
# Clone the repository
git clone https://github.com/H0NEYP0T-466/cuda-mpi-matmul-engine.git
cd cuda-mpi-matmul-engine

# Build all available targets
make all

# Or build individually:
make cpu            # CPU sequential + OpenMP CPU
make mpi            # MPI distributed
make cuda           # CUDA naïve + tiled (requires nvcc)
make openmp-gpu     # OpenMP GPU offload (requires gcc-offload)
```

### Docker Setup (MPI Cluster)

```bash
cd docker
docker-compose up --build
```

### Google Colab (CUDA)

Open `colab/matmul_gpu.ipynb` in Google Colab with a GPU runtime (free T4 available).

---

## ⚡ Usage

### CLI Interface

All modes are **explicitly selected** — no auto-detection or silent fallback.

```bash
# CPU sequential baseline
./build/matmul_cpu --mode cpu --size medium

# OpenMP CPU parallel
./build/matmul_cpu --mode openmp-cpu --size 1024

# CUDA naïve kernel (requires GPU)
./build/matmul_cuda --mode cuda-naive --size large

# CUDA tiled shared-memory kernel (requires GPU)
./build/matmul_cuda --mode cuda-tiled --size xlarge

# OpenMP GPU offload (requires GPU)
./build/matmul_openmp_gpu --mode openmp-gpu --size large

# MPI distributed (separate binary)
mpirun -np 4 ./build/matmul_mpi 512
```

### Manual Input Mode (Correctness Testing)

```bash
# Enter matrix elements manually (limited to 8×8)
./build/matmul_cpu --mode cpu --size 4 --manual
```

### Size Presets

| Preset  | Dimensions  |
|---------|-------------|
| `small` | 256 × 256   |
| `medium`| 512 × 512   |
| `large` | 1024 × 1024 |
| `xlarge`| 2048 × 2048 |

You can also pass a custom integer: `--size 512`

### Help

```bash
./build/matmul_cpu --help
```

---

## 📂 Folder Structure

```
cuda-mpi-matmul-engine/
├── CLAUDE.md                    # AI assistant project instructions
├── Makefile                     # Build system (cpu/cuda/mpi/openmp-gpu/all)
├── README.md                    # This file
├── LICENSE                      # MIT License
├── CONTRIBUTING.md              # Contribution guidelines
├── SECURITY.md                  # Security policy
├── CODE_OF_CONDUCT.md           # Code of Conduct
├── .gitignore                   # Git ignore rules
├── benchmarks.txt               # Raw benchmark results
├── execution_cpu.txt            # CPU execution logs
├── execution_mpi_docker.txt     # Docker MPI execution logs
├── execution_mpi_local.txt      # Local MPI execution logs
│
├── src/                         # Source code
│   ├── main.c                   # Unified CLI entry point
│   ├── core/
│   │   ├── matrix.c             # Matrix alloc/init/verify/free
│   │   ├── matrix.h             # Matrix utility header
│   │   ├── timer.c              # Timing & GFLOPS calculation
│   │   └── timer.h              # Timer utility header
│   ├── cpu/
│   │   ├── sequential.c         # CPU sequential baseline
│   │   └── sequential.h         # Sequential header
│   ├── cuda/
│   │   ├── naive.cu             # CUDA naïve kernel
│   │   ├── tiled.cu             # CUDA tiled shared-memory kernel
│   │   └── cuda_wrapper.h       # CUDA function declarations
│   ├── openmp/
│   │   ├── offload_cpu.c        # OpenMP CPU parallel implementation
│   │   └── offload_gpu.c        # OpenMP GPU offload implementation
│   └── mpi/
│       └── distributed.c        # MPI distributed (separate entry point)
│
├── scripts/
│   ├── benchmark.sh             # Automated benchmark runner
│   ├── benchmark.py             # Chart generation (matplotlib)
│   └── cloud_scaling.sh         # AWS cloud scaling script
│
├── docker/
│   ├── Dockerfile               # MPI + OpenMP build environment
│   ├── docker-compose.yml       # 4-node static MPI cluster
│   └── hostfile                 # MPI hostfile (4 nodes)
│
├── aws/
│   ├── setup_ec2.sh             # EC2 instance setup
│   ├── run_mpi_cloud.sh         # Cloud MPI execution
│   ├── cloud_hostfile           # AWS MPI hostfile
│   ├── SSH_commands.txt         # SSH reference commands
│   └── matmul-project-keypair.pem # AWS keypair (DO NOT commit)
│
├── awsProofs/                   # AWS deployment proof screenshots
│   ├── 1-Instances.PNG
│   ├── 2-SecurityRules.PNG
│   ├── final1.PNG
│   ├── final2.PNG
│   └── FINAL_LOGS.txt
│
├── colab/
│   ├── matmul_gpu.ipynb         # Google Colab CUDA notebook
│   └── fix_notebook.py          # Notebook fix utility
│
├── results/
│   ├── timings.csv              # Benchmark timing data
│   ├── scaling.csv              # AWS scaling data
│   └── charts/                  # Generated performance charts
│       ├── 1_exec_time.png
│       ├── 2_speedup.png
│       ├── 3_gflops.png
│       ├── 4_mpi_strong_scaling.png
│       ├── 5_mpi_weak_scaling.png
│       └── 6_cost_efficiency.png
│
├── report/
│   └── report.md                # Technical report
│
├── build/                       # Compiled binaries (gitignored)
│   ├── matmul_cpu
│   └── matmul_mpi
│
└── .github/
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.yml
    │   ├── feature_request.yml
    │   └── config.yml
    └── pull_request_template.md
```

---

## 📊 Performance Charts

Generated from `results/timings.csv` via `python3 scripts/benchmark.py`.

### 1️⃣ Execution Time vs Matrix Size
![Execution Time vs Matrix Size](results/charts/1_exec_time.png)

*Grouped bar chart comparing execution time (log scale) across all implementations for 256, 512, 1024, and 2048 matrix sizes.*

### 2️⃣ Speedup vs CPU Sequential Baseline
![Speedup vs CPU Baseline](results/charts/2_speedup.png)

*Speedup relative to CPU sequential. The red dashed line at 1× marks the baseline. Higher is better.*

### 3️⃣ Computational Throughput (GFLOPS)
![GFLOPS Comparison](results/charts/3_gflops.png)

*GFLOPS achieved by each implementation. Higher values indicate better computational throughput.*

### 4️⃣ MPI Strong Scaling
![MPI Strong Scaling](results/charts/4_mpi_strong_scaling.png)

*Speedup vs process count for fixed problem sizes. The red dashed line shows ideal linear scaling.*

### 5️⃣ MPI Execution Time vs Problem Size
![MPI Weak Scaling](results/charts/5_mpi_weak_scaling.png)

*Execution time comparison across MPI process counts as problem size grows.*

### 6️⃣ Cost Efficiency Analysis
![Cost Efficiency](results/charts/6_cost_efficiency.png)

*Estimated cost per run in milli-dollars for AWS EC2 c5.large instances ($0.085/hr). Lower is better.*

---

## 🏋️ Benchmarking

Run the complete benchmark suite:

```bash
# Build first
make cpu && make mpi

# Run benchmarks
bash scripts/benchmark.sh

# Generate charts
python3 scripts/benchmark.py
```

### Benchmark Output

Results are saved to `results/timings.csv` with columns:
- `mode` — Implementation name
- `matrix_size` — N for N×N matrix
- `exec_time_ms` — Execution time in milliseconds
- `gflops` — Computational throughput
- `verified` — Correctness check (PASS/FAIL)

### Sample Results

| Mode | Size | Time (ms) | GFLOPS | Verified |
|------|------|-----------|--------|----------|
| CPU Sequential | 256 | 43.732 | 0.7673 | ✅ PASS |
| CPU Sequential | 512 | 315.637 | 0.8505 | ✅ PASS |
| CPU Sequential | 1024 | 7811.235 | 0.2749 | ✅ PASS |
| CPU Sequential | 2048 | 126842.005 | 0.1354 | ✅ PASS |
| OpenMP CPU Parallel | 256 | 29.982 | 1.1191 | ✅ PASS |
| OpenMP CPU Parallel | 2048 | 84620.725 | 0.2030 | ✅ PASS |
| MPI (1 proc) | 256 | 34.412 | 0.9751 | ✅ PASS |
| MPI (2 proc) | 256 | 21.411 | 1.5672 | ✅ PASS |
| MPI (4 proc) | 2048 | ~19180* | ~0.8957* | ✅ PASS |

*\*From AWS cloud scaling results*

---

## 🐳 Docker MPI Cluster

A static 4-node MPI cluster using Docker Compose:

```bash
cd docker
docker-compose up --build
```

**How it works:**
- 4 containers (`node0`–`node3`) on a bridge network with fixed IPs
- `node0` is the master — it waits for workers, then runs `mpirun -np 4`
- Pre-configured SSH with passwordless keys and hostfile
- Results volume-mounted to `../results/`

---

## 📐 Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Deterministic matrices** — `srand(42)` everywhere | Same size = same matrix across Local, Docker, Colab, and AWS |
| 2 | **Explicit modes** — no auto-detection | Missing hardware produces clear errors, not silent fallbacks |
| 3 | **Separate OpenMP targets** | GPU offload and CPU parallel are different compile targets |
| 4 | **Row-wise MPI decomposition** | Minimizes communication: broadcast B, scatter A rows |
| 5 | **Static Docker cluster** | Fixed compose + hostfile, no dynamic SSH discovery |
| 6 | **Tile size = 16×16** | Balances shared memory usage (2KB/block) against occupancy on 48KB SM |
| 7 | **Tolerance = 1e-3** | Accounts for floating-point non-association across implementations |

---

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- How to fork and submit PRs
- Code style and linting rules
- Bug reporting and feature request processes
- Testing requirements

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🛡 Security

Please review our [Security Policy](SECURITY.md) for information on reporting vulnerabilities responsibly.

---

## 📏 Code of Conduct

We are committed to providing a welcoming and inclusive experience. Please read our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 🌍 Deployment Environments

| Environment | Purpose | Cost |
|-------------|---------|------|
| Local Docker | MPI cluster simulation (4 static nodes) | Free |
| Google Colab | CUDA GPU experiments (free T4) | Free |
| AWS EC2 (c5.large) | Cloud scaling demonstration | ~$0.09/experiment |

---

<p align="center">Made with ❤ by <a href="https://github.com/H0NEYP0T-466">H0NEYP0T-466</a></p>
