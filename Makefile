# ==============================================================================
# Makefile — Distributed Matrix Multiplication Framework
#
# Targets:
#   make cpu          - CPU-only build (sequential + OpenMP CPU)
#   make cuda         - CUDA build (naïve + tiled + CPU)   [requires nvcc]
#   make openmp-gpu   - OpenMP GPU offload build            [requires nvc/gcc-offload]
#   make mpi          - MPI distributed build               [requires mpicc]
#   make all          - Build all available targets
#   make clean        - Remove build artifacts
# ==============================================================================

# Compilers
CC       = gcc
NVCC     = nvcc
MPICC    = mpicc

# Flags
CFLAGS   = -O2 -Wall -Wextra -std=c99
NVFLAGS  = -O2 -std=c++14
OMPFLAGS = -fopenmp
LDFLAGS  = -lm

# Directories
SRC      = src
CORE     = $(SRC)/core
CPU      = $(SRC)/cpu
CUDA_DIR = $(SRC)/cuda
OMP      = $(SRC)/openmp
MPI_DIR  = $(SRC)/mpi
BUILD    = build

# Core objects (always needed)
CORE_SRC = $(CORE)/matrix.c $(CORE)/timer.c
CPU_SRC  = $(CPU)/sequential.c

# Output binaries
BIN_CPU       = $(BUILD)/matmul_cpu
BIN_CUDA      = $(BUILD)/matmul_cuda
BIN_OMP_GPU   = $(BUILD)/matmul_openmp_gpu
BIN_MPI       = $(BUILD)/matmul_mpi

# ==============================================================================
# Default target
# ==============================================================================
.PHONY: all cpu cuda openmp-gpu openmp-cpu mpi clean help

help:
	@echo ""
	@echo "  Distributed Matrix Multiplication Framework"
	@echo "  ============================================"
	@echo ""
	@echo "  Targets:"
	@echo "    make cpu          CPU sequential + OpenMP CPU"
	@echo "    make cuda         CUDA naive + tiled (requires nvcc)"
	@echo "    make openmp-gpu   OpenMP GPU offload (requires nvc)"
	@echo "    make mpi          MPI distributed (requires mpicc)"
	@echo "    make all          Build all available targets"
	@echo "    make clean        Remove build artifacts"
	@echo ""

# ==============================================================================
# CPU-only build (sequential + OpenMP CPU threads)
# ==============================================================================
cpu: $(BUILD)
	$(CC) $(CFLAGS) $(OMPFLAGS) \
		-DENABLE_OPENMP_CPU \
		$(SRC)/main.c $(CORE_SRC) $(CPU_SRC) $(OMP)/offload_cpu.c \
		-o $(BIN_CPU) $(LDFLAGS)
	@echo "[BUILD] $(BIN_CPU) — CPU sequential + OpenMP CPU"

# ==============================================================================
# CUDA build (naïve + tiled + CPU baseline)
# ==============================================================================
cuda: $(BUILD)
	$(NVCC) $(NVFLAGS) \
		-DENABLE_CUDA \
		-x cu $(SRC)/main.c \
		-x cu $(CORE)/matrix.c $(CORE)/timer.c $(CPU)/sequential.c \
		$(CUDA_DIR)/naive.cu $(CUDA_DIR)/tiled.cu \
		-o $(BIN_CUDA) $(LDFLAGS)
	@echo "[BUILD] $(BIN_CUDA) — CUDA naïve + tiled"

# ==============================================================================
# OpenMP GPU offload build
# ==============================================================================
openmp-gpu: $(BUILD)
	$(CC) $(CFLAGS) $(OMPFLAGS) -foffload=nvptx-none \
		-DENABLE_OPENMP_GPU \
		$(SRC)/main.c $(CORE_SRC) $(CPU_SRC) $(OMP)/offload_gpu.c \
		-o $(BIN_OMP_GPU) $(LDFLAGS)
	@echo "[BUILD] $(BIN_OMP_GPU) — OpenMP GPU offload"

# ==============================================================================
# MPI build (separate entry point)
# ==============================================================================
mpi: $(BUILD)
	$(MPICC) $(CFLAGS) \
		$(MPI_DIR)/distributed.c $(CORE_SRC) $(CPU_SRC) \
		-o $(BIN_MPI) $(LDFLAGS)
	@echo "[BUILD] $(BIN_MPI) — MPI distributed"

# ==============================================================================
# Build all (skip targets whose compilers are missing)
# ==============================================================================
all: $(BUILD)
	@echo "=== Building all available targets ==="
	@$(MAKE) cpu 2>/dev/null        && echo "[OK] cpu"        || echo "[SKIP] cpu (gcc not found)"
	@$(MAKE) mpi 2>/dev/null        && echo "[OK] mpi"        || echo "[SKIP] mpi (mpicc not found)"
	@$(MAKE) cuda 2>/dev/null       && echo "[OK] cuda"       || echo "[SKIP] cuda (nvcc not found)"
	@$(MAKE) openmp-gpu 2>/dev/null && echo "[OK] openmp-gpu" || echo "[SKIP] openmp-gpu (offload not supported)"
	@echo "=== Build complete ==="

# ==============================================================================
# Create build directory
# ==============================================================================
$(BUILD):
	@mkdir -p $(BUILD)

# ==============================================================================
# Clean
# ==============================================================================
clean:
	rm -rf $(BUILD)
	@echo "[CLEAN] Build directory removed"
