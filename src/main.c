/*
 * main.c — Unified CLI Entry Point for the Distributed MatMul Framework
 *
 * All execution modes are EXPLICITLY selected via --mode flag.
 * No automatic hardware detection or silent fallback.
 *
 * Usage:
 *   ./matmul --mode <MODE> --size <N|preset> [--manual]
 *
 * Modes: cpu, cuda-naive, cuda-tiled, openmp-gpu, openmp-cpu
 * (MPI mode has its own entry point: src/mpi/distributed.c)
 *
 * Presets: small(256), medium(512), large(1024), xlarge(2048)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/matrix.h"
#include "core/timer.h"
#include "cpu/sequential.h"

/* Forward declarations — linked from respective .o / .cu.o files */
#ifdef ENABLE_CUDA
extern double cuda_naive_matmul(const float* A, const float* B, float* C,
                                int M, int N, int K, double* kernel_time_ms);
extern double cuda_tiled_matmul(const float* A, const float* B, float* C,
                                int M, int N, int K, double* kernel_time_ms);
#endif

#ifdef ENABLE_OPENMP_GPU
extern double openmp_gpu_matmul(const float* A, const float* B, float* C,
                                int M, int N, int K);
#endif

#ifdef ENABLE_OPENMP_CPU
extern double openmp_cpu_matmul(const float* A, const float* B, float* C,
                                int M, int N, int K);
#endif

/* Mode enumeration */
typedef enum {
    MODE_NONE = 0,
    MODE_CPU,
    MODE_CUDA_NAIVE,
    MODE_CUDA_TILED,
    MODE_OPENMP_GPU,
    MODE_OPENMP_CPU
} ExecMode;

static void print_usage(const char* prog) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║   Distributed Matrix Multiplication Framework        ║\n");
    printf("║   All modes are explicit — no auto-detection         ║\n");
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Usage: %s --mode <MODE> --size <SIZE> [--manual]    \n", prog);
    printf("║                                                      ║\n");
    printf("║  Modes:                                              ║\n");
    printf("║    cpu          CPU sequential baseline               ║\n");
    printf("║    cuda-naive   CUDA naïve kernel (requires GPU)      ║\n");
    printf("║    cuda-tiled   CUDA tiled shared-mem (requires GPU)  ║\n");
    printf("║    openmp-gpu   OpenMP target offload (requires GPU)  ║\n");
    printf("║    openmp-cpu   OpenMP CPU parallel threads           ║\n");
    printf("║                                                      ║\n");
    printf("║  Size presets:                                       ║\n");
    printf("║    small   = 256×256                                  ║\n");
    printf("║    medium  = 512×512                                  ║\n");
    printf("║    large   = 1024×1024                                ║\n");
    printf("║    xlarge  = 2048×2048                                ║\n");
    printf("║    <N>     = N×N (custom integer)                     ║\n");
    printf("║                                                      ║\n");
    printf("║  Options:                                            ║\n");
    printf("║    --manual   Enter matrix elements manually (≤8×8)   ║\n");
    printf("║                                                      ║\n");
    printf("║  MPI mode: use src/mpi/distributed.c separately      ║\n");
    printf("║    mpirun -np 4 ./matmul_mpi <N>                     ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");
}

static int parse_size(const char* str) {
    if (strcmp(str, "small") == 0) return 256;
    if (strcmp(str, "medium") == 0) return 512;
    if (strcmp(str, "large") == 0) return 1024;
    if (strcmp(str, "xlarge") == 0) return 2048;

    int n = atoi(str);
    if (n <= 0) {
        fprintf(stderr, "[ERROR] Invalid size: '%s'. Use small/medium/large/xlarge or a positive integer.\n", str);
        return -1;
    }
    return n;
}

static ExecMode parse_mode(const char* str) {
    if (strcmp(str, "cpu") == 0)         return MODE_CPU;
    if (strcmp(str, "cuda-naive") == 0)  return MODE_CUDA_NAIVE;
    if (strcmp(str, "cuda-tiled") == 0)  return MODE_CUDA_TILED;
    if (strcmp(str, "openmp-gpu") == 0)  return MODE_OPENMP_GPU;
    if (strcmp(str, "openmp-cpu") == 0)  return MODE_OPENMP_CPU;
    return MODE_NONE;
}

static const char* mode_name(ExecMode mode) {
    switch (mode) {
        case MODE_CPU:         return "CPU Sequential";
        case MODE_CUDA_NAIVE:  return "CUDA Naïve";
        case MODE_CUDA_TILED:  return "CUDA Tiled (Shared Memory)";
        case MODE_OPENMP_GPU:  return "OpenMP GPU Offload";
        case MODE_OPENMP_CPU:  return "OpenMP CPU Parallel";
        default:               return "Unknown";
    }
}

int main(int argc, char** argv) {
    ExecMode mode = MODE_NONE;
    int N = -1;
    int manual = 0;

    /* Parse command-line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = parse_mode(argv[++i]);
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            N = parse_size(argv[++i]);
        } else if (strcmp(argv[i], "--manual") == 0) {
            manual = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }

    /* Validate arguments */
    if (mode == MODE_NONE) {
        fprintf(stderr, "[ERROR] No mode specified. Use --mode <MODE>.\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (manual) {
        if (N < 0) {
            printf("Enter matrix dimension N (≤8 for manual): ");
            if (scanf("%d", &N) != 1 || N <= 0) {
                fprintf(stderr, "[ERROR] Invalid dimension.\n");
                return EXIT_FAILURE;
            }
        }
        if (N > 8) {
            fprintf(stderr, "[ERROR] Manual input limited to 8×8. Use --size for larger.\n");
            return EXIT_FAILURE;
        }
    } else if (N < 0) {
        fprintf(stderr, "[ERROR] No size specified. Use --size <SIZE>.\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    printf("\n[MODE] %s\n", mode_name(mode));
    printf("[MATRIX] %d×%d (seed=%d)\n", N, N, MATRIX_SEED);

    /* Allocate matrices */
    float* A = matrix_alloc(N, N);
    float* B = matrix_alloc(N, N);
    float* C = matrix_alloc(N, N);
    float* C_ref = matrix_alloc(N, N);  /* CPU reference for verification */

    if (!A || !B || !C || !C_ref) {
        fprintf(stderr, "[ERROR] Memory allocation failed for %d×%d matrices.\n", N, N);
        return EXIT_FAILURE;
    }

    /* Initialize matrices */
    if (manual) {
        matrix_read_manual(A, N, N, "Matrix A");
        matrix_read_manual(B, N, N, "Matrix B");
        matrix_print(A, N, N, "A");
        matrix_print(B, N, N, "B");
    } else {
        matrix_init_deterministic(A, N, N, 0);
        matrix_init_deterministic(B, N, N, 1);
        if (N <= 8) {
            matrix_print(A, N, N, "A");
            matrix_print(B, N, N, "B");
        }
    }

    /* Always compute CPU reference for verification */
    printf("[INFO] Computing CPU reference for verification...\n");
    double cpu_time = cpu_sequential_matmul(A, B, C_ref, N, N, N);
    printf("[CPU REF] %.3f ms\n", cpu_time);

    /* Execute selected mode */
    double exec_time = -1.0;
#ifdef ENABLE_CUDA
    double kernel_time = 0.0;
#endif
    int verified = 0;

    switch (mode) {
    case MODE_CPU:
        /* Already computed as reference */
        exec_time = cpu_time;
        memcpy(C, C_ref, (size_t)N * N * sizeof(float));
        verified = 1;
        break;

    case MODE_CUDA_NAIVE:
#ifdef ENABLE_CUDA
        exec_time = cuda_naive_matmul(A, B, C, N, N, N, &kernel_time);
        if (exec_time < 0) {
            fprintf(stderr, "[ERROR] CUDA naïve execution failed.\n");
            goto cleanup;
        }
        printf("[CUDA] Kernel time: %.3f ms | Total (with transfers): %.3f ms\n",
               kernel_time, exec_time);
#else
        fprintf(stderr, "[ERROR] CUDA support not compiled. Rebuild with ENABLE_CUDA=1.\n");
        goto cleanup;
#endif
        break;

    case MODE_CUDA_TILED:
#ifdef ENABLE_CUDA
        exec_time = cuda_tiled_matmul(A, B, C, N, N, N, &kernel_time);
        if (exec_time < 0) {
            fprintf(stderr, "[ERROR] CUDA tiled execution failed.\n");
            goto cleanup;
        }
        printf("[CUDA] Kernel time: %.3f ms | Total (with transfers): %.3f ms\n",
               kernel_time, exec_time);
#else
        fprintf(stderr, "[ERROR] CUDA support not compiled. Rebuild with ENABLE_CUDA=1.\n");
        goto cleanup;
#endif
        break;

    case MODE_OPENMP_GPU:
#ifdef ENABLE_OPENMP_GPU
        exec_time = openmp_gpu_matmul(A, B, C, N, N, N);
        if (exec_time < 0) {
            fprintf(stderr, "[ERROR] OpenMP GPU execution failed (no GPU found).\n");
            goto cleanup;
        }
#else
        fprintf(stderr, "[ERROR] OpenMP GPU offload not compiled. Rebuild with ENABLE_OPENMP_GPU=1.\n");
        goto cleanup;
#endif
        break;

    case MODE_OPENMP_CPU:
#ifdef ENABLE_OPENMP_CPU
        exec_time = openmp_cpu_matmul(A, B, C, N, N, N);
        if (exec_time < 0) {
            fprintf(stderr, "[ERROR] OpenMP CPU execution failed.\n");
            goto cleanup;
        }
#else
        fprintf(stderr, "[ERROR] OpenMP CPU not compiled. Rebuild with ENABLE_OPENMP_CPU=1.\n");
        goto cleanup;
#endif
        break;

    default:
        fprintf(stderr, "[ERROR] Invalid mode.\n");
        goto cleanup;
    }

    /* Verify result against CPU reference */
    if (mode != MODE_CPU) {
        verified = matrix_verify(C_ref, C, N, N);
    }

    /* Print small result matrices */
    if (N <= 8) {
        matrix_print(C, N, N, "C (Result)");
        if (mode != MODE_CPU) {
            matrix_print(C_ref, N, N, "C (CPU Reference)");
        }
    }

    /* Print final results */
    double gflops = timer_calc_gflops(N, N, N, exec_time);
    double speedup = (mode == MODE_CPU) ? 1.0 : cpu_time / exec_time;

    timer_print_result(mode_name(mode), N, exec_time, gflops, verified, speedup);

    /* Output CSV line for automated benchmarking */
    printf("CSV,%s,%d,%.3f,%.4f,%s\n",
           mode_name(mode), N, exec_time, gflops, verified ? "PASS" : "FAIL");

cleanup:
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
    matrix_free(C_ref);

    return (exec_time >= 0 && verified) ? EXIT_SUCCESS : EXIT_FAILURE;
}
