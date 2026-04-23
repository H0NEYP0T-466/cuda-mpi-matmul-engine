/*
 * offload_gpu.c — OpenMP Target Offload (GPU ONLY)
 *
 * Uses #pragma omp target to offload matrix multiplication to the GPU.
 * This mode REQUIRES a GPU with OpenMP offload support.
 * If no GPU is found, it prints an error and exits — NO silent fallback.
 *
 * Compile with:
 *   nvc -mp=gpu -o matmul_openmp_gpu offload_gpu.c    (NVIDIA HPC SDK)
 *   gcc -fopenmp -foffload=nvptx-none                  (GCC with offload)
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../core/matrix.h"
#include "../core/timer.h"

/*
 * OpenMP GPU target offload matrix multiplication.
 * Returns execution time in milliseconds.
 * Returns -1.0 on failure (no GPU found).
 */
double openmp_gpu_matmul(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    /*
     * EXPLICIT CHECK: Do NOT silently fall back to CPU.
     * If no GPU device is available, fail loudly.
     */
    int num_devices = omp_get_num_devices();
    printf("[OpenMP GPU] Devices detected: %d\n", num_devices);

    if (num_devices == 0) {
        fprintf(stderr, "[ERROR] OpenMP GPU mode selected but no GPU devices found.\n");
        fprintf(stderr, "[ERROR] Use --mode openmp-cpu for CPU-only OpenMP. Exiting.\n");
        return -1.0;
    }

    printf("[OpenMP GPU] Using device 0 of %d\n", num_devices);
    printf("[OpenMP GPU] Host threads: %d\n", omp_get_max_threads());

    int size_A = M * K;
    int size_B = K * N;
    int size_C = M * N;

    double start = timer_get_ms();

    /*
     * Map data to GPU device:
     *   - A and B: to (read-only on device)
     *   - C: from (write-only on device, copy back after)
     */
    #pragma omp target data map(to: A[0:size_A], B[0:size_B]) \
                            map(from: C[0:size_C])
    {
        #pragma omp target teams distribute parallel for collapse(2) \
                schedule(static)
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    double end = timer_get_ms();
    return end - start;
}
