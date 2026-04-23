/*
 * offload_cpu.c — OpenMP CPU Parallel Matrix Multiplication
 *
 * Uses #pragma omp parallel for to distribute work across CPU threads.
 * This mode uses CPU threads ONLY — it never attempts GPU offload.
 *
 * Compile with:
 *   gcc -fopenmp -o matmul_openmp_cpu offload_cpu.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../core/matrix.h"
#include "../core/timer.h"

/*
 * OpenMP CPU parallel matrix multiplication.
 * Returns execution time in milliseconds.
 */
double openmp_cpu_matmul(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    int num_threads = omp_get_max_threads();
    printf("[OpenMP CPU] Using %d threads (CPU only, no GPU offload)\n", num_threads);

    double start = timer_get_ms();

    /*
     * Parallelize the outer two loops across CPU threads.
     * collapse(2) merges i and j loops for better load balancing.
     * schedule(static) divides iterations evenly among threads.
     */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    double end = timer_get_ms();
    return end - start;
}
