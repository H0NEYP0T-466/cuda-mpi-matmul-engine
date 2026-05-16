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

    /* Zero C since the i-k-j loop uses += accumulation */
    matrix_zero(C, M, N);

    double start = timer_get_ms();

    /*
     * Parallelize the i-loop across CPU threads.
     * Each thread gets a contiguous block of rows (schedule(static)).
     * Loop order i-k-j for better cache locality on B (sequential access).
     * No collapse — outer i-loop alone gives enough parallelism
     * and avoids false sharing on adjacent C writes.
     */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }

    double end = timer_get_ms();
    return end - start;
}
