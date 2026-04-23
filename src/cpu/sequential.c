/*
 * sequential.c — CPU sequential matrix multiplication (baseline)
 *
 * Standard triple-nested-loop: O(M*N*K)
 * This serves as the ground truth for correctness verification
 * and the baseline for speedup calculations.
 */

#include "sequential.h"
#include "../core/timer.h"

double cpu_sequential_matmul(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    double start = timer_get_ms();

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
