/*
 * cuda_wrapper.h — Host-side wrappers for CUDA kernels
 *
 * These functions are implemented in .cu files and called from main.c.
 * They handle device memory allocation, H2D/D2H transfers, and kernel launch.
 */

#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

/*
 * CUDA Naïve matrix multiplication.
 * Each thread computes one element of C.
 *
 * Returns total execution time in ms (including memory transfers).
 * kernel_time_ms is set to the kernel-only execution time.
 *
 * Returns -1.0 if CUDA is not available or an error occurs.
 */
double cuda_naive_matmul(const float* A, const float* B, float* C,
                         int M, int N, int K, double* kernel_time_ms);

/*
 * CUDA Tiled matrix multiplication using shared memory.
 * Uses TILE_SIZE×TILE_SIZE shared memory tiles with __syncthreads().
 * Handles edge cases where dimensions are not multiples of TILE_SIZE.
 *
 * Returns total execution time in ms (including memory transfers).
 * kernel_time_ms is set to the kernel-only execution time.
 *
 * Returns -1.0 if CUDA is not available or an error occurs.
 */
double cuda_tiled_matmul(const float* A, const float* B, float* C,
                         int M, int N, int K, double* kernel_time_ms);

#endif /* CUDA_WRAPPER_H */
