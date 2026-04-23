/*
 * naive.cu — CUDA Naïve Matrix Multiplication Kernel
 *
 * Each thread computes exactly one element of the output matrix C.
 * Grid/Block: dim3 block(16,16), dim3 grid(ceil(N/16), ceil(M/16))
 *
 * This is the unoptimized baseline GPU kernel. It reads all data
 * from global memory with no caching or tiling strategy.
 *
 * Boundary checks ensure correctness for non-multiple-of-16 dimensions.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_wrapper.h"

/* Block dimension for naïve kernel */
#define BLOCK_SIZE 16

/*
 * Macro to check CUDA calls and print clear error messages.
 */
#define CUDA_CHECK(call)                                                        \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA ERROR] %s:%d — %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            return -1.0;                                                         \
        }                                                                        \
    } while (0)

/*
 * Naïve kernel: each thread computes C[row][col] = sum(A[row][k] * B[k][col])
 */
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    /* Boundary check for non-multiple-of-BLOCK_SIZE dimensions */
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/*
 * Host-side wrapper: allocates device memory, transfers data,
 * launches kernel, and copies result back.
 */
double cuda_naive_matmul(const float* h_A, const float* h_B, float* h_C,
                         int M, int N, int K, double* kernel_time_ms) {
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;

    /* Verify CUDA device is available */
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "[ERROR] CUDA naïve mode selected but no GPU found. Exiting.\n");
        return -1.0;
    }

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    /* Create timing events */
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    /* Start total timer (includes transfers) */
    CUDA_CHECK(cudaEventRecord(start_total));

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    /* Host → Device transfer */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    /* Configure grid and block dimensions */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* Launch kernel with timing */
    CUDA_CHECK(cudaEventRecord(start_kernel));
    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Device → Host transfer */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    /* Stop total timer */
    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));

    /* Calculate elapsed times */
    float total_ms = 0.0f, kern_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start_total, stop_total));
    CUDA_CHECK(cudaEventElapsedTime(&kern_ms, start_kernel, stop_kernel));
    *kernel_time_ms = (double)kern_ms;

    /* Cleanup */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    return (double)total_ms;
}
