/*
 * tiled.cu — CUDA Tiled Matrix Multiplication with Shared Memory
 *
 * Uses TILE_SIZE×TILE_SIZE shared memory tiles to reduce global memory
 * accesses. Each thread block loads tiles of A and B into shared memory,
 * synchronizes, computes partial products, then moves to the next tile.
 *
 * This is the optimized GPU kernel that exploits data locality in the
 * GPU memory hierarchy (shared memory is ~100x faster than global).
 *
 * Edge-case handling: boundary checks when M, N, or K are not exact
 * multiples of TILE_SIZE — out-of-bounds threads load 0.0f.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_wrapper.h"

/* Tile dimension — must match block dimensions */
#define TILE_SIZE 16

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
 * Tiled shared-memory kernel.
 *
 * Algorithm:
 *   1. Load one element each of tile_A and tile_B from global → shared memory
 *   2. __syncthreads() to ensure tile is fully loaded
 *   3. Each thread accumulates partial dot product from the shared tiles
 *   4. __syncthreads() before loading next tile pair
 *   5. After all tiles processed, write result to global memory
 */
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    /* Shared memory tiles */
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    /* Number of tile phases needed to cover K dimension */
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        /*
         * Load tile_A: element at (row, t*TILE_SIZE + threadIdx.x)
         * Load tile_B: element at (t*TILE_SIZE + threadIdx.y, col)
         * Out-of-bounds threads load 0.0f (handles non-multiple-of-TILE_SIZE)
         */
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        if (row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        /* Synchronize: ensure entire tile is loaded before computation */
        __syncthreads();

        /* Accumulate partial dot product from this tile */
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        /* Synchronize: ensure all threads are done before loading next tile */
        __syncthreads();
    }

    /* Write final result (boundary check) */
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/*
 * Host-side wrapper for tiled kernel.
 */
double cuda_tiled_matmul(const float* h_A, const float* h_B, float* h_C,
                         int M, int N, int K, double* kernel_time_ms) {
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;

    /* Verify CUDA device is available */
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "[ERROR] CUDA tiled mode selected but no GPU found. Exiting.\n");
        return -1.0;
    }

    /* Print device info */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("[CUDA] Device: %s\n", prop.name);
    printf("[CUDA] Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("[CUDA] Tile size: %d × %d\n", TILE_SIZE, TILE_SIZE);
    printf("[CUDA] Shared memory used: %zu bytes (2 tiles)\n",
           2 * TILE_SIZE * TILE_SIZE * sizeof(float));

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    /* Create timing events */
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    CUDA_CHECK(cudaEventRecord(start_total));

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    /* Host → Device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    /* Configure grid */
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    /* Launch tiled kernel */
    CUDA_CHECK(cudaEventRecord(start_kernel));
    matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Device → Host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));

    /* Calculate times */
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
