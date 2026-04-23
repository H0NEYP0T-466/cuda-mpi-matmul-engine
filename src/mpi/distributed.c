/*
 * distributed.c — MPI Distributed Matrix Multiplication
 *
 * Uses row-wise decomposition of matrix A across MPI ranks:
 *   1. Rank 0 initializes A and B deterministically (seed=42)
 *   2. MPI_Bcast sends B to all ranks
 *   3. MPI_Scatterv distributes rows of A (handles uneven splits)
 *   4. Each rank computes its portion of C = local_A × B
 *   5. MPI_Gatherv collects results back to rank 0
 *   6. Rank 0 verifies against CPU sequential result
 *
 * Compile: mpicc -o matmul_mpi distributed.c ../core/matrix.c ../core/timer.c ../cpu/sequential.c
 * Run:     mpirun -np 4 --hostfile hostfile ./matmul_mpi <N>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "../core/matrix.h"
#include "../core/timer.h"
#include "../cpu/sequential.h"

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Parse matrix size from CLI */
    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: mpirun -np <P> ./matmul_mpi <N>\n");
            fprintf(stderr, "  N: matrix dimension (N×N)\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        if (rank == 0)
            fprintf(stderr, "[ERROR] Invalid matrix size: %d\n", N);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        printf("[MPI] Matrix size: %d×%d\n", N, N);
        printf("[MPI] Number of processes: %d\n", size);
    }

    /*
     * Calculate row distribution — handle uneven splits.
     * Each rank gets base_rows, the last rank gets remainder too.
     */
    int base_rows = N / size;
    int remainder = N % size;

    /* sendcounts and displacements for Scatterv/Gatherv */
    int* sendcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    int* recvcounts = (int*)malloc(size * sizeof(int));
    int* rdispls = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        int rows_i = base_rows + (i < remainder ? 1 : 0);
        sendcounts[i] = rows_i * N;  /* elements of A to send */
        recvcounts[i] = rows_i * N;  /* elements of C to receive */
    }
    displs[0] = 0;
    rdispls[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + sendcounts[i - 1];
        rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
    }

    int my_rows = base_rows + (rank < remainder ? 1 : 0);

    /* Allocate matrices */
    float* A = NULL;  /* Full A on rank 0 only */
    float* B = matrix_alloc(N, N);    /* B on all ranks */
    float* C = NULL;  /* Full C on rank 0 only */
    float* local_A = matrix_alloc(my_rows, N);
    float* local_C = matrix_alloc(my_rows, N);

    if (rank == 0) {
        A = matrix_alloc(N, N);
        C = matrix_alloc(N, N);

        /* Deterministic initialization */
        matrix_init_deterministic(A, N, N, 0);
        matrix_init_deterministic(B, N, N, 1);
    }

    /* === Timing starts here === */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* Broadcast B to all ranks */
    double t_comm_start = MPI_Wtime();
    MPI_Bcast(B, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Scatter rows of A to all ranks */
    MPI_Scatterv(A, sendcounts, displs, MPI_FLOAT,
                 local_A, my_rows * N, MPI_FLOAT,
                 0, MPI_COMM_WORLD);
    double t_comm_scatter = MPI_Wtime();

    /* Local computation: local_C = local_A × B */
    double t_comp_start = MPI_Wtime();
    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += local_A[i * N + k] * B[k * N + j];
            }
            local_C[i * N + j] = sum;
        }
    }
    double t_comp_end = MPI_Wtime();

    /* Gather results back to rank 0 */
    double t_gather_start = MPI_Wtime();
    MPI_Gatherv(local_C, my_rows * N, MPI_FLOAT,
                C, recvcounts, rdispls, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    double t_gather_end = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    /* Rank 0: verify and report */
    if (rank == 0) {
        double total_ms = (t_end - t_start) * 1000.0;
        double comm_ms = (t_comm_scatter - t_comm_start) * 1000.0 +
                         (t_gather_end - t_gather_start) * 1000.0;
        double comp_ms = (t_comp_end - t_comp_start) * 1000.0;
        double gflops = timer_calc_gflops(N, N, N, total_ms);

        /* Verify against CPU sequential */
        float* C_ref = matrix_alloc(N, N);
        double cpu_time = cpu_sequential_matmul(A, B, C_ref, N, N, N);
        int verified = matrix_verify(C_ref, C, N, N);
        double speedup = cpu_time / total_ms;

        printf("╔══════════════════════════════════════════╗\n");
        printf("║  MPI Distributed (%d processes)           ║\n", size);
        printf("╠══════════════════════════════════════════╣\n");
        printf("║  Matrix Size   : %4d × %4d             ║\n", N, N);
        printf("║  Total Time    : %10.3f ms            ║\n", total_ms);
        printf("║  Compute Time  : %10.3f ms            ║\n", comp_ms);
        printf("║  Comm Time     : %10.3f ms            ║\n", comm_ms);
        printf("║  GFLOPS        : %10.4f               ║\n", gflops);
        printf("║  Speedup vs CPU: %10.2fx              ║\n", speedup);
        printf("║  Efficiency    : %10.2f%%              ║\n",
               (speedup / size) * 100.0);
        printf("║  Verified      : %s                     ║\n",
               verified ? "PASS ✓" : "FAIL ✗");
        printf("╚══════════════════════════════════════════╝\n");

        /* Output CSV line for benchmarking */
        printf("CSV,mpi-%d,%d,%.3f,%.4f,%s\n",
               size, N, total_ms, gflops, verified ? "PASS" : "FAIL");

        matrix_free(C_ref);
        matrix_free(A);
        matrix_free(C);
    }

    /* Cleanup */
    matrix_free(B);
    matrix_free(local_A);
    matrix_free(local_C);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(rdispls);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
