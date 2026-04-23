/*
 * matrix.h — Matrix utilities for the Distributed MatMul Framework
 * 
 * Provides allocation, deterministic initialization (seed=42),
 * verification, and printing for flat (row-major) float matrices.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Fixed seed for deterministic, reproducible matrix generation */
#define MATRIX_SEED 42

/* Tolerance for floating-point comparison across implementations */
#define VERIFY_TOLERANCE 1e-3f

/*
 * Allocate a rows×cols float matrix (row-major, contiguous).
 * Returns NULL on failure.
 */
float* matrix_alloc(int rows, int cols);

/*
 * Free a matrix allocated by matrix_alloc.
 */
void matrix_free(float* M);

/*
 * Initialize matrix with deterministic random values in [0, 10).
 * Always uses MATRIX_SEED — same dimensions produce the same matrix
 * across Local, Docker, Colab, and AWS.
 *
 * seed_offset: use 0 for matrix A, 1 for matrix B, etc.
 *              This ensures A and B are different even with same dimensions.
 */
void matrix_init_deterministic(float* M, int rows, int cols, int seed_offset);

/*
 * Initialize matrix to zero.
 */
void matrix_zero(float* M, int rows, int cols);

/*
 * Verify two matrices are element-wise equal within VERIFY_TOLERANCE.
 * Returns 1 if PASS, 0 if FAIL.
 * On failure, prints the first mismatched element.
 */
int matrix_verify(const float* expected, const float* actual, int rows, int cols);

/*
 * Print a matrix to stdout (for small matrices only, ≤8×8 recommended).
 */
void matrix_print(const float* M, int rows, int cols, const char* name);

/*
 * Read a matrix from stdin (manual input mode).
 * Prompts the user for each element.
 * Matrix must be pre-allocated.
 */
void matrix_read_manual(float* M, int rows, int cols, const char* name);

#endif /* MATRIX_H */
