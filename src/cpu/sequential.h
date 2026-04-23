/*
 * sequential.h — CPU sequential matrix multiplication (baseline)
 */

#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

/*
 * CPU sequential matrix multiplication: C = A × B
 * A is M×K, B is K×N, C is M×N (all row-major).
 * Returns execution time in milliseconds.
 */
double cpu_sequential_matmul(const float* A, const float* B, float* C,
                             int M, int N, int K);

#endif /* SEQUENTIAL_H */
