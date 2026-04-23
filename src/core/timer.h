/*
 * timer.h — High-resolution cross-platform timing
 *
 * Uses clock_gettime (POSIX) or QueryPerformanceCounter (Windows).
 */

#ifndef TIMER_H
#define TIMER_H

/*
 * Get current time in milliseconds (high resolution).
 */
double timer_get_ms(void);

/*
 * Calculate GFLOPS for matrix multiplication.
 * For M×K × K×N multiplication: ops = 2*M*N*K
 */
double timer_calc_gflops(int M, int N, int K, double time_ms);

/*
 * Print a formatted timing result line.
 */
void timer_print_result(const char* mode, int N, double time_ms, double gflops,
                        int verified, double speedup);

#endif /* TIMER_H */
