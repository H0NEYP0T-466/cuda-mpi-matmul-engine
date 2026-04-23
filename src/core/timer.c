/*
 * timer.c — High-resolution timing implementation
 */

#include "timer.h"
#include <stdio.h>

#ifdef _WIN32
  #include <windows.h>
  double timer_get_ms(void) {
      LARGE_INTEGER freq, count;
      QueryPerformanceFrequency(&freq);
      QueryPerformanceCounter(&count);
      return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
  }
#else
  #include <time.h>
  double timer_get_ms(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
  }
#endif

double timer_calc_gflops(int M, int N, int K, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    double ops = 2.0 * (double)M * (double)N * (double)K;
    return ops / (time_ms / 1000.0) / 1.0e9;
}

void timer_print_result(const char* mode, int N, double time_ms, double gflops,
                        int verified, double speedup) {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  %-40s║\n", mode);
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  Matrix Size : %4d × %4d               ║\n", N, N);
    printf("║  Time        : %10.3f ms              ║\n", time_ms);
    printf("║  GFLOPS      : %10.4f                 ║\n", gflops);
    printf("║  Speedup     : %10.2fx                ║\n", speedup);
    printf("║  Verified    : %s                       ║\n",
           verified ? "PASS ✓" : "FAIL ✗");
    printf("╚══════════════════════════════════════════╝\n\n");
}
