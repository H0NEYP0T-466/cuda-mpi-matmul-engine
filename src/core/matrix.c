/*
 * matrix.c — Matrix utility implementations
 */

#include "matrix.h"

float* matrix_alloc(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "[ERROR] Invalid matrix dimensions: %d×%d\n", rows, cols);
        return NULL;
    }
    float* M = (float*)calloc((size_t)rows * cols, sizeof(float));
    if (!M) {
        fprintf(stderr, "[ERROR] Failed to allocate %d×%d matrix (%zu bytes)\n",
                rows, cols, (size_t)rows * cols * sizeof(float));
    }
    return M;
}

void matrix_free(float* M) {
    if (M) free(M);
}

void matrix_init_deterministic(float* M, int rows, int cols, int seed_offset) {
    /*
     * Use MATRIX_SEED + seed_offset so that matrix A (offset=0) and
     * matrix B (offset=1) produce different values even for same dimensions.
     * srand() is called here to ensure full reproducibility regardless
     * of any prior random state.
     */
    srand(MATRIX_SEED + seed_offset);
    for (int i = 0; i < rows * cols; i++) {
        M[i] = (float)rand() / (float)RAND_MAX * 10.0f;  /* [0, 10) */
    }
}

void matrix_zero(float* M, int rows, int cols) {
    memset(M, 0, (size_t)rows * cols * sizeof(float));
}

int matrix_verify(const float* expected, const float* actual, int rows, int cols) {
    int pass = 1;
    int first_fail_printed = 0;
    int fail_count = 0;

    for (int i = 0; i < rows * cols; i++) {
        float diff = fabsf(expected[i] - actual[i]);
        if (diff > VERIFY_TOLERANCE) {
            fail_count++;
            if (!first_fail_printed) {
                int r = i / cols;
                int c = i % cols;
                fprintf(stderr, "[VERIFY FAIL] First mismatch at (%d,%d): "
                        "expected=%.6f, actual=%.6f, diff=%.6f\n",
                        r, c, expected[i], actual[i], diff);
                first_fail_printed = 1;
            }
            pass = 0;
        }
    }

    if (!pass) {
        fprintf(stderr, "[VERIFY FAIL] Total mismatches: %d / %d elements\n",
                fail_count, rows * cols);
    }

    return pass;
}

void matrix_print(const float* M, int rows, int cols, const char* name) {
    printf("\n=== %s (%d×%d) ===\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("  [");
        for (int j = 0; j < cols; j++) {
            printf("%8.3f", M[i * cols + j]);
            if (j < cols - 1) printf(", ");
        }
        printf("]\n");
    }
    printf("\n");
}

void matrix_read_manual(float* M, int rows, int cols, const char* name) {
    printf("\nEnter elements for %s (%d×%d), row by row:\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("  Row %d (%d values): ", i, cols);
        for (int j = 0; j < cols; j++) {
            if (scanf("%f", &M[i * cols + j]) != 1) {
                fprintf(stderr, "[ERROR] Invalid input at (%d,%d)\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }
}
