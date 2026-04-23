import json

file_path = r'x:\file\Projects\CUDA-MPI MatMul Engine\colab\matmul_gpu.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data['cells']:
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        if len(source) > 0 and source[0].startswith('%%writefile matrix.h'):
            cell['source'] = [
                '%%writefile matrix.h\n',
                '#ifndef MATRIX_H\n',
                '#define MATRIX_H\n',
                '#include <stdio.h>\n',
                '#include <stdlib.h>\n',
                '#include <math.h>\n',
                '#include <string.h>\n',
                '#define MATRIX_SEED 42\n',
                '#define VERIFY_TOLERANCE 1e-3f\n',
                '#ifdef __cplusplus\n',
                'extern "C" {\n',
                '#endif\n',
                'float* matrix_alloc(int rows, int cols);\n',
                'void matrix_free(float* M);\n',
                'void matrix_init_deterministic(float* M, int rows, int cols, int seed_offset);\n',
                'void matrix_zero(float* M, int rows, int cols);\n',
                'int matrix_verify(const float* expected, const float* actual, int rows, int cols);\n',
                'void matrix_print(const float* M, int rows, int cols, const char* name);\n',
                '#ifdef __cplusplus\n',
                '}\n',
                '#endif\n',
                '#endif'
            ]

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print('Patched successfully!')
