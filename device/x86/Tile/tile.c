#include "tile.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include "memory.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs)
{
    TILE_CONFIG_S * cfg = (TILE_CONFIG_S*)(params[0].addr);

    float* input_ptr = (float*)(inputs[0].addr);
    float* output_ptr = (float*)(outputs[0].addr);

    OPERAND_S* in_tensor = (OPERAND_S*)(params[1].addr);
    int32_t in_elem_size = 1;
    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
        in_elem_size *= in_tensor->shapes[dim_i];
    }

    memcpy(output_ptr, input_ptr, in_elem_size * sizeof(float));

    return 0;
}