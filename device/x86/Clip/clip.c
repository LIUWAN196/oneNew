#include "clip.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include "stdint.h"
#include <omp.h>

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs)
{
    CLIP_CONFIG_S* cfg = (CLIP_CONFIG_S*)(params[0].addr);

    float* input_ptr = (float*)(inputs[0].addr);
    float* output_ptr = (float*)(outputs[0].addr);

    OPERAND_S* in_tensor = (OPERAND_S*)(params[1].addr);
    int32_t in_elem_size = 1;
    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
        in_elem_size *= in_tensor->shapes[dim_i];
    }

    float min = cfg->min;
    float max = cfg->max;
    float tmp;
    for (int i = 0; i < in_elem_size; ++i) {
        tmp = (input_ptr[i] > min) ? input_ptr[i] : min;
        output_ptr[i] = (tmp < max) ? tmp : max;
    }

    return 0;
}