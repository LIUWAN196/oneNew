#include "gelu.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include "stdint.h"
#include <omp.h>

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs)
{
    GELU_CONFIG_S* cfg = (GELU_CONFIG_S*)(params[0].addr);

    float* input_ptr = (float*)(inputs[0].addr);
    float* output_ptr = (float*)(outputs[0].addr);

    OPERAND_S* in_tensor = (OPERAND_S*)(params[1].addr);
    int32_t in_elem_size = 1;
    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
        in_elem_size *= in_tensor->shapes[dim_i];
    }

    float *gelu_lut = (float *) (inputs[BUF_MAXNUM - 1].addr);
    float single_limit = 8.0f;
    float inv_single_limit = -1 * single_limit;
    float step = 1.0f / 512;
    float inv_step = 1.0f / step;
    for (int elem_i = 0; elem_i < in_elem_size; ++elem_i) {
        float cur_val = input_ptr[elem_i];
        if (cur_val < inv_single_limit) {
            output_ptr[elem_i] = 0;
        } else if (cur_val >= single_limit) {
            output_ptr[elem_i] = cur_val;
        } else {
            // cur_val >= inv_single_limit && cur_val < single_limit
            int idx = (int)((cur_val + single_limit) * inv_step);
            output_ptr[elem_i] = gelu_lut[idx];
        }
    }

    return 0;
}

