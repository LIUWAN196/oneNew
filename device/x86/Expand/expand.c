#include "expand.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs)
{
    RESHAPE_CONFIG_S * cfg = (RESHAPE_CONFIG_S*)(params[0].addr);

    float* input_ptr = (float*)(inputs[0].addr);
    float* output_ptr = (float*)(outputs[0].addr);

    OPERAND_S* in_tensor = (OPERAND_S*)(params[1].addr);
    OPERAND_S* out_tensor = (OPERAND_S*)(params[2].addr);

    int32_t in_elem_size = 1;
    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
        in_elem_size *= in_tensor->shapes[dim_i];
    }

    int32_t out_elem_size = 1;
    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
        out_elem_size *= out_tensor->shapes[dim_i];
    }

    int32_t ratio = out_elem_size / in_elem_size;
    for (int in_elem_i = 0; in_elem_i < in_elem_size; ++in_elem_i) {
        float elem = input_ptr[in_elem_i];
        for (int ratio_i = 0; ratio_i < ratio; ++ratio_i) {
            output_ptr[in_elem_i * ratio + ratio_i] = elem;
        }
    }

    return 0;
}