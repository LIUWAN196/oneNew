#include "pow.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include "stdint.h"
#include <omp.h>
#include "math.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs)
{
    POW_CONFIG_S* cfg = (POW_CONFIG_S*)(params[0].addr);
    float power_num = cfg->power_num;
//    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    float* input_ptr = (float*)(inputs[0].addr);
    float* output_ptr = (float*)(outputs[0].addr);

    OPERAND_S* in_tensor = (OPERAND_S*)(params[1].addr);
    int32_t in_elem_size = 1;
    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
        in_elem_size *= in_tensor->shapes[dim_i];
    }

    for (int i = 0; i < in_elem_size; ++i) {
        output_ptr[i] = powf(input_ptr[i], power_num);
    }

    return 0;
}