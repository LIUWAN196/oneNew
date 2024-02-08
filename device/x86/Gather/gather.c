#include "gather.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "string.h"
#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    GATHER_CONFIG_S *cfg = (GATHER_CONFIG_S *) (params[0].addr);
//    printf("yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    int32_t axis = cfg->axis;
    int32_t indices = cfg->indices;

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

    int32_t outer_elem_size = 1;
    for (int dim_i = 0; dim_i < axis; ++dim_i) {
        outer_elem_size *= in_tensor->shapes[dim_i];
    }

    int32_t inner_elem_size = 1;
    for (int dim_i = axis + 1; dim_i < SHAPE_LEN; ++dim_i) {
        inner_elem_size *= in_tensor->shapes[dim_i];
    }

    float *ifmap_useful_st = input_ptr + indices * in_tensor->shapes[axis] * inner_elem_size;
    memcpy(output_ptr, ifmap_useful_st, inner_elem_size * sizeof(float));

    return 0;
}