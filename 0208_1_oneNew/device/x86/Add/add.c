#include "add.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    ADD_CONFIG_S *cfg = (ADD_CONFIG_S *) (params[0].addr);
//    printf("yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    float *input0_ptr = (float *) (inputs[0].addr);
    float *input1_ptr = (float *) (inputs[1].addr);
    float *output_ptr = (float *) (outputs->addr);

    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *in1_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

    int32_t in_n = in0_tensor->shape.N;
    int32_t in_c = in0_tensor->shape.C;
    int32_t in_h = in0_tensor->shape.H;
    int32_t in_w = in0_tensor->shape.W;

    int32_t out_n = out_tensor->shape.N;
    int32_t out_c = out_tensor->shape.C;
    int32_t out_h = out_tensor->shape.H;
    int32_t out_w = out_tensor->shape.W;

    int32_t in_elem_size = in0_tensor->shape.N * in0_tensor->shape.C * in0_tensor->shape.H * in0_tensor->shape.W;

    for (int i = 0; i < in_elem_size; ++i) {
        output_ptr[i] = input0_ptr[i] + input1_ptr[i];
    }

    int c = 101;
    return 0;
}