#include "gemm.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs)
{
    GEMM_CONFIG_S* cfg = (GEMM_CONFIG_S*)(params[0].addr);
//    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *weight_tensor = (OPERAND_S *) (params[3].addr);
    OPERAND_S *bias_tensor = (OPERAND_S *) (params[4].addr);

    float* input_ptr = (float*)(inputs[0].addr);
    float* weight_ptr = (float*)(inputs[1].addr);
    float* bias_ptr = (float*)(inputs[2].addr);
    float* output_ptr = (float*)(outputs[0].addr);

    int32_t in_elem_size = in_tensor->shape.N * in_tensor->shape.C * in_tensor->shape.H * in_tensor->shape.W;
    int32_t out_elem_size = out_tensor->shape.N * out_tensor->shape.C * out_tensor->shape.H * out_tensor->shape.W;

    for (int out_i = 0; out_i < out_elem_size; ++out_i) {
        float psum = 0;
        for (int in_i = 0; in_i < in_elem_size; ++in_i) {
            psum += input_ptr[in_i] * weight_ptr[out_i * in_elem_size + in_i];
        }
        output_ptr[out_i] = psum + bias_ptr[out_i];
    }

//    // write_bin(replace_char(cfg->out_operand_name[0]), out_elem_size * sizeof(float), output_ptr);


    return 0;
}