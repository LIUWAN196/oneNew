#include "softmax.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "math.h"
#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    SOFTMAX_CONFIG_S *cfg = (SOFTMAX_CONFIG_S *) (params[0].addr);
//    printf("yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs->addr);

    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

    int32_t in_n = in0_tensor->shape.N;
    int32_t in_c = in0_tensor->shape.C;
    int32_t in_h = in0_tensor->shape.H;
    int32_t in_w = in0_tensor->shape.W;

    int32_t out_n = out_tensor->shape.N;
    int32_t out_c = out_tensor->shape.C;
    int32_t out_h = out_tensor->shape.H;
    int32_t out_w = out_tensor->shape.W;

    int32_t in_elem_size = in0_tensor->shape.N * in0_tensor->shape.C * in0_tensor->shape.H * in0_tensor->shape.W;

    // step 1: find max
    float max_elem = 32768;
    for (int i = 0; i < in_elem_size; ++i) {
        max_elem = (input_ptr[i] < max_elem) ? input_ptr[i] : max_elem;
    }

    // step 2: calc exp(x - max(x))
    float total_exp = 0;
    for (int i = 0; i < in_elem_size; ++i) {
        output_ptr[i] = expf(output_ptr[i] - max_elem);
        total_exp += output_ptr[i];
    }

    // step 3: calc exp(x) / sum(exp(x))
    float total_exp_inv = 1.0f / total_exp;
    for (int i = 0; i < in_elem_size; ++i) {
        output_ptr[i] = output_ptr[i] * total_exp_inv;
    }

    int c = 101;
    return 0;
}