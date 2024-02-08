#include "argmax.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "math.h"
#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    ARGMAX_CONFIG_S *cfg = (ARGMAX_CONFIG_S *) (params[0].addr);
//    printf("yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    float *input_ptr = (float *) (inputs[0].addr);
    int32_t *output_ptr = (int32_t *) (outputs[0].addr);

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


    int32_t in_elem_size = 1000;

    // step 1: find max and the idx
    float max_elem = -32768;
    int32_t max_idx = 0;
    int32_t topk = cfg->topk;

    for (int k_i = 0; k_i < topk; ++k_i) {
        max_elem = -32768;
        for (int32_t i = 0; i < in_elem_size; ++i) {
            max_idx = (input_ptr[i] > max_elem) ? i : max_idx;
            max_elem = (input_ptr[i] > max_elem) ? input_ptr[i] : max_elem;
        }
        output_ptr[k_i] = max_idx;
        printf("argmax num.%d is %f\n", k_i, max_elem);
        input_ptr[max_idx] = -32768;
    }

    int c = 101;
    return 0;
}