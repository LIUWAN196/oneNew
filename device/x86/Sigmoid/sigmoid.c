#include "sigmoid.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "math.h"
#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
    // printf("this is x86 sigmoid start\n");
    SIGMOID_CONFIG_S *cfg = (SIGMOID_CONFIG_S *) (params[0].addr);
//    // printf("yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    float *input0_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

    int32_t in_n = in0_tensor->shapes[0];
    int32_t in_c = in0_tensor->shapes[1];
    int32_t in_h = in0_tensor->shapes[2];
    int32_t in_w = in0_tensor->shapes[3];

    int32_t out_n = out_tensor->shapes[0];
    int32_t out_c = out_tensor->shapes[1];
    int32_t out_h = out_tensor->shapes[2];
    int32_t out_w = out_tensor->shapes[3];

    int32_t in_elem_size = 1;
    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
        in_elem_size *= in0_tensor->shapes[dim_i];
    }

//    int32_t in_elem_size = in0_tensor->shape.N * in0_tensor->shape.C * in0_tensor->shape.H * in0_tensor->shape.W;

//    int32_t in_elem_size = operand_elem_size(in0_tensor);
    // printf("aaaa\n");
    for (int i = 0; i < in_elem_size; ++i) {
//        // printf("bbb\n");

        output_ptr[i] = 1 / (1 + expf(-1 * input0_ptr[i]));
    }
    // printf("ccc\n");

    int32_t out_elem_size = 1;
    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
        out_elem_size *= out_tensor->shapes[dim_i];
    }

//    int32_t out_elem_size = out_tensor->shape.N * out_tensor->shape.C * out_tensor->shape.H * out_tensor->shape.W;
    // printf("this op name is %s\n", cfg->out_operand_name[0]);
    // printf("sigmoid op, n is %d, c is %d, h is %d, w is %d\n", out_tensor->shape.N, out_tensor->shape.C, out_tensor->shape.H, out_tensor->shape.W);
//    write_bin("yes", 12 * sizeof(float), output_ptr);
    // printf("sigmoid op, n is %d, c is %d, h is %d, w is %d\n", out_tensor->shape.N, out_tensor->shape.C, out_tensor->shape.H, out_tensor->shape.W);
    // write_bin(replace_char(cfg->out_operand_name[0]), out_elem_size * sizeof(float), output_ptr);
    // printf("this is x86 sigmoid end\n");

    return 0;
}