#include "concat.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONCAT_CONFIG_S *cfg = (CONCAT_CONFIG_S *) (params[0].addr);

    int axis = (int) cfg->axis;
    int in_tensor_num = (int) cfg->op_base_cfg.in_operand_num;
    // printf("=============in cancat op, the in_tensor_num is %d\n", in_tensor_num);

    float *output_ptr = (float *) (outputs[0].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[in_tensor_num + 1].addr);
    int32_t out_n = out_tensor->shapes[0];
    int32_t out_c = out_tensor->shapes[1];
    int32_t out_h = out_tensor->shapes[2];
    int32_t out_w = out_tensor->shapes[3];
    // printf("the fist tensor is %s, 2 tensor is %s\n", cfg->in_operand_name[0], cfg->in_operand_name[1]);
    // printf("concat op, n is %d, c is %d, h is %d, w is %d\n", out_tensor->shape.N, out_tensor->shape.C, out_tensor->shape.H, out_tensor->shape.W);

    int fill_st_offset = 0;
    for (int tenor_i = 0; tenor_i < in_tensor_num; ++tenor_i) {
        float *input_ptr = (float *) (inputs[tenor_i].addr);
        OPERAND_S *in_tensor = (OPERAND_S *) (params[tenor_i + 1].addr);
        int32_t in_n = in_tensor->shapes[0];
        int32_t in_c = in_tensor->shapes[1];
        int32_t in_h = in_tensor->shapes[2];
        int32_t in_w = in_tensor->shapes[3];
        // printf("concat op, n is %d, c is %d, h is %d, w is %d\n", in_tensor->shape.N, in_tensor->shape.C, in_tensor->shape.H, in_tensor->shape.W);

        int continue_arrange_elem_size;
        int repetitions;
        int stride;
        if(axis == 0){
            repetitions = 1;
            continue_arrange_elem_size = in_n * in_c * in_h * in_w;
            stride = out_n * in_c * in_h * in_w;
        } else if(axis == 1){
            repetitions = in_n;
            continue_arrange_elem_size = in_c * in_h * in_w;
            stride = out_c * in_h * in_w;
        } else if(axis == 2){
            repetitions = in_n * in_c;
            continue_arrange_elem_size = in_h * in_w;
            stride = out_h * in_w;
        } else if(axis == 3){
            repetitions = in_n * in_c * in_h;
            continue_arrange_elem_size = in_w;
            stride = out_w;
        }

        for (int rep_i = 0; rep_i < repetitions; ++rep_i) {
            for (int continue_i = 0; continue_i < continue_arrange_elem_size; ++continue_i) {
                output_ptr[rep_i * stride + fill_st_offset + continue_i] = input_ptr[rep_i * continue_arrange_elem_size + continue_i];
            }
        }

        fill_st_offset += continue_arrange_elem_size;
    }

//    int32_t out_elem_size = out_tensor->shape.N * out_tensor->shape.C * out_tensor->shape.H * out_tensor->shape.W;
    // write_bin(replace_char(cfg->out_operand_name[0]), out_elem_size * sizeof(float), output_ptr);

    return 0;
}