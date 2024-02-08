#include "split.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "string.h"
#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    SPLIT_CONFIG_S *cfg = (SPLIT_CONFIG_S *) (params[0].addr);

    if (strcmp(cfg->op_base_cfg.op_name, "/model.10/attn/Split") == 0) {
        int a = 101;
    }

    int axis = (int) cfg->axis;
    int out_tensor_num = (int) cfg->op_base_cfg.out_operand_num;

    float *input_ptr = (float *) (inputs[0].addr);
    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);


    int32_t axis_total_outer = 1;
    int32_t axis_total_inner = 1;
    if (axis == -1) {
        for (int axis_i = 0; axis_i < in_tensor->dim_num_of_shapes - 1; ++axis_i) {
            axis_total_outer *= in_tensor->shapes[axis_i];
        }
        axis = in_tensor->dim_num_of_shapes - 1;
    } else {
        for (int axis_i = 0; axis_i < axis; ++axis_i) {
            axis_total_outer *= in_tensor->shapes[axis_i];
        }

        for (int axis_i = axis + 1; axis_i < SHAPE_LEN; ++axis_i) {
            axis_total_inner *= in_tensor->shapes[axis_i];
        }
    }

    for (int out_loop_i = 0; out_loop_i < axis_total_outer; ++out_loop_i) {
        float *cur_ifmap_ptr =  input_ptr + out_loop_i * in_tensor->shapes[axis] * axis_total_inner;
        for (int ofmap_tensor_i = 0; ofmap_tensor_i < out_tensor_num; ++ofmap_tensor_i) {
            float *cur_ofmap_ptr = (float *) ((float *)outputs[ofmap_tensor_i].addr + out_loop_i * cfg->split[ofmap_tensor_i] * axis_total_inner);
            memcpy(cur_ofmap_ptr, cur_ifmap_ptr, cfg->split[ofmap_tensor_i] * axis_total_inner * sizeof(float));
            cur_ifmap_ptr += cfg->split[ofmap_tensor_i] * axis_total_inner;
        }
    }

    return 0;
}


