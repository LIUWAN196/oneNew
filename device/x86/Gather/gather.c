#include "gather.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "string.h"
#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

//    show_dev_input(params);
    GATHER_CONFIG_S *cfg = (GATHER_CONFIG_S *) (params[0].addr);
//    printf("yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

//    if (strcmp(cfg->op_base_cfg.op_name, "/model.28/Gather_10") == 0) {
//        show_dev_input(params);
//    }
    if (cfg->indices_from_ifmap == FALSE) {
        int32_t axis = cfg->axis;
        int32_t indices = cfg->indices;

        float *input_ptr = (float *) (inputs[0].addr);
        float *output_ptr = (float *) (outputs[0].addr);

        OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);

        int32_t outer_elem_size = 1;
        for (int dim_i = 0; dim_i < axis; ++dim_i) {
            outer_elem_size *= in_tensor->shapes[dim_i];
        }

        int32_t inner_elem_size = 1;
        for (int dim_i = axis + 1; dim_i < SHAPE_LEN; ++dim_i) {
            inner_elem_size *= in_tensor->shapes[dim_i];
        }

        for (int i = 0; i < outer_elem_size; ++i) {
            float *ifmap_useful_st = input_ptr + i * in_tensor->shapes[axis] * inner_elem_size + indices * inner_elem_size;
            float * cur_ofmap = output_ptr + i * inner_elem_size;
            memcpy(cur_ofmap, ifmap_useful_st, inner_elem_size * sizeof(float));
        }

//        float *ifmap_useful_st = input_ptr + indices * in_tensor->shapes[axis] * inner_elem_size;
//        memcpy(output_ptr, ifmap_useful_st, inner_elem_size * sizeof(float));
    } else {
        float *input_ptr = (float *) (inputs[0].addr);
        int32_t *indices_ptr = (int32_t *) (inputs[1].addr);
        float *output_ptr = (float *) (outputs[0].addr);

        OPERAND_S *data_tensor = (OPERAND_S *) (params[1].addr);
        OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

        int32_t inner_elem_size = 1;
        for (int dim_i = SHAPE_LEN - 1; dim_i >= 0; --dim_i) {
            if (data_tensor->shapes[dim_i] != 1) {
                inner_elem_size = data_tensor->shapes[dim_i];
//                LOG_MSG("data_tensor->shapes[dim_i] is %d\n", data_tensor->shapes[dim_i]);
                break;
            }
        }

        int32_t out_elem_size = 1;
        for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
            out_elem_size *= out_tensor->shapes[dim_i];
        }
        out_elem_size /= inner_elem_size;

        float *cur_output_ptr, *cur_input_ptr;
        for (int out_i = 0; out_i < out_elem_size; ++out_i) {
//            LOG_MSG("out_i is %d, out_elem_size is %d, inner_elem_size is %d", out_i, out_elem_size, inner_elem_size);
            cur_output_ptr = output_ptr + out_i * inner_elem_size;
            cur_input_ptr = input_ptr + indices_ptr[out_i] * inner_elem_size;
            memcpy(cur_output_ptr, cur_input_ptr, inner_elem_size * sizeof(float));
        }
    }

//    LOG_ERR("end of first gather\n");
    return 0;
}