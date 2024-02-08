#include "reduce_mean.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    REDUCE_MEAN_CONFIG_S *cfg = (REDUCE_MEAN_CONFIG_S *) (params[0].addr);
//    printf("yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);


    if (cfg->axes_num == 1) {
        int32_t axes = cfg->axes[0];

        float *input_ptr = (float *) (inputs[0].addr);
        float *output_ptr = (float *) (outputs[0].addr);

        OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
        OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

        if (axes != -1) {
            printf("cur, just support axes == -1, in reduce mean\n");
        }

        int32_t outer_elem_size = 1;
        for (int dim_i = 0; dim_i < in_tensor->dim_num_of_shapes - 1; ++dim_i) {
            outer_elem_size *= in_tensor->shapes[dim_i];
        }
        int32_t inner_elem_size = in_tensor->shapes[in_tensor->dim_num_of_shapes - 1];

        float coeff = 1.0f / inner_elem_size;
        for (int outer_i = 0; outer_i < outer_elem_size; ++outer_i) {
            float *cur_ifmap_ptr = input_ptr + outer_i * inner_elem_size;
            float *cur_ofmap_ptr = output_ptr;
            float psum = 0;
            for (int inner_i = 0; inner_i < inner_elem_size; ++inner_i) {
                psum += cur_ifmap_ptr[inner_i];
            }
            cur_ofmap_ptr[outer_i] = psum * coeff;
        }
    } else if (cfg->axes[0] == -2 && cfg->axes[1] == -1) {
        // reduce the last dims -1 and -2
        float *input_ptr = (float *) (inputs[0].addr);
        float *output_ptr = (float *) (outputs[0].addr);

        OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
        OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

//        printf("this is reduce mean\n");
//        for (int i = 0; i < SHAPE_LEN; ++i) {
//            printf("%d  ", in_tensor->shapes[i]);
//        }
//        printf("\n");
//        for (int i = 0; i < SHAPE_LEN; ++i) {
//            printf("%d  ", out_tensor->shapes[i]);
//        }
//        printf("\n");
//        printf("cfg->axes_num is %d\n", cfg->axes_num);


        int32_t psum_elem_size = 1;
        for (int i = in_tensor->dim_num_of_shapes - 2; i < in_tensor->dim_num_of_shapes; ++i) {
            psum_elem_size *= in_tensor->shapes[i];
        }
        float inv_ratio = 1.0f / (psum_elem_size * 1.0f);

        int32_t outter_elem_size = 1;
        for (int i = 0; i < in_tensor->dim_num_of_shapes - 2; ++i) {
            outter_elem_size *= in_tensor->shapes[i];
        }

        for (int i = 0; i < outter_elem_size; ++i) {
            float psum = 0;
            for (int j = 0; j < psum_elem_size; ++j) {
                psum += input_ptr[i * psum_elem_size + j];
            }
            output_ptr[i] = psum * inv_ratio;
        }
    } else {
        // cfg->axes_num != 1
        int32_t axes = cfg->axes[0];

        float *input_ptr = (float *) (inputs[0].addr);
        float *output_ptr = (float *) (outputs[0].addr);

        OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
        OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

        int32_t psum_elem_size = 1;
        int32_t max_axis = 0;
        for (int axis_idx = 0; axis_idx < cfg->axes_num; ++axis_idx) {
            psum_elem_size *= in_tensor->shapes[cfg->axes[axis_idx]];
            max_axis = cfg->axes[axis_idx] > max_axis ? cfg->axes[axis_idx] : max_axis;
        }
        float inv_ratio = 1.0f / (psum_elem_size * 1.0f);

        int32_t inner_elem_size = 1;
        for (int i = max_axis + 1; i < SHAPE_LEN; ++i) {
            inner_elem_size *= in_tensor->shapes[i];
        }

        for (int i = 0; i < inner_elem_size; ++i) {
            float psum = 0;
            for (int j = 0; j < psum_elem_size; ++j) {
                psum += input_ptr[j * inner_elem_size + i];
            }
            output_ptr[i] = psum * inv_ratio;
        }

        int a = 101;
    }


    return 0;
}