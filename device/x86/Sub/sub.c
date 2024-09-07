#include "sub.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
//    show_dev_input(params);
//    printf("this is x86 mul start\n");
    SUB_CONFIG_S *cfg = (SUB_CONFIG_S *) (params[0].addr);
//    printf("this is device, the op type is mul\n");

    float *input0_ptr = (float *) (inputs[0].addr);
    float *input1_ptr = (float *) (inputs[1].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *in1_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

    int32_t in_n = in0_tensor->shapes[0];
    int32_t in_c = in0_tensor->shapes[1];
    int32_t in_h = in0_tensor->shapes[2];
    int32_t in_w = in0_tensor->shapes[3];

    int32_t in1_n = in1_tensor->shapes[0];
    int32_t in1_c = in1_tensor->shapes[1];
    int32_t in1_h = in1_tensor->shapes[2];
    int32_t in1_w = in1_tensor->shapes[3];

    int32_t out_n = out_tensor->shapes[0];
    int32_t out_c = out_tensor->shapes[1];
    int32_t out_h = out_tensor->shapes[2];
    int32_t out_w = out_tensor->shapes[3];

    int32_t in_elem_size = 1;
    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
        in_elem_size *= in0_tensor->shapes[dim_i];
    }

    int32_t out_elem_size = 1;
    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
        out_elem_size *= out_tensor->shapes[dim_i];
    }

    if (in0_tensor->dim_num_of_shapes == 1) {
        int32_t psum_elem_size = in0_tensor->shapes[0];
        for (int outc_i = 0; outc_i < out_elem_size / psum_elem_size; ++outc_i) {
            for (int inner_i = 0; inner_i < psum_elem_size; ++inner_i) {
                output_ptr[outc_i * psum_elem_size + inner_i]
                        = input0_ptr[inner_i] - input1_ptr[outc_i * psum_elem_size + inner_i];
            }
        }

        return 0;
    }

    if (in1_n * in1_c * in1_h * in1_w == 1) {   // in1 tensor should to be expand
        for (int elem_i = 0; elem_i < in_elem_size; ++elem_i) {
            output_ptr[elem_i] = input0_ptr[elem_i] - input1_ptr[0];
        }
    } else if (in_h == 1 && in_w == 1) {   // in0 tensor should to be expand
        for (int outc_i = 0; outc_i < out_n * out_c; ++outc_i) {
            for (int outhxw_i = 0; outhxw_i < out_h * out_w; ++outhxw_i) {
                output_ptr[outc_i * out_h * out_w + outhxw_i] = input0_ptr[outc_i] - input1_ptr[outc_i * out_h * out_w + outhxw_i];
            }
        }
    } else if (in1_h * in1_w == 1) {   // in1 tensor should to be expand
        for (int outc_i = 0; outc_i < out_n * out_c; ++outc_i) {
            for (int outhxw_i = 0; outhxw_i < out_h * out_w; ++outhxw_i) {
                output_ptr[outc_i * out_h * out_w + outhxw_i] = input0_ptr[outc_i * out_h * out_w + outhxw_i] - input1_ptr[outc_i];
            }
        }
    } else if (in1_w == 1) {   // in1 tensor should to be expand
        for (int outc_i = 0; outc_i < out_n * out_c * out_h; ++outc_i) {
            for (int outw_i = 0; outw_i < out_w; ++outw_i) {
                output_ptr[outc_i * out_w + outw_i] = input0_ptr[outc_i * out_w + outw_i] - input1_ptr[outc_i];
            }
        }
    } else {   // in0 and in1 tensor have equal shape
        for (int i = 0; i < in_elem_size; ++i) {
            output_ptr[i] = input0_ptr[i] - input1_ptr[i];
        }
    }

    return 0;
}