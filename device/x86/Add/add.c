#include "add.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval_dim_num_4(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

//    printf("this is x86 mul start\n");
    ADD_CONFIG_S *cfg = (ADD_CONFIG_S *) (params[0].addr);
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
    if (in0_tensor->dim_num_of_shapes == 0) {
        for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
            in_elem_size *= in1_tensor->shapes[dim_i];
        }
    } else if (in1_tensor->dim_num_of_shapes == 0) {
        for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
            in_elem_size *= in0_tensor->shapes[dim_i];
        }
    } else {    // 到这个分支说明两个输入的 shape 一样，拿其中任意一个来计算 elem size 即可
        for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
            in_elem_size *= in0_tensor->shapes[dim_i];
        }
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
                        = input0_ptr[inner_i] + input1_ptr[outc_i * psum_elem_size + inner_i];
            }
        }

        return 0;
    }

    if (in1_tensor->dim_num_of_shapes == 1) {
        int32_t psum_elem_size = in1_tensor->shapes[0];
        for (int outc_i = 0; outc_i < out_elem_size / psum_elem_size; ++outc_i) {
            for (int inner_i = 0; inner_i < psum_elem_size; ++inner_i) {
                output_ptr[outc_i * psum_elem_size + inner_i]
                        = input0_ptr[outc_i * psum_elem_size + inner_i] + input1_ptr[inner_i];
            }
        }
    } else if (in1_n * in1_c * in1_h * in1_w == 1) {   // in1 tensor should to be expand
        for (int elem_i = 0; elem_i < in_elem_size; ++elem_i) {
            output_ptr[elem_i] = input0_ptr[elem_i] + input1_ptr[0];
        }
    } else if (in_h == 1 && in_w == 1) {   // in0 tensor should to be expand
        for (int outc_i = 0; outc_i < out_c; ++outc_i) {
            for (int outhxw_i = 0; outhxw_i < out_h * out_w; ++outhxw_i) {
                output_ptr[outc_i * out_h * out_w + outhxw_i] = input0_ptr[outc_i] + input1_ptr[outc_i * out_h * out_w + outhxw_i];
            }
        }
    } else if (in1_h * in1_w == 1) {   // in1 tensor should to be expand
        for (int outc_i = 0; outc_i < out_c; ++outc_i) {
            for (int outhxw_i = 0; outhxw_i < out_h * out_w; ++outhxw_i) {
                output_ptr[outc_i * out_h * out_w + outhxw_i] = input0_ptr[outc_i * out_h * out_w + outhxw_i] + input1_ptr[outc_i];
            }
        }
    }  else if (in1_n == 1) {   // in1 tensor should to be expand
        for (int outter = 0; outter < out_n; ++outter) {
            for (int inner = 0; inner < out_h * out_w * out_c; ++inner) {
                output_ptr[outter * out_h * out_w * out_c + inner] = input0_ptr[outter * out_h * out_w * out_c + inner] + input1_ptr[inner];
            }
        }
    } else {   // in0 and in1 tensor have equal shape
        for (int i = 0; i < in_elem_size; ++i) {
            output_ptr[i] = input0_ptr[i] + input1_ptr[i];
        }
    }

    return 0;
}

int eval_dim_num_5(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    ADD_CONFIG_S *cfg = (ADD_CONFIG_S *) (params[0].addr);

    float *input0_ptr = (float *) (inputs[0].addr);
    float *input1_ptr = (float *) (inputs[1].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *in1_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

//    printf("==============\n");
//    for (int i = 0; i < SHAPE_LEN; ++i) {
//        printf("%d  ", in0_tensor->shapes[i]);
//    }
//    printf("\n");
//    for (int i = 0; i < SHAPE_LEN; ++i) {
//        printf("%d  ", in1_tensor->shapes[i]);
//    }
//    printf("\n");
//    for (int i = 0; i < SHAPE_LEN; ++i) {
//        printf("%d  ", out_tensor->shapes[i]);
//    }
//    printf("\n");

    int32_t expand_dims = -1;
    for (int i = 0; i < SHAPE_LEN; ++i) {
        if (in0_tensor->shapes[i] != in1_tensor->shapes[i]){
            expand_dims = i;
        }
    }

    int32_t outter_elem_size = 1;
    for (int i = 0; i < expand_dims; ++i) {
        outter_elem_size *= out_tensor->shapes[i];
    }

    int32_t expand_elem_size = out_tensor->shapes[expand_dims];

    int32_t inner_elem_size = 1;
    for (int i = expand_dims + 1; i < SHAPE_LEN; ++i) {
        inner_elem_size *= out_tensor->shapes[i];
    }
//    printf("expand_dims is %d, outter_elem_size is %d, expand_elem_size is %d, inner_elem_size is %d\n",
//           expand_dims, outter_elem_size, expand_elem_size, inner_elem_size);

    for (int outter_i = 0; outter_i < outter_elem_size; ++outter_i) {
        float *cur_ifmap0 = input0_ptr + outter_i * expand_elem_size * inner_elem_size;
        float *cur_ifmap1 = input1_ptr + outter_i * inner_elem_size;
        float *cur_ofmap = output_ptr + outter_i * expand_elem_size * inner_elem_size;
        for (int expand_i = 0; expand_i < expand_elem_size; ++expand_i) {
            float* tmp_cur_ifmap0 = cur_ifmap0 + expand_i * inner_elem_size;
            float* tmp_cur_ofmap = cur_ofmap + expand_i * inner_elem_size;
            for (int inner_i = 0; inner_i < inner_elem_size; ++inner_i) {
                tmp_cur_ofmap[inner_i] = tmp_cur_ifmap0[inner_i] + cur_ifmap1[inner_i];
            }
        }
    }
    return 0;
}

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    ADD_CONFIG_S *cfg = (ADD_CONFIG_S *) (params[0].addr);

    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);

    if (in0_tensor->dim_num_of_shapes <= 4) {
        eval_dim_num_4(params, inputs, outputs);
    } else if (in0_tensor->dim_num_of_shapes == 5) {
        eval_dim_num_5(params, inputs, outputs);
    }

    return 0;
}



