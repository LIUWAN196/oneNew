#include "transpose.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "string.h"
#include "stdint.h"

int eval_perm_num4(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    TRANSPOSE_CONFIG_S *cfg = (TRANSPOSE_CONFIG_S *) (params[0].addr);

    int perm_num = (int) cfg->perm_num;
    int64_t* perm = cfg->perm;

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

    // perm_num == 4
    int32_t in_n = in_tensor->shapes[0];
    int32_t in_c = in_tensor->shapes[1];
    int32_t in_h = in_tensor->shapes[2];
    int32_t in_w = in_tensor->shapes[3];

    int32_t out_n = out_tensor->shapes[0];
    int32_t out_c = out_tensor->shapes[1];
    int32_t out_h = out_tensor->shapes[2];
    int32_t out_w = out_tensor->shapes[3];

    int32_t in_stride[8] = {in_c * in_h * in_w, in_h * in_w, in_w, 1};
    int32_t out_stride[8] = {out_c * out_h * out_w, out_h * out_w, out_w, 1};

    /*
     * 原始的 in stride 不能直接使用，如果直接使用的话，等于仅仅是将输入数据拷贝了一份到输出的地址上。
     * 因为这个算子是 transpose，需要跳跃着将输入数据给输出数据进行赋值。这里跳跃多少，就是通过 perm 来重新排序的
     */
    int32_t correct_in_stride[8];
    for (int i = 0; i < perm_num; ++i) {
        correct_in_stride[i] = in_stride[perm[i]];
    }

    float* cur_input_ptr, *cur_output_ptr;
    for (int outn_i = 0; outn_i < out_n; ++outn_i) {
        for (int outc_i = 0; outc_i < out_c; ++outc_i) {
            for (int outh_i = 0; outh_i < out_h; ++outh_i) {
                cur_input_ptr = input_ptr + outn_i * correct_in_stride[0] + outc_i * correct_in_stride[1] + outh_i * correct_in_stride[2];
                cur_output_ptr = output_ptr + outn_i * out_stride[0] + outc_i * out_stride[1] + outh_i * out_stride[2];
                for (int outw_i = 0; outw_i < out_w; ++outw_i) {
                    cur_output_ptr[outw_i * out_stride[3]] = cur_input_ptr[outw_i * correct_in_stride[3]];
                }
            }
        }
    }

    return 0;
}

int eval_perm_num6(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    TRANSPOSE_CONFIG_S *cfg = (TRANSPOSE_CONFIG_S *) (params[0].addr);

    int perm_num = (int) cfg->perm_num;
    int64_t* perm = cfg->perm;

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

    // perm_num == 4
    int32_t in_n = in_tensor->shapes[0];
    int32_t in_c = in_tensor->shapes[1];
    int32_t in_h = in_tensor->shapes[2];
    int32_t in_w = in_tensor->shapes[3];
    int32_t in_w1 = in_tensor->shapes[4];
    int32_t in_w2 = in_tensor->shapes[5];

    int32_t out_n = out_tensor->shapes[0];
    int32_t out_c = out_tensor->shapes[1];
    int32_t out_h = out_tensor->shapes[2];
    int32_t out_w = out_tensor->shapes[3];
    int32_t out_w1 = out_tensor->shapes[4];
    int32_t out_w2 = out_tensor->shapes[5];

    int32_t in_stride[8] = {in_c * in_h * in_w * in_w1 * in_w2, in_h * in_w * in_w1 * in_w2, in_w * in_w1 * in_w2, in_w1 * in_w2, in_w2, 1};
    int32_t out_stride[8] = {out_c * out_h * out_w * out_w1 * out_w2, out_h * out_w * out_w1 * out_w2, out_w * out_w1 * out_w2,  out_w1 * out_w2, out_w2, 1};

    /*
     * 原始的 in stride 不能直接使用，如果直接使用的话，等于仅仅是将输入数据拷贝了一份到输出的地址上。
     * 因为这个算子是 transpose，需要跳跃着将输入数据给输出数据进行赋值。这里跳跃多少，就是通过 perm 来重新排序的
     */
    int32_t correct_in_stride[8];
    for (int i = 0; i < perm_num; ++i) {
        correct_in_stride[i] = in_stride[perm[i]];
    }

    float* cur_input_ptr, *cur_output_ptr;
    for (int outn_i = 0; outn_i < out_n; ++outn_i) {
        for (int outc_i = 0; outc_i < out_c; ++outc_i) {
            for (int outh_i = 0; outh_i < out_h; ++outh_i) {
                for (int outw_i = 0; outw_i < out_w; ++outw_i) {
                    for (int outw1_i = 0; outw1_i < out_w1; ++outw1_i) {
                        cur_input_ptr = input_ptr + outn_i * correct_in_stride[0] + outc_i * correct_in_stride[1] + outh_i * correct_in_stride[2]
                                + outw_i * correct_in_stride[3] + outw1_i * correct_in_stride[4];
                        cur_output_ptr = output_ptr + outn_i * out_stride[0] + outc_i * out_stride[1] + outh_i * out_stride[2]
                                + outw_i * out_stride[3] + outw1_i * out_stride[4];
                        for (int outw2_i = 0; outw2_i < out_w2; ++outw2_i) {
                            cur_output_ptr[outw2_i * out_stride[5]] = cur_input_ptr[outw2_i * correct_in_stride[5]];
                        }
                    }
                }
            }
        }
    }

    return 0;
}

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

//    show_dev_input(params);
    TRANSPOSE_CONFIG_S *cfg = (TRANSPOSE_CONFIG_S *) (params[0].addr);

    if (cfg->perm_num <= 4) {
        eval_perm_num4(params, inputs, outputs);
    } else if (cfg->perm_num <= 6) {
        eval_perm_num6(params, inputs, outputs);
    }

    return 0;
}


