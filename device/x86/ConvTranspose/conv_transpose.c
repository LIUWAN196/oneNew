#include "conv_transpose.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "stdint.h"
#include "string.h"
#include <immintrin.h>
#include <omp.h>
#include "math.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
//    show_dev_input(params);

    CONV_TRANSPOSE_CONFIG_S *cfg = (CONV_TRANSPOSE_CONFIG_S *) (params[0].addr);

    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

    float *input_ptr = (float *) (inputs[0].addr);
    float *weight_ptr = (float *) (inputs[1].addr);

    float *bias_ptr;
    if (cfg->has_bias) {
        bias_ptr = (float *) (inputs[2].addr);
    }

    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *weight_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *bias_tensor = (OPERAND_S *) (params[3].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[4].addr);

    int32_t kernel_c = weight_tensor->shapes[1];
    int32_t kernel_h = weight_tensor->shapes[2];
    int32_t kernel_w = weight_tensor->shapes[3];

    int32_t in_n = in_tensor->shapes[0];
    int32_t in_c = in_tensor->shapes[1];
    int32_t in_h = in_tensor->shapes[2];
    int32_t in_w = in_tensor->shapes[3];

    int32_t out_n = out_tensor->shapes[0];
    int32_t out_c = out_tensor->shapes[1];
    int32_t out_h = out_tensor->shapes[2];
    int32_t out_w = out_tensor->shapes[3];

#pragma omp parallel for num_threads(THREADS_NUM)
    for (int outc_i = 0; outc_i < out_c; ++outc_i) {
        float * cur_input_ptr = input_ptr;
        float * cur_output_ptr = output_ptr + outc_i * out_h * out_w;
        // 特别注意： 转置卷积很坑的一点是：假如 weight 的排列为 nchw = 37 53 4 7。那么在做累加的时候，是卷积
        // 核 4 7 在 H W 平面上去计算同一个点，然后再在 37 这个纵深上去累加，最后把这 4 x 7 个结果累加到 output 上
        float * cur_weight_ptr = weight_ptr + outc_i * kernel_h * kernel_w;
        for (int inh_i = 0; inh_i < in_h; ++inh_i) {
            for (int inw_i = 0; inw_i < in_w; ++inw_i) {
                float psum0 = 0.f;
                float psum1 = 0.f;
                float psum2 = 0.f;
                float psum3 = 0.f;
                float* ifmap_point_ptr = cur_input_ptr + inh_i * in_w + inw_i;
#pragma unroll 8
                for (int inc_i = 0; inc_i < in_c; ++inc_i) {
                    psum0 += ifmap_point_ptr[inc_i * in_h * in_w] * cur_weight_ptr[inc_i * out_c * kernel_h * kernel_w + 0];
                    psum1 += ifmap_point_ptr[inc_i * in_h * in_w] * cur_weight_ptr[inc_i * out_c * kernel_h * kernel_w + 1];
                    psum2 += ifmap_point_ptr[inc_i * in_h * in_w] * cur_weight_ptr[inc_i * out_c * kernel_h * kernel_w + 2];
                    psum3 += ifmap_point_ptr[inc_i * in_h * in_w] * cur_weight_ptr[inc_i * out_c * kernel_h * kernel_w + 3];
                }

                // save output
                cur_output_ptr[(inh_i * 2 + 0) * out_w + (inw_i * 2 + 0)] = psum0 + bias_ptr[outc_i];
                cur_output_ptr[(inh_i * 2 + 0) * out_w + (inw_i * 2 + 1)] = psum1 + bias_ptr[outc_i];
                cur_output_ptr[(inh_i * 2 + 1) * out_w + (inw_i * 2 + 0)] = psum2 + bias_ptr[outc_i];
                cur_output_ptr[(inh_i * 2 + 1) * out_w + (inw_i * 2 + 1)] = psum3 + bias_ptr[outc_i];
            }
        }
    }

    return 0;

}

