#include "conv.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

    float *input_ptr = (float *) (inputs[0].addr);
    float *weight_ptr = (float *) (inputs[1].addr);

    printf("weight conv is ===========================\n");
    for (int i = 0; i < 48; ++i) {
        printf("%f ", weight_ptr[i]);
    }
    printf("\n weight conv is ===========================\n");

    float *bias_ptr;
    if (cfg->has_bias){
        bias_ptr = (float *) (inputs[2].addr);
    }

    float *output_ptr = (float *) (outputs[0].addr);
    output_ptr[0] = 101.0f;
    output_ptr[12] = 11.0f;

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *weight_tensor = (OPERAND_S *) (params[3].addr);
    OPERAND_S *bias_tensor;

    if (cfg->has_bias){
        bias_tensor = (OPERAND_S *) (params[4].addr);
    }


//    weight_tensor->shape.N = 4;
//    weight_tensor->shape.C = 3;
//    weight_tensor->shape.H = 2;
//    weight_tensor->shape.W = 2;

    int32_t kernel_c = weight_tensor->shape.C;
    int32_t kernel_h = weight_tensor->shape.H;
    int32_t kernel_w = weight_tensor->shape.W;

//    int32_t weight_n = weight_tensor->shape.N;
//    int32_t weight_c = weight_tensor->shape.C;
//    int32_t weight_h = weight_tensor->shape.H;
//    int32_t weight_w = weight_tensor->shape.W;

    int32_t in_n = in_tensor->shape.N;
    int32_t in_c = in_tensor->shape.C;
    int32_t in_h = in_tensor->shape.H;
    int32_t in_w = in_tensor->shape.W;

    int32_t out_n = out_tensor->shape.N;
    int32_t out_c = out_tensor->shape.C;
    int32_t out_h = out_tensor->shape.H;
    int32_t out_w = out_tensor->shape.W;

    // loop params
    float *tmp_input_ptr;
    float *tmp_output_ptr;
    float *tmp_weight_ptr;
    float *cur_input_ptr;
    float *cur_output_ptr;
    for (int n_i = 0; n_i < out_n; ++n_i) {
        for (int c_i = 0; c_i < out_c; ++c_i) {
            for (int h_i = 0; h_i < out_h; ++h_i) {
                for (int w_i = 0; w_i < out_w; ++w_i) {
                    tmp_output_ptr = output_ptr + n_i * out_c * out_h * out_w + c_i * out_h * out_w + h_i * out_w + w_i;
                    tmp_input_ptr = input_ptr + h_i * out_w * stride_x * stride_y + w_i * stride_x;
                    tmp_weight_ptr = weight_ptr + c_i * kernel_c * kernel_h * kernel_w;
                    float psum = 0;
                    for (int k_c = 0; k_c < kernel_c; ++k_c) {
                        cur_input_ptr = tmp_input_ptr + k_c * in_h * in_w;
                        for (int k_h = 0; k_h < kernel_h; ++k_h) {
                            for (int k_w = 0; k_w < kernel_w; ++k_w) {
                                psum += cur_input_ptr[k_h * in_w + k_w] * tmp_weight_ptr[k_c * kernel_h * kernel_w + k_h * kernel_w + k_w];
                            }
                        }
                    }
                    tmp_output_ptr[0] = psum;
                }
            }
        }
    }


    printf("=========output of conv==================\n");
    for (int i = 0; i < 4 * 2 * 2; ++i) {
        if (i % 2 == 0){
            printf("\n");
        }
        printf("%f  ", output_ptr[i]);
    }

    int c = 101;
    return 0;
}