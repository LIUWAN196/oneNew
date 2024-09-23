#include "conv.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "pad_conv.h"
#include "stdint.h"
#include "string.h"
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

typedef struct {
    int32_t FH, FW, stride_h, stride_w;
    int32_t in_c, in_h, in_w;
    int32_t out_h, out_w;
    int32_t GEMM_K;
} ARGS_S;

int eval_1x1j1(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
//    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    float *input_ptr = (float *) (inputs[0].addr);
    float *weight_ptr = (float *) (inputs[1].addr);

    float *bias_ptr;
    if (cfg->has_bias){
        bias_ptr = (float *) (inputs[2].addr);
    }

    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *weight_tensor = (OPERAND_S *) (params[3].addr);
    OPERAND_S *bias_tensor;

    if (cfg->has_bias){
        bias_tensor = (OPERAND_S *) (params[4].addr);
    }

    int32_t in_n = in_tensor->shape.N;
    int32_t in_c = in_tensor->shape.C;
    int32_t in_h = in_tensor->shape.H;
    int32_t in_w = in_tensor->shape.W;

    int32_t out_n = out_tensor->shape.N;
    int32_t out_c = out_tensor->shape.C;
    int32_t out_h = out_tensor->shape.H;
    int32_t out_w = out_tensor->shape.W;

    void *src_pad_ptr;
    if (cfg->pads[0] != 0){
        // do pad
        src_pad_ptr = aligned_alloc(32, in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(float ));
        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv((char*)src_pad_ptr, (char*)input_ptr, (OPERAND_S *)in_tensor, (PAD_INNER_CONFIG_S *)&pad_cfg);

        input_ptr = (float *)src_pad_ptr;
    }

    // loop params
    register float psum0_reg, psum1_reg, psum2_reg, psum3_reg;

    // gemm
    float psum;
    float *w0_ptr, *w1_ptr, *w2_ptr, *w3_ptr;
    register float in_data;

//    for (int hw_i = 0; hw_i < out_h * out_w; ++hw_i) {
//        for (int outc_i = 0; outc_i < out_c; ++outc_i) {
//            psum_reg = 0;
//            w_ptr = weight_ptr + outc_i * in_c;
//            for (int inc_i = 0; inc_i < in_c; ++inc_i) {
//                psum_reg += *w_ptr++ * input_ptr[inc_i * in_h * in_w + hw_i];
//            }
//            output_ptr[outc_i * in_h * in_w + hw_i] = psum_reg + bias_ptr[outc_i];
//        }
//    }

// todo: mybe the out_c % 4 != 0
    for (int hw_i = 0; hw_i < out_h * out_w; ++hw_i) {
        for (int outc_i = 0; outc_i < out_c; outc_i += 4) {  // todo: mybe the out_c % 4 != 0
            psum0_reg = 0, psum1_reg = 0, psum2_reg = 0, psum3_reg = 0;
            w0_ptr = weight_ptr + outc_i * in_c;
            w1_ptr = w0_ptr + in_c;
            w2_ptr = w0_ptr + 2 * in_c;
            w3_ptr = w0_ptr + 3 * in_c;
            for (int inc_i = 0; inc_i < in_c; ++inc_i) {
                in_data = input_ptr[inc_i * in_h * in_w + hw_i];
                psum0_reg += *w0_ptr++ * in_data;
                psum1_reg += *w1_ptr++ * in_data;
                psum2_reg += *w2_ptr++ * in_data;
                psum3_reg += *w3_ptr++ * in_data;
            }
            output_ptr[outc_i * in_h * in_w + hw_i] = psum0_reg + bias_ptr[outc_i];
            output_ptr[(outc_i + 1) * in_h * in_w + hw_i] = psum1_reg + bias_ptr[outc_i + 1];
            output_ptr[(outc_i + 2) * in_h * in_w + hw_i] = psum2_reg + bias_ptr[outc_i + 2];
            output_ptr[(outc_i + 3) * in_h * in_w + hw_i] = psum3_reg + bias_ptr[outc_i + 3];
        }
    }


    return 0;
}

int eval_mxn(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
//    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

    float *input_ptr = (float *) (inputs[0].addr);
    float *weight_ptr = (float *) (inputs[1].addr);

    float *bias_ptr;
    if (cfg->has_bias){
        bias_ptr = (float *) (inputs[2].addr);
    }

    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *weight_tensor = (OPERAND_S *) (params[3].addr);
    OPERAND_S *bias_tensor;

    if (cfg->has_bias){
        bias_tensor = (OPERAND_S *) (params[4].addr);
    }

    int32_t kernel_c = weight_tensor->shape.C;
    int32_t kernel_h = weight_tensor->shape.H;
    int32_t kernel_w = weight_tensor->shape.W;

    int32_t in_n = in_tensor->shape.N;
    int32_t in_c = in_tensor->shape.C;
    int32_t in_h = in_tensor->shape.H;
    int32_t in_w = in_tensor->shape.W;

    int32_t out_n = out_tensor->shape.N;
    int32_t out_c = out_tensor->shape.C;
    int32_t out_h = out_tensor->shape.H;
    int32_t out_w = out_tensor->shape.W;

    void *src_pad_ptr;
    if (cfg->pads[0] != 0){
        // do pad
        src_pad_ptr = aligned_alloc(32, in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(float ));
        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv((char*)src_pad_ptr, (char*)input_ptr, (OPERAND_S *)in_tensor, (PAD_INNER_CONFIG_S *)&pad_cfg);

        input_ptr = (float *)src_pad_ptr;
    }

    // loop params
    float *tmp_input_ptr;
    float *cur_input_ptr;

    float *tmp0_weight_ptr;
    float *tmp1_weight_ptr;
    float *tmp2_weight_ptr;
    float *tmp3_weight_ptr;
    float *tmp4_weight_ptr;
    float *tmp5_weight_ptr;
    float *tmp6_weight_ptr;
    float *tmp7_weight_ptr;

    float *cur0_weight_ptr;
    float *cur1_weight_ptr;
    float *cur2_weight_ptr;
    float *cur3_weight_ptr;
    float *cur4_weight_ptr;
    float *cur5_weight_ptr;
    float *cur6_weight_ptr;
    float *cur7_weight_ptr;

    float *tmp_output_ptr;

    float *output_batch_ptr;

    float *cur_output_ptr;

    register float psum0, psum1, psum2, psum3, psum4, psum5, psum6, psum7;

    register float in_data;

    for (int n_i = 0; n_i < out_n; ++n_i) {
        output_batch_ptr = output_ptr + n_i * out_c * out_h * out_w;
        for (int c_i = 0; c_i < out_c; c_i += 8) {  // todo: maybe the out_c % 8 != 0
            for (int h_i = 0; h_i < out_h; ++h_i) {
                for (int w_i = 0; w_i < out_w; ++w_i) {
                    tmp_input_ptr = input_ptr + h_i * stride_y * in_w + w_i * stride_x;
                    tmp0_weight_ptr = weight_ptr + c_i * kernel_c * kernel_h * kernel_w;
                    tmp1_weight_ptr = tmp0_weight_ptr + 1 * kernel_c * kernel_h * kernel_w;
                    tmp2_weight_ptr = tmp0_weight_ptr + 2 * kernel_c * kernel_h * kernel_w;
                    tmp3_weight_ptr = tmp0_weight_ptr + 3 * kernel_c * kernel_h * kernel_w;
                    tmp4_weight_ptr = tmp0_weight_ptr + 4 * kernel_c * kernel_h * kernel_w;
                    tmp5_weight_ptr = tmp0_weight_ptr + 5 * kernel_c * kernel_h * kernel_w;
                    tmp6_weight_ptr = tmp0_weight_ptr + 6 * kernel_c * kernel_h * kernel_w;
                    tmp7_weight_ptr = tmp0_weight_ptr + 7 * kernel_c * kernel_h * kernel_w;
                    psum0 = 0, psum1 = 0, psum2 = 0, psum3 = 0;
                    psum4 = 0, psum5 = 0, psum6 = 0, psum7 = 0;
                    for (int k_c = 0; k_c < kernel_c; ++k_c) {
                        cur0_weight_ptr = tmp0_weight_ptr + k_c * kernel_h * kernel_w;
                        cur1_weight_ptr = tmp1_weight_ptr + k_c * kernel_h * kernel_w;
                        cur2_weight_ptr = tmp2_weight_ptr + k_c * kernel_h * kernel_w;
                        cur3_weight_ptr = tmp3_weight_ptr + k_c * kernel_h * kernel_w;
                        cur4_weight_ptr = tmp4_weight_ptr + k_c * kernel_h * kernel_w;
                        cur5_weight_ptr = tmp5_weight_ptr + k_c * kernel_h * kernel_w;
                        cur6_weight_ptr = tmp6_weight_ptr + k_c * kernel_h * kernel_w;
                        cur7_weight_ptr = tmp7_weight_ptr + k_c * kernel_h * kernel_w;
                        cur_input_ptr = tmp_input_ptr + k_c * in_h * in_w;
                        for (int k_h = 0; k_h < kernel_h; ++k_h) {
                            for (int k_w = 0; k_w < kernel_w; ++k_w) {
                                in_data = cur_input_ptr[k_h * in_w + k_w];
                                psum0 += *cur0_weight_ptr++ * in_data;
                                psum1 += *cur1_weight_ptr++ * in_data;
                                psum2 += *cur2_weight_ptr++ * in_data;
                                psum3 += *cur3_weight_ptr++ * in_data;
                                psum4 += *cur4_weight_ptr++ * in_data;
                                psum5 += *cur5_weight_ptr++ * in_data;
                                psum6 += *cur6_weight_ptr++ * in_data;
                                psum7 += *cur7_weight_ptr++ * in_data;
                            }
                        }
                    }
                    output_batch_ptr[c_i * out_h * out_w + h_i * out_w + w_i] = psum0 + bias_ptr[c_i];
                    output_batch_ptr[(c_i + 1) * out_h * out_w + h_i * out_w + w_i] = psum1 + bias_ptr[c_i + 1];
                    output_batch_ptr[(c_i + 2) * out_h * out_w + h_i * out_w + w_i] = psum2 + bias_ptr[c_i + 2];
                    output_batch_ptr[(c_i + 3) * out_h * out_w + h_i * out_w + w_i] = psum3 + bias_ptr[c_i + 3];
                    output_batch_ptr[(c_i + 4) * out_h * out_w + h_i * out_w + w_i] = psum4 + bias_ptr[c_i + 4];
                    output_batch_ptr[(c_i + 5) * out_h * out_w + h_i * out_w + w_i] = psum5 + bias_ptr[c_i + 5];
                    output_batch_ptr[(c_i + 6) * out_h * out_w + h_i * out_w + w_i] = psum6 + bias_ptr[c_i + 6];
                    output_batch_ptr[(c_i + 7) * out_h * out_w + h_i * out_w + w_i] = psum7 + bias_ptr[c_i + 7];
                }
            }
        }
    }


    return 0;

    //    // write_bin(replace_char(cfg->out_operand_name[0]), out_n * out_c * out_h * out_w * sizeof(float), output_ptr);

    int c = 101;
    return 0;
}

int eval_mxn_quant(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
//    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

    float *input_ptr = (float *) (inputs[0].addr);
    float *weight_ptr = (float *) (inputs[1].addr);

    float *bias_ptr;
    if (cfg->has_bias){
        bias_ptr = (float *) (inputs[2].addr);
    }

    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *weight_tensor = (OPERAND_S *) (params[3].addr);
    OPERAND_S *bias_tensor;

    if (cfg->has_bias){
        bias_tensor = (OPERAND_S *) (params[4].addr);
    }

    int32_t kernel_c = weight_tensor->shape.C;
    int32_t kernel_h = weight_tensor->shape.H;
    int32_t kernel_w = weight_tensor->shape.W;

    int32_t in_n = in_tensor->shape.N;
    int32_t in_c = in_tensor->shape.C;
    int32_t in_h = in_tensor->shape.H;
    int32_t in_w = in_tensor->shape.W;

//    char* tar_str = "input.1";
//    if (strcmp(cfg->in_operand_name[0],tar_str) == 0){
//        // write_input_bin
//        write_bin(cfg->in_operand_name[0], in_n * in_c * in_h * in_w * sizeof(float), input_ptr);
//    }

    int32_t out_n = out_tensor->shape.N;
    int32_t out_c = out_tensor->shape.C;
    int32_t out_h = out_tensor->shape.H;
    int32_t out_w = out_tensor->shape.W;

    void *src_pad_ptr;
    if (cfg->pads[0] != 0){
        // do pad
        src_pad_ptr = aligned_alloc(32, in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(float ));
        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv((char*)src_pad_ptr, (char*)input_ptr, (OPERAND_S *)in_tensor, (PAD_INNER_CONFIG_S *)&pad_cfg);

        input_ptr = (float *)src_pad_ptr;
    }

    const float input_sacle = 51.2998;
//    const float input_sacle = 450.894;
    // trans input from float to int8
    int8_t *input_s8_ptr = (int8_t *)aligned_alloc(32, in_n * in_c * in_h * in_w * sizeof(int8_t));
    for (int i = 0; i < in_n * in_c * in_h * in_w; ++i) {
        int32_t quant_s8 = (int32_t)(input_ptr[i] * input_sacle);
        quant_s8 = quant_s8 > 127 ? 127 : quant_s8;
        quant_s8 = quant_s8 < -128 ? -128 : quant_s8;
        input_s8_ptr[i] = quant_s8;
    }
//    input_ptr = input_s8_ptr;

//    int8_t a = (int8_t)(245);
//    int8_t b = (int8_t)(-3245);
//    printf("a is %d, b is %d\n", a, b);


    // loop params
    int8_t *tmp_input_ptr;
    int8_t *cur_input_ptr;

    float *tmp0_weight_ptr;
    float *tmp1_weight_ptr;
    float *tmp2_weight_ptr;
    float *tmp3_weight_ptr;
    float *tmp4_weight_ptr;
    float *tmp5_weight_ptr;
    float *tmp6_weight_ptr;
    float *tmp7_weight_ptr;

    float *cur0_weight_ptr;
    float *cur1_weight_ptr;
    float *cur2_weight_ptr;
    float *cur3_weight_ptr;
    float *cur4_weight_ptr;
    float *cur5_weight_ptr;
    float *cur6_weight_ptr;
    float *cur7_weight_ptr;

    float *tmp_output_ptr;

    float *output_batch_ptr;

    float *cur_output_ptr;

    register float psum0, psum1, psum2, psum3, psum4, psum5, psum6, psum7;

    register float in_data;

    for (int n_i = 0; n_i < out_n; ++n_i) {
        output_batch_ptr = output_ptr + n_i * out_c * out_h * out_w;
        for (int c_i = 0; c_i < out_c; c_i += 8) {  // todo: maybe the out_c % 8 != 0
            for (int h_i = 0; h_i < out_h; ++h_i) {
                for (int w_i = 0; w_i < out_w; ++w_i) {
                    tmp_input_ptr = input_s8_ptr + h_i * stride_y * in_w + w_i * stride_x;
                    tmp0_weight_ptr = weight_ptr + c_i * kernel_c * kernel_h * kernel_w;
                    tmp1_weight_ptr = tmp0_weight_ptr + 1 * kernel_c * kernel_h * kernel_w;
                    tmp2_weight_ptr = tmp0_weight_ptr + 2 * kernel_c * kernel_h * kernel_w;
                    tmp3_weight_ptr = tmp0_weight_ptr + 3 * kernel_c * kernel_h * kernel_w;
                    tmp4_weight_ptr = tmp0_weight_ptr + 4 * kernel_c * kernel_h * kernel_w;
                    tmp5_weight_ptr = tmp0_weight_ptr + 5 * kernel_c * kernel_h * kernel_w;
                    tmp6_weight_ptr = tmp0_weight_ptr + 6 * kernel_c * kernel_h * kernel_w;
                    tmp7_weight_ptr = tmp0_weight_ptr + 7 * kernel_c * kernel_h * kernel_w;
                    psum0 = 0, psum1 = 0, psum2 = 0, psum3 = 0;
                    psum4 = 0, psum5 = 0, psum6 = 0, psum7 = 0;
                    for (int k_c = 0; k_c < kernel_c; ++k_c) {
                        cur0_weight_ptr = tmp0_weight_ptr + k_c * kernel_h * kernel_w;
                        cur1_weight_ptr = tmp1_weight_ptr + k_c * kernel_h * kernel_w;
                        cur2_weight_ptr = tmp2_weight_ptr + k_c * kernel_h * kernel_w;
                        cur3_weight_ptr = tmp3_weight_ptr + k_c * kernel_h * kernel_w;
                        cur4_weight_ptr = tmp4_weight_ptr + k_c * kernel_h * kernel_w;
                        cur5_weight_ptr = tmp5_weight_ptr + k_c * kernel_h * kernel_w;
                        cur6_weight_ptr = tmp6_weight_ptr + k_c * kernel_h * kernel_w;
                        cur7_weight_ptr = tmp7_weight_ptr + k_c * kernel_h * kernel_w;
                        cur_input_ptr = tmp_input_ptr + k_c * in_h * in_w;
                        for (int k_h = 0; k_h < kernel_h; ++k_h) {
                            for (int k_w = 0; k_w < kernel_w; ++k_w) {
                                in_data = cur_input_ptr[k_h * in_w + k_w];
                                psum0 += *cur0_weight_ptr++ * (float)in_data;
                                psum1 += *cur1_weight_ptr++ * (float)in_data;
                                psum2 += *cur2_weight_ptr++ * (float)in_data;
                                psum3 += *cur3_weight_ptr++ * (float)in_data;
                                psum4 += *cur4_weight_ptr++ * (float)in_data;
                                psum5 += *cur5_weight_ptr++ * (float)in_data;
                                psum6 += *cur6_weight_ptr++ * (float)in_data;
                                psum7 += *cur7_weight_ptr++ * (float)in_data;
                            }
                        }
                    }
                    output_batch_ptr[c_i * out_h * out_w + h_i * out_w + w_i] = psum0 + bias_ptr[c_i] * input_sacle;
                    output_batch_ptr[(c_i + 1) * out_h * out_w + h_i * out_w + w_i] = psum1 + bias_ptr[c_i + 1] * input_sacle;
                    output_batch_ptr[(c_i + 2) * out_h * out_w + h_i * out_w + w_i] = psum2 + bias_ptr[c_i + 2] * input_sacle;
                    output_batch_ptr[(c_i + 3) * out_h * out_w + h_i * out_w + w_i] = psum3 + bias_ptr[c_i + 3] * input_sacle;
                    output_batch_ptr[(c_i + 4) * out_h * out_w + h_i * out_w + w_i] = psum4 + bias_ptr[c_i + 4] * input_sacle;
                    output_batch_ptr[(c_i + 5) * out_h * out_w + h_i * out_w + w_i] = psum5 + bias_ptr[c_i + 5] * input_sacle;
                    output_batch_ptr[(c_i + 6) * out_h * out_w + h_i * out_w + w_i] = psum6 + bias_ptr[c_i + 6] * input_sacle;
                    output_batch_ptr[(c_i + 7) * out_h * out_w + h_i * out_w + w_i] = psum7 + bias_ptr[c_i + 7] * input_sacle;
                }
            }
        }
    }


    for (int i = 0; i < out_n * out_c * out_h * out_w; ++i) {
        output_ptr[i] = output_ptr[i] / input_sacle;
//        int32_t quant_s8 = (int32_t)(input_ptr[i] * input_sacle);
//        quant_s8 = quant_s8 > 127 ? 127 : quant_s8;
//        quant_s8 = quant_s8 < -128 ? -128 : quant_s8;
//        input_s8_ptr[i] = quant_s8;
    }

    return 0;

    //    // write_bin(replace_char(cfg->out_operand_name[0]), out_n * out_c * out_h * out_w * sizeof(float), output_ptr);

    int c = 101;
    return 0;
}


//int eval_mxn(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
//
//    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
////    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);
//
//    int32_t stride_x = cfg->strides[0];
//    int32_t stride_y = cfg->strides[1];
//
//    float *input_ptr = (float *) (inputs[0].addr);
//    float *weight_ptr = (float *) (inputs[1].addr);
//
//    float *bias_ptr;
//    if (cfg->has_bias){
//        bias_ptr = (float *) (inputs[2].addr);
//    }
//
//    float *output_ptr = (float *) (outputs[0].addr);
//
//    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
//    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);
//    OPERAND_S *weight_tensor = (OPERAND_S *) (params[3].addr);
//    OPERAND_S *bias_tensor;
//
//    if (cfg->has_bias){
//        bias_tensor = (OPERAND_S *) (params[4].addr);
//    }
//
//    int32_t kernel_c = weight_tensor->shape.C;
//    int32_t kernel_h = weight_tensor->shape.H;
//    int32_t kernel_w = weight_tensor->shape.W;
//
//    int32_t in_n = in_tensor->shape.N;
//    int32_t in_c = in_tensor->shape.C;
//    int32_t in_h = in_tensor->shape.H;
//    int32_t in_w = in_tensor->shape.W;
//
//    int32_t out_n = out_tensor->shape.N;
//    int32_t out_c = out_tensor->shape.C;
//    int32_t out_h = out_tensor->shape.H;
//    int32_t out_w = out_tensor->shape.W;
//
//    void *src_pad_ptr;
//    if (cfg->pads[0] != 0){
//        // do pad
//        src_pad_ptr = aligned_alloc(32, in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(float ));
//        PAD_INNER_CONFIG_S pad_cfg;
//        pad_cfg.h = cfg->pads[0];
//        pad_cfg.w = cfg->pads[0];
//
//        in_h = in_h + 2 * cfg->pads[0];
//        in_w = in_w + 2 * cfg->pads[0];
//
//                do_pad_conv((char*)src_pad_ptr, (char*)input_ptr, (OPERAND_S *)in_tensor, (PAD_INNER_CONFIG_S *)&pad_cfg);
//        input_ptr = (float *)src_pad_ptr;
//    }
//
//    // loop params
//    float *tmp_input_ptr;
//    float *cur_input_ptr;
//
//    float *tmp0_weight_ptr;
//    float *tmp1_weight_ptr;
//    float *tmp2_weight_ptr;
//    float *tmp3_weight_ptr;
//
//    float *cur0_weight_ptr;
//    float *cur1_weight_ptr;
//    float *cur2_weight_ptr;
//    float *cur3_weight_ptr;
//
//    float *tmp_output_ptr;
//
//    float *output_batch_ptr;
//
//    float *cur_output_ptr;
//
//    register float psum0, psum1, psum2, psum3;
//
//    float in_data;
//
//    for (int n_i = 0; n_i < out_n; ++n_i) {
//        output_batch_ptr = output_ptr + n_i * out_c * out_h * out_w;
//        for (int c_i = 0; c_i < out_c; c_i += 4) {  // todo: mybe the out_c % 4 != 0
//            for (int h_i = 0; h_i < out_h; ++h_i) {
//                for (int w_i = 0; w_i < out_w; ++w_i) {
//                    tmp_input_ptr = input_ptr + h_i * stride_y * in_w + w_i * stride_x;
//                    tmp0_weight_ptr = weight_ptr + c_i * kernel_c * kernel_h * kernel_w;
//                    tmp1_weight_ptr = tmp0_weight_ptr + 1 * kernel_c * kernel_h * kernel_w;
//                    tmp2_weight_ptr = tmp0_weight_ptr + 2 * kernel_c * kernel_h * kernel_w;
//                    tmp3_weight_ptr = tmp0_weight_ptr + 3 * kernel_c * kernel_h * kernel_w;
//                    psum0 = 0, psum1 = 0, psum2 = 0, psum3 = 0;
//                    for (int k_c = 0; k_c < kernel_c; ++k_c) {
//                        cur0_weight_ptr = tmp0_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur1_weight_ptr = tmp1_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur2_weight_ptr = tmp2_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur3_weight_ptr = tmp3_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur_input_ptr = tmp_input_ptr + k_c * in_h * in_w;
//                        for (int k_h = 0; k_h < kernel_h; ++k_h) {
//                            for (int k_w = 0; k_w < kernel_w; ++k_w) {
//                                in_data = cur_input_ptr[k_h * in_w + k_w];
//                                psum0 += *cur0_weight_ptr++ * in_data;
//                                psum1 += *cur1_weight_ptr++ * in_data;
//                                psum2 += *cur2_weight_ptr++ * in_data;
//                                psum3 += *cur3_weight_ptr++ * in_data;
//                            }
//                        }
//                    }
//                    output_batch_ptr[c_i * out_h * out_w + h_i * out_w + w_i] = psum0 + bias_ptr[c_i];
//                    output_batch_ptr[(c_i + 1) * out_h * out_w + h_i * out_w + w_i] = psum1 + bias_ptr[c_i + 1];
//                    output_batch_ptr[(c_i + 2) * out_h * out_w + h_i * out_w + w_i] = psum2 + bias_ptr[c_i + 2];
//                    output_batch_ptr[(c_i + 3) * out_h * out_w + h_i * out_w + w_i] = psum3 + bias_ptr[c_i + 3];
//                }
//            }
//        }
//    }
//
//
//    return 0;
//
//    //    // write_bin(replace_char(cfg->out_operand_name[0]), out_n * out_c * out_h * out_w * sizeof(float), output_ptr);
//
//    int c = 101;
//    return 0;
//}

int eval_mxn_naive_gemm_c(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
//    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

    float *input_ptr = (float *) (inputs[0].addr);
    float *weight_ptr = (float *) (inputs[1].addr);

    float *bias_ptr;
    if (cfg->has_bias){
        bias_ptr = (float *) (inputs[2].addr);
    }

    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *weight_tensor = (OPERAND_S *) (params[3].addr);
    OPERAND_S *bias_tensor;

    if (cfg->has_bias){
        bias_tensor = (OPERAND_S *) (params[4].addr);
    }

    int32_t kernel_c = weight_tensor->shape.C;
    int32_t kernel_h = weight_tensor->shape.H;
    int32_t kernel_w = weight_tensor->shape.W;

    int32_t in_n = in_tensor->shape.N;
    int32_t in_c = in_tensor->shape.C;
    int32_t in_h = in_tensor->shape.H;
    int32_t in_w = in_tensor->shape.W;

    int32_t out_n = out_tensor->shape.N;
    int32_t out_c = out_tensor->shape.C;
    int32_t out_h = out_tensor->shape.H;
    int32_t out_w = out_tensor->shape.W;

    void *src_pad_ptr;
    if (cfg->pads[0] != 0){
        // do pad
        src_pad_ptr = aligned_alloc(32, in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(float ));
        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv((char*)src_pad_ptr, (char*)input_ptr, (OPERAND_S *)in_tensor, (PAD_INNER_CONFIG_S *)&pad_cfg);

        input_ptr = (float *)src_pad_ptr;
    }


    const int32_t N = in_tensor->shape.N;
    const int32_t IC = in_tensor->shape.C;

    const int32_t OC = out_tensor->shape.C;
    const int32_t OH = out_tensor->shape.H;
    const int32_t OW = out_tensor->shape.W;

    const int32_t FH = weight_tensor->shape.H;
    const int32_t FW = weight_tensor->shape.W;

    const int32_t stride_h = cfg->strides[0];
    const int32_t stride_w = cfg->strides[1];

    // do conv as gemm
    const int32_t GEMM_M = OC;
    const int32_t GEMM_N = N * OH * OW;
    const int32_t GEMM_K = IC * FH * FW;

    for (int i = 0; i < GEMM_M; ++i)
    {
        int32_t oc = i;
        for (int j = 0; j < GEMM_N; ++j)
        {
            float accumulator = bias_ptr[i];
            int32_t n = j / (OH * OW);
            int32_t j_res = j % (OH * OW);
            int32_t oh = j_res / OW;
            int32_t ow = j_res % OW;

            // calc single output's elem
            for (int k = 0; k < GEMM_K; ++k)
            {
                int32_t ic = k / (FH * FW);
                int32_t k_res = k % (FH * FW);
                int32_t fh = k_res / FW;
                int32_t fw = k_res % FW;
                int32_t ih = oh * stride_h + fh;
                int32_t iw = ow * stride_w + fw;
//                accumulator = accumulator + in_array[n][ic][ih][iw] * weight_array[oc][ic][fh][fw];
                accumulator = accumulator + input_ptr[ic * in_h * in_w + ih * in_w + iw] *
                        weight_ptr[oc * in_c * kernel_h * kernel_w + ic * kernel_h * kernel_w + fh * kernel_w + fw];
            }
//            out_array[n][oc][oh][ow] = accumulator;
            output_ptr[oc * out_h * out_w + oh * out_w + ow] = accumulator;
        }
    }



//
//    // loop params
//    float *tmp_input_ptr;
//    float *cur_input_ptr;
//
//    float *tmp0_weight_ptr;
//    float *tmp1_weight_ptr;
//    float *tmp2_weight_ptr;
//    float *tmp3_weight_ptr;
//    float *tmp4_weight_ptr;
//    float *tmp5_weight_ptr;
//    float *tmp6_weight_ptr;
//    float *tmp7_weight_ptr;
//
//    float *cur0_weight_ptr;
//    float *cur1_weight_ptr;
//    float *cur2_weight_ptr;
//    float *cur3_weight_ptr;
//    float *cur4_weight_ptr;
//    float *cur5_weight_ptr;
//    float *cur6_weight_ptr;
//    float *cur7_weight_ptr;
//
//    float *tmp_output_ptr;
//
//    float *output_batch_ptr;
//
//    float *cur_output_ptr;
//
//    register float psum0, psum1, psum2, psum3, psum4, psum5, psum6, psum7;
//
//    register float in_data;
//
//    for (int n_i = 0; n_i < out_n; ++n_i) {
//        output_batch_ptr = output_ptr + n_i * out_c * out_h * out_w;
//        for (int c_i = 0; c_i < out_c; c_i += 8) {  // todo: maybe the out_c % 8 != 0
//            for (int h_i = 0; h_i < out_h; ++h_i) {
//                for (int w_i = 0; w_i < out_w; ++w_i) {
//                    tmp_input_ptr = input_ptr + h_i * stride_y * in_w + w_i * stride_x;
//                    tmp0_weight_ptr = weight_ptr + c_i * kernel_c * kernel_h * kernel_w;
//                    tmp1_weight_ptr = tmp0_weight_ptr + 1 * kernel_c * kernel_h * kernel_w;
//                    tmp2_weight_ptr = tmp0_weight_ptr + 2 * kernel_c * kernel_h * kernel_w;
//                    tmp3_weight_ptr = tmp0_weight_ptr + 3 * kernel_c * kernel_h * kernel_w;
//                    tmp4_weight_ptr = tmp0_weight_ptr + 4 * kernel_c * kernel_h * kernel_w;
//                    tmp5_weight_ptr = tmp0_weight_ptr + 5 * kernel_c * kernel_h * kernel_w;
//                    tmp6_weight_ptr = tmp0_weight_ptr + 6 * kernel_c * kernel_h * kernel_w;
//                    tmp7_weight_ptr = tmp0_weight_ptr + 7 * kernel_c * kernel_h * kernel_w;
//                    psum0 = 0, psum1 = 0, psum2 = 0, psum3 = 0;
//                    psum4 = 0, psum5 = 0, psum6 = 0, psum7 = 0;
//                    for (int k_c = 0; k_c < kernel_c; ++k_c) {
//                        cur0_weight_ptr = tmp0_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur1_weight_ptr = tmp1_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur2_weight_ptr = tmp2_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur3_weight_ptr = tmp3_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur4_weight_ptr = tmp4_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur5_weight_ptr = tmp5_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur6_weight_ptr = tmp6_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur7_weight_ptr = tmp7_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur_input_ptr = tmp_input_ptr + k_c * in_h * in_w;
//                        for (int k_h = 0; k_h < kernel_h; ++k_h) {
//                            for (int k_w = 0; k_w < kernel_w; ++k_w) {
//                                in_data = cur_input_ptr[k_h * in_w + k_w];
//                                psum0 += *cur0_weight_ptr++ * in_data;
//                                psum1 += *cur1_weight_ptr++ * in_data;
//                                psum2 += *cur2_weight_ptr++ * in_data;
//                                psum3 += *cur3_weight_ptr++ * in_data;
//                                psum4 += *cur4_weight_ptr++ * in_data;
//                                psum5 += *cur5_weight_ptr++ * in_data;
//                                psum6 += *cur6_weight_ptr++ * in_data;
//                                psum7 += *cur7_weight_ptr++ * in_data;
//                            }
//                        }
//                    }
//                    output_batch_ptr[c_i * out_h * out_w + h_i * out_w + w_i] = psum0 + bias_ptr[c_i];
//                    output_batch_ptr[(c_i + 1) * out_h * out_w + h_i * out_w + w_i] = psum1 + bias_ptr[c_i + 1];
//                    output_batch_ptr[(c_i + 2) * out_h * out_w + h_i * out_w + w_i] = psum2 + bias_ptr[c_i + 2];
//                    output_batch_ptr[(c_i + 3) * out_h * out_w + h_i * out_w + w_i] = psum3 + bias_ptr[c_i + 3];
//                    output_batch_ptr[(c_i + 4) * out_h * out_w + h_i * out_w + w_i] = psum4 + bias_ptr[c_i + 4];
//                    output_batch_ptr[(c_i + 5) * out_h * out_w + h_i * out_w + w_i] = psum5 + bias_ptr[c_i + 5];
//                    output_batch_ptr[(c_i + 6) * out_h * out_w + h_i * out_w + w_i] = psum6 + bias_ptr[c_i + 6];
//                    output_batch_ptr[(c_i + 7) * out_h * out_w + h_i * out_w + w_i] = psum7 + bias_ptr[c_i + 7];
//                }
//            }
//        }
//    }
//
//
//    return 0;
//
//    //    // write_bin(replace_char(cfg->out_operand_name[0]), out_n * out_c * out_h * out_w * sizeof(float), output_ptr);
//
//    int c = 101;
    return 0;
}


__global__ void calc_conv_as_gemm_gpu(float *input_ptr, float *weight_ptr, float *bias_ptr, float *output_ptr, float *args)
{
    int block_idx = blockIdx.x;
    int th_idx = threadIdx.x;

    ARGS_S * arg = (ARGS_S*)args;

    int32_t ic, k_res, fh, fw, ih, iw;

    float accumulator = bias_ptr[block_idx];
    int32_t j_res = th_idx % (arg->out_h * arg->out_w);
    int32_t oh = j_res / arg->out_w;
    int32_t ow = j_res % arg->out_w;

    // calc single output's elem
    for (int k = 0; k < arg->GEMM_K; ++k)
    {
        ic = k / (arg->FH * arg->FW);
        k_res = k % (arg->FH * arg->FW);
        fh = k_res / arg->FW;
        fw = k_res % arg->FW;
        ih = oh * arg->stride_h + fh;
        iw = ow * arg->stride_w + fw;
        accumulator = accumulator + input_ptr[ic * arg->in_h * arg->in_w + ih * arg->in_w + iw] *
                                    weight_ptr[block_idx * arg->in_c * arg->FH * arg->FW + ic * arg->FH * arg->FW + fh * arg->FW + fw];
    }
    output_ptr[block_idx * arg->out_h * arg->out_w + oh * arg->out_w + ow] = accumulator;


}


int eval_mxn_naive_gemm_cuda(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
//    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

    float *input_ptr = (float *) (inputs[0].addr);
    float *weight_ptr = (float *) (inputs[1].addr);

    float *bias_ptr;
    if (cfg->has_bias){
        bias_ptr = (float *) (inputs[2].addr);
    }

    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *weight_tensor = (OPERAND_S *) (params[3].addr);
    OPERAND_S *bias_tensor;

    if (cfg->has_bias){
        bias_tensor = (OPERAND_S *) (params[4].addr);
    }

    int32_t kernel_c = weight_tensor->shape.C;
    int32_t kernel_h = weight_tensor->shape.H;
    int32_t kernel_w = weight_tensor->shape.W;

    int32_t in_n = in_tensor->shape.N;
    int32_t in_c = in_tensor->shape.C;
    int32_t in_h = in_tensor->shape.H;
    int32_t in_w = in_tensor->shape.W;

    int32_t out_n = out_tensor->shape.N;
    int32_t out_c = out_tensor->shape.C;
    int32_t out_h = out_tensor->shape.H;
    int32_t out_w = out_tensor->shape.W;

    void *src_pad_ptr;
    if (cfg->pads[0] != 0){
        // do pad
        src_pad_ptr = aligned_alloc(32, in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(float ));
        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv((char*)src_pad_ptr, (char*)input_ptr, (OPERAND_S *)in_tensor, (PAD_INNER_CONFIG_S *)&pad_cfg);
        input_ptr = (float *)src_pad_ptr;
    }


    const int32_t N = in_tensor->shape.N;
    const int32_t IC = in_tensor->shape.C;

    const int32_t OC = out_tensor->shape.C;
    const int32_t OH = out_tensor->shape.H;
    const int32_t OW = out_tensor->shape.W;

    const int32_t FH = weight_tensor->shape.H;
    const int32_t FW = weight_tensor->shape.W;

    const int32_t stride_h = cfg->strides[0];
    const int32_t stride_w = cfg->strides[1];

    // do conv as gemm
    const int32_t GEMM_M = OC;
    const int32_t GEMM_N = N * OH * OW;
    const int32_t GEMM_K = IC * FH * FW;



    ARGS_S args;
    args.FH = FH, args.FW = FW;
    args.stride_h = stride_h, args.stride_w = stride_w;
    args.in_c = in_c, args.in_h = in_h, args.in_w = in_w;
    args.out_h = out_h, args.out_w = out_w;
    args.GEMM_K = GEMM_K;

    dim3 grid(GEMM_M);
    dim3 block(GEMM_N);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    calc_conv_as_gemm_gpu<<<grid, block>>>(input_ptr, weight_ptr, bias_ptr, output_ptr, (float *)&args);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); // host 等待 gpu 侧计算完毕




//    int32_t ic, k_res, fh, fw, ih, iw;
//    for (int i = 0; i < GEMM_M; ++i)
//    {
//        int32_t oc = i;
//        for (int j = 0; j < GEMM_N; ++j)
//        {
//            float accumulator = bias_ptr[i];
//            int32_t j_res = j % (OH * OW);
//            int32_t oh = j_res / OW;
//            int32_t ow = j_res % OW;
//
//            // calc single output's elem
//            for (int k = 0; k < GEMM_K; ++k)
//            {
//                ic = k / (FH * FW);
//                k_res = k % (FH * FW);
//                fh = k_res / FW;
//                fw = k_res % FW;
//                ih = oh * stride_h + fh;
//                iw = ow * stride_w + fw;
//                accumulator = accumulator + input_ptr[ic * in_h * in_w + ih * in_w + iw] *
//                                            weight_ptr[oc * in_c * FH * FW + ic * FH * FW + fh * FW + fw];
//            }
//            output_ptr[oc * out_h * out_w + oh * out_w + ow] = accumulator;
//        }
//    }




//
//    // loop params
//    float *tmp_input_ptr;
//    float *cur_input_ptr;
//
//    float *tmp0_weight_ptr;
//    float *tmp1_weight_ptr;
//    float *tmp2_weight_ptr;
//    float *tmp3_weight_ptr;
//    float *tmp4_weight_ptr;
//    float *tmp5_weight_ptr;
//    float *tmp6_weight_ptr;
//    float *tmp7_weight_ptr;
//
//    float *cur0_weight_ptr;
//    float *cur1_weight_ptr;
//    float *cur2_weight_ptr;
//    float *cur3_weight_ptr;
//    float *cur4_weight_ptr;
//    float *cur5_weight_ptr;
//    float *cur6_weight_ptr;
//    float *cur7_weight_ptr;
//
//    float *tmp_output_ptr;
//
//    float *output_batch_ptr;
//
//    float *cur_output_ptr;
//
//    register float psum0, psum1, psum2, psum3, psum4, psum5, psum6, psum7;
//
//    register float in_data;
//
//    for (int n_i = 0; n_i < out_n; ++n_i) {
//        output_batch_ptr = output_ptr + n_i * out_c * out_h * out_w;
//        for (int c_i = 0; c_i < out_c; c_i += 8) {  // todo: maybe the out_c % 8 != 0
//            for (int h_i = 0; h_i < out_h; ++h_i) {
//                for (int w_i = 0; w_i < out_w; ++w_i) {
//                    tmp_input_ptr = input_ptr + h_i * stride_y * in_w + w_i * stride_x;
//                    tmp0_weight_ptr = weight_ptr + c_i * kernel_c * kernel_h * kernel_w;
//                    tmp1_weight_ptr = tmp0_weight_ptr + 1 * kernel_c * kernel_h * kernel_w;
//                    tmp2_weight_ptr = tmp0_weight_ptr + 2 * kernel_c * kernel_h * kernel_w;
//                    tmp3_weight_ptr = tmp0_weight_ptr + 3 * kernel_c * kernel_h * kernel_w;
//                    tmp4_weight_ptr = tmp0_weight_ptr + 4 * kernel_c * kernel_h * kernel_w;
//                    tmp5_weight_ptr = tmp0_weight_ptr + 5 * kernel_c * kernel_h * kernel_w;
//                    tmp6_weight_ptr = tmp0_weight_ptr + 6 * kernel_c * kernel_h * kernel_w;
//                    tmp7_weight_ptr = tmp0_weight_ptr + 7 * kernel_c * kernel_h * kernel_w;
//                    psum0 = 0, psum1 = 0, psum2 = 0, psum3 = 0;
//                    psum4 = 0, psum5 = 0, psum6 = 0, psum7 = 0;
//                    for (int k_c = 0; k_c < kernel_c; ++k_c) {
//                        cur0_weight_ptr = tmp0_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur1_weight_ptr = tmp1_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur2_weight_ptr = tmp2_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur3_weight_ptr = tmp3_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur4_weight_ptr = tmp4_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur5_weight_ptr = tmp5_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur6_weight_ptr = tmp6_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur7_weight_ptr = tmp7_weight_ptr + k_c * kernel_h * kernel_w;
//                        cur_input_ptr = tmp_input_ptr + k_c * in_h * in_w;
//                        for (int k_h = 0; k_h < kernel_h; ++k_h) {
//                            for (int k_w = 0; k_w < kernel_w; ++k_w) {
//                                in_data = cur_input_ptr[k_h * in_w + k_w];
//                                psum0 += *cur0_weight_ptr++ * in_data;
//                                psum1 += *cur1_weight_ptr++ * in_data;
//                                psum2 += *cur2_weight_ptr++ * in_data;
//                                psum3 += *cur3_weight_ptr++ * in_data;
//                                psum4 += *cur4_weight_ptr++ * in_data;
//                                psum5 += *cur5_weight_ptr++ * in_data;
//                                psum6 += *cur6_weight_ptr++ * in_data;
//                                psum7 += *cur7_weight_ptr++ * in_data;
//                            }
//                        }
//                    }
//                    output_batch_ptr[c_i * out_h * out_w + h_i * out_w + w_i] = psum0 + bias_ptr[c_i];
//                    output_batch_ptr[(c_i + 1) * out_h * out_w + h_i * out_w + w_i] = psum1 + bias_ptr[c_i + 1];
//                    output_batch_ptr[(c_i + 2) * out_h * out_w + h_i * out_w + w_i] = psum2 + bias_ptr[c_i + 2];
//                    output_batch_ptr[(c_i + 3) * out_h * out_w + h_i * out_w + w_i] = psum3 + bias_ptr[c_i + 3];
//                    output_batch_ptr[(c_i + 4) * out_h * out_w + h_i * out_w + w_i] = psum4 + bias_ptr[c_i + 4];
//                    output_batch_ptr[(c_i + 5) * out_h * out_w + h_i * out_w + w_i] = psum5 + bias_ptr[c_i + 5];
//                    output_batch_ptr[(c_i + 6) * out_h * out_w + h_i * out_w + w_i] = psum6 + bias_ptr[c_i + 6];
//                    output_batch_ptr[(c_i + 7) * out_h * out_w + h_i * out_w + w_i] = psum7 + bias_ptr[c_i + 7];
//                }
//            }
//        }
//    }
//
//
//    return 0;
//
//    //    // write_bin(replace_char(cfg->out_operand_name[0]), out_n * out_c * out_h * out_w * sizeof(float), output_ptr);
//
//    int c = 101;
    return 0;
}


int eval_impl(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);

    OPERAND_S *weight_tensor = (OPERAND_S *) (params[3].addr);

    int32_t kernel_c = weight_tensor->shape.C;
    int32_t kernel_h = weight_tensor->shape.H;
    int32_t kernel_w = weight_tensor->shape.W;

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);


    int32_t out_n = out_tensor->shape.N;
    int32_t out_c = out_tensor->shape.C;
    int32_t out_h = out_tensor->shape.H;
    int32_t out_w = out_tensor->shape.W;
    float *output_ptr = (float *) (outputs[0].addr);


////    eval_mxn(params, inputs, outputs);
//
//    char* tar_str = "input.1";
//    if (strcmp(cfg->in_operand_name[0],tar_str) == 0){
//        eval_mxn_quant(params, inputs, outputs);
//    } else {
//        eval_mxn(params, inputs, outputs);
//    }


//    printf("end this conv op\n");


    eval_mxn_naive_gemm_cuda(params, inputs, outputs);


//    if (cfg->kernel_shape[0] == 3 && cfg->kernel_shape[1] == 3){
//        eval_mxn(params, inputs, outputs);
//    } else if (cfg->kernel_shape[0] == 1 && cfg->kernel_shape[1] == 1 && cfg->strides[0] == 1 && cfg->strides[1] == 1){
//        eval_mxn_naive_gemm_c(params, inputs, outputs);
//
////        eval_1x1j1(params, inputs, outputs);
//    } else {
//        eval_mxn(params, inputs, outputs);
//    }



    int c = 101;
//    printf("this is cuda\n");
    return 0;
}

#include <stdio.h>
extern "C" __attribute__((visibility("default"))) int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) { return eval_impl(params, inputs, outputs); }




