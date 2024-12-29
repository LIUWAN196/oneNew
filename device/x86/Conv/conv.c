#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "string.h"
#include <immintrin.h>
#include <omp.h>

#include "conv.h"
#include "conv_pad.h"
#include "math.h"
#include "../../x86_utils/opt_gemm.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);

    // 一、不带 bias 的 conv 实现
    if (cfg->has_bias == FALSE) {
        if (cfg->ifmap_quant2 == TYPE_INT8) {
            // ifmap with input scale, weight is s8, bias is s32
            eval_mxn_img2col_W8A32_with_input_scale_no_bias(params, inputs, outputs);
        } else {
            // ifmap and ofmap, weight, bias all is float32
            eval_mxn_img2col_no_bias(params, inputs, outputs);
        }
        return 0;
    }

    if (cfg->group != 1) {  // 二、 depth wise conv 的实现
        // depth wise conv
        if (cfg->ifmap_quant2 == TYPE_INT8) {
            eval_depthwise_conv_mxn_img2col_W8A32(params, inputs, outputs);
        } else {
            if (cfg->kernel_shape[0] == 3 && cfg->kernel_shape[1] == 3
                && cfg->pads[0] == 1 && cfg->pads[1] == 1 && cfg->pads[2] == 1 && cfg->pads[3] == 1) {
                // 这段代码不太容易理解，理解不了，可以直接使用 eval_depthwise_conv_mxn_img2col 的代码
                eval_depthwise_conv_3x3_pad1(params, inputs, outputs);
            } else {
                eval_depthwise_conv_mxn_img2col(params, inputs, outputs);
            }
        }
    } else {        // 三、普通 conv 的实现
        // normal conv
        if (cfg->ifmap_quant2 == TYPE_INT8) {
            // ifmap with input scale, weight is s8, bias is s32
            eval_mxn_img2col_W8A32_with_input_scale(params, inputs, outputs);
        } else {
            // ifmap and ofmap, weight, bias all is float32
            if (cfg->kernel_shape[0] == 3 && cfg->kernel_shape[1] == 3) {
                eval_mxn_img2col(params, inputs, outputs);
            } else if (cfg->kernel_shape[0] == 1 && cfg->kernel_shape[1] == 1 && cfg->strides[0] == 1 &&
                       cfg->strides[1] == 1) {
                eval_1x1j1(params, inputs, outputs);
            } else {
                eval_mxn_img2col(params, inputs, outputs);
            }
        }
    }

    // do act if need
    if (cfg->act_type != NOT_ACT) {
        do_act(params, inputs, outputs);
    }

    return 0;
}

int eval_1x1j1(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);

    USEFUL_INFO_S* useful_info = (USEFUL_INFO_S *) (params[BUF_MAXNUM - 1].addr);
    int64_t public_buf_size = useful_info->public_buf_info.public_buf_size;
    int64_t public_buf_ptr = useful_info->public_buf_info.public_buf_ptr;
    int64_t rem_buf_size = public_buf_size;
    int64_t rem_buf_ptr = public_buf_ptr;

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

    void *src_pad_ptr;
    if (cfg->pads[0] != 0) {
        // do pad
        int64_t pad_need_buf_size = in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) *
                                   sizeof(float);

        if (pad_need_buf_size < rem_buf_size) {
            src_pad_ptr = (void *)rem_buf_ptr;
            rem_buf_ptr += (pad_need_buf_size + 31) & (~32);
            rem_buf_size -= (pad_need_buf_size + 31) & (~32);
        } else {
            LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)pad_need_buf_size);
        }

        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv((char *)src_pad_ptr, (char *)input_ptr, in_tensor, &pad_cfg);
        input_ptr = (float *) src_pad_ptr;
    }

    GEMM_TILE_INFO gemm_tile_info;
    gemm_tile_info.M = out_c;
    gemm_tile_info.N = out_h * out_w;
    gemm_tile_info.K = in_c;

    // 目前认为的 sgemm 的最佳配置为 m_tile_size = 32，n_tile_size = 1024，k_tile_size = 8；
    const int32_t best_n_tile = useful_info->block_info.x86_gemm_multi_threads_tile_n;
    const int32_t best_k_tile = useful_info->block_info.x86_gemm_multi_threads_tile_k;
    const int32_t num_threads = 8;
    // 这里因为开了 num_threads 个线程，所以要保证 m_tile 为 num_threads 的倍数，不让有的线程创建了但是没有使用
    int32_t best_m_tile = (gemm_tile_info.M / num_threads > 32) ? 32 : gemm_tile_info.M / num_threads;
    best_m_tile = (best_m_tile == 0) ? gemm_tile_info.M : best_m_tile;

    gemm_tile_info.m_tile_size = best_m_tile;
    gemm_tile_info.n_tile_size = best_n_tile;
    gemm_tile_info.k_tile_size = best_k_tile;

    const int32_t avx2_align_size = 32;
    if (gemm_tile_info.N % avx2_align_size == 0 && gemm_tile_info.K % avx2_align_size == 0) {
        opt_gemm_aligned_multi_threads(output_ptr, weight_ptr, input_ptr, gemm_tile_info);
    } else {
        opt_gemm_multi_threads(output_ptr, weight_ptr, input_ptr, gemm_tile_info);
    }

    // 加偏置
#pragma omp parallel for num_threads(THREADS_NUM)
    for (int i = 0; i < gemm_tile_info.M; i++) {
        int idx = i * gemm_tile_info.N;
#pragma unroll 8
        for (int z = 0; z < gemm_tile_info.N; ++z) {
            output_ptr[idx + z] += bias_ptr[i];
        }
    }

    return 0;
}

int
im2col(float *input_col_ptr, float *input_ptr, OPERAND_S *in_tensor, OPERAND_S *out_tensor, OPERAND_S *weight_tensor,
       CONV_CONFIG_S *cfg) {

    int32_t kernel_c = weight_tensor->shapes[1];
    int32_t kernel_h = weight_tensor->shapes[2];
    int32_t kernel_w = weight_tensor->shapes[3];

    int32_t in_n = in_tensor->shapes[0];
    int32_t in_c = in_tensor->shapes[1];
    int32_t in_h = in_tensor->shapes[2] + cfg->pads[0] + cfg->pads[2];
    int32_t in_w = in_tensor->shapes[3] + cfg->pads[1] + cfg->pads[3];

    int32_t out_n = out_tensor->shapes[0];
    int32_t out_c = out_tensor->shapes[1];
    int32_t out_h = out_tensor->shapes[2];
    int32_t out_w = out_tensor->shapes[3];

    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

    int32_t outHxoutH = out_h * out_h;
    int32_t kWxoutHxoutH = kernel_w * out_h * out_h;
    int32_t kHxkWxoutHxoutw = kernel_h * kernel_w * out_h * out_w;

    if (kernel_h == 3 && kernel_w == 3) {
#pragma omp parallel for num_threads(THREADS_NUM)
        for (int col_h1 = 0; col_h1 < in_c; ++col_h1) {
            int32_t cur_in_h, cur_in_w;
            int32_t ifmap_offset, ofmap_offset;
            int32_t cur_ifmap_offset, cur_ofmap_offset;
            for (int col_h = 0; col_h < out_h; ++col_h) {
                cur_in_h = col_h * stride_y;
                ifmap_offset = col_h1 * kHxkWxoutHxoutw + col_h * out_w;
                ofmap_offset = col_h1 * in_h * in_w + cur_in_h * in_w;
#pragma unroll 2
                for (int col_w = 0; col_w < out_w; ++col_w) {
                    cur_in_w = col_w * stride_x;
                    cur_ifmap_offset = ifmap_offset + col_w;
                    cur_ofmap_offset = ofmap_offset + cur_in_w;
                    *(input_col_ptr + cur_ifmap_offset + 0 * kWxoutHxoutH + 0 * outHxoutH)
                            = *(input_ptr + cur_ofmap_offset + 0 * in_w + 0);
                    *(input_col_ptr + cur_ifmap_offset + 0 * kWxoutHxoutH + 1 * outHxoutH)
                            = *(input_ptr + cur_ofmap_offset + 0 * in_w + 1);
                    *(input_col_ptr + cur_ifmap_offset + 0 * kWxoutHxoutH + 2 * outHxoutH)
                            = *(input_ptr + cur_ofmap_offset + 0 * in_w + 2);

                    *(input_col_ptr + cur_ifmap_offset + 1 * kWxoutHxoutH + 0 * outHxoutH)
                            = *(input_ptr + cur_ofmap_offset + 1 * in_w + 0);
                    *(input_col_ptr + cur_ifmap_offset + 1 * kWxoutHxoutH + 1 * outHxoutH)
                            = *(input_ptr + cur_ofmap_offset + 1 * in_w + 1);
                    *(input_col_ptr + cur_ifmap_offset + 1 * kWxoutHxoutH + 2 * outHxoutH)
                            = *(input_ptr + cur_ofmap_offset + 1 * in_w + 2);

                    *(input_col_ptr + cur_ifmap_offset + 2 * kWxoutHxoutH + 0 * outHxoutH)
                            = *(input_ptr + cur_ofmap_offset + 2 * in_w + 0);
                    *(input_col_ptr + cur_ifmap_offset + 2 * kWxoutHxoutH + 1 * outHxoutH)
                            = *(input_ptr + cur_ofmap_offset + 2 * in_w + 1);
                    *(input_col_ptr + cur_ifmap_offset + 2 * kWxoutHxoutH + 2 * outHxoutH)
                            = *(input_ptr + cur_ofmap_offset + 2 * in_w + 2);
                }
            }
        }
    } else {
#pragma omp parallel for num_threads(THREADS_NUM)
        for (int col_h1 = 0; col_h1 < in_c; ++col_h1) {
            int32_t cur_in_h, cur_in_w;
            int32_t ifmap_offset, ofmap_offset;
            int32_t cur_ifmap_offset, cur_ofmap_offset;
            for (int col_h = 0; col_h < out_h; ++col_h) {
                cur_in_h = col_h * stride_y;
                ifmap_offset = col_h1 * kHxkWxoutHxoutw + col_h * out_w;
                ofmap_offset = col_h1 * in_h * in_w + cur_in_h * in_w;
#pragma unroll 4
                for (int col_w = 0; col_w < out_w; ++col_w) {
                    cur_in_w = col_w * stride_x;
                    cur_ifmap_offset = ifmap_offset + col_w;
                    cur_ofmap_offset = ofmap_offset + cur_in_w;
                    for (int col_h2 = 0; col_h2 < kernel_h; ++col_h2) {
                        for (int col_h3 = 0; col_h3 < kernel_w; ++col_h3) {
                            input_col_ptr[cur_ifmap_offset + col_h2 * kWxoutHxoutH + col_h3 * outHxoutH]
                                    = input_ptr[cur_ofmap_offset + col_h2 * in_w + col_h3];
                        }
                    }
                }
            }
        }
    }

    return 0;
}

int im2col_s8(int8_t *input_col_ptr, int8_t *input_ptr, OPERAND_S *in_tensor, OPERAND_S *out_tensor,
              OPERAND_S *weight_tensor,
              CONV_CONFIG_S *cfg) {

    int32_t kernel_c = weight_tensor->shapes[1];
    int32_t kernel_h = weight_tensor->shapes[2];
    int32_t kernel_w = weight_tensor->shapes[3];

    int32_t in_n = in_tensor->shapes[0];
    int32_t in_c = in_tensor->shapes[1];
    int32_t in_h = in_tensor->shapes[2] + cfg->pads[0] + cfg->pads[2];
    int32_t in_w = in_tensor->shapes[3] + cfg->pads[1] + cfg->pads[3];

    int32_t out_n = out_tensor->shapes[0];
    int32_t out_c = out_tensor->shapes[1];
    int32_t out_h = out_tensor->shapes[2];
    int32_t out_w = out_tensor->shapes[3];

    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

#pragma omp parallel for num_threads(THREADS_NUM)
    for (int col_h1 = 0; col_h1 < in_c; ++col_h1) {
        for (int col_h2 = 0; col_h2 < kernel_h; ++col_h2) {
            for (int col_h3 = 0; col_h3 < kernel_w; ++col_h3) {
#pragma unroll 8
                for (int col_w = 0; col_w < out_h * out_w; ++col_w) {
                    int cur_in_h = col_w / out_w * stride_y + col_h2;
                    int cur_in_w = col_w % out_w * stride_x + col_h3;
                    input_col_ptr[col_h1 * kernel_h * kernel_w * out_h * out_w + col_h2 * kernel_w * out_h * out_h +
                                  col_h3 * out_h * out_h + col_w] = input_ptr[col_h1 * in_h * in_w + cur_in_h * in_w +
                                                                              cur_in_w];
                }
            }
        }
    }

    return 0;
}

int eval_mxn_img2col(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);

    USEFUL_INFO_S* useful_info = (USEFUL_INFO_S *) (params[BUF_MAXNUM - 1].addr);
    int64_t public_buf_size = useful_info->public_buf_info.public_buf_size;
    int64_t public_buf_ptr = useful_info->public_buf_info.public_buf_ptr;
    int64_t rem_buf_size = public_buf_size;
    int64_t rem_buf_ptr = public_buf_ptr;

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

    void *src_pad_ptr;
    // conv pads:  top  left  bottom  right
    if (cfg->pads[0] != 0 || cfg->pads[1] != 0 || cfg->pads[2] != 0 || cfg->pads[3] != 0) {

        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.top_pad = cfg->pads[0];
        pad_cfg.left_pad = cfg->pads[1];
        pad_cfg.bottom_pad = cfg->pads[2];
        pad_cfg.right_pad = cfg->pads[3];

        // do pad
        int64_t pad_need_buf_size = in_n * in_c * (in_h + pad_cfg.top_pad + pad_cfg.bottom_pad) *
                                    (in_w + pad_cfg.left_pad + pad_cfg.right_pad) * sizeof(float);

        if (pad_need_buf_size < rem_buf_size) {
            src_pad_ptr = (void *)rem_buf_ptr;
            rem_buf_ptr += (pad_need_buf_size + 31) & (~32);
            rem_buf_size -= (pad_need_buf_size + 31) & (~32);
        } else {
            LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)pad_need_buf_size);
        }

        in_h = in_h + pad_cfg.top_pad + pad_cfg.bottom_pad;
        in_w = in_w + pad_cfg.left_pad + pad_cfg.right_pad;

        do_pad_conv_new((char *)src_pad_ptr, (char *)input_ptr, in_tensor, &pad_cfg);
        input_ptr = (float *) src_pad_ptr;
    }

    float *input_col_ptr;
    int64_t col_need_buf_size = in_c * kernel_w * kernel_h * out_h * out_w * sizeof(float);

    if (col_need_buf_size < rem_buf_size) {
        input_col_ptr = (float *)rem_buf_ptr;
        rem_buf_ptr += (col_need_buf_size + 31) & (~32);
        rem_buf_size -= (col_need_buf_size + 31) & (~32);
    } else {
        LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)col_need_buf_size);
    }
    im2col(input_col_ptr, input_ptr, in_tensor, out_tensor, weight_tensor, cfg);
    input_ptr = input_col_ptr;


    GEMM_TILE_INFO gemm_tile_info;
    gemm_tile_info.M = out_c;
    gemm_tile_info.N = out_h * out_w;
    gemm_tile_info.K = in_c * kernel_h * kernel_w;

    // 目前认为的 sgemm 的最佳配置为 m_tile_size = 32，n_tile_size = 1024，k_tile_size = 8；
    const int32_t best_n_tile = useful_info->block_info.x86_gemm_multi_threads_tile_n;
    const int32_t best_k_tile = useful_info->block_info.x86_gemm_multi_threads_tile_k;
    const int32_t num_threads = 8;

    int32_t best_m_tile = (gemm_tile_info.M / num_threads > 32) ? 32 : gemm_tile_info.M / num_threads;
    best_m_tile = (best_m_tile == 0) ? gemm_tile_info.M : best_m_tile;

    gemm_tile_info.m_tile_size = best_m_tile;
    gemm_tile_info.n_tile_size = best_n_tile;
    gemm_tile_info.k_tile_size = best_k_tile;

    opt_gemm_multi_threads(output_ptr, weight_ptr, input_ptr, gemm_tile_info);

    // 加偏置
#pragma omp parallel for num_threads(THREADS_NUM)
    for (int i = 0; i < gemm_tile_info.M; i++) {
        int idx = i * gemm_tile_info.N;
#pragma unroll 8
        for (int z = 0; z < gemm_tile_info.N; ++z) {
            output_ptr[idx + z] += bias_ptr[i];
        }
    }

    return 0;
}

int eval_mxn_img2col_no_bias(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
    USEFUL_INFO_S* useful_info = (USEFUL_INFO_S *) (params[BUF_MAXNUM - 1].addr);
    int64_t public_buf_size = useful_info->public_buf_info.public_buf_size;
    int64_t public_buf_ptr = useful_info->public_buf_info.public_buf_ptr;
    int64_t rem_buf_size = public_buf_size;
    int64_t rem_buf_ptr = public_buf_ptr;

    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

    float *input_ptr = (float *) (inputs[0].addr);
    float *weight_ptr = (float *) (inputs[1].addr);

    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *weight_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

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

    void *src_pad_ptr;
    // conv pads:  top  left  bottom  right
    if (cfg->pads[0] != 0 || cfg->pads[1] != 0 || cfg->pads[2] != 0 || cfg->pads[3] != 0) {

        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.top_pad = cfg->pads[0];
        pad_cfg.left_pad = cfg->pads[1];
        pad_cfg.bottom_pad = cfg->pads[2];
        pad_cfg.right_pad = cfg->pads[3];

        // do pad
        int64_t pad_need_buf_size = in_n * in_c * (in_h + pad_cfg.top_pad + pad_cfg.bottom_pad) *
                                    (in_w + pad_cfg.left_pad + pad_cfg.right_pad) * sizeof(float);

        if (pad_need_buf_size < rem_buf_size) {
            src_pad_ptr = (void *)rem_buf_ptr;
            rem_buf_ptr += (pad_need_buf_size + 31) & (~32);
            rem_buf_size -= (pad_need_buf_size + 31) & (~32);
        } else {
            LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)pad_need_buf_size);
        }

        in_h = in_h + pad_cfg.top_pad + pad_cfg.bottom_pad;
        in_w = in_w + pad_cfg.left_pad + pad_cfg.right_pad;

        do_pad_conv_new((char *)src_pad_ptr, (char *)input_ptr, in_tensor, &pad_cfg);
        input_ptr = (float *) src_pad_ptr;
    }

    float *input_col_ptr;
    int64_t col_need_buf_size = in_c * kernel_w * kernel_h * out_h * out_w * sizeof(float);

    if (col_need_buf_size < rem_buf_size) {
        input_col_ptr = (float *)rem_buf_ptr;
        rem_buf_ptr += (col_need_buf_size + 31) & (~32);
        rem_buf_size -= (col_need_buf_size + 31) & (~32);
    } else {
        LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)col_need_buf_size);
    }
    im2col(input_col_ptr, input_ptr, in_tensor, out_tensor, weight_tensor, cfg);
    input_ptr = input_col_ptr;

    const int M = out_c;
    const int N = out_h * out_w;
    const int K = in_c * kernel_h * kernel_w;

#pragma omp parallel for num_threads(THREADS_NUM)
    for (int i = 0; i < M; i++) {
        memset(&output_ptr[i * N], 0, N * sizeof(float));
        int idx_a, idx_b, idx_c;
        __m256 psum;
        __m256 weight_vec;
        __m256 sum_pre;
        idx_c = i * N;
#pragma unroll 8
        for (int k = 0; k < K; k++) {
            idx_a = i * K + k;
            idx_b = k * N;
            int j = 0;
            weight_vec = _mm256_set1_ps(weight_ptr[idx_a]);
#pragma unroll 2
            for (; j < N - 7; j += 8) {
                sum_pre = _mm256_loadu_ps(&output_ptr[idx_c + j]);
                sum_pre = _mm256_fmadd_ps(weight_vec, _mm256_loadu_ps(&input_ptr[idx_b + j]), sum_pre);
                _mm256_storeu_ps(&output_ptr[idx_c + j], sum_pre);
            }
            for (; j < N; ++j) {
                output_ptr[idx_c + j] += weight_ptr[idx_a] * input_ptr[idx_b + j];
            }
        }
    }

    return 0;
}

int eval_depthwise_conv_mxn_img2col(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
    USEFUL_INFO_S* useful_info = (USEFUL_INFO_S *) (params[BUF_MAXNUM - 1].addr);
    int64_t public_buf_size = useful_info->public_buf_info.public_buf_size;
    int64_t public_buf_ptr = useful_info->public_buf_info.public_buf_ptr;
    int64_t rem_buf_size = public_buf_size;
    int64_t rem_buf_ptr = public_buf_ptr;

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

    void *src_pad_ptr;
    // conv pads:  top  left  bottom  right
    if (cfg->pads[0] != 0 || cfg->pads[1] != 0 || cfg->pads[2] != 0 || cfg->pads[3] != 0) {

        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.top_pad = cfg->pads[0];
        pad_cfg.left_pad = cfg->pads[1];
        pad_cfg.bottom_pad = cfg->pads[2];
        pad_cfg.right_pad = cfg->pads[3];

        // do pad
        int64_t pad_need_buf_size = in_n * in_c * (in_h + pad_cfg.top_pad + pad_cfg.bottom_pad) *
                                    (in_w + pad_cfg.left_pad + pad_cfg.right_pad) * sizeof(float);

        if (pad_need_buf_size < rem_buf_size) {
            src_pad_ptr = (void *)rem_buf_ptr;
            rem_buf_ptr += (pad_need_buf_size + 31) & (~32);
            rem_buf_size -= (pad_need_buf_size + 31) & (~32);
        } else {
            LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)pad_need_buf_size);
        }

        in_h = in_h + pad_cfg.top_pad + pad_cfg.bottom_pad;
        in_w = in_w + pad_cfg.left_pad + pad_cfg.right_pad;

        do_pad_conv_new((char *)src_pad_ptr, (char *)input_ptr, in_tensor, &pad_cfg);
        input_ptr = (float *) src_pad_ptr;
    }

    for (int outc_i = 0; outc_i < out_c; ++outc_i) {
        float *cur_weight_ptr = weight_ptr + outc_i * kernel_h * kernel_w;
        for (int outh_i = 0; outh_i < out_h; ++outh_i) {
            for (int outw_i = 0; outw_i < out_w; ++outw_i) {
                float psum = 0;
                float *cur_ofmp_ptr = output_ptr + outc_i * out_h * out_w + outh_i * out_w + outw_i;
                float *cur_ifmp_ptr = input_ptr + outc_i * in_h * in_w + outh_i * stride_y * in_w + outw_i * stride_x;
                for (int kh_i = 0; kh_i < kernel_h; ++kh_i) {
                    for (int kw_i = 0; kw_i < kernel_w; ++kw_i) {
                        psum += cur_weight_ptr[kh_i * kernel_w + kw_i] * cur_ifmp_ptr[kh_i * in_w + kw_i];
                    }
                }
                cur_ofmp_ptr[0] = psum + bias_ptr[outc_i];
            }
        }
    }

    return 0;
}

int eval_depthwise_conv_mxn_img2col_W8A32(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
    USEFUL_INFO_S* useful_info = (USEFUL_INFO_S *) (params[BUF_MAXNUM - 1].addr);
    int64_t public_buf_size = useful_info->public_buf_info.public_buf_size;
    int64_t public_buf_ptr = useful_info->public_buf_info.public_buf_ptr;
    int64_t rem_buf_size = public_buf_size;
    int64_t rem_buf_ptr = public_buf_ptr;

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

    // 将输入数据量化为 int8
    float tmp;
    float ifmap_scale = cfg->input_scale;
    float opposite_ifmap_scale = 1.0f / ifmap_scale;

    int32_t dequant_need_buf_size = in_c * in_h * in_w * sizeof(float) + in_c * in_h * in_w * sizeof(int8_t);

    float *input_f32_ptr;
    int8_t *input_s8_ptr;
    if (dequant_need_buf_size < rem_buf_size) {
        input_f32_ptr = (void *)rem_buf_ptr;
        rem_buf_ptr += (dequant_need_buf_size + 31) & (~32);
        rem_buf_size -= (dequant_need_buf_size + 31) & (~32);
        input_s8_ptr = (void *)rem_buf_ptr;
        rem_buf_ptr += (dequant_need_buf_size + 31) & (~32);
        rem_buf_size -= (dequant_need_buf_size + 31) & (~32);
    } else {
        LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)dequant_need_buf_size);
    }


#pragma unroll 4
    for (int i = 0; i < in_c * in_h * in_w; ++i) {
        tmp = input_ptr[i] * opposite_ifmap_scale;
        input_f32_ptr[i] = (tmp < -128) ? -128 : (tmp > 127) ? 127 : tmp;
    }

#pragma omp parallel for num_threads(THREADS_NUM)
    for (int i = 0; i < in_c * in_h * in_w; i += 4) {
        input_s8_ptr[i] = (int8_t) (roundf(input_f32_ptr[i]));
        input_s8_ptr[i + 1] = (int8_t) (roundf(input_f32_ptr[i + 1]));
        input_s8_ptr[i + 2] = (int8_t) (roundf(input_f32_ptr[i + 2]));
        input_s8_ptr[i + 3] = (int8_t) (roundf(input_f32_ptr[i + 3]));
    }

    int8_t *ifmap_ptr = input_s8_ptr;
    int8_t *src_s8_pad_ptr;

    void *src_pad_ptr;
    // conv pads:  top  left  bottom  right
    if (cfg->pads[0] != 0 || cfg->pads[1] != 0 || cfg->pads[2] != 0 || cfg->pads[3] != 0) {

        int64_t pad_need_buf_size = in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(int8_t);

        if (pad_need_buf_size < rem_buf_size) {
            src_s8_pad_ptr = (int8_t *)rem_buf_ptr;
            rem_buf_ptr += (pad_need_buf_size + 31) & (~32);
            rem_buf_size -= (pad_need_buf_size + 31) & (~32);
        } else {
            LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)pad_need_buf_size);
        }

        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv_s8((char *) src_s8_pad_ptr, (char *) input_s8_ptr, in_tensor, &pad_cfg);
        ifmap_ptr = (int8_t *) src_s8_pad_ptr;
    }


    int8_t *weight_s8_ptr = (int8_t *) weight_ptr;
    int32_t *bias_s32_ptr = (int32_t *) bias_ptr;

    float *output_f32_ptr = (float *) output_ptr;

    for (int outc_i = 0; outc_i < out_c; ++outc_i) {
        int8_t *cur_weight_ptr = weight_s8_ptr + outc_i * kernel_h * kernel_w;
        for (int outh_i = 0; outh_i < out_h; ++outh_i) {
            for (int outw_i = 0; outw_i < out_w; ++outw_i) {
                int32_t psum = 0;
                float *cur_ofmp_ptr = output_f32_ptr + outc_i * out_h * out_w + outh_i * out_w + outw_i;
                int8_t *cur_ifmp_ptr = ifmap_ptr + outc_i * in_h * in_w + outh_i * stride_y * in_w + outw_i * stride_x;
                for (int kh_i = 0; kh_i < kernel_h; ++kh_i) {
                    for (int kw_i = 0; kw_i < kernel_w; ++kw_i) {
                        psum += cur_weight_ptr[kh_i * kernel_w + kw_i] * cur_ifmp_ptr[kh_i * in_w + kw_i];
                    }
                }
                cur_ofmp_ptr[0] = (float)psum * ifmap_scale  + (float)bias_s32_ptr[outc_i];
            }
        }
    }

    // trans output to float
    float *cur_output_ptr;
    for (int outc_i = 0; outc_i < out_c; ++outc_i) {
        cur_output_ptr = output_ptr + outc_i * out_h * out_w;
        float dequant_coeff = cfg->weight_aux[outc_i] * 1 / 127.0f;
        for (int i = 0; i < out_h * out_w; ++i) {
            cur_output_ptr[i] = cur_output_ptr[i] * dequant_coeff;
        }
    }

    return 0;
}

int eval_depthwise_conv_3x3_pad1(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);

    USEFUL_INFO_S* useful_info = (USEFUL_INFO_S *) (params[BUF_MAXNUM - 1].addr);
    int64_t public_buf_size = useful_info->public_buf_info.public_buf_size;
    int64_t public_buf_ptr = useful_info->public_buf_info.public_buf_ptr;
    int64_t rem_buf_size = public_buf_size;
    int64_t rem_buf_ptr = public_buf_ptr;

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

    int32_t top_pad = cfg->pads[0];
    int32_t left_pad = cfg->pads[1];
    int32_t bottom_pad = cfg->pads[2];

    /*
     * 不对 depth wise 做之前的 pad 上下左右，然后做 im2col。因为 depth wise 计算量本来就很小，如果这样做 pad 上下左右会占用 80% 以上时间
     * 所以采用的方法如下所示：
     * 1、以 in h * in w 为一个组合，只在 pad top 和 bottom。然后使用常规的方法计算 conv;
     * 2、在步骤 1 计算完毕后，可以发现，out_w_i = 0 和 = out_w - 1 因为没有做 left pad 和 right pad，所以的结果是错误的，再对这两列进行计算即可
     */
    // 做 top bottom 的 pad
    float *input_pad_h_ptr;
    int64_t pad_need_buf_size = in_c * (top_pad + in_h + bottom_pad) * in_w * sizeof(float);
    if (pad_need_buf_size < rem_buf_size) {
        input_pad_h_ptr = (void *)rem_buf_ptr;
        rem_buf_ptr += (pad_need_buf_size + 31) & (~32);
        rem_buf_size -= (pad_need_buf_size + 31) & (~32);
    } else {
        LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)pad_need_buf_size);
    }

#pragma omp parallel for num_threads(THREADS_NUM)
    for (int c_i = 0; c_i < in_c; ++c_i) {
        float *cur_src_f32 = input_ptr + c_i * in_h * in_w;
        float *cur_dst_using_f32 = input_pad_h_ptr + c_i * (top_pad + in_h + bottom_pad) * in_w + top_pad * in_w;
        float *cur_dst_st_f32 = input_pad_h_ptr + c_i * (top_pad + in_h + bottom_pad) * in_w;
        float *cur_dst_ed_f32 = input_pad_h_ptr + c_i * (top_pad + in_h + bottom_pad) * in_w + (top_pad + in_h) * in_w;
        // 填充 top 和 bottom 的 pad
        for (int i = 0; i < in_w; ++i) {
            cur_dst_st_f32[i] = 0;
            cur_dst_ed_f32[i] = 0;
        }
        memcpy(cur_dst_using_f32, cur_src_f32, in_h * in_w * sizeof(float));
    }

    // 更新输入数据指针以及 in h
    input_ptr = input_pad_h_ptr;
    in_h = top_pad + in_h + bottom_pad;

    // 可能右侧的 pad 是不需要的
    int32_t real_right_pad = (out_w - 1) * stride_x + kernel_w - in_w - left_pad;
    int32_t out_w_align = out_w - 1;
    if (real_right_pad == 0) {
        out_w_align = out_w;
    }

#pragma omp parallel for num_threads(THREADS_NUM)
    for (int outc_i = 0; outc_i < out_c; ++outc_i) {
        for (int outh_i = 0; outh_i < out_h; ++outh_i) {
            float *cur_weight_ptr = weight_ptr + outc_i * kernel_h * kernel_w;
            float *cur_ofmp_ptr = output_ptr + outc_i * out_h * out_w + outh_i * out_w;
            float *cur_ifmp_ptr = input_ptr + outc_i * in_h * in_w + outh_i * stride_y * in_w - left_pad;
            // 计算主体部分
#pragma unroll 4
            for (int outw_i = 1; outw_i < out_w_align; ++outw_i) {
                float psum = 0;

                psum += cur_weight_ptr[0] * cur_ifmp_ptr[0 * in_w + 0 + outw_i * stride_x];
                psum += cur_weight_ptr[1] * cur_ifmp_ptr[0 * in_w + 1 + outw_i * stride_x];
                psum += cur_weight_ptr[2] * cur_ifmp_ptr[0 * in_w + 2 + outw_i * stride_x];

                psum += cur_weight_ptr[3] * cur_ifmp_ptr[1 * in_w + 0 + outw_i * stride_x];
                psum += cur_weight_ptr[4] * cur_ifmp_ptr[1 * in_w + 1 + outw_i * stride_x];
                psum += cur_weight_ptr[5] * cur_ifmp_ptr[1 * in_w + 2 + outw_i * stride_x];

                psum += cur_weight_ptr[6] * cur_ifmp_ptr[2 * in_w + 0 + outw_i * stride_x];
                psum += cur_weight_ptr[7] * cur_ifmp_ptr[2 * in_w + 1 + outw_i * stride_x];
                psum += cur_weight_ptr[8] * cur_ifmp_ptr[2 * in_w + 2 + outw_i * stride_x];

                cur_ofmp_ptr[outw_i] = psum + bias_ptr[outc_i];
            }

            // 计算 out_w_i = 0 和 = out_w - 1
            // outw_i == 0
            float psum0 = 0;
            psum0 += cur_weight_ptr[1] * cur_ifmp_ptr[0 * in_w + 1];
            psum0 += cur_weight_ptr[2] * cur_ifmp_ptr[0 * in_w + 2];
            psum0 += cur_weight_ptr[4] * cur_ifmp_ptr[1 * in_w + 1];
            psum0 += cur_weight_ptr[5] * cur_ifmp_ptr[1 * in_w + 2];
            psum0 += cur_weight_ptr[7] * cur_ifmp_ptr[2 * in_w + 1];
            psum0 += cur_weight_ptr[8] * cur_ifmp_ptr[2 * in_w + 2];

            cur_ofmp_ptr[0] = psum0 + bias_ptr[outc_i];

            // outw_i == out_w - 1
            if (out_w_align != out_w) {
                float psum1 = 0;
                psum1 += cur_weight_ptr[0] * cur_ifmp_ptr[0 * in_w + 0 + (out_w - 1) * stride_x];
                psum1 += cur_weight_ptr[1] * cur_ifmp_ptr[0 * in_w + 1 + (out_w - 1) * stride_x];
                psum1 += cur_weight_ptr[3] * cur_ifmp_ptr[1 * in_w + 0 + (out_w - 1) * stride_x];
                psum1 += cur_weight_ptr[4] * cur_ifmp_ptr[1 * in_w + 1 + (out_w - 1) * stride_x];
                psum1 += cur_weight_ptr[6] * cur_ifmp_ptr[2 * in_w + 0 + (out_w - 1) * stride_x];
                psum1 += cur_weight_ptr[7] * cur_ifmp_ptr[2 * in_w + 1 + (out_w - 1) * stride_x];

                cur_ofmp_ptr[out_w - 1] = psum1 + bias_ptr[outc_i];
            }
        }
    }

    return 0;
}

int eval_mxn_img2col_W8A32_with_input_scale(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);

    USEFUL_INFO_S* useful_info = (USEFUL_INFO_S *) (params[BUF_MAXNUM - 1].addr);
    int64_t public_buf_size = useful_info->public_buf_info.public_buf_size;
    int64_t public_buf_ptr = useful_info->public_buf_info.public_buf_ptr;
    int64_t rem_buf_size = public_buf_size;
    int64_t rem_buf_ptr = public_buf_ptr;

    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

    float *input_ptr = (float *) (inputs[0].addr);
    float *weight_ptr = (float *) (inputs[1].addr);
    float *bias_ptr = (float *) (inputs[2].addr);

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


    // 将输入数据量化为 int8
    float tmp;
    float ifmap_scale = cfg->input_scale;
    float opposite_ifmap_scale = 1.0f / ifmap_scale;

    int32_t dequant_need_buf_size = in_c * in_h * in_w * sizeof(float) + in_c * in_h * in_w * sizeof(int8_t);

    float *input_f32_ptr;
    int8_t *input_s8_ptr;
    if (dequant_need_buf_size < rem_buf_size) {
        input_f32_ptr = (void *)rem_buf_ptr;
        rem_buf_ptr += (dequant_need_buf_size + 31) & (~32);
        rem_buf_size -= (dequant_need_buf_size + 31) & (~32);
        input_s8_ptr = (void *)rem_buf_ptr;
        rem_buf_ptr += (dequant_need_buf_size + 31) & (~32);
        rem_buf_size -= (dequant_need_buf_size + 31) & (~32);
    } else {
        LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)dequant_need_buf_size);
    }


#pragma unroll 4
    for (int i = 0; i < in_c * in_h * in_w; ++i) {
        tmp = input_ptr[i] * opposite_ifmap_scale;
        input_f32_ptr[i] = (tmp < -128) ? -128 : (tmp > 127) ? 127 : tmp;
    }

#pragma omp parallel for num_threads(THREADS_NUM)
    for (int i = 0; i < in_c * in_h * in_w; i += 4) {
        input_s8_ptr[i] = (int8_t) (roundf(input_f32_ptr[i]));
        input_s8_ptr[i + 1] = (int8_t) (roundf(input_f32_ptr[i + 1]));
        input_s8_ptr[i + 2] = (int8_t) (roundf(input_f32_ptr[i + 2]));
        input_s8_ptr[i + 3] = (int8_t) (roundf(input_f32_ptr[i + 3]));
    }

    int8_t *ifmap_ptr = input_s8_ptr;

    int8_t *src_s8_pad_ptr;
    // do pad
    if (cfg->pads[0] != 0) {
        int64_t pad_need_buf_size = in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(int8_t);

        if (pad_need_buf_size < rem_buf_size) {
            src_s8_pad_ptr = (int8_t *)rem_buf_ptr;
            rem_buf_ptr += (pad_need_buf_size + 31) & (~32);
            rem_buf_size -= (pad_need_buf_size + 31) & (~32);
        } else {
            LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)pad_need_buf_size);
        }

        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv_s8((char *) src_s8_pad_ptr, (char *) input_s8_ptr, in_tensor, &pad_cfg);
        ifmap_ptr = (int8_t *) src_s8_pad_ptr;
    }

    int8_t *input_col_ptr;
    int64_t col_need_buf_size = in_c * kernel_w * kernel_h * out_h * out_w * sizeof(int8_t);

    if (col_need_buf_size < rem_buf_size) {
        input_col_ptr = (int8_t *)rem_buf_ptr;
        rem_buf_ptr += (col_need_buf_size + 31) & (~32);
        rem_buf_size -= (col_need_buf_size + 31) & (~32);
    } else {
        LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)col_need_buf_size);
    }

    im2col_s8(input_col_ptr, ifmap_ptr, in_tensor, out_tensor, weight_tensor, cfg);

    // weight and ifmap all s8
    const int M = out_c;
    const int N = out_h * out_w;
    const int K = in_c * kernel_h * kernel_w;

    int8_t *weight_s8_ptr = (int8_t *) weight_ptr;
    int32_t *bias_s32_ptr = (int32_t *) bias_ptr;

    int32_t *output_s32_ptr = (int32_t *) output_ptr;

#pragma omp parallel for num_threads(THREADS_NUM)
    for (int i = 0; i < M; i++) {
        memset(&output_ptr[i * N], 0, N * sizeof(float));
        int idx_a, idx_b, idx_c;
        idx_c = i * N;

        for (int k = 0; k < K; k++) {
            idx_a = i * K + k;
            idx_b = k * N;

            // s8 * s8
#pragma unroll 4
            for (int j = 0; j < N; ++j) {
                output_s32_ptr[idx_c + j] += weight_s8_ptr[idx_a] * input_col_ptr[idx_b + j];
            }
        }

#pragma unroll 8
        for (int z = 0; z < N; ++z) {
            output_ptr[idx_c + z] = output_s32_ptr[idx_c + z] * ifmap_scale;
            output_ptr[idx_c + z] += bias_s32_ptr[i];
        }
    }

    // trans output to float
    float *cur_output_ptr;
    for (int outc_i = 0; outc_i < out_c; ++outc_i) {
        cur_output_ptr = output_ptr + outc_i * out_h * out_w;
        float dequant_coeff = cfg->weight_aux[outc_i] * 1 / 127.0f;
        for (int i = 0; i < out_h * out_w; ++i) {
            cur_output_ptr[i] = cur_output_ptr[i] * dequant_coeff;
        }
    }

    return 0;
}

int
eval_mxn_img2col_W8A32_with_input_scale_no_bias(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);

    USEFUL_INFO_S* useful_info = (USEFUL_INFO_S *) (params[BUF_MAXNUM - 1].addr);
    int64_t public_buf_size = useful_info->public_buf_info.public_buf_size;
    int64_t public_buf_ptr = useful_info->public_buf_info.public_buf_ptr;
    int64_t rem_buf_size = public_buf_size;
    int64_t rem_buf_ptr = public_buf_ptr;

    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

    float *input_ptr = (float *) (inputs[0].addr);
    float *weight_ptr = (float *) (inputs[1].addr);

    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *weight_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

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

    // 将输入数据量化为 int8
    float tmp;
    float ifmap_scale = cfg->input_scale;
    float opposite_ifmap_scale = 1.0f / ifmap_scale;

    int32_t dequant_need_buf_size = in_c * in_h * in_w * sizeof(float) + in_c * in_h * in_w * sizeof(int8_t);

    float *input_f32_ptr;
    int8_t *input_s8_ptr;
    if (dequant_need_buf_size < rem_buf_size) {
        input_f32_ptr = (void *)rem_buf_ptr;
        rem_buf_ptr += (dequant_need_buf_size + 31) & (~32);
        rem_buf_size -= (dequant_need_buf_size + 31) & (~32);
        input_s8_ptr = (void *)rem_buf_ptr;
        rem_buf_ptr += (dequant_need_buf_size + 31) & (~32);
        rem_buf_size -= (dequant_need_buf_size + 31) & (~32);
    } else {
        LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)dequant_need_buf_size);
    }

#pragma unroll 4
    for (int i = 0; i < in_c * in_h * in_w; ++i) {
        tmp = input_ptr[i] * opposite_ifmap_scale;
        input_f32_ptr[i] = (tmp < -128) ? -128 : (tmp > 127) ? 127 : tmp;
    }

#pragma omp parallel for num_threads(THREADS_NUM)
    for (int i = 0; i < in_c * in_h * in_w; i += 4) {
        input_s8_ptr[i] = (int8_t) (roundf(input_f32_ptr[i]));
        input_s8_ptr[i + 1] = (int8_t) (roundf(input_f32_ptr[i + 1]));
        input_s8_ptr[i + 2] = (int8_t) (roundf(input_f32_ptr[i + 2]));
        input_s8_ptr[i + 3] = (int8_t) (roundf(input_f32_ptr[i + 3]));
    }

    int8_t *ifmap_ptr = input_s8_ptr;

    int8_t *src_s8_pad_ptr;
    // do pad
    if (cfg->pads[0] != 0) {
        int64_t pad_need_buf_size = in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(int8_t);

        if (pad_need_buf_size < rem_buf_size) {
            src_s8_pad_ptr = (int8_t *)rem_buf_ptr;
            rem_buf_ptr += (pad_need_buf_size + 31) & (~32);
            rem_buf_size -= (pad_need_buf_size + 31) & (~32);
        } else {
            LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)pad_need_buf_size);
        }

        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv_s8((char *) src_s8_pad_ptr, (char *) input_s8_ptr, in_tensor, &pad_cfg);
        ifmap_ptr = (int8_t *) src_s8_pad_ptr;
    }

    int8_t *input_col_ptr;
    int64_t col_need_buf_size = in_c * kernel_w * kernel_h * out_h * out_w * sizeof(int8_t);

    if (col_need_buf_size < rem_buf_size) {
        input_col_ptr = (int8_t *)rem_buf_ptr;
        rem_buf_ptr += (col_need_buf_size + 31) & (~32);
        rem_buf_size -= (col_need_buf_size + 31) & (~32);
    } else {
        LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)col_need_buf_size);
    }

    im2col_s8(input_col_ptr, ifmap_ptr, in_tensor, out_tensor, weight_tensor, cfg);

    // weight and ifmap all s8
    const int M = out_c;
    const int N = out_h * out_w;
    const int K = in_c * kernel_h * kernel_w;

    int8_t *weight_s8_ptr = (int8_t *) weight_ptr;

    int32_t *output_s32_ptr = (int32_t *) output_ptr;

#pragma omp parallel for num_threads(THREADS_NUM)
    for (int i = 0; i < M; i++) {
        memset(&output_ptr[i * N], 0, N * sizeof(float));
        int idx_a, idx_b, idx_c;
        idx_c = i * N;

        for (int k = 0; k < K; k++) {
            idx_a = i * K + k;
            idx_b = k * N;

            // s8 * s8
#pragma unroll 4
            for (int j = 0; j < N; ++j) {
                output_s32_ptr[idx_c + j] += weight_s8_ptr[idx_a] * input_col_ptr[idx_b + j];
            }
        }

#pragma unroll 8
        for (int z = 0; z < N; ++z) {
            output_ptr[idx_c + z] = output_s32_ptr[idx_c + z] * ifmap_scale;
        }
    }

    // trans output to float
    float *cur_output_ptr;
    for (int outc_i = 0; outc_i < out_c; ++outc_i) {
        cur_output_ptr = output_ptr + outc_i * out_h * out_w;
        float dequant_coeff = cfg->weight_aux[outc_i] * 1 / 127.0f;
        for (int i = 0; i < out_h * out_w; ++i) {
            cur_output_ptr[i] = cur_output_ptr[i] * dequant_coeff;
        }
    }

    return 0;
}

int do_act(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs){

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);

    float *output_ptr = (float *) (outputs[0].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[4].addr);
    int32_t out_n = out_tensor->shapes[0];
    int32_t out_c = out_tensor->shapes[1];
    int32_t out_h = out_tensor->shapes[2];
    int32_t out_w = out_tensor->shapes[3];
    int32_t ofmap_elem_size = out_n * out_c * out_h * out_w;

    if (cfg->act_type == RELU) {
        int aligned_size = (ofmap_elem_size / 8) * 8;

        float min =  0.0f;
        __m256 min_vec = _mm256_set1_ps(min);  // 将min广播到所有元素

#pragma omp parallel for num_threads(THREADS_NUM)
        for (int elem_i = 0; elem_i < aligned_size; elem_i += 8) {
            __m256 vec = _mm256_loadu_ps(&output_ptr[elem_i]);
            // 执行clip操作
            vec = _mm256_max_ps(vec, min_vec);
            // 存储结果
            _mm256_storeu_ps(&output_ptr[elem_i], vec);
        }

        for (int32_t elem_i = aligned_size; elem_i < ofmap_elem_size; elem_i++) {
            output_ptr[elem_i] = (output_ptr[elem_i] > min) ? output_ptr[elem_i] : min;
        }
    } else if (cfg->act_type == SILU) {
        float *silu_lut = (float *) (inputs[BUF_MAXNUM - 1].addr);
        float single_limit = 8.0f;
        float inv_single_limit = -1 * single_limit;
        float step = 1.0f / 512;
        float inv_step = 1.0f / step;
        for (int elem_i = 0; elem_i < ofmap_elem_size; ++elem_i) {
            float cur_val = output_ptr[elem_i];
            if (cur_val < inv_single_limit) {
                output_ptr[elem_i] = 0;
            } else if (cur_val >= single_limit) { ;
            } else if (cur_val >= inv_single_limit && cur_val < single_limit) {
                int idx = (int) ((cur_val + single_limit) * inv_step);
                output_ptr[elem_i] = silu_lut[idx];
            }
        }
    } else if (cfg->act_type == CLIP) {
        int aligned_size = (ofmap_elem_size / 8) * 8;

        float max = cfg->clip_max;
        float min = cfg->clip_min;
        __m256 min_vec = _mm256_set1_ps(min);  // 将min广播到所有元素
        __m256 max_vec = _mm256_set1_ps(max);  // 将max广播到所有元素

#pragma omp parallel for num_threads(THREADS_NUM)
        for (int elem_i = 0; elem_i < aligned_size; elem_i += 8) {
            __m256 vec = _mm256_loadu_ps(&output_ptr[elem_i]);
            // 执行clip操作
            vec = _mm256_max_ps(vec, min_vec);
            vec = _mm256_min_ps(vec, max_vec);
            // 存储结果
            _mm256_storeu_ps(&output_ptr[elem_i], vec);
        }

        for (int32_t elem_i = aligned_size; elem_i < ofmap_elem_size; elem_i++) {
            output_ptr[elem_i] = (output_ptr[elem_i] > min) ? output_ptr[elem_i] : min;
            output_ptr[elem_i] = (output_ptr[elem_i] < max) ? output_ptr[elem_i] : max;
        }

    } else if (cfg->act_type == LEAKYRELU) {
        float alpha = cfg->leaky_relu_alpha;
        for (int elem_i = 0; elem_i < ofmap_elem_size; ++elem_i) {
            output_ptr[elem_i] = (output_ptr[elem_i] > 0.f) ? output_ptr[elem_i] : alpha * output_ptr[elem_i];
        }
    } else if (cfg->act_type == HARDSILU) {
        float tmp;
        float alpha = cfg->hard_sigmoid_alpha;
        float beta = cfg->hard_sigmoid_beta;
        // follow min and max value is define of hard sigmoid, you can see https://onnx.ai/onnx/operators/onnx__HardSigmoid.html
        float min = 0, max = 1;
        for (int elem_i = 0; elem_i < ofmap_elem_size; ++elem_i) {
            tmp = alpha * output_ptr[elem_i] + beta;
            tmp = (tmp > min) ? tmp : min;
            output_ptr[elem_i] = output_ptr[elem_i] * ((tmp < max) ? tmp : max);
        }
    }

    return 0;
}

