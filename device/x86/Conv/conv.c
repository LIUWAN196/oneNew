#include "conv.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
//#include "pad_conv.h"
#include "stdint.h"
#include "string.h"
#include <immintrin.h>
#include <omp.h>
#include "math.h"


typedef struct {
    int32_t M;
    int32_t N;
    int32_t K;

    int32_t m_tile_size;
    int32_t n_tile_size;
    int32_t k_tile_size;
} GEMM_TILE_INFO;

int opt_gemm(float *ofmap_ptr, float *ifmap0_ptr, float *ifmap1_ptr, GEMM_TILE_INFO gemm_tile_info) {

    int32_t M = gemm_tile_info.M;
    int32_t N = gemm_tile_info.N;
    int32_t K = gemm_tile_info.K;

    const int32_t best_m_tile = gemm_tile_info.m_tile_size;
    const int32_t best_n_tile = gemm_tile_info.n_tile_size;
    const int32_t best_k_tile = gemm_tile_info.k_tile_size;
    int32_t m_tile_size, n_tile_size, k_tile_size;
    if (M < best_m_tile) {
        m_tile_size = M;
    } else {
        int32_t m_tile_cnt = (int32_t) ceilf((float) M / (float) best_m_tile);
        m_tile_size = M / m_tile_cnt;
    }

    if (N < best_n_tile) {
        // 因为在 matmul 中对 n_tile 使用了 avx2 的并行计算，并行度为 8, 所以这里 n_tile_size 需要是 8 的整数倍
        n_tile_size = (N >> 3) << 3;
    } else {
        int32_t n_tile_cnt = (int32_t) ceilf((float) N / (float) best_n_tile);
        n_tile_size = ((N / n_tile_cnt) >> 3) << 3;
    }

    if (K < best_k_tile) {
        // 因为在 sgemm 中对 k_tile 做了 unroll = 4 的展开, 所以这里 k_tile_size 需要是 4 的整数倍
        k_tile_size = (K >> 2) << 2;
    } else {
        k_tile_size = best_k_tile;
    }

    // init ofmap buf
    memset(ofmap_ptr, 0, M * N * sizeof(float));

    // set tile params
    int m_tile_cnt = M / m_tile_size;
    int n_tile_cnt = (n_tile_size == 0) ? 0 : N / n_tile_size;
    int k_tile_cnt = (k_tile_size == 0) ? 0 : K / k_tile_size;

    // step 1: 这个循环处理 M、N、K 的整块部分
#pragma omp parallel for num_threads(8)
    for (int m_tile_i = 0; m_tile_i < m_tile_cnt; m_tile_i++) {
        register int ofmap_offset = 0, ifmap0_offset = 0, ifmap1_offset = 0;
        register float *cur_ofmap_ptr, *cur_ifmap0_st, *cur_ifmap1_st, *cur_ifmap1_ptr[4];
        register __m256 psum_vec, ifmap0_vec[4], mul_res_vec[4];
        for (int n_tile_i = 0; n_tile_i < n_tile_cnt; n_tile_i++) {
            for (int k_tile_i = 0; k_tile_i < k_tile_cnt; k_tile_i++) {
                ofmap_offset = (m_tile_i * m_tile_size) * N + n_tile_i * n_tile_size;
                ifmap0_offset = (m_tile_i * m_tile_size) * K + k_tile_i * k_tile_size;
                ifmap1_offset = (k_tile_i * k_tile_size) * N + n_tile_i * n_tile_size;
                for (int m_i = 0; m_i < m_tile_size; m_i++) {
                    cur_ofmap_ptr = ofmap_ptr + ofmap_offset + m_i * N;
                    for (int k_i = 0; k_i < k_tile_size; k_i += 4) {
                        cur_ifmap0_st = ifmap0_ptr + ifmap0_offset + m_i * K + k_i;
                        cur_ifmap1_st = ifmap1_ptr + ifmap1_offset + k_i * N;
                        ifmap0_vec[0] = _mm256_set1_ps(*(cur_ifmap0_st + 0));
                        ifmap0_vec[1] = _mm256_set1_ps(*(cur_ifmap0_st + 1));
                        ifmap0_vec[2] = _mm256_set1_ps(*(cur_ifmap0_st + 2));
                        ifmap0_vec[3] = _mm256_set1_ps(*(cur_ifmap0_st + 3));
                        cur_ifmap1_ptr[0] = cur_ifmap1_st + 0 * N;
                        cur_ifmap1_ptr[1] = cur_ifmap1_st + 1 * N;
                        cur_ifmap1_ptr[2] = cur_ifmap1_st + 2 * N;
                        cur_ifmap1_ptr[3] = cur_ifmap1_st + 3 * N;
                        for (int n_i = 0; n_i < n_tile_size; n_i += 8) {
                            mul_res_vec[0] = _mm256_mul_ps(ifmap0_vec[0], _mm256_loadu_ps(cur_ifmap1_ptr[0] + n_i));
                            mul_res_vec[1] = _mm256_mul_ps(ifmap0_vec[1], _mm256_loadu_ps(cur_ifmap1_ptr[1] + n_i));
                            mul_res_vec[2] = _mm256_mul_ps(ifmap0_vec[2], _mm256_loadu_ps(cur_ifmap1_ptr[2] + n_i));
                            mul_res_vec[3] = _mm256_mul_ps(ifmap0_vec[3], _mm256_loadu_ps(cur_ifmap1_ptr[3] + n_i));
                            psum_vec = _mm256_loadu_ps(cur_ofmap_ptr + n_i);
                            psum_vec = _mm256_add_ps(psum_vec, mul_res_vec[0]);
                            psum_vec = _mm256_add_ps(psum_vec, mul_res_vec[1]);
                            psum_vec = _mm256_add_ps(psum_vec, mul_res_vec[2]);
                            psum_vec = _mm256_add_ps(psum_vec, mul_res_vec[3]);
                            _mm256_storeu_ps(cur_ofmap_ptr + n_i, psum_vec);
                        }
                    }
                }
            }
        }
    }

    // step 2: 这个循环处理 N 的尾部部分，即 [N - n_tile_cnt * n_tile_size, N] 部分
    const int32_t n_tail = N - n_tile_cnt * n_tile_size;
    if (n_tail != 0) {
#pragma omp parallel for num_threads(8)
        for (int m_tile_i = 0; m_tile_i < m_tile_cnt; m_tile_i++) {
            register int ofmap_offset = 0, ifmap0_offset = 0, ifmap1_offset = 0;
            register float *cur_ofmap_ptr, *cur_ifmap0_st, *cur_ifmap1_st, *cur_ifmap1_ptr;
            register float psum_vec;
            register float ifmap0;
            for (int k_tile_i = 0; k_tile_i < k_tile_cnt; k_tile_i++) {
                ofmap_offset = (m_tile_i * m_tile_size) * N + n_tile_cnt * n_tile_size;
                ifmap0_offset = (m_tile_i * m_tile_size) * K + k_tile_i * k_tile_size;
                ifmap1_offset = (k_tile_i * k_tile_size) * N + n_tile_cnt * n_tile_size;
                for (int m_i = 0; m_i < m_tile_size; m_i++) {
                    cur_ofmap_ptr = ofmap_ptr + ofmap_offset + m_i * N;
                    for (int k_i = 0; k_i < k_tile_size; k_i ++) {
                        cur_ifmap0_st = ifmap0_ptr + ifmap0_offset + m_i * K + k_i;
                        cur_ifmap1_st = ifmap1_ptr + ifmap1_offset + k_i * N;
                        ifmap0 = *(cur_ifmap0_st + 0);
                        cur_ifmap1_ptr = cur_ifmap1_st + 0 * N;
                        for (int n_i = 0; n_i < n_tail; n_i++) {
                            psum_vec = ifmap0 * (*(cur_ifmap1_ptr + n_i));
                            *(cur_ofmap_ptr + n_i) += psum_vec;
                        }
                    }
                }
            }
        }
    }

    // step 3: 这个循环处理 M 的尾部部分，即 [M - m_tile_cnt * m_tile_size, M] 部分
    const int32_t m_tail = M - m_tile_cnt * m_tile_size;
    if (m_tail != 0) {
#pragma omp parallel for num_threads(8)
        for (int m_i = 0; m_i < m_tail; m_i++) {
            register int ofmap_offset = (m_tile_cnt * m_tile_size) * N;
            register int ifmap0_offset = (m_tile_cnt * m_tile_size) * K;
            register float *cur_ofmap_ptr;
            register float ifmap0_val;
            register int k_i, n_i;
            register __m256 ifmap0_vec, ifmap1_vec, mul_res_vec, psum_vec;
            // 注意这里的 k 只需要处理到 k_tile_cnt * k_tile_size，后面的 k 在 step4 中计算
            for (k_i = 0; k_i < k_tile_cnt * k_tile_size; k_i++) {
                ifmap0_val = ifmap0_ptr[ifmap0_offset + m_i * K + k_i];
                ifmap0_vec = _mm256_set1_ps(*(ifmap0_ptr + ifmap0_offset + m_i * K + k_i));
                cur_ofmap_ptr = ofmap_ptr + ofmap_offset + m_i * N;
                for (n_i = 0; n_i < N - 7; n_i += 8) {
                    ifmap1_vec = _mm256_loadu_ps(ifmap1_ptr + k_i * N + n_i);
                    mul_res_vec = _mm256_mul_ps(ifmap0_vec, ifmap1_vec);
                    psum_vec = _mm256_loadu_ps(cur_ofmap_ptr + n_i);
                    _mm256_storeu_ps(cur_ofmap_ptr + n_i, _mm256_add_ps(psum_vec, mul_res_vec));
                }
                for (; n_i < N; ++n_i) {
                    *(cur_ofmap_ptr + n_i) += ifmap0_val * ifmap1_ptr[k_i * N + n_i];
                }
            }
        }
    }

    // step 4: 这个循环处理 K 的尾部部分，即 [K - k_tile_cnt * k_tile_size, K] 部分
    const int32_t k_tail = K - k_tile_cnt * k_tile_size;
    int32_t k_tail_st = k_tile_cnt * k_tile_size;
    if (k_tail != 0) {
#pragma omp parallel for num_threads(8)
        for (int m_i = 0; m_i < M; ++m_i) {
            for (int k_i = k_tail_st; k_i < K; ++k_i) {
                register float ifmap0_val = ifmap0_ptr[m_i * K + k_i];
                for (int n_i = 0; n_i < N; ++n_i) {
                    ofmap_ptr[m_i * N + n_i] += ifmap0_val * ifmap1_ptr[k_i * N + n_i];
                }
            }
        }
    }
    return 0;
}


int do_pad_conv(char *dst_ptr, char *src_ptr, OPERAND_S *src_data_desc, PAD_INNER_CONFIG_S *cfg) {
    float pad_value = 0;
    float *dst_f32 = (float *) dst_ptr;
    float *src_f32 = (float *) src_ptr;

    int32_t src_c = src_data_desc->shapes[1];
    int32_t src_h = src_data_desc->shapes[2];
    int32_t src_w = src_data_desc->shapes[3];

    int32_t dst_c = src_data_desc->shapes[1];
    int32_t dst_h = src_h + 2 * cfg->h;
    int32_t dst_w = src_w + 2 * cfg->w;

    for (int i = 0; i < dst_c * dst_h * dst_w; ++i) {
        dst_f32[i] = pad_value;
    }
//    memset(dst_ptr, dst_c * dst_h * dst_w * sizeof(float), pad_value);

    for (int c_i = 0; c_i < dst_c; ++c_i) {
        for (int h_i = cfg->h; h_i < dst_h - cfg->h; ++h_i) {
            for (int w_i = cfg->w; w_i < dst_w - cfg->w; ++w_i) {
                dst_f32[c_i * dst_h * dst_w + h_i * dst_w + w_i] = src_f32[c_i * src_h * src_w +
                                                                           (h_i - cfg->h) * src_w + (w_i - cfg->w)];
            }

        }
    }


    return 0;
}

int do_pad_conv_s8(char *dst_ptr, char *src_ptr, OPERAND_S *src_data_desc, PAD_INNER_CONFIG_S *cfg) {
    int8_t pad_value = 0;
    int8_t *dst_f32 = (int8_t *) dst_ptr;
    int8_t *src_f32 = (int8_t *) src_ptr;

    int32_t src_c = src_data_desc->shapes[1];
    int32_t src_h = src_data_desc->shapes[2];
    int32_t src_w = src_data_desc->shapes[3];

    int32_t dst_c = src_data_desc->shapes[1];
    int32_t dst_h = src_h + 2 * cfg->h;
    int32_t dst_w = src_w + 2 * cfg->w;

    for (int i = 0; i < dst_c * dst_h * dst_w; ++i) {
        dst_f32[i] = pad_value;
    }
//    memset(dst_ptr, dst_c * dst_h * dst_w * sizeof(float), pad_value);

    for (int c_i = 0; c_i < dst_c; ++c_i) {
        for (int h_i = cfg->h; h_i < dst_h - cfg->h; ++h_i) {
            for (int w_i = cfg->w; w_i < dst_w - cfg->w; ++w_i) {
                dst_f32[c_i * dst_h * dst_w + h_i * dst_w + w_i] = src_f32[c_i * src_h * src_w +
                                                                           (h_i - cfg->h) * src_w + (w_i - cfg->w)];
            }

        }
    }


    return 0;
}


int do_pad_conv_new(char *dst_ptr, char *src_ptr, OPERAND_S *src_data_desc, PAD_INNER_CONFIG_S *pad_cfg) {

    float *dst_f32 = (float *) dst_ptr;
    float *src_f32 = (float *) src_ptr;

    int32_t src_c = src_data_desc->shapes[1];
    int32_t src_h = src_data_desc->shapes[2];
    int32_t src_w = src_data_desc->shapes[3];

    int32_t dst_c = src_data_desc->shapes[1];
    int32_t dst_h = src_h + pad_cfg->top_pad + pad_cfg->bottom_pad;
    int32_t dst_w = src_w + pad_cfg->left_pad + pad_cfg->right_pad;

    memset(dst_ptr, 0, dst_c * dst_h * dst_w * sizeof(float));

    const int32_t top_pad = pad_cfg->top_pad;
    const int32_t bottom_pad = pad_cfg->bottom_pad;
    const int32_t left_pad = pad_cfg->left_pad;

    for (int c_i = 0; c_i < dst_c; ++c_i) {
        float* cur_src_f32 = src_f32 + c_i * src_h * src_w - top_pad * src_w;
        float* cur_dst_f32 = dst_f32 + c_i * dst_h * dst_w + left_pad;
        int h_i = top_pad;
        for (h_i = top_pad; h_i < dst_h - bottom_pad - 3; h_i += 4) {
            memcpy(cur_dst_f32 + (h_i + 0) * dst_w, cur_src_f32 + (h_i + 0) * src_w, src_w * sizeof(float));
            memcpy(cur_dst_f32 + (h_i + 1) * dst_w, cur_src_f32 + (h_i + 1) * src_w, src_w * sizeof(float));
            memcpy(cur_dst_f32 + (h_i + 2) * dst_w, cur_src_f32 + (h_i + 2) * src_w, src_w * sizeof(float));
            memcpy(cur_dst_f32 + (h_i + 3) * dst_w, cur_src_f32 + (h_i + 3) * src_w, src_w * sizeof(float));
        }
        for (; h_i < dst_h - bottom_pad; h_i++) {
            memcpy(cur_dst_f32 + h_i * dst_w, cur_src_f32 + h_i * src_w, src_w * sizeof(float));
        }
    }

    return 0;
}


int eval_1x1j1(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
//     printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

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
        src_pad_ptr = malloc(in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(float));
        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv(src_pad_ptr, input_ptr, in_tensor, &pad_cfg);
        input_ptr = (float *) src_pad_ptr;
    }

    GEMM_TILE_INFO gemm_tile_info;
    gemm_tile_info.M = out_c;
    gemm_tile_info.N = out_h * out_w;
    gemm_tile_info.K = in_c;

    // 目前认为的 sgemm 的最佳配置为 m_tile_size = 32，n_tile_size = 1024，k_tile_size = 8；
    const int32_t best_n_tile = 1024, best_k_tile = 8;
    const int32_t num_threads = 8;
    // 这里因为开了 num_threads 个线程，所以要保证 m_tile 为 num_threads 的倍数，不让有的线程创建了但是没有使用
    int32_t best_m_tile = (gemm_tile_info.M / num_threads > 32) ? 32 : gemm_tile_info.M / num_threads;
    best_m_tile = (best_m_tile == 0) ? gemm_tile_info.M : best_m_tile;

    gemm_tile_info.m_tile_size = best_m_tile;
    gemm_tile_info.n_tile_size = best_n_tile;
    gemm_tile_info.k_tile_size = best_k_tile;
    opt_gemm(output_ptr, weight_ptr, input_ptr, gemm_tile_info);

    // 加偏置
#pragma omp parallel for num_threads(8)
    for (int i = 0; i < gemm_tile_info.M; i++)
    {
        int idx = i * gemm_tile_info.N;
#pragma unroll 8
        for (int z = 0; z < gemm_tile_info.N; ++z) {
            output_ptr[idx + z] += bias_ptr[i];
        }
    }



//
////    memset(output_ptr, 0, M * N * sizeof(float));
//
//#pragma omp parallel for num_threads(8)
//    for (int i = 0; i < M; i++)
//    {
//        memset(&output_ptr[i * N], 0, N * sizeof(float));
//        int idx_a, idx_b, idx_c;
//        idx_c = i * N;
//#pragma unroll 8
//        for (int k = 0; k < K; k++)
//        {
//            idx_a = i * K + k;
//            idx_b = k * N;
//
//            int j = 0;
//            for (; j < N; j++)
//            {
//                output_ptr[idx_c + j] += weight_ptr[idx_a] * input_ptr[idx_b + j];
//            }
//        }
//#pragma unroll 8
//        for (int z = 0; z < N; ++z) {
//            output_ptr[idx_c + z] += bias_ptr[i];
//        }
//    }

    if (cfg->pads[0] != 0) {
        free(src_pad_ptr);
    }

    return 0;
}

int im2col(float *input_col_ptr, float *input_ptr, OPERAND_S *in_tensor, OPERAND_S *out_tensor, OPERAND_S *weight_tensor,
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
#pragma omp parallel for num_threads(8)
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
#pragma omp parallel for num_threads(8)
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


int im2col_s8(int8_t *input_col_ptr, int8_t *input_ptr, OPERAND_S *in_tensor, OPERAND_S *out_tensor, OPERAND_S *weight_tensor,
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

#pragma omp parallel for num_threads(8)
    for (int col_h1 = 0; col_h1 <in_c; ++col_h1) {
        for (int col_h2 = 0; col_h2 < kernel_h; ++col_h2) {
            for (int col_h3 = 0; col_h3 < kernel_w; ++col_h3) {
#pragma unroll 8
                for (int col_w = 0; col_w < out_h * out_w; ++col_w) {
                    int cur_in_h = col_w / out_w * stride_y + col_h2;
                    int cur_in_w = col_w % out_w * stride_x + col_h3;
                    input_col_ptr[col_h1 * kernel_h * kernel_w * out_h * out_w + col_h2 * kernel_w * out_h * out_h +
                                  col_h3 * out_h * out_h + col_w] = input_ptr[col_h1 * in_h * in_w + cur_in_h * in_w + cur_in_w];
                }
            }
        }
    }

    return 0;
}


int eval_mxn_img2col(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
//    // printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

//    if (strcmp(cfg->op_base_cfg.op_name, "Conv_80") == 0) {
//
//        int a = 101;
//    }
    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

    float *input_ptr = (float *) (inputs[0].addr);
    float *weight_ptr = (float *) (inputs[1].addr);

    // printf("input is %f, %f\n", input_ptr[0], input_ptr[1]);
    // printf("weight is %f, %f\n", weight_ptr[0], weight_ptr[1]);

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
        src_pad_ptr = malloc(in_n * in_c * (in_h + pad_cfg.top_pad + pad_cfg.bottom_pad) * (in_w + pad_cfg.left_pad + pad_cfg.right_pad) * sizeof(float));

        in_h = in_h + pad_cfg.top_pad + pad_cfg.bottom_pad;
        in_w = in_w + pad_cfg.left_pad + pad_cfg.right_pad;

        do_pad_conv_new(src_pad_ptr, input_ptr, in_tensor, &pad_cfg);
        input_ptr = (float *) src_pad_ptr;
    }

    float *input_col_ptr = malloc(in_c * kernel_w * kernel_h * out_h * out_w * sizeof(float));
    im2col(input_col_ptr, input_ptr, in_tensor, out_tensor, weight_tensor, cfg);
    input_ptr = input_col_ptr;


    GEMM_TILE_INFO gemm_tile_info;
    gemm_tile_info.M = out_c;
    gemm_tile_info.N = out_h * out_w;
    gemm_tile_info.K = in_c * kernel_h * kernel_w;

    // 目前认为的 sgemm 的最佳配置为 m_tile_size = 32，n_tile_size = 1024，k_tile_size = 8；
    const int32_t best_n_tile = 1024, best_k_tile = 8;
    const int32_t num_threads = 8;
    // 这里因为开了 num_threads 个线程，所以要保证 m_tile 为 num_threads 的倍数，不让有的线程创建了但是没有使用
    int32_t best_m_tile = (gemm_tile_info.M / num_threads > 32) ? 32 : gemm_tile_info.M / num_threads;
    best_m_tile = (best_m_tile == 0) ? gemm_tile_info.M : best_m_tile;

    gemm_tile_info.m_tile_size = best_m_tile;
    gemm_tile_info.n_tile_size = best_n_tile;
    gemm_tile_info.k_tile_size = best_k_tile;
    opt_gemm(output_ptr, weight_ptr, input_ptr, gemm_tile_info);

    // 加偏置
#pragma omp parallel for num_threads(8)
    for (int i = 0; i < gemm_tile_info.M; i++)
    {
        int idx = i * gemm_tile_info.N;
#pragma unroll 8
        for (int z = 0; z < gemm_tile_info.N; ++z) {
            output_ptr[idx + z] += bias_ptr[i];
        }
    }



//    const int M = out_c;
//    const int N = out_h * out_w;
//    const int K = in_c * kernel_h * kernel_w;
//
//#pragma omp parallel for num_threads(8)
//    for (int i = 0; i < M; i++)
//    {
//        memset(&output_ptr[i * N], 0, N * sizeof(float));
//        int idx_a, idx_b, idx_c;
//        __m256 psum;
//        __m256 weight_vec;
//        __m256 sum_pre;
//        idx_c = i * N;
//#pragma unroll 8
//        for (int k = 0; k < K; k++)
//        {
//            idx_a = i * K + k;
//            idx_b = k * N;
//            int j = 0;
//            weight_vec = _mm256_set1_ps(weight_ptr[idx_a]);
//#pragma unroll 2
//            for (; j < N - 7; j+=8)
//            {
//                sum_pre = _mm256_loadu_ps(&output_ptr[idx_c + j]);
//                sum_pre = _mm256_fmadd_ps(weight_vec, _mm256_loadu_ps(&input_ptr[idx_b + j]), sum_pre);
//                _mm256_storeu_ps(&output_ptr[idx_c + j], sum_pre);
//            }
//            for (; j < N; ++j) {
//                output_ptr[idx_c + j] += weight_ptr[idx_a] * input_ptr[idx_b + j];
//            }
//        }
//#pragma unroll 8
//        for (int z = 0; z < N; ++z) {
//            output_ptr[idx_c + z] += bias_ptr[i];
//        }
//    }

    if (cfg->pads[0] != 0) {
        free(src_pad_ptr);
    }
    free(input_col_ptr);

    return 0;
}

int eval_mxn_img2col_no_bias(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
//    // printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

//    if (strcmp(cfg->op_base_cfg.op_name, "Conv_80") == 0) {
//
//        int a = 101;
//    }
    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

    float *input_ptr = (float *) (inputs[0].addr);
    float *weight_ptr = (float *) (inputs[1].addr);

    // printf("input is %f, %f\n", input_ptr[0], input_ptr[1]);
    // printf("weight is %f, %f\n", weight_ptr[0], weight_ptr[1]);

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
        src_pad_ptr = malloc(in_n * in_c * (in_h + pad_cfg.top_pad + pad_cfg.bottom_pad) * (in_w + pad_cfg.left_pad + pad_cfg.right_pad) * sizeof(float));

        in_h = in_h + pad_cfg.top_pad + pad_cfg.bottom_pad;
        in_w = in_w + pad_cfg.left_pad + pad_cfg.right_pad;

        do_pad_conv_new(src_pad_ptr, input_ptr, in_tensor, &pad_cfg);
        input_ptr = (float *) src_pad_ptr;
    }

    float *input_col_ptr = malloc(in_c * kernel_w * kernel_h * out_h * out_w * sizeof(float));
    im2col(input_col_ptr, input_ptr, in_tensor, out_tensor, weight_tensor, cfg);
    input_ptr = input_col_ptr;


    const int M = out_c;
    const int N = out_h * out_w;
    const int K = in_c * kernel_h * kernel_w;

#pragma omp parallel for num_threads(8)
    for (int i = 0; i < M; i++)
    {
        memset(&output_ptr[i * N], 0, N * sizeof(float));
        int idx_a, idx_b, idx_c;
        __m256 psum;
        __m256 weight_vec;
        __m256 sum_pre;
        idx_c = i * N;
#pragma unroll 8
        for (int k = 0; k < K; k++)
        {
            idx_a = i * K + k;
            idx_b = k * N;
            int j = 0;
            weight_vec = _mm256_set1_ps(weight_ptr[idx_a]);
#pragma unroll 2
            for (; j < N - 7; j+=8)
            {
                sum_pre = _mm256_loadu_ps(&output_ptr[idx_c + j]);
                sum_pre = _mm256_fmadd_ps(weight_vec, _mm256_loadu_ps(&input_ptr[idx_b + j]), sum_pre);
                _mm256_storeu_ps(&output_ptr[idx_c + j], sum_pre);
            }
            for (; j < N; ++j) {
                output_ptr[idx_c + j] += weight_ptr[idx_a] * input_ptr[idx_b + j];
            }
        }
    }

    if (cfg->pads[0] != 0) {
        free(src_pad_ptr);
    }
    free(input_col_ptr);

    return 0;
}


int eval_depthwise_conv_mxn_img2col(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);

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
        src_pad_ptr = malloc(in_n * in_c * (in_h + pad_cfg.top_pad + pad_cfg.bottom_pad) * (in_w + pad_cfg.left_pad + pad_cfg.right_pad) * sizeof(float));

        in_h = in_h + pad_cfg.top_pad + pad_cfg.bottom_pad;
        in_w = in_w + pad_cfg.left_pad + pad_cfg.right_pad;

        do_pad_conv_new(src_pad_ptr, input_ptr, in_tensor, &pad_cfg);
        input_ptr = (float *) src_pad_ptr;
    }

    for (int outc_i = 0; outc_i < out_c; ++outc_i) {
        float *cur_weight_ptr = weight_ptr + outc_i * kernel_h * kernel_w;
        for (int outh_i = 0; outh_i < out_h; ++outh_i) {
            for (int outw_i = 0; outw_i < out_w; ++outw_i) {
                float psum = 0;
                float* cur_ofmp_ptr = output_ptr + outc_i * out_h * out_w + outh_i * out_w + outw_i;
                float* cur_ifmp_ptr = input_ptr + outc_i * in_h * in_w + outh_i * stride_y * in_w + outw_i * stride_x;
                for (int kh_i = 0; kh_i < kernel_h; ++kh_i) {
                    for (int kw_i = 0; kw_i < kernel_w; ++kw_i) {
                        psum += cur_weight_ptr[kh_i * kernel_w + kw_i] * cur_ifmp_ptr[kh_i * in_w + kw_i];
                    }
                }
                cur_ofmp_ptr[0] = psum + bias_ptr[outc_i];
            }
        }
    }


    if (cfg->pads[0] != 0) {
        free(src_pad_ptr);
    }

    return 0;
}

int eval_depthwise_conv_3x3_img2col(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);

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
        src_pad_ptr = malloc(in_n * in_c * (in_h + pad_cfg.top_pad + pad_cfg.bottom_pad) * (in_w + pad_cfg.left_pad + pad_cfg.right_pad) * sizeof(float));

        in_h = in_h + pad_cfg.top_pad + pad_cfg.bottom_pad;
        in_w = in_w + pad_cfg.left_pad + pad_cfg.right_pad;

        do_pad_conv_new(src_pad_ptr, input_ptr, in_tensor, &pad_cfg);
        input_ptr = (float *) src_pad_ptr;
    }

#pragma omp parallel for num_threads(8)
    for (int outc_i = 0; outc_i < out_c; ++outc_i) {
        for (int outh_i = 0; outh_i < out_h; ++outh_i) {
            float *cur_weight_ptr = weight_ptr + outc_i * kernel_h * kernel_w;
            float* cur_ofmp_ptr = output_ptr + outc_i * out_h * out_w + outh_i * out_w;
            float* cur_ifmp_ptr = input_ptr + outc_i * in_h * in_w + outh_i * stride_y * in_w;
#pragma unroll 4
            for (int outw_i = 0; outw_i < out_w; ++outw_i) {
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
        }
    }


    if (cfg->pads[0] != 0) {
        free(src_pad_ptr);
    }

    return 0;
}

int eval_mxn_img2col_W8A32_with_input_scale(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);

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
    float *input_f32_ptr = (float *)malloc(in_c * in_h * in_w * sizeof(float));
    int8_t *input_s8_ptr = (int8_t *)malloc(in_c * in_h * in_w * sizeof(int8_t));

#pragma unroll 4
    for (int i = 0; i < in_c * in_h * in_w; ++i) {
        tmp = input_ptr[i] * opposite_ifmap_scale;
        input_f32_ptr[i] = (tmp < -128) ? -128 : (tmp > 127) ? 127 : tmp;
    }

#pragma omp parallel for num_threads(8)
    for (int i = 0; i < in_c * in_h * in_w; i +=4) {
        input_s8_ptr[i] = (int8_t)(roundf(input_f32_ptr[i]));
        input_s8_ptr[i + 1] = (int8_t)(roundf(input_f32_ptr[i + 1]));
        input_s8_ptr[i + 2] = (int8_t)(roundf(input_f32_ptr[i + 2]));
        input_s8_ptr[i + 3] = (int8_t)(roundf(input_f32_ptr[i + 3]));
    }

    int8_t* ifmap_ptr = input_s8_ptr;

    int8_t *src_s8_pad_ptr;
    // do pad
    if (cfg->pads[0] != 0) {
        src_s8_pad_ptr = (int8_t *)malloc(in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(int8_t));
        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv_s8((char *)src_s8_pad_ptr, (char *)input_s8_ptr, in_tensor, &pad_cfg);
        ifmap_ptr = (int8_t *) src_s8_pad_ptr;
    }

    int8_t *input_col_ptr = malloc(in_c * kernel_w * kernel_h * out_h * out_w * sizeof(int8_t));
    im2col_s8(input_col_ptr, ifmap_ptr, in_tensor, out_tensor, weight_tensor, cfg);

    // weight and ifmap all s8
    const int M = out_c;
    const int N = out_h * out_w;
    const int K = in_c * kernel_h * kernel_w;

    int8_t * weight_s8_ptr = (int8_t*)weight_ptr;
    int32_t * bias_s32_ptr = (int32_t*)bias_ptr;

    int32_t *output_s32_ptr = (int32_t*)output_ptr;

#pragma omp parallel for num_threads(8)
    for (int i = 0; i < M; i++)
    {
        memset(&output_ptr[i * N], 0, N * sizeof(float));
        int idx_a, idx_b, idx_c;
        idx_c = i * N;

        for (int k = 0; k < K; k++)
        {
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

    if (cfg->pads[0] != 0) {
        free(src_s8_pad_ptr);
    }
    free(input_f32_ptr);
    free(input_s8_ptr);
    free(input_col_ptr);

    // ==============================================================================
    // trans output to float
    float *cur_output_ptr;
    for (int outc_i = 0; outc_i < out_c; ++outc_i) {
        cur_output_ptr = output_ptr + outc_i * out_h * out_w;
        float dequant_coeff = cfg->weight_aux[outc_i] * 1 / 127.0f;
        for (int i = 0; i < out_h * out_w; ++i) {
            cur_output_ptr[i] = cur_output_ptr[i] * dequant_coeff;
        }
    }
    // ==============================================================================

    return 0;
}

int eval_mxn_img2col_W8A32_with_input_scale_no_bias(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);

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
    float *input_f32_ptr = (float *)malloc(in_c * in_h * in_w * sizeof(float));
    int8_t *input_s8_ptr = (int8_t *)malloc(in_c * in_h * in_w * sizeof(int8_t));

#pragma unroll 4
    for (int i = 0; i < in_c * in_h * in_w; ++i) {
        tmp = input_ptr[i] * opposite_ifmap_scale;
        input_f32_ptr[i] = (tmp < -128) ? -128 : (tmp > 127) ? 127 : tmp;
    }

#pragma omp parallel for num_threads(8)
    for (int i = 0; i < in_c * in_h * in_w; i +=4) {
        input_s8_ptr[i] = (int8_t)(roundf(input_f32_ptr[i]));
        input_s8_ptr[i + 1] = (int8_t)(roundf(input_f32_ptr[i + 1]));
        input_s8_ptr[i + 2] = (int8_t)(roundf(input_f32_ptr[i + 2]));
        input_s8_ptr[i + 3] = (int8_t)(roundf(input_f32_ptr[i + 3]));
    }

    int8_t* ifmap_ptr = input_s8_ptr;

    int8_t *src_s8_pad_ptr;
    // do pad
    if (cfg->pads[0] != 0) {
        src_s8_pad_ptr = (int8_t *)malloc(in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(int8_t));
        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv_s8((char *)src_s8_pad_ptr, (char *)input_s8_ptr, in_tensor, &pad_cfg);
        ifmap_ptr = (int8_t *) src_s8_pad_ptr;
    }

    int8_t *input_col_ptr = malloc(in_c * kernel_w * kernel_h * out_h * out_w * sizeof(int8_t));
    im2col_s8(input_col_ptr, ifmap_ptr, in_tensor, out_tensor, weight_tensor, cfg);

    // weight and ifmap all s8
    const int M = out_c;
    const int N = out_h * out_w;
    const int K = in_c * kernel_h * kernel_w;

    int8_t * weight_s8_ptr = (int8_t*)weight_ptr;

    int32_t *output_s32_ptr = (int32_t*)output_ptr;

#pragma omp parallel for num_threads(8)
    for (int i = 0; i < M; i++)
    {
        memset(&output_ptr[i * N], 0, N * sizeof(float));
        int idx_a, idx_b, idx_c;
        idx_c = i * N;

        for (int k = 0; k < K; k++)
        {
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

    if (cfg->pads[0] != 0) {
        free(src_s8_pad_ptr);
    }
    free(input_f32_ptr);
    free(input_s8_ptr);
    free(input_col_ptr);

    // ==============================================================================
    // trans output to float
    float *cur_output_ptr;
    for (int outc_i = 0; outc_i < out_c; ++outc_i) {
        cur_output_ptr = output_ptr + outc_i * out_h * out_w;
        float dequant_coeff = cfg->weight_aux[outc_i] * 1 / 127.0f;
        for (int i = 0; i < out_h * out_w; ++i) {
            cur_output_ptr[i] = cur_output_ptr[i] * dequant_coeff;
        }
    }
    // ==============================================================================

    return 0;
}


int eval_mxn_openmp(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
//    // printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

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
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *weight_tensor = (OPERAND_S *) (params[3].addr);
    OPERAND_S *bias_tensor;

    if (cfg->has_bias) {
        bias_tensor = (OPERAND_S *) (params[4].addr);
    }

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
        src_pad_ptr = malloc(in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(float));
        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv(src_pad_ptr, input_ptr, in_tensor, &pad_cfg);
        input_ptr = (float *) src_pad_ptr;
    }

    const int th_num = 8;
    const int outc_per_th = out_c / th_num;

#pragma omp parallel for num_threads(8)
    for (int th_i = 0; th_i < th_num; ++th_i) {
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

        float *output_batch_ptr;
        float psum0, psum1, psum2, psum3, psum4, psum5, psum6, psum7;

        float in_data;
        for (int c_i = outc_per_th * th_i;
             c_i < outc_per_th * (th_i + 1); c_i += 8) {  // todo: maybe the out_c % 8 != 0
            output_batch_ptr = output_ptr + 0 * out_c * out_h * out_w;

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

    if (cfg->pads[0] != 0) {
        free(src_pad_ptr);
    }


    return 0;

    //    // write_bin(replace_char(cfg->out_operand_name[0]), out_n * out_c * out_h * out_w * sizeof(float), output_ptr);

    int c = 101;
    return 0;
}


int eval_mxn_quant(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
//    // printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

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
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *weight_tensor = (OPERAND_S *) (params[3].addr);
    OPERAND_S *bias_tensor;

    if (cfg->has_bias) {
        bias_tensor = (OPERAND_S *) (params[4].addr);
    }

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
        src_pad_ptr = malloc(in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(float));
        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv(src_pad_ptr, input_ptr, in_tensor, &pad_cfg);
        input_ptr = (float *) src_pad_ptr;
    }

    const float input_sacle = 51.2998;
//    const float input_sacle = 450.894;
    // trans input from float to int8
    int8_t *input_s8_ptr = malloc(in_n * in_c * in_h * in_w * sizeof(int8_t));
    for (int i = 0; i < in_n * in_c * in_h * in_w; ++i) {
        int32_t quant_s8 = (int32_t) (input_ptr[i] * input_sacle);
        quant_s8 = quant_s8 > 127 ? 127 : quant_s8;
        quant_s8 = quant_s8 < -128 ? -128 : quant_s8;
        input_s8_ptr[i] = quant_s8;
    }
//    input_ptr = input_s8_ptr;

//    int8_t a = (int8_t)(245);
//    int8_t b = (int8_t)(-3245);
//    // printf("a is %d, b is %d\n", a, b);


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
                                psum0 += *cur0_weight_ptr++ * (float) in_data;
                                psum1 += *cur1_weight_ptr++ * (float) in_data;
                                psum2 += *cur2_weight_ptr++ * (float) in_data;
                                psum3 += *cur3_weight_ptr++ * (float) in_data;
                                psum4 += *cur4_weight_ptr++ * (float) in_data;
                                psum5 += *cur5_weight_ptr++ * (float) in_data;
                                psum6 += *cur6_weight_ptr++ * (float) in_data;
                                psum7 += *cur7_weight_ptr++ * (float) in_data;
                            }
                        }
                    }
                    output_batch_ptr[c_i * out_h * out_w + h_i * out_w + w_i] = psum0 + bias_ptr[c_i] * input_sacle;
                    output_batch_ptr[(c_i + 1) * out_h * out_w + h_i * out_w + w_i] =
                            psum1 + bias_ptr[c_i + 1] * input_sacle;
                    output_batch_ptr[(c_i + 2) * out_h * out_w + h_i * out_w + w_i] =
                            psum2 + bias_ptr[c_i + 2] * input_sacle;
                    output_batch_ptr[(c_i + 3) * out_h * out_w + h_i * out_w + w_i] =
                            psum3 + bias_ptr[c_i + 3] * input_sacle;
                    output_batch_ptr[(c_i + 4) * out_h * out_w + h_i * out_w + w_i] =
                            psum4 + bias_ptr[c_i + 4] * input_sacle;
                    output_batch_ptr[(c_i + 5) * out_h * out_w + h_i * out_w + w_i] =
                            psum5 + bias_ptr[c_i + 5] * input_sacle;
                    output_batch_ptr[(c_i + 6) * out_h * out_w + h_i * out_w + w_i] =
                            psum6 + bias_ptr[c_i + 6] * input_sacle;
                    output_batch_ptr[(c_i + 7) * out_h * out_w + h_i * out_w + w_i] =
                            psum7 + bias_ptr[c_i + 7] * input_sacle;
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

    if (cfg->pads[0] != 0) {
        free(src_pad_ptr);
    }
    free(input_s8_ptr);

    return 0;

    //    // write_bin(replace_char(cfg->out_operand_name[0]), out_n * out_c * out_h * out_w * sizeof(float), output_ptr);

    int c = 101;
    return 0;
}


int eval_mxn_naive_gemm_c(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
//    // printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

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
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *weight_tensor = (OPERAND_S *) (params[3].addr);
    OPERAND_S *bias_tensor;

    if (cfg->has_bias) {
        bias_tensor = (OPERAND_S *) (params[4].addr);
    }

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
        src_pad_ptr = malloc(in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(float));
        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv(src_pad_ptr, input_ptr, in_tensor, &pad_cfg);
        input_ptr = (float *) src_pad_ptr;
    }


    const int32_t N = in_tensor->shapes[0];
    const int32_t IC = in_tensor->shapes[1];

    const int32_t OC = out_tensor->shapes[1];
    const int32_t OH = out_tensor->shapes[2];
    const int32_t OW = out_tensor->shapes[3];

    const int32_t FH = weight_tensor->shapes[2];
    const int32_t FW = weight_tensor->shapes[3];

    const int32_t stride_h = cfg->strides[0];
    const int32_t stride_w = cfg->strides[1];

    // do conv as gemm
    const int32_t GEMM_M = OC;
    const int32_t GEMM_N = N * OH * OW;
    const int32_t GEMM_K = IC * FH * FW;

    for (int i = 0; i < GEMM_M; ++i) {
        int32_t oc = i;
        for (int j = 0; j < GEMM_N; ++j) {
            float accumulator = bias_ptr[i];
            int32_t n = j / (OH * OW);
            int32_t j_res = j % (OH * OW);
            int32_t oh = j_res / OW;
            int32_t ow = j_res % OW;

            // calc single output's elem
            for (int k = 0; k < GEMM_K; ++k) {
                int32_t ic = k / (FH * FW);
                int32_t k_res = k % (FH * FW);
                int32_t fh = k_res / FW;
                int32_t fw = k_res % FW;
                int32_t ih = oh * stride_h + fh;
                int32_t iw = ow * stride_w + fw;
//                accumulator = accumulator + in_array[n][ic][ih][iw] * weight_array[oc][ic][fh][fw];
                accumulator = accumulator + input_ptr[ic * in_h * in_w + ih * in_w + iw] *
                                            weight_ptr[oc * in_c * kernel_h * kernel_w + ic * kernel_h * kernel_w +
                                                       fh * kernel_w + fw];
            }
//            out_array[n][oc][oh][ow] = accumulator;
            output_ptr[oc * out_h * out_w + oh * out_w + ow] = accumulator;
        }
    }

    if (cfg->pads[0] != 0) {
        free(src_pad_ptr);
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


int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);

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

    if (cfg->group != 1) {
        // depth wise conv
        if (cfg->kernel_shape[0] == 3 && cfg->kernel_shape[1] == 3) {
            eval_depthwise_conv_3x3_img2col(params, inputs, outputs);
        } else {
            eval_depthwise_conv_mxn_img2col(params, inputs, outputs);
        }
    } else {
        // normal conv
        if (cfg->ifmap_quant2 == TYPE_INT8) {
            // ifmap with input scale, weight is s8, bias is s32
            eval_mxn_img2col_W8A32_with_input_scale(params, inputs, outputs);
        } else {
            // ifmap and ofmap, weight, bias all is float32
            if (cfg->kernel_shape[0] == 3 && cfg->kernel_shape[1] == 3) {
                eval_mxn_img2col(params, inputs, outputs);
            } else if (cfg->kernel_shape[0] == 1 && cfg->kernel_shape[1] == 1 && cfg->strides[0] == 1 && cfg->strides[1] == 1) {
                eval_1x1j1(params, inputs, outputs);
            } else {
                eval_mxn_img2col(params, inputs, outputs);
            }
        }
    }

    // do act if need
    float *output_ptr = (float *) (outputs[0].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[4].addr);
    int32_t out_n = out_tensor->shapes[0];
    int32_t out_c = out_tensor->shapes[1];
    int32_t out_h = out_tensor->shapes[2];
    int32_t out_w = out_tensor->shapes[3];
    int32_t ofmap_elem_size = out_n * out_c * out_h * out_w;

    if (cfg->act_type == RELU) {
        for (int elem_i = 0; elem_i < ofmap_elem_size; ++elem_i) {
            output_ptr[elem_i] = output_ptr[elem_i] > 0 ? output_ptr[elem_i] : 0;
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
            } else if (cur_val >= single_limit) {
                ;
            } else if (cur_val >= inv_single_limit && cur_val < single_limit) {
                int idx = (int)((cur_val + single_limit) * inv_step);
                output_ptr[elem_i] = silu_lut[idx];
            }
        }
    } else if (cfg->act_type == CLIP) {
        int aligned_size = (ofmap_elem_size / 8) * 8;

        float max = cfg->clip_max;
        float min = cfg->clip_min;
        __m256 min_vec = _mm256_set1_ps(min);  // 将min广播到所有元素
        __m256 max_vec = _mm256_set1_ps(max);  // 将max广播到所有元素

#pragma omp parallel for num_threads(8)
        for (int elem_i = 0; elem_i < aligned_size; elem_i += 8) {
            __m256 vec = _mm256_loadu_ps(&output_ptr[elem_i]);

            // 执行clamp操作
            vec = _mm256_max_ps(vec, min_vec);
            vec = _mm256_min_ps(vec, max_vec);

            // 存储结果
            _mm256_storeu_ps(&output_ptr[elem_i], vec);
        }

        for (int32_t elem_i = aligned_size; elem_i < ofmap_elem_size; elem_i ++) {
            output_ptr[elem_i] = (output_ptr[elem_i] > min) ? output_ptr[elem_i] : min;
            output_ptr[elem_i] = (output_ptr[elem_i] < max) ? output_ptr[elem_i] : max;
        }

//#pragma unroll 8
//        for (int elem_i = 0; elem_i < ofmap_elem_size; ++elem_i) {
//            output_ptr[elem_i] = (output_ptr[elem_i] > min) ? output_ptr[elem_i] : min;
//            output_ptr[elem_i] = (output_ptr[elem_i] < max) ? output_ptr[elem_i] : max;
//        }
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
            output_ptr[elem_i] = output_ptr[elem_i] *((tmp < max) ? tmp : max);
        }
    }

    return 0;
}





