//
// Created by wanzai on 24-9-20.
//

#ifndef ONENEW_OPT_GEMM_H
#define ONENEW_OPT_GEMM_H

#include "stdint.h"
#include "math.h"

typedef struct {
    int32_t M;
    int32_t N;
    int32_t K;

    int32_t m_tile_size;
    int32_t n_tile_size;
    int32_t k_tile_size;
} GEMM_TILE_INFO;

int opt_gemm_multi_threads(float *ofmap_ptr, float *ifmap0_ptr, float *ifmap1_ptr, GEMM_TILE_INFO gemm_tile_info) {

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
                    for (int k_i = 0; k_i < k_tile_size; k_i++) {
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

int opt_gemm_single_threads(float *ofmap_ptr, float *ifmap0_ptr, float *ifmap1_ptr, GEMM_TILE_INFO gemm_tile_info) {

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
//#pragma omp parallel for num_threads(8)
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
//#pragma omp parallel for num_threads(8)
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
                    for (int k_i = 0; k_i < k_tile_size; k_i++) {
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
//#pragma omp parallel for num_threads(8)
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
//#pragma omp parallel for num_threads(8)
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

int opt_gemm_aligned_multi_threads(float *ofmap_ptr, float *ifmap0_ptr, float *ifmap1_ptr, GEMM_TILE_INFO gemm_tile_info) {

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
                            mul_res_vec[0] = _mm256_mul_ps(ifmap0_vec[0], _mm256_load_ps(cur_ifmap1_ptr[0] + n_i));
                            mul_res_vec[1] = _mm256_mul_ps(ifmap0_vec[1], _mm256_load_ps(cur_ifmap1_ptr[1] + n_i));
                            mul_res_vec[2] = _mm256_mul_ps(ifmap0_vec[2], _mm256_load_ps(cur_ifmap1_ptr[2] + n_i));
                            mul_res_vec[3] = _mm256_mul_ps(ifmap0_vec[3], _mm256_load_ps(cur_ifmap1_ptr[3] + n_i));
                            psum_vec = _mm256_load_ps(cur_ofmap_ptr + n_i);
                            psum_vec = _mm256_add_ps(psum_vec, mul_res_vec[0]);
                            psum_vec = _mm256_add_ps(psum_vec, mul_res_vec[1]);
                            psum_vec = _mm256_add_ps(psum_vec, mul_res_vec[2]);
                            psum_vec = _mm256_add_ps(psum_vec, mul_res_vec[3]);
                            _mm256_store_ps(cur_ofmap_ptr + n_i, psum_vec);
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
                    for (int k_i = 0; k_i < k_tile_size; k_i++) {
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
                    ifmap1_vec = _mm256_load_ps(ifmap1_ptr + k_i * N + n_i);
                    mul_res_vec = _mm256_mul_ps(ifmap0_vec, ifmap1_vec);
                    psum_vec = _mm256_load_ps(cur_ofmap_ptr + n_i);
                    _mm256_store_ps(cur_ofmap_ptr + n_i, _mm256_add_ps(psum_vec, mul_res_vec));
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

int opt_gemm_aligned_single_threads(float *ofmap_ptr, float *ifmap0_ptr, float *ifmap1_ptr, GEMM_TILE_INFO gemm_tile_info) {

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
//#pragma omp parallel for num_threads(8)
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
                            mul_res_vec[0] = _mm256_mul_ps(ifmap0_vec[0], _mm256_load_ps(cur_ifmap1_ptr[0] + n_i));
                            mul_res_vec[1] = _mm256_mul_ps(ifmap0_vec[1], _mm256_load_ps(cur_ifmap1_ptr[1] + n_i));
                            mul_res_vec[2] = _mm256_mul_ps(ifmap0_vec[2], _mm256_load_ps(cur_ifmap1_ptr[2] + n_i));
                            mul_res_vec[3] = _mm256_mul_ps(ifmap0_vec[3], _mm256_load_ps(cur_ifmap1_ptr[3] + n_i));
                            psum_vec = _mm256_load_ps(cur_ofmap_ptr + n_i);
                            psum_vec = _mm256_add_ps(psum_vec, mul_res_vec[0]);
                            psum_vec = _mm256_add_ps(psum_vec, mul_res_vec[1]);
                            psum_vec = _mm256_add_ps(psum_vec, mul_res_vec[2]);
                            psum_vec = _mm256_add_ps(psum_vec, mul_res_vec[3]);
                            _mm256_store_ps(cur_ofmap_ptr + n_i, psum_vec);
                        }
                    }
                }
            }
        }
    }

    // step 2: 这个循环处理 N 的尾部部分，即 [N - n_tile_cnt * n_tile_size, N] 部分
    const int32_t n_tail = N - n_tile_cnt * n_tile_size;
    if (n_tail != 0) {
//#pragma omp parallel for num_threads(8)
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
                    for (int k_i = 0; k_i < k_tile_size; k_i++) {
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
//#pragma omp parallel for num_threads(8)
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
                    ifmap1_vec = _mm256_load_ps(ifmap1_ptr + k_i * N + n_i);
                    mul_res_vec = _mm256_mul_ps(ifmap0_vec, ifmap1_vec);
                    psum_vec = _mm256_load_ps(cur_ofmap_ptr + n_i);
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
//#pragma omp parallel for num_threads(8)
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

#endif //ONENEW_OPT_GEMM_H
