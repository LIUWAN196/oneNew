#ifndef ONENEW_MODEL_AUTO_TUNING_HPP
#define ONENEW_MODEL_AUTO_TUNING_HPP

#include "optimize.hpp"
#include <sys/time.h>
#include <immintrin.h>
#include "../device/x86_utils/opt_gemm.h"

int32_t auto_tuning(BLOCK_INFO_S* block_info) {

    // step 1: set size of ifmap and ofmap
    const int M = 1024, N = 1024, K = 1024;

    // step 2: malloc buf and set init val
    float *ifmap0_ptr = (float *) malloc( M * K * sizeof(float));
    float *ifmap1_ptr = (float *) malloc(N * K * sizeof(float));
    float *ofmap_ptr = (float *) malloc(M * N * sizeof(float));

    // step 3: do optimize gemm
    GEMM_TILE_INFO gemm_tile_info;
    gemm_tile_info.M = M;
    gemm_tile_info.N = N;
    gemm_tile_info.K = K;
    const int32_t m_option_cnt = 8;
    const int32_t n_option_cnt = 8;
    const int32_t k_option_cnt = 8;
    int m_tile_size_arr[m_option_cnt] = {8, 16, 32, 64, 128, 256, 512, 1024};
    int n_tile_size_arr[n_option_cnt] = {8, 16, 32, 64, 128, 256, 512, 1024};
    int k_tile_size_arr[k_option_cnt] = {8, 16, 32, 64, 128, 256, 512, 1024};

    double min_single_th_time = 1e16;
    double min_multi_th_time = 1e16;
    int best_single_th_m = 0, best_single_th_n = 0, best_single_th_k = 0;
    int best_multi_th_m = 0, best_multi_th_n = 0, best_multi_th_k = 0;
    for (int m_tile_i = 0; m_tile_i < m_option_cnt; ++m_tile_i) {
        for (int n_tile_i = 0; n_tile_i < n_option_cnt; ++n_tile_i) {
            for (int k_tile_i = 0; k_tile_i < k_option_cnt; ++k_tile_i) {
                gemm_tile_info.m_tile_size = m_tile_size_arr[m_tile_i];
                gemm_tile_info.n_tile_size = n_tile_size_arr[n_tile_i];
                gemm_tile_info.k_tile_size = k_tile_size_arr[k_tile_i];

                const double ratio = 0.98f; // 这是为了不让 tile k m n 太大，所以在耗时接近的情况下，选择 tile m n k 较小的配置
                // 寻找最佳的单线程 tile m n k
                double single_th_st = omp_get_wtime();
                opt_gemm_single_threads(ofmap_ptr, ifmap0_ptr, ifmap1_ptr, gemm_tile_info);
                double single_th_ed = omp_get_wtime();
                double single_time = single_th_ed - single_th_st;
                best_single_th_m = single_time < ratio * min_single_th_time ? m_tile_size_arr[m_tile_i] : best_single_th_m;
                best_single_th_n = single_time < ratio * min_single_th_time ? n_tile_size_arr[n_tile_i] : best_single_th_n;
                best_single_th_k = single_time < ratio * min_single_th_time ? k_tile_size_arr[k_tile_i] : best_single_th_k;
                min_single_th_time = single_time < ratio * min_single_th_time ? single_time : min_single_th_time;

                // 寻找最佳的多线程 tile m n k
                double multi_th_st = omp_get_wtime();
                opt_gemm_multi_threads(ofmap_ptr, ifmap0_ptr, ifmap1_ptr, gemm_tile_info);
                double multi_th_ed = omp_get_wtime();
                double multi_time = multi_th_ed - multi_th_st;
                best_multi_th_m = multi_time < ratio * min_multi_th_time ? m_tile_size_arr[m_tile_i] : best_multi_th_m;
                best_multi_th_n = multi_time < ratio * min_multi_th_time ? n_tile_size_arr[n_tile_i] : best_multi_th_n;
                best_multi_th_k = multi_time < ratio * min_multi_th_time ? k_tile_size_arr[k_tile_i] : best_multi_th_k;
                min_multi_th_time = multi_time < ratio * min_multi_th_time ? multi_time : min_multi_th_time;

            }
        }
    }

    block_info->x86_gemm_single_threads_tile_m = best_single_th_m;
    block_info->x86_gemm_single_threads_tile_n = best_single_th_n;
    block_info->x86_gemm_single_threads_tile_k = best_single_th_k;

    block_info->x86_gemm_multi_threads_tile_m = best_multi_th_m;
    block_info->x86_gemm_multi_threads_tile_n = best_multi_th_n;
    block_info->x86_gemm_multi_threads_tile_k = best_multi_th_k;
    return 0;
}

#endif //ONENEW_MODEL_AUTO_TUNING_HPP
