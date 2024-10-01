#include "mat_mul.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include <stdlib.h>
#include "stdint.h"
#include "string.h"
#include "../../x86_utils/opt_gemm.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

//    show_dev_input(params);
//    printf("this is gemm eval in x86\n");
    MATMUL_CONFIG_S *cfg = (MATMUL_CONFIG_S *) (params[0].addr);
//    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *in1_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

    float *input0_ptr = (float *) (inputs[0].addr);
    float *input1_ptr = (float *) (inputs[1].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    int32_t out_loop = 1;
    for (int i = 0; i < in0_tensor->dim_num_of_shapes - 2; ++i) {
        out_loop *= in0_tensor->shapes[i];
    }

//    if (strcmp(cfg->op_base_cfg.op_name, "/image_encoder/layers.1/blocks.0/mlp/fc1/MatMul") == 0) {
//        int a = 1;
//
//        printf("cfg->op_base_cfg.op_name is %c\n", cfg->op_base_cfg.op_name);
//        printf("aaaa is %d, cfg->op_base_cfg.op_name is %c\n", a, cfg->op_base_cfg.op_name);
//    }
    int32_t M = in0_tensor->shapes[in0_tensor->dim_num_of_shapes - 2];
    int32_t K = in0_tensor->shapes[in0_tensor->dim_num_of_shapes - 1];
    int32_t N = out_tensor->shapes[out_tensor->dim_num_of_shapes - 1];

    int32_t ifmap1_total_elem_size = operand_elem_size(in1_tensor);

    const int32_t num_threads = 8;
    // 如果 matmul 这个算子的最外层循环很大或者为 num_threads 整数倍， 则多线程在 opt_gemm 外部，避免线程的创建开销
    if (out_loop % num_threads == 0 || out_loop > 8 * num_threads) {
#pragma omp parallel for num_threads(8)
        for (int out_loop_i = 0; out_loop_i < out_loop; ++out_loop_i) {
            float *cur_input0_ptr, *cur_input1_ptr, *cur_output_ptr;
            cur_input0_ptr = input0_ptr + out_loop_i * M * K;
            if (ifmap1_total_elem_size == K * N) {
                cur_input1_ptr = input1_ptr;
            } else {
                cur_input1_ptr = input1_ptr + out_loop_i * K * N;
            }
            cur_output_ptr = output_ptr + out_loop_i * M * N;

            GEMM_TILE_INFO gemm_tile_info;
            gemm_tile_info.M = M;
            gemm_tile_info.N = N;
            gemm_tile_info.K = K;

            // 目前认为的 sgemm 的最佳配置为 m_tile_size = 32，n_tile_size = 1024，k_tile_size = 8；
            const int32_t best_n_tile = 1024, best_k_tile = 8;
            int32_t best_m_tile = 32;

            gemm_tile_info.m_tile_size = best_m_tile;
            gemm_tile_info.n_tile_size = best_n_tile;
            gemm_tile_info.k_tile_size = best_k_tile;

            const int32_t avx2_align_size = 32;
            if (gemm_tile_info.M % avx2_align_size == 0
            && gemm_tile_info.N % avx2_align_size == 0
            && gemm_tile_info.K % avx2_align_size == 0) {
                opt_gemm_aligned_single_threads(cur_output_ptr, cur_input0_ptr, cur_input1_ptr, gemm_tile_info);
            } else {
                opt_gemm_single_threads(cur_output_ptr, cur_input0_ptr, cur_input1_ptr, gemm_tile_info);
            }
        }
    } else {
        float *cur_input0_ptr, *cur_input1_ptr, *cur_output_ptr;
        for (int out_loop_i = 0; out_loop_i < out_loop; ++out_loop_i) {
            cur_input0_ptr = input0_ptr + out_loop_i * M * K;
            if (ifmap1_total_elem_size == K * N) {
                cur_input1_ptr = input1_ptr;
            } else {
                cur_input1_ptr = input1_ptr + out_loop_i * K * N;
            }
            cur_output_ptr = output_ptr + out_loop_i * M * N;

            GEMM_TILE_INFO gemm_tile_info;
            gemm_tile_info.M = M;
            gemm_tile_info.N = N;
            gemm_tile_info.K = K;

            // 目前认为的 sgemm 的最佳配置为 m_tile_size = 32，n_tile_size = 1024，k_tile_size = 8；
            const int32_t best_n_tile = 1024, best_k_tile = 8;
            // 这里因为开了 num_threads 个线程，所以要保证 m_tile 为 num_threads 的倍数，不让有的线程创建了但是没有使用
            int32_t best_m_tile = (gemm_tile_info.M / num_threads > 32) ? 32 : gemm_tile_info.M / num_threads;
            best_m_tile = (best_m_tile == 0) ? gemm_tile_info.M : best_m_tile;

            gemm_tile_info.m_tile_size = best_m_tile;
            gemm_tile_info.n_tile_size = best_n_tile;
            gemm_tile_info.k_tile_size = best_k_tile;
            const int32_t avx2_align_size = 32;
            opt_gemm_multi_threads(cur_output_ptr, cur_input0_ptr, cur_input1_ptr, gemm_tile_info);
//            if (gemm_tile_info.M % avx2_align_size == 0
//                && gemm_tile_info.N % avx2_align_size == 0
//                && gemm_tile_info.K % avx2_align_size == 0) {
//                opt_gemm_aligned_multi_threads(cur_output_ptr, cur_input0_ptr, cur_input1_ptr, gemm_tile_info);
//            } else {
//                opt_gemm_multi_threads(cur_output_ptr, cur_input0_ptr, cur_input1_ptr, gemm_tile_info);
//            }
        }
    }


    return 0;
}