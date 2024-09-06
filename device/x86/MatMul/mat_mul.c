#include "mat_mul.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include <stdlib.h>
#include "stdint.h"
#include "string.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs)
{

//    show_dev_input(params);
//    printf("this is gemm eval in x86\n");
    MATMUL_CONFIG_S * cfg = (MATMUL_CONFIG_S*)(params[0].addr);
//    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *in1_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

    float* input0_ptr = (float*)(inputs[0].addr);
    float* input1_ptr = (float*)(inputs[1].addr);
    float* output_ptr = (float*)(outputs[0].addr);

    int32_t out_loop = 1;
    for (int i = 0; i < in0_tensor->dim_num_of_shapes - 2; ++i) {
        out_loop *= in0_tensor->shapes[i];
//        printf("in0_tensor->shapes[i] is %d, out_loop is %d\n", in0_tensor->shapes[i], out_loop);
    }

//    printf("==============\n");
////    printf("in0_tensor->dim_num_of_shapes is %d, out_loop is %d\n", in0_tensor->dim_num_of_shapes, out_loop);
//    for (int i = 0; i < 8; ++i) {
//        printf("shape %d  is %d\n", i, in0_tensor->shapes[i]);
//    }

    int32_t M = in0_tensor->shapes[in0_tensor->dim_num_of_shapes - 2];
    int32_t K = in0_tensor->shapes[in0_tensor->dim_num_of_shapes - 1];
    int32_t N = out_tensor->shapes[out_tensor->dim_num_of_shapes - 1];

    int32_t ifmap1_total_elem_size = operand_elem_size(in1_tensor);

    float *cur_input0_ptr, *cur_input1_ptr, *cur_output_ptr;
    for (int out_loop_i = 0; out_loop_i < out_loop; ++out_loop_i) {
        cur_input0_ptr = input0_ptr + out_loop_i * M * K;
        if (ifmap1_total_elem_size == K * N) {
            cur_input1_ptr = input1_ptr;
        } else {
            cur_input1_ptr = input1_ptr + out_loop_i * K * N;
        }
        cur_output_ptr = output_ptr + out_loop_i  * M * N;

#pragma omp parallel for num_threads(8)
        for (int i = 0; i < M; i++)
        {
            memset(&cur_output_ptr[i * N], 0, N * sizeof(float));
            int idx_a, idx_b, idx_c;
            __m256 psum;
            __m256 weight_vec;
            __m256 sum_pre;
            idx_c = i * N;
#pragma unroll 4
            for (int k = 0; k < K; k++)
            {
                idx_a = i * K + k;
                idx_b = k * N;
                int j = 0;
                weight_vec = _mm256_set1_ps(cur_input0_ptr[idx_a]);
#pragma unroll 2
                for (; j < N - 7; j+=8)
                {
                    sum_pre = _mm256_loadu_ps(&cur_output_ptr[idx_c + j]);
                    sum_pre = _mm256_fmadd_ps(weight_vec, _mm256_loadu_ps(&cur_input1_ptr[idx_b + j]), sum_pre);
                    _mm256_storeu_ps(&cur_output_ptr[idx_c + j], sum_pre);
                }
                for (; j < N; ++j) {
                    cur_output_ptr[idx_c + j] += cur_input0_ptr[idx_a] * cur_input1_ptr[idx_b + j];
                }
            }

        }

    }


//
//    float *cur_input0_ptr, *cur_input1_ptr;
//    for (int out_loopi = 0; out_loopi < out_loop; ++out_loopi) {
//        for (int m_i = 0; m_i < M; ++m_i) {
//            for (int n_i = 0; n_i < N; ++n_i) {
//                cur_input0_ptr = input0_ptr + out_loopi * M * K + m_i * K;
//                if (ifmap1_total_elem_size == K * N) {
//                    cur_input1_ptr = input1_ptr;
//                } else {
//                    cur_input1_ptr = input1_ptr + out_loopi * K * N;
//                }
//                float psum = 0;
//                for (int k_i = 0; k_i < K; ++k_i) {
//                    psum += cur_input0_ptr[k_i] * cur_input1_ptr[k_i * N + n_i];
//                }
//                output_ptr[out_loopi * M * N + m_i * N + n_i] = psum;
//            }
//        }
//    }
//    LOG_DBG("end this op");

    return 0;
}