#include "gemm.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include "string.h"
#include "../../x86_utils/opt_gemm.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs)
{
//    show_dev_input(params);

    GEMM_CONFIG_S* cfg = (GEMM_CONFIG_S*)(params[0].addr);

    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *in1_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *in2_tensor = (OPERAND_S *) (params[3].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[4].addr);

    float* input0_ptr = (float*)(inputs[0].addr);
    float* input1_ptr = (float*)(inputs[1].addr);
    float* input2_ptr = (float*)(inputs[2].addr);
    float* output_ptr = (float*)(outputs[0].addr);

    int32_t out_loop = 1;
    for (int i = 0; i < in0_tensor->dim_num_of_shapes - 2; ++i) {
        out_loop *= in0_tensor->shapes[i];
    }

    int32_t M = in0_tensor->shapes[in0_tensor->dim_num_of_shapes - 2];
    int32_t K = in0_tensor->shapes[in0_tensor->dim_num_of_shapes - 1];
    int32_t N = out_tensor->shapes[out_tensor->dim_num_of_shapes - 1];


#pragma omp parallel for num_threads(THREADS_NUM)
    for (int m_i = 0; m_i < M; ++m_i) {
        float *cur_input0_ptr, *cur_input1_ptr;
        int n_i = 0;
        for (; n_i < N - 7; n_i += 8) {
            cur_input0_ptr = input0_ptr + m_i * K;
            cur_input1_ptr = input1_ptr + n_i * K;
            float psum_vec[8];
            psum_vec[0] = 0, psum_vec[1] = 0, psum_vec[2] = 0, psum_vec[3] = 0;
            psum_vec[4] = 0, psum_vec[5] = 0, psum_vec[6] = 0, psum_vec[7] = 0;

            for (int k_i = 0; k_i < K; ++k_i) {
                psum_vec[0] += cur_input0_ptr[k_i] * cur_input1_ptr[k_i];
                psum_vec[1] += cur_input0_ptr[k_i] * cur_input1_ptr[k_i + 1 * K];
                psum_vec[2] += cur_input0_ptr[k_i] * cur_input1_ptr[k_i + 2 * K];
                psum_vec[3] += cur_input0_ptr[k_i] * cur_input1_ptr[k_i + 3 * K];
                psum_vec[4] += cur_input0_ptr[k_i] * cur_input1_ptr[k_i + 4 * K];
                psum_vec[5] += cur_input0_ptr[k_i] * cur_input1_ptr[k_i + 5 * K];
                psum_vec[6] += cur_input0_ptr[k_i] * cur_input1_ptr[k_i + 6 * K];
                psum_vec[7] += cur_input0_ptr[k_i] * cur_input1_ptr[k_i + 7 * K];
            }
            output_ptr[m_i * N + n_i] = psum_vec[0] + input2_ptr[n_i];
            output_ptr[m_i * N + n_i + 1] = psum_vec[1] + input2_ptr[n_i + 1];
            output_ptr[m_i * N + n_i + 2] = psum_vec[2] + input2_ptr[n_i + 2];
            output_ptr[m_i * N + n_i + 3] = psum_vec[3] + input2_ptr[n_i + 3];
            output_ptr[m_i * N + n_i + 4] = psum_vec[4] + input2_ptr[n_i + 4];
            output_ptr[m_i * N + n_i + 5] = psum_vec[5] + input2_ptr[n_i + 5];
            output_ptr[m_i * N + n_i + 6] = psum_vec[6] + input2_ptr[n_i + 6];
            output_ptr[m_i * N + n_i + 7] = psum_vec[7] + input2_ptr[n_i + 7];
        }

        for (; n_i < N; ++n_i) {
            cur_input0_ptr = input0_ptr + m_i * K;
            cur_input1_ptr = input1_ptr + n_i * K;
            float psum = 0;
#pragma unroll 2
            for (int k_i = 0; k_i < K; ++k_i) {
                psum += cur_input0_ptr[k_i] * cur_input1_ptr[k_i];
            }
            output_ptr[m_i * N + n_i] = psum + input2_ptr[n_i];
        }
    }

    return 0;
}