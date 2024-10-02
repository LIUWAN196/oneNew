#include "einsum.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "math.h"
#include "float.h"
#include "stdint.h"
#include "string.h"

int eval_ofmap_bmhwn(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
    USEFUL_INFO_S* useful_info = (USEFUL_INFO_S *) (params[BUF_MAXNUM - 1].addr);
    int64_t public_buf_size = useful_info->public_buf_info.public_buf_size;
    int64_t public_buf_ptr = useful_info->public_buf_info.public_buf_ptr;
    int64_t rem_buf_size = public_buf_size;
    int64_t rem_buf_ptr = public_buf_ptr;

    // in0:  bmchw --> bmhwc          in1: bnmc  --> bmnc
    // do  ofmap = gemm(in0, in1)   --> bmhwn
    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *in1_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

    float *input0_ptr = (float *) (inputs[0].addr);
    float *input1_ptr = (float *) (inputs[1].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    int32_t in0_elem_size = 1;
    for (int i = 0; i < SHAPE_LEN; ++i) {
        in0_elem_size *= in0_tensor->shapes[i];
    }

    int32_t in1_elem_size = 1;
    for (int i = 0; i < SHAPE_LEN; ++i) {
        in1_elem_size *= in1_tensor->shapes[i];
    }

    int64_t ifmap_need_buf_size = in0_elem_size * sizeof(float) + in1_elem_size * sizeof(float);

    float *in0_transed_ptr;
    float *in1_transed_ptr;
    if (ifmap_need_buf_size < rem_buf_size) {
        in0_transed_ptr = (void *)rem_buf_ptr;
        rem_buf_ptr += (in0_elem_size * sizeof(float) + 31) & (~32);
        rem_buf_size -= (in0_elem_size * sizeof(float) + 31) & (~32);

        in1_transed_ptr = (void *)rem_buf_ptr;
        rem_buf_ptr += (in1_elem_size * sizeof(float) + 31) & (~32);
        rem_buf_size -= (in1_elem_size * sizeof(float) + 31) & (~32);
    } else {
        LOG_ERR("remaining buf size is %d, but need buf size is %d", rem_buf_size, ifmap_need_buf_size);
    }

    // step 1: do in0:  bmchw --> bmhwc
    {
        int32_t perm_num = 4;
        int32_t perm[4] = {0, 2, 3, 1};
        int32_t in_n = in0_tensor->shapes[1];   // m
        int32_t in_c = in0_tensor->shapes[2];   // c
        int32_t in_h = in0_tensor->shapes[3];   // h
        int32_t in_w = in0_tensor->shapes[4];   // w

        int32_t m = in_n;
        int32_t c = in_c;
        int32_t h = in_h;
        int32_t w = in_w;

        int32_t out_n = m;
        int32_t out_c = h;
        int32_t out_h = w;
        int32_t out_w = c;

        int32_t in_stride[8] = {in_c * in_h * in_w, in_h * in_w, in_w, 1};
        int32_t out_stride[8] = {out_c * out_h * out_w, out_h * out_w, out_w, 1};

        /*
        * 原始的 in stride 不能直接使用，如果直接使用的话，等于仅仅是将输入数据拷贝了一份到输出的地址上。
        * 因为这个算子是 transpose，需要跳跃着将输入数据给输出数据进行赋值。这里跳跃多少，就是通过 perm 来重新排序的
        */
        int32_t correct_in_stride[8];
        for (int i = 0; i < perm_num; ++i) {
            correct_in_stride[i] = in_stride[perm[i]];
        }

        float *cur_input_ptr, *cur_output_ptr;
        for (int outn_i = 0; outn_i < out_n; ++outn_i) {
            for (int outc_i = 0; outc_i < out_c; ++outc_i) {
                for (int outh_i = 0; outh_i < out_h; ++outh_i) {
                    cur_input_ptr = input0_ptr + outn_i * correct_in_stride[0] + outc_i * correct_in_stride[1] +
                                    outh_i * correct_in_stride[2];
                    cur_output_ptr =
                            in0_transed_ptr + outn_i * out_stride[0] + outc_i * out_stride[1] + outh_i * out_stride[2];
                    for (int outw_i = 0; outw_i < out_w; ++outw_i) {
                        cur_output_ptr[outw_i * out_stride[3]] = cur_input_ptr[outw_i * correct_in_stride[3]];
                    }
                }
            }
        }
    }

    // step 2: do in1: bnmc  --> bmnc
    {
        int32_t perm_num = 4;
        int32_t perm[4] = {0, 2, 1, 3};
        int32_t in_n = in1_tensor->shapes[0];   // b
        int32_t in_c = in1_tensor->shapes[1];   // n
        int32_t in_h = in1_tensor->shapes[2];   // m
        int32_t in_w = in1_tensor->shapes[3];   // c

        int32_t b = in_n;
        int32_t n = in_c;
        int32_t m = in_h;
        int32_t c = in_w;

        int32_t out_n = b;
        int32_t out_c = m;
        int32_t out_h = n;
        int32_t out_w = c;

        int32_t in_stride[8] = {in_c * in_h * in_w, in_h * in_w, in_w, 1};
        int32_t out_stride[8] = {out_c * out_h * out_w, out_h * out_w, out_w, 1};

        /*
        * 原始的 in stride 不能直接使用，如果直接使用的话，等于仅仅是将输入数据拷贝了一份到输出的地址上。
        * 因为这个算子是 transpose，需要跳跃着将输入数据给输出数据进行赋值。这里跳跃多少，就是通过 perm 来重新排序的
        */
        int32_t correct_in_stride[8];
        for (int i = 0; i < perm_num; ++i) {
            correct_in_stride[i] = in_stride[perm[i]];
        }

        float *cur_input_ptr, *cur_output_ptr;
        for (int outn_i = 0; outn_i < out_n; ++outn_i) {
            for (int outc_i = 0; outc_i < out_c; ++outc_i) {
                for (int outh_i = 0; outh_i < out_h; ++outh_i) {
                    cur_input_ptr = input1_ptr + outn_i * correct_in_stride[0] + outc_i * correct_in_stride[1] +
                                    outh_i * correct_in_stride[2];
                    cur_output_ptr =
                            in1_transed_ptr + outn_i * out_stride[0] + outc_i * out_stride[1] + outh_i * out_stride[2];
                    for (int outw_i = 0; outw_i < out_w; ++outw_i) {
                        cur_output_ptr[outw_i * out_stride[3]] = cur_input_ptr[outw_i * correct_in_stride[3]];
                    }
                }
            }
        }
    }

    // step 3: do gemm
    int32_t out_loop = 1; // = b * n
    for (int i = 0; i < 2; ++i) {
        out_loop *= out_tensor->shapes[i];
    }
    int32_t M = out_tensor->shapes[2] * in0_tensor->shapes[3];  // = h * w
    int32_t K = in0_tensor->shapes[2]; // = c
    int32_t N = out_tensor->shapes[4];  // = n

    int32_t ifmap0_inner_loop = M * K;
    int32_t ifmap1_inner_loop = K * N;

    for (int out_loop_i = 0; out_loop_i < out_loop; ++out_loop_i) {
        float * ifmap0_ptr = in0_transed_ptr + out_loop_i * ifmap0_inner_loop;
        float * ifmap1_ptr = in1_transed_ptr + out_loop_i * ifmap1_inner_loop;
        float * ofmap_ptr = output_ptr + out_loop_i * M * N;
#pragma omp parallel for num_threads(8)
        for (int m_i = 0; m_i < M; ++m_i) {
            float *cur_input0_ptr, *cur_input1_ptr;
            for (int n_i = 0; n_i < N; ++n_i) {
                cur_input0_ptr = ifmap0_ptr + m_i * K;
                cur_input1_ptr = ifmap1_ptr + n_i * K;
                float psum = 0;
#pragma unroll 2
                for (int k_i = 0; k_i < K; ++k_i) {
                    psum += cur_input0_ptr[k_i] * cur_input1_ptr[k_i];
                }
                ofmap_ptr[m_i * N + n_i] = psum;
            }
        }
    }

    return 0;
}

int eval_ofmap_bkhw(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
    USEFUL_INFO_S* useful_info = (USEFUL_INFO_S *) (params[BUF_MAXNUM - 1].addr);
    int64_t public_buf_size = useful_info->public_buf_info.public_buf_size;
    int64_t public_buf_ptr = useful_info->public_buf_info.public_buf_ptr;
    int64_t rem_buf_size = public_buf_size;
    int64_t rem_buf_ptr = public_buf_ptr;

    // in0:  bchw --> bhwc          in1: bkc
    // do  ofmap = gemm(in1, in0)   --> bkc * bhwc --> bkhw
    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *in1_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

    float *input0_ptr = (float *) (inputs[0].addr);
    float *input1_ptr = (float *) (inputs[1].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    int32_t in0_elem_size = 1;
    for (int i = 0; i < SHAPE_LEN; ++i) {
        in0_elem_size *= in0_tensor->shapes[i];
    }

    int32_t in1_elem_size = 1;
    for (int i = 0; i < SHAPE_LEN; ++i) {
        in1_elem_size *= in1_tensor->shapes[i];
    }


    int64_t ifmap_need_buf_size = in0_elem_size * sizeof(float);
    float *in0_transed_ptr;
    if (ifmap_need_buf_size < rem_buf_size) {
        in0_transed_ptr = (void *)rem_buf_ptr;
        rem_buf_ptr += (in0_elem_size * sizeof(float) + 31) & (~32);
        rem_buf_size -= (in0_elem_size * sizeof(float) + 31) & (~32);
    } else {
        LOG_ERR("remaining buf size is %d, but need buf size is %d", rem_buf_size, ifmap_need_buf_size);
    }

    // step 1: do in0:  bchw --> bhwc
    {
        int32_t perm_num = 4;
        int32_t perm[4] = {0, 2, 3, 1};
        int32_t in_n = in0_tensor->shapes[0];   // b
        int32_t in_c = in0_tensor->shapes[1];   // c
        int32_t in_h = in0_tensor->shapes[2];   // h
        int32_t in_w = in0_tensor->shapes[3];   // w

        int32_t b = in_n;
        int32_t c = in_c;
        int32_t h = in_h;
        int32_t w = in_w;

        int32_t out_n = b;
        int32_t out_c = h;
        int32_t out_h = w;
        int32_t out_w = c;

        int32_t in_stride[8] = {in_c * in_h * in_w, in_h * in_w, in_w, 1};
        int32_t out_stride[8] = {out_c * out_h * out_w, out_h * out_w, out_w, 1};

        /*
        * 原始的 in stride 不能直接使用，如果直接使用的话，等于仅仅是将输入数据拷贝了一份到输出的地址上。
        * 因为这个算子是 transpose，需要跳跃着将输入数据给输出数据进行赋值。这里跳跃多少，就是通过 perm 来重新排序的
        */
        int32_t correct_in_stride[8];
        for (int i = 0; i < perm_num; ++i) {
            correct_in_stride[i] = in_stride[perm[i]];
        }

        float *cur_input_ptr, *cur_output_ptr;
        for (int outn_i = 0; outn_i < out_n; ++outn_i) {
            for (int outc_i = 0; outc_i < out_c; ++outc_i) {
                for (int outh_i = 0; outh_i < out_h; ++outh_i) {
                    cur_input_ptr = input0_ptr + outn_i * correct_in_stride[0] + outc_i * correct_in_stride[1] +
                                    outh_i * correct_in_stride[2];
                    cur_output_ptr =
                            in0_transed_ptr + outn_i * out_stride[0] + outc_i * out_stride[1] + outh_i * out_stride[2];
                    for (int outw_i = 0; outw_i < out_w; ++outw_i) {
                        cur_output_ptr[outw_i * out_stride[3]] = cur_input_ptr[outw_i * correct_in_stride[3]];
                    }
                }
            }
        }
    }


    // step 2: do  ofmap = gemm(in1, in0)   --> bkc * bhwc --> bkhw
    int32_t out_loop = 1; // = b
    for (int i = 0; i < 1; ++i) {
        out_loop *= out_tensor->shapes[i];
    }
    int32_t M = out_tensor->shapes[1];  // = k
    int32_t K = in1_tensor->shapes[2]; // = c
    int32_t N = out_tensor->shapes[2] * out_tensor->shapes[3];  // = h * w

    int32_t ifmap1_inner_loop = M * K;
    int32_t ifmap0_inner_loop = K * N;

    for (int out_loop_i = 0; out_loop_i < out_loop; ++out_loop_i) {
        float * ifmap1_ptr = input1_ptr + out_loop_i * ifmap1_inner_loop;
        float * ifmap0_ptr = in0_transed_ptr + out_loop_i * ifmap0_inner_loop;
        float * ofmap_ptr = output_ptr + out_loop_i * M * N;
#pragma omp parallel for num_threads(8)
        for (int m_i = 0; m_i < M; ++m_i) {
            float *cur_input0_ptr, *cur_input1_ptr;
            for (int n_i = 0; n_i < N; ++n_i) {
                cur_input1_ptr = ifmap1_ptr + m_i * K;
                cur_input0_ptr = ifmap0_ptr + n_i * K;
                float psum = 0;
#pragma unroll 2
                for (int k_i = 0; k_i < K; ++k_i) {
                    psum += cur_input0_ptr[k_i] * cur_input1_ptr[k_i];
                }
                ofmap_ptr[m_i * N + n_i] = psum;
            }
        }
    }

    return 0;
}

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
//    show_dev_input(params);

    EINSUM_CONFIG_S *cfg = (EINSUM_CONFIG_S *) (params[0].addr);

    if (strcmp(cfg->equation, "bmchw,bnmc->bmhwn") == 0) {
//        LOG_DBG("the cfg->equation is %s", cfg->equation);
        eval_ofmap_bmhwn(params, inputs, outputs);
    } else if (strcmp(cfg->equation, "bchw,bkc->bkhw") == 0) {
//        LOG_DBG("the cfg->equation is %s", cfg->equation);
        eval_ofmap_bkhw(params, inputs, outputs);
    }


//    LOG_ERR("sorry, the einsum have not be achieve， cur op equation is: %s", cfg->equation);

    return 0;
}



