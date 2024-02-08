#include "conv.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "pad_conv.h"
#include "stdint.h"
#include "string.h"
//#include "../../../common/nn_common.h"
//#include <immintrin.h>
//#include <omp.h>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

int
im2col(float *input_col_ptr, float *input_ptr, OPERAND_S *in_tensor, OPERAND_S *out_tensor, OPERAND_S *weight_tensor,
       CONV_CONFIG_S *cfg) {
    int32_t kernel_c = weight_tensor->shape.C;
    int32_t kernel_h = weight_tensor->shape.H;
    int32_t kernel_w = weight_tensor->shape.W;
    int32_t in_n = in_tensor->shape.N;
    int32_t in_c = in_tensor->shape.C;
    int32_t in_h = in_tensor->shape.H + 2 * cfg->pads[0];
    int32_t in_w = in_tensor->shape.W + 2 * cfg->pads[0];

    int32_t out_h = out_tensor->shape.H;
    int32_t out_w = out_tensor->shape.W;

    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

#pragma omp parallel for num_threads(8)
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


__global__ void
calc_conv_as_gemm_gpu(float *__restrict__ output_ptr, float *__restrict__ input_ptr, float *__restrict__ weight_ptr,
                      float *__restrict__ bias_ptr, int M, int N, int K) {
    int block_idx = blockIdx.x;
    int th_idx = threadIdx.x;

    int out_elem_per_th_calc = (N + blockDim.x - 1) / blockDim.x;

////    memset(&output_ptr[block_idx * N], 0, N * sizeof(float));
//    int idx_a, idx_b, idx_c;
//    idx_c = block_idx * N;
//#pragma unroll 8
//    for (int k = 0; k < K; k++)
//    {
//        idx_a = block_idx * K + k;
//        idx_b = k * N;
//#pragma unroll 8
//        for (int j = th_idx * out_elem_per_th_calc; j < (th_idx + 1) * out_elem_per_th_calc; j++)
//        {
//            if (j < N) {
//                output_ptr[idx_c + j] += weight_ptr[idx_a] * input_ptr[idx_b + j];
//            }
//        }
//    }
//
//#pragma unroll 8
//    for (int j = th_idx * out_elem_per_th_calc; j < (th_idx + 1) * out_elem_per_th_calc; j++) {
//        if (j < N) {
//            output_ptr[idx_c + j] += bias_ptr[block_idx];
//        }
//    }


    int idx_a, idx_b, idx_c;
    idx_c = block_idx * N;
#pragma unroll
    for (int j = th_idx * out_elem_per_th_calc; j < (th_idx + 1) * out_elem_per_th_calc; j++) {
        if (j < N) {
            float psum = 0;
#pragma unroll
            for (int k = 0; k < K; k++) {
                idx_a = block_idx * K + k;
                idx_b = k * N;
                psum += weight_ptr[idx_a] * input_ptr[idx_b + j];
            }
            output_ptr[idx_c + j] = psum + bias_ptr[block_idx];
        }
    }


//    int idx_a, idx_b, idx_c;
//    idx_c = block_idx * N;
//    for (int j = th_idx * out_elem_per_th_calc; j < (th_idx + 1) * out_elem_per_th_calc; j++) {
//        if (j < N) {
//            float psum = 0;
//            for (int k = 0; k < K; k++) {
//                idx_a = block_idx * K + k;
//                idx_b = k * N;
//                psum += weight_ptr[idx_a] * input_ptr[idx_b + j];
//            }
//            output_ptr[idx_c + j] = psum + bias_ptr[block_idx];
//        }
//    }

}

int eval_1x1j1(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
//    // // printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

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

    int32_t in_n = in_tensor->shape.N;
    int32_t in_c = in_tensor->shape.C;
    int32_t in_h = in_tensor->shape.H;
    int32_t in_w = in_tensor->shape.W;

    int32_t out_n = out_tensor->shape.N;
    int32_t out_c = out_tensor->shape.C;
    int32_t out_h = out_tensor->shape.H;
    int32_t out_w = out_tensor->shape.W;

    void *src_pad_ptr;
    if (cfg->pads[0] != 0) {
        // do pad
        src_pad_ptr = malloc(in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(float));
        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv((char *) src_pad_ptr, (char *) input_ptr, in_tensor, &pad_cfg);
        input_ptr = (float *) src_pad_ptr;
    }

    const int M = out_c;
    const int N = out_h * out_w;
    const int K = in_c;


    int32_t in_elem_size = in_c * in_h * in_w;
    int32_t out_elem_size = out_c * out_h * out_w;
    int32_t weight_elem_size = out_c * in_c;
    int32_t bias_elem_size = out_c;

    float *d_in, *d_weight, *d_bias, *d_out, *d_args;
    cudaMalloc((void **) &d_in, in_elem_size * sizeof(float));
    cudaMalloc((void **) &d_weight, weight_elem_size * sizeof(float));
    cudaMalloc((void **) &d_bias, bias_elem_size * sizeof(float));

    cudaMalloc((void **) &d_out, out_elem_size * sizeof(float));

    cudaMemcpy(d_in, &input_ptr[0], in_elem_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, &weight_ptr[0], weight_elem_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, &bias_ptr[0], bias_elem_size * sizeof(float), cudaMemcpyHostToDevice);

    const int th_num = 128;
    dim3 grid(M);
    dim3 block(th_num);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    calc_conv_as_gemm_gpu<<<grid, block>>>(d_out, d_in, d_weight, d_bias, M, N, K);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); // host 等待 gpu 侧计算完毕

    cudaMemcpy(&output_ptr[0], d_out, out_elem_size * sizeof(float), cudaMemcpyDeviceToHost);


    return 0;
}

int eval_mxn_img2col(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
//    // // printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

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
    if (cfg->pads[0] != 0) {
        // do pad
        src_pad_ptr = malloc(in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(float));
        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_conv((char *) src_pad_ptr, (char *) input_ptr, in_tensor, &pad_cfg);
        input_ptr = (float *) src_pad_ptr;
    }

    float *input_col_ptr = (float *) malloc(in_c * kernel_w * kernel_h * out_h * out_w * sizeof(float));
    im2col(input_col_ptr, input_ptr, in_tensor, out_tensor, weight_tensor, cfg);
    input_ptr = input_col_ptr;


    const int M = out_c;
    const int N = out_h * out_w;
    const int K = in_c * kernel_h * kernel_w;


    int32_t in_elem_size = in_c * kernel_h * kernel_w * out_h * out_w;
    int32_t out_elem_size = out_c * out_h * out_w;
    int32_t weight_elem_size = out_c * in_c * kernel_h * kernel_w;
    int32_t bias_elem_size = out_c;

    float *d_in, *d_weight, *d_bias, *d_out, *d_args;
    cudaMalloc((void **) &d_in, in_elem_size * sizeof(float));
    cudaMalloc((void **) &d_weight, weight_elem_size * sizeof(float));
    cudaMalloc((void **) &d_bias, bias_elem_size * sizeof(float));

    cudaMalloc((void **) &d_out, out_elem_size * sizeof(float));

    cudaMemcpy(d_in, &input_ptr[0], in_elem_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, &weight_ptr[0], weight_elem_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, &bias_ptr[0], bias_elem_size * sizeof(float), cudaMemcpyHostToDevice);

    const int th_num = 128;
    dim3 grid(M);
    dim3 block(th_num);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    calc_conv_as_gemm_gpu<<<grid, block>>>(d_out, d_in, d_weight, d_bias, M, N, K);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop); // host 等待 gpu 侧计算完毕

    cudaMemcpy(&output_ptr[0], d_out, out_elem_size * sizeof(float), cudaMemcpyDeviceToHost);



    return 0;
}

int eval_impl(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    // printf("start conv dve\n");

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);

    if (cfg->kernel_shape[0] == 3 && cfg->kernel_shape[1] == 3) {
        eval_mxn_img2col(params, inputs, outputs);
    } else if (cfg->kernel_shape[0] == 1 && cfg->kernel_shape[1] == 1 && cfg->strides[0] == 1 && cfg->strides[1] == 1) {
        eval_1x1j1(params, inputs, outputs);
    } else {
        eval_mxn_img2col(params, inputs, outputs);
    }

    float *output_ptr = (float *) (outputs[0].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);
    int32_t out_elem_size = out_tensor->shape.N * out_tensor->shape.C * out_tensor->shape.H * out_tensor->shape.W;
    // write_bin(replace_char(cfg->out_operand_name[0]), out_elem_size * sizeof(float), (char *)output_ptr);

    // // printf("end conv dve\n");

    return 0;
}

#include <stdio.h>

extern "C" __attribute__((visibility("default"))) int
eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
    return eval_impl(params, inputs, outputs);
}





