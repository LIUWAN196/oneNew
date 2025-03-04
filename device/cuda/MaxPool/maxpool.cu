#include "maxpool.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

// 定义核函数，执行最大池化操作
__global__ void maxPoolKernel(float *output_feat, float *input_feat,
                              int in_h, int in_w, int out_h, int out_w, MAX_POOL_CONFIG_S *cfg) {
    int out_h_i = blockIdx.x, out_w_i = blockIdx.y;
    int out_c_i = threadIdx.x;
    float* cur_input_feat = input_feat + out_c_i * in_h * in_w;
    float* cur_output_feat = output_feat + out_c_i * out_h * out_w;

    // 初始化最大值
    float max_val = -3.40282347e+38;

    int stride_h = cfg->strides[0];
    int stride_w = cfg->strides[1];
    int k_h = cfg->kernel_shape[0];
    int k_w = cfg->kernel_shape[1];
    int pad_h = cfg->pads[0];
    int pad_w = cfg->pads[1];

    // 遍历池化窗口内的元素
    for (int i = 0; i < k_h; ++i) {
        for (int j = 0; j < k_w; ++j) {
            int in_h_i = out_h_i * stride_h + i - pad_h;
            int in_w_i = out_w_i * stride_w + j - pad_w;
            // 更新最大值
            if ((in_h_i >= 0 && in_h_i < in_h) &&
                    (in_w_i >= 0 && in_w_i < in_w)) {
                max_val = max_val > 0 ? max_val : 0;
                max_val = fmaxf(max_val, cur_input_feat[in_h_i * in_w + in_w_i]);
            }
        }
    }

    // 将结果写入输出
    cur_output_feat[out_h_i * out_w + out_w_i] = max_val;
}

int eval_impl(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    MAX_POOL_CONFIG_S *cfg = (MAX_POOL_CONFIG_S *) (params[0].addr);

    int32_t kernel_h = cfg->kernel_shape[0];
    int32_t kernel_w = cfg->kernel_shape[1];

    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs->addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

    int32_t in_n = in_tensor->shapes[0];
    int32_t in_c = in_tensor->shapes[1];
    int32_t in_h = in_tensor->shapes[2];
    int32_t in_w = in_tensor->shapes[3];

    int32_t out_n = out_tensor->shapes[0];
    int32_t out_c = out_tensor->shapes[1];
    int32_t out_h = out_tensor->shapes[2];
    int32_t out_w = out_tensor->shapes[3];

    // 设置线程块和网格尺寸
    dim3 BlocksPerGrid(out_h, out_w);
    dim3 threadsPerBlock(out_c);

    // 启动核函数
    maxPoolKernel<<<BlocksPerGrid, threadsPerBlock>>>(output_ptr, input_ptr, in_h, in_w, out_h, out_w, cfg);

    // 等待所有内核完成
    cudaDeviceSynchronize();

    return 0;
}

#include <stdio.h>

extern "C" __attribute__((visibility("default"))) int
eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
    return eval_impl(params, inputs, outputs);
}

