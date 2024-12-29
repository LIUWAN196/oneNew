#include "global_avgpool.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#define GAP_SHARE_ELEM (25 * 1024 / sizeof(float))

// 定义核函数，执行全局平均池化操作
__global__ void globalAvgPoolKernel(float *output_feat, float *input_feat, int32_t reduce_num, int32_t reduce_size) {
    int block_idx = blockIdx.x;
    int th_idx = threadIdx.x;

    __shared__ float smem_data[GAP_SHARE_ELEM];
    int32_t cur_idx = block_idx * blockDim.x + th_idx;
    if (cur_idx < reduce_num) {
        for (int i = 0; i < reduce_size; i++) {
            smem_data[th_idx * reduce_size + i] = input_feat[cur_idx * reduce_size + i];
        }
    }

    __syncthreads();

    float coeff = 1.0f / reduce_size;
    if (cur_idx < reduce_num) {
        float psum = 0;
        for (int i = 0; i < reduce_size; i++) {
            psum += smem_data[th_idx * reduce_size + i];
        }
        output_feat[cur_idx] = psum * coeff;
    }

    return;
}

int eval_impl(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);

    int32_t in_n = in_tensor->shapes[0];
    int32_t in_c = in_tensor->shapes[1];
    int32_t in_h = in_tensor->shapes[2];
    int32_t in_w = in_tensor->shapes[3];

    int32_t reduce_num = in_n * in_c;
    int32_t reduce_size = in_h * in_w;
    int32_t thread_pre_block = GAP_SHARE_ELEM / reduce_size;
    int32_t block_per_grid = (reduce_num + thread_pre_block - 1) / thread_pre_block;

    // 设置线程块和网格尺寸
    dim3 BlocksPerGrid(block_per_grid);
    dim3 threadsPerBlock(thread_pre_block);

    // 启动核函数
    globalAvgPoolKernel<<<BlocksPerGrid, threadsPerBlock>>>(output_ptr, input_ptr, reduce_num, reduce_size);

    cudaDeviceSynchronize();
    return 0;
}

#include <stdio.h>

extern "C" __attribute__((visibility("default"))) int
eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
    return eval_impl(params, inputs, outputs);
}
