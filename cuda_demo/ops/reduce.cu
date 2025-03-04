#include <iostream>
#include "cstdio"
#include <cuda_runtime.h>
#include <stdint.h>


template<int BLOCK_SIZE, int NUM_PER_THREAD>
__global__ void reduce_kernel(float *input, float *output) {

    // 在 shared mem 声明空间，用于存放 block 内的数据
    volatile __shared__ float smem_data[BLOCK_SIZE];

    uint32_t th_idx = threadIdx.x;
    float *cur_block_input = input + blockIdx.x * blockDim.x * NUM_PER_THREAD;
    // 在每个线程从 global mem 读取数据时，同时做 NUM_PER_THREAD 次加法，避免大部分线程只做数据搬运，造成浪费
    float tmp = 0.0f;
#pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        tmp += cur_block_input[th_idx + blockDim.x * i];
    }
    smem_data[th_idx] = tmp;

    // block 内做线程同步，保证 smem_data 所需的数据完全从 global mem 加载进来了
    __syncthreads();

    for (int act_th = blockDim.x / 2; act_th > 32; act_th >>= 1) {    // 每次循环活跃的线程数减半
        if (threadIdx.x < act_th) { // block 内只有部分线程活跃
            // 活跃的线程将对应线程的数据以及跳跃 act_th 步长的数据累加
            smem_data[threadIdx.x] += smem_data[threadIdx.x + act_th];
        }
        // 依然需要做 block 内的线程同步，保证 smem_data 数据的正确性
        __syncthreads();
    }

    /*
     * 上面 for 循环的终止是 act_th <= 32，此时实际上已经将该 block 数据 reduce 到一个 warp 了，由于 warp 内天然同步，
     * 不需要 __syncthreads() 来同步，所以直接将最后 32 个数据的 reduce 展开
     */
    if (th_idx < 32) {
        smem_data[th_idx] += smem_data[th_idx + 32];
        smem_data[th_idx] += smem_data[th_idx + 16];
        smem_data[th_idx] += smem_data[th_idx + 8];
        smem_data[th_idx] += smem_data[th_idx + 4];
        smem_data[th_idx] += smem_data[th_idx + 2];
        smem_data[th_idx] += smem_data[th_idx + 1];
    }

    // 经过上面操作，该 block 内所需要 reduce 的数据，已经完全归约到 smem_data[0] 了，此时写出即可
    if (th_idx == 0) {
        output[blockIdx.x] = smem_data[0];
    }

}

int reduce_test() {
    const int test_loop = 1024; // 设定 kernel 测试循环次数

    const int N = 1 << 20; // 计算 1M 数据的 reduce sum 结果
    int ifmap_buf_size = N * sizeof(float);

    // 1、分配 host 内存
    float *h_ifmap, *h_ofmap;
    float *cuda_ofmap;  //  用于存放 dev 侧计算完，复制回 host 侧的输出数据

    h_ifmap = (float *) malloc(ifmap_buf_size);
    h_ofmap = (float *) malloc(sizeof(float));
    cuda_ofmap = (float *) malloc(sizeof(float));

    // 2、初始化输入数据 h_ifmap
    for (int i = 0; i < N; i++) {
        h_ifmap[i] = 1.0f;
    }
    h_ofmap[0] = 0.0f;
    cuda_ofmap[0] = 0.0f;

    // 3、分配 dev 内存
    const int NUM_PER_THREAD = 4;
    const int STAGE0_BLOCK_SIZE = 1024;
    const int STAGE1_BLOCK_SIZE = N / STAGE0_BLOCK_SIZE / NUM_PER_THREAD / NUM_PER_THREAD;

    float *d_ifmap, *d_tmp, *d_ofmap;
    cudaMalloc(&d_ifmap, ifmap_buf_size);
    cudaMalloc(&d_tmp, STAGE0_BLOCK_SIZE * sizeof(float));
    cudaMalloc(&d_ofmap, sizeof(float));

    // 4、将输入数据 h_ifmap 从 host 复制到 dev
    cudaMemcpy(d_ifmap, h_ifmap, ifmap_buf_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int test_loop_i = 0; test_loop_i < test_loop; test_loop_i++) {
        // 5、运行 kernel
        {   // stage 0
            dim3 grid(N / STAGE0_BLOCK_SIZE / NUM_PER_THREAD);
            dim3 block(STAGE0_BLOCK_SIZE);
            reduce_kernel<STAGE0_BLOCK_SIZE, NUM_PER_THREAD><<<grid, block>>>(d_ifmap, d_tmp);
        }
        {   // stage 1
            dim3 grid(1);
            dim3 block(STAGE1_BLOCK_SIZE);
            reduce_kernel<STAGE1_BLOCK_SIZE, NUM_PER_THREAD><<<grid, block>>>(d_tmp, d_ofmap);
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA synchronization failed: %s\n", cudaGetErrorString(err));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float seconds = milliseconds / 1000.0;
    float band_width = test_loop * (ifmap_buf_size / seconds / 1e9);
    printf("reduce_sum: using time is %6.3f ms, band width is %6.3f GB/s\n", milliseconds, band_width);

    // 6、将结果从 dev 复制回 host
    cudaMemcpy(cuda_ofmap, d_ofmap, sizeof(float), cudaMemcpyDeviceToHost);

    // 7、使用 cpp 代码，完成 reference 的计算
    h_ofmap[0] = 0.0f;
    for (int i = 0; i < N; i++) {
        h_ofmap[0] += h_ifmap[i];
    }

    // 8、验证 cuda 计算结果的正确性
    float err_ = abs(cuda_ofmap[0] - h_ofmap[0]);
    if (err_ < 1e-3f) {
        printf("reduce_sum op test correct, CUDA output is %f, ref is %f, and the error is %f\n",
               cuda_ofmap[0], h_ofmap[0], err_);
    } else {
        printf("reduce_sum op test incorrect, CUDA output is %f, ref is %f, and the error is %f\n",
               cuda_ofmap[0], h_ofmap[0], err_);
    }
    printf("\n");

    // 9、释放 host 和 dev 内存
    free(h_ifmap);
    free(h_ofmap);
    free(cuda_ofmap);
    cudaFree(d_ifmap);
    cudaFree(d_ofmap);
    cudaFree(d_tmp);

    return 0;
}

