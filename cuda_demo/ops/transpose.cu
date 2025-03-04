#include <iostream>
#include "cstdio"
#include <cuda_runtime.h>
#include <stdint.h>

template <int BLOCK_SIZE>
__global__ void transpose_kernel(float *input, float *output, int M, int N)
{
    // 二维的矩阵转置操作，每个 block 处理 BLOCK_SIZE * BLOCK_SIZE 个数据

    // 申请 smem_data 时，对 w 方向上 +1 的 pad 长度，用于避免 bank conflict
    __shared__ float smem_data[BLOCK_SIZE][BLOCK_SIZE + 1];

    const int block_x = blockIdx.x, block_y = blockIdx.y;
    const int th_x = threadIdx.x, th_y = threadIdx.y;

    // 获取在原始的 MN 矩阵的行列索引
    int src_x_idx = block_x * BLOCK_SIZE + th_x;
    int src_y_idx = block_y * BLOCK_SIZE + th_y;

    // 将数据从 global mem 拷贝到 smem data
    if (src_y_idx < M && src_x_idx < N)
    {
        smem_data[th_y][th_x] = input[src_y_idx * N + src_x_idx];
    }

    // block 内做线程同步，保证 smem_data 所需的数据完全从 global mem 加载进来了
    __syncthreads();

    /**
     * 原始是这种格式来输出，但是采用这种情况，将 output 写出到 global mem 时没有保持线程的访存连续
     * int dst_x_idx = block_y * BLOCK_SIZE + th_y;    // 转置后矩阵的列索引
     * int dst_y_idx = block_x * BLOCK_SIZE + th_x;    // 转置后矩阵的行索引
     * if (dst_y_idx < N && dst_x_idx < M)
     * {
     *     output[dst_y_idx * M + dst_x_idx] = smem_data[th_y][th_x];
     * }
    */

    // 调整线程块顺序，保证相邻线程写出数据也是连续的，实现访存合并
    int dst_x_idx = block_y * BLOCK_SIZE + th_x;    // 转置后矩阵的列索引
    int dst_y_idx = block_x * BLOCK_SIZE + th_y;    // 转置后矩阵的行索引
    if (dst_y_idx < N && dst_x_idx < M)
    {
        output[dst_y_idx * M + dst_x_idx] = smem_data[th_x][th_y];
    }
}

int transpose_test()
{
    const int test_loop = 1024; // 设定 kernel 测试循环次数

    const int M = 1024, N = 1024;
    int fmap_buf_size = M * N * sizeof(float);

    // 1、分配 host 内存
    float *h_ifmap, *h_ofmap;
    float *cuda_ofmap; //  用于存放 dev 侧计算完，复制回 host 侧的输出数据

    h_ifmap = (float *)malloc(fmap_buf_size);
    h_ofmap = (float *)malloc(fmap_buf_size);
    cuda_ofmap = (float *)malloc(fmap_buf_size);

    // 2、初始化输入数据 h_ifmap
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_ifmap[i * N + j] = i * N + j;
        }
    }

    // 3、分配 dev 内存
    float *d_ifmap, *d_ofmap;
    cudaMalloc(&d_ifmap, fmap_buf_size);
    cudaMalloc(&d_ofmap, fmap_buf_size);

    // 4、将输入数据 h_ifmap 从 host 复制到 dev
    cudaMemcpy(d_ifmap, h_ifmap, fmap_buf_size, cudaMemcpyHostToDevice);

    // 5、运行 kernel
    const int BLOCK_SIZE = 16;
    dim3 grid(ceil(M / BLOCK_SIZE), ceil(N / BLOCK_SIZE));
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int test_loop_i = 0; test_loop_i < test_loop; test_loop_i++)
    {
        transpose_kernel<BLOCK_SIZE><<<grid, block>>>(d_ifmap, d_ofmap, M, N);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA synchronization failed: %s\n", cudaGetErrorString(err));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float seconds = milliseconds / 1000.0;
    float band_width = test_loop * (fmap_buf_size / seconds / 1e9);
    printf("transpose: using time is %6.3f ms, band width is %6.3f GB/s\n", milliseconds, band_width);

    // 6、将结果从 dev 复制回 host
    cudaMemcpy(cuda_ofmap, d_ofmap, fmap_buf_size, cudaMemcpyDeviceToHost);

    // 7、使用 cpp 代码，完成 reference 的计算
    h_ofmap[0] = 0.0f;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_ofmap[i * N + j] = h_ifmap[j * M + i];
        }
    }

    // 8、验证 cuda 计算结果的正确性
    float err_max = 0.0f;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float cur_err = abs(cuda_ofmap[i * N + j] - h_ofmap[i * N + j]);
            err_max = std::max(cur_err, err_max);
        }
    }
    if (err_max < 1e-6f)
    {
        printf("transpose op test correct, the max error is %f\n", err_max);
    }
    else
    {
        printf("transpose op test incorrect, the max error is %f\n", err_max);
    }
    printf("\n");

    // 9、释放 host 和 dev 内存
    free(h_ifmap);
    free(h_ofmap);
    free(cuda_ofmap);
    cudaFree(d_ifmap);
    cudaFree(d_ofmap);

    return 0;
}

