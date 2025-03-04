#include <iostream>
#include "cstdio"
#include <cuda_runtime.h>
#include <random>
#include <time.h>

#define TH_PER_BLOCK 256
#define TILE_M 128
#define TILE_N 128
#define TILE_K 16

#define X_PER_TH 8 // per thread calc x dirct
#define Y_PER_TH 8 // per thread calc y dirct

/*
 * 就整体而言，每个 block 使用 256 个线程来计算 128 * 128 个数据，所以每个线程计算 64 个数据。
 * 然后每个 block 内部，由于 smem 有限制，同时为了做 Prefetch。对 K 维度上再次进行分块，每个小块为 TILE_K，由此形成了每个线程内部的 loop_i 这个循环
 */
__global__ void gemm_kernel(float *C, float *A, float *B, int M, int N, int K) {
    int block_idx = blockIdx.x;
    int th_idx = threadIdx.x;

    // 使用 float4 来一次从 gmem 读取 4 个 float
    const float4 *A_f4 = (float4 *) A;
    const float4 *B_f4 = (float4 *) B;
    float4 *C_f4 = (float4 *) C;

    const int float4_size = sizeof(float4) / sizeof(float);  // 这里的 float4_size == 4，指的是一次加载 4 个 float，而不是一个 float4 的字节数

    // 申请 smem，用于存放每个 block 需要搬入的 A 和 B 部分数据。 __shared__ 的数据是一个 block 公用的
    __shared__ float4 smem_part_a_f4[2][TILE_K * TILE_M / float4_size];
    __shared__ float4 smem_part_b_f4[2][TILE_N * TILE_K / float4_size];

    float *smem_part_a;
    float *smem_part_b;

    // 每个线程自己开辟寄存器空间，来临时存放自己线程算好的 C 矩阵部分
    float c_reg[Y_PER_TH][X_PER_TH] = {{0}};
    float4 *c_reg_f4 = (float4 *) &c_reg[0];

    int cp_in_flag = 0;
    int calc_flag = 0;

    int loop = K / TILE_K;
    int elem_per_th_load_a = TILE_K * TILE_M / TH_PER_BLOCK;  // 每个线程需要搬运的 A 矩阵元素个数
    int elem_per_th_load_b = TILE_N * TILE_K / TH_PER_BLOCK;  // 每个线程需要搬运的 B 矩阵元素个数
    int th_per_row = 16; //  = (线程数)^0.5 = 256^0.5 = 16。表明每个 block 中，会输出 128 行 128 列数据。而每行和每列划分为 16 * 16 个小格子，分别由 256 个线程负责，即每个线程负责计算 8 * 8 个输出元素

    // 下面的 prefetch 逻辑和 dsp 的 pingpong 逻辑一模一样
    // step 1: load A part and B part to smem from gmem
    int loop_i = 0;
    {
        // 计算本个线程需要搬运的 A 矩阵元素的起始索引
        int a_copy_st = (block_idx / 8 * TILE_M + th_idx * elem_per_th_load_a / TILE_K) * K + loop_i * TILE_K +
                        th_idx * elem_per_th_load_a % TILE_K;
        a_copy_st = a_copy_st / float4_size;  // 由于是按照 float4 来搬运，所以如果把 float 数据看作 float4 的话，起始索引应该 / 4

#pragma unroll
        // 开始搬运 A 矩阵的部分内容到 smem
        for (size_t i = 0; i < elem_per_th_load_a / float4_size; i++) {
            smem_part_a_f4[cp_in_flag][th_idx * elem_per_th_load_a / float4_size + i] = A_f4[a_copy_st + i];
        }

        // 这里的内容和上面搬运 A 矩阵的步骤一模一样
        int b_copy_st = (loop_i * TILE_K + th_idx * elem_per_th_load_a / TILE_N) * N + block_idx % 8 * TILE_N +
                        th_idx * elem_per_th_load_a % TILE_N;
        b_copy_st = b_copy_st / float4_size;

#pragma unroll
        for (size_t i = 0; i < elem_per_th_load_b / float4_size; i++) {
            smem_part_b_f4[cp_in_flag][th_idx * elem_per_th_load_b / float4_size + i] = B_f4[b_copy_st + i];
        }
    }

    __syncthreads(); // sync share mem       需要对 block 中的数据做一次同步

    for (loop_i = 0; loop_i < loop; loop_i++) {
        // step 1: load A part and B part to smem from gmem        这段除了这里有一个 (loop_i + 1)，和上面搬运第一次 A 和 B 矩阵到 smem 的代码基本完全一致
        cp_in_flag = cp_in_flag ^ 0x1;
        if (loop_i < loop - 1) {
            int a_copy_st =
                    (block_idx / 8 * TILE_M + th_idx * elem_per_th_load_a / TILE_K) * K + (loop_i + 1) * TILE_K +
                    th_idx * elem_per_th_load_a % TILE_K;
            a_copy_st = a_copy_st / float4_size;

#pragma unroll
            for (size_t i = 0; i < elem_per_th_load_a / float4_size; i++) {
                smem_part_a_f4[cp_in_flag][th_idx * elem_per_th_load_a / float4_size + i] = A_f4[a_copy_st + i];
            }

            int b_copy_st =
                    ((loop_i + 1) * TILE_K + th_idx * elem_per_th_load_a / TILE_N) * N + block_idx % 8 * TILE_N +
                    th_idx * elem_per_th_load_a % TILE_N;
            b_copy_st = b_copy_st / float4_size;

#pragma unroll
            for (size_t i = 0; i < elem_per_th_load_b / float4_size; i++) {
                smem_part_b_f4[cp_in_flag][th_idx * elem_per_th_load_b / float4_size + i] = B_f4[b_copy_st + i];
            }
        }

        // step 2: calc, each thread calc X_PER_TH * Y_PER_TH size output elem
        // 这里为了方便，我用 float 的指针 smem_part_a、smem_part_b 指向 smem_part_a_f4、smem_part_b_f4
        smem_part_a = (float *) &smem_part_a_f4[calc_flag][0];
        smem_part_b = (float *) &smem_part_b_f4[calc_flag][0];
        {
            int y_calc_st = th_idx / th_per_row * Y_PER_TH;
            int x_calc_st = th_idx % th_per_row * X_PER_TH;

            // 一个 block 计算 128 * 128 个输出元素，一共 256 个线程。所以每个线程计算 128 * 128 / 256 = 64 = Y_PER_TH * X_PER_TH 个元素
#pragma unroll
            for (size_t y = 0; y < Y_PER_TH; y++) {
                for (size_t x = 0; x < X_PER_TH; x++) {
                    for (size_t k_i = 0; k_i < TILE_K; k_i++) {
                        c_reg[y][x] +=
                                smem_part_a[(y_calc_st + y) * TILE_K + k_i] * smem_part_b[k_i * TILE_N + x_calc_st + x];
                    }
                }
            }
        }

        calc_flag = calc_flag ^ 0x1;
        __syncthreads(); // sync share mem
    }

    // step 3: save c_reg to gmem
    // 由这里来再次理解 block_idx / 8 和 th_idx / th_per_row  (这里的 th_per_row 表示每个 block 中的每行输出的 128 个元素，由 16 个线程来负责输出)。首先输出的 1024 * 1024 被分为了 8 * 8 个大格子，每个大格子由一个 block 来计算，所以一共有 64 个 block，每个 block 计算 128 * 128 个输出数据；接着每个大格子被分为 16 * 16 个小格子，，每个小格子由一个线程来计算，所以一共有 256 个线程，每个线程计算 8 * 8 个输出数据，每行或者每列都有 16 个线程。所以这里的 y_store_st 和 x_store_st 就是通过当前 block 和线程的 idx 再 * TILE_M 或者 * TILE_N (这个值就是大格子的宽或者高 128) 和  * Y_PER_TH 或者  * X_PER_TH (这个值就是小格子的宽或者高 8) ，来得到当前线程所计算的输出的起始位置
    int y_store_st = block_idx / 8 * TILE_M + th_idx / th_per_row * Y_PER_TH;
    int x_store_st = block_idx % 8 * TILE_N + th_idx % th_per_row * X_PER_TH;

    // 每个线程将自己 c_reg_f4 寄存器中的数据写出到 gmem 中
#pragma unroll
    for (size_t y = 0; y < Y_PER_TH; y++) {
#pragma unroll
        for (size_t x = 0; x < X_PER_TH / float4_size; x++) {
            C_f4[(y_store_st + y) * N / float4_size + x_store_st / float4_size + x] = c_reg_f4[
                    y * X_PER_TH / float4_size + x];
        }
    }
}

int gemm_test() {
    const int test_loop = 512; // 设定 kernel 测试循环次数

    int M = 1024;
    int K = 1024;
    int N = 1024;

    // 1、分配 host 内存
    float *h_ifmap_a, *h_ifmap_b, *h_ofmap;
    float *cuda_ofmap;  //  用于存放 dev 侧计算完，复制回 host 侧的输出数据

    h_ifmap_a = (float *) malloc(M * K * sizeof(float));
    h_ifmap_b = (float *) malloc(K * N * sizeof(float));
    h_ofmap = (float *) malloc(M * N * sizeof(float));
    cuda_ofmap = (float *) malloc(M * N * sizeof(float));

    // 2、初始化输入数据 h_ifmap
    std::random_device rd;
    std::mt19937 gen(1);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    const int matrix_max = 32; // 设置 A 和 B 两个矩阵元素的范围为 [0, matrix_max]

    for (int i = 0; i < M * K; i++) {
        h_ifmap_a[i] = dis(gen) * matrix_max;
    }

    for (int i = 0; i < K * N; i++) {
        h_ifmap_b[i] = dis(gen) * matrix_max;
    }

    for (int i = 0; i < M * N; i++) {
        h_ofmap[i] = 0;
    }

    // 3、分配 dev 内存
    float *d_ifmap_a, *d_ifmap_b, *d_ofmap;
    cudaMalloc(&d_ifmap_a, M * K * sizeof(float));
    cudaMalloc(&d_ifmap_b, K * N * sizeof(float));
    cudaMalloc(&d_ofmap, M * N * sizeof(float));

    // 4、将输入数据 h_ifmap 从 host 复制到 dev
    cudaMemcpy(d_ifmap_a, h_ifmap_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ifmap_b, h_ifmap_b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int test_loop_i = 0; test_loop_i < test_loop; test_loop_i++) {
        // 运行 kernel
        dim3 grid(M / TILE_M * N / TILE_N);
        dim3 block(TH_PER_BLOCK);
        gemm_kernel<<<grid, block>>>(d_ofmap, d_ifmap_a, d_ifmap_b, M, N, K);
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
    float ops = (long) test_loop * M * N * K * 2 / 1e12;
    printf("gemm: using time is %6.3f ms, computing power is %6.3f TFLOPS, \n", milliseconds, ops / seconds);

    // 6、将结果从 dev 复制回 host
    cudaMemcpy(cuda_ofmap, d_ofmap, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 7、使用 cpp 代码，完成 reference 的计算
    for (size_t m_i = 0; m_i < M; m_i++) {
        for (size_t k_i = 0; k_i < K; k_i++) {
            for (size_t n_i = 0; n_i < N; n_i++) {
                h_ofmap[m_i * N + n_i] += h_ifmap_a[m_i * K + k_i] * h_ifmap_b[k_i * N + n_i];
            }
        }
    }

    // 8、验证 cuda 计算结果的正确性
    float err_max = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float cur_err = abs(cuda_ofmap[i * N + j] - h_ofmap[i * N + j]);
            err_max = std::max(cur_err, err_max);
        }
    }
    if (err_max < 0.1f) {
        printf("gemm op test correct, the max error is %f\n", err_max);
    } else {
        printf("gemm op test incorrect, the max error is %f\n", err_max);
    }
    printf("\n");

    // 9、释放 host 和 dev 内存
    free(h_ifmap_a);
    free(h_ifmap_b);
    free(h_ofmap);
    free(cuda_ofmap);
    cudaFree(d_ifmap_a);
    cudaFree(d_ifmap_b);
    cudaFree(d_ofmap);

    return 0;
}
