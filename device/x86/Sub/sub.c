#include "sub.h"
#include <stdlib.h>
#include <stdio.h>

/*
 * 可以参考 https://blog.csdn.net/weixin_42575020/article/details/106947188 这里的 numpy 的 4 种广播方法
numpy 四则运算的广播规则：
规则 1：如果两个数组的维度数不相同，那么小维度数组的形状将会在最左边补 1
规则 2：如果两个数组的形状在任何一个维度上都不匹配，那么数组的形状会沿着维度 为 1 的维度扩展以匹配另外一个数组的形状
规则 3：如果两个数组的形状在任何一个维度上都不匹配并且没有任何一个维度等于 1， 那么会引发异常。
 */

int32_t fun_0(OPERAND_S ofmap_tensor, OPERAND_S small_tensor, OPERAND_S large_tensor, SUB_CONFIG_S* cfg);
int32_t fun_1(OPERAND_S ofmap_tensor, OPERAND_S small_tensor, OPERAND_S large_tensor, SUB_CONFIG_S* cfg);
int32_t fun_2(OPERAND_S ofmap_tensor, OPERAND_S small_tensor, OPERAND_S large_tensor, SUB_CONFIG_S* cfg);

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
//    show_dev_input(params);

    SUB_CONFIG_S *cfg = (SUB_CONFIG_S *) (params[0].addr);

    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *in1_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

    int32_t in0_elem_size = operand_elem_size(in0_tensor);
    int32_t in1_elem_size = operand_elem_size(in1_tensor);

    // 将 elem size 较大和较小的 tensor 和指针分别命名为 large 和 small
    OPERAND_S ofmap_tensor, small_tensor, large_tensor;
    ofmap_tensor = *out_tensor;
    ofmap_tensor.p_data = outputs[0].addr;
    if (in0_elem_size < in1_elem_size) {
        small_tensor = *in0_tensor;
        small_tensor.p_data = inputs[0].addr;
        large_tensor = *in1_tensor;
        large_tensor.p_data = inputs[1].addr;
    } else {
        small_tensor = *in1_tensor;
        small_tensor.p_data = inputs[1].addr;
        large_tensor = *in0_tensor;
        large_tensor.p_data = inputs[0].addr;
    }

    if (in0_elem_size == in1_elem_size) {
        // 1、两个数组 elem size 完全相同，直接相加即可；
        fun_0(ofmap_tensor, small_tensor, large_tensor, cfg);
    } else if (operand_elem_size(&small_tensor) == 1) {
        // 2、small 的 elem size 为 1，也就是要把这个标量广播到任意维度
        fun_1(ofmap_tensor, small_tensor, large_tensor, cfg);
    } else {
        // 注意进入这个分支需要修改 shape 信息，所以要将 tensor 复制一份，不要在原始的 tensor 上修改
        if (small_tensor.dim_num_of_shapes != large_tensor.dim_num_of_shapes) {
            OPERAND_S tmp;
            for (int i = 0; i < SHAPE_LEN; ++i) {
                tmp.shapes[i] = 1;
            }
            int32_t large_dims_num = large_tensor.dim_num_of_shapes;
            int32_t small_dims_num = small_tensor.dim_num_of_shapes;

            for (int32_t small_idx = 0; small_idx < small_dims_num; ++small_idx) {
                tmp.shapes[small_idx + (large_dims_num - small_dims_num)] = small_tensor.shapes[small_idx];
            }
            tmp.dim_num_of_shapes = large_tensor.dim_num_of_shapes;
            tmp.p_data = small_tensor.p_data;
            small_tensor = tmp;
        }
        // 此时两个 tensor 的维度长度都相同了, 做某些维度的广播减法，这些广播的维度可能相邻也可能不相邻
        // 根据上面规则 3，依次对比 tensor0 和 tensor1 的每个维度
        int32_t dims_num = large_tensor.dim_num_of_shapes;
        BOOL dims_correct = TRUE;
        for (int dim_i = 0; dim_i < dims_num; ++dim_i) {
            if (small_tensor.shapes[dim_i] != large_tensor.shapes[dim_i] && small_tensor.shapes[dim_i] != 1) {
                dims_correct = FALSE;
                break;
            }
        }
        if (dims_correct == FALSE) {
            LOG_ERR("sub op 的 shape 不符合广播规则\n");
        }
        fun_2(ofmap_tensor, small_tensor, large_tensor, cfg);
    }

    // 由于有的特殊的 case (例如 rt detr 模型中的部分 sub 算子)，是 scalar - vector。所以这里需要考虑是否需要对数据 * -1
    if (in0_elem_size < in1_elem_size) {
        float *output_ptr = (float *) (ofmap_tensor.p_data);
        int32_t ofmap_elem_size = operand_elem_size(&ofmap_tensor);
#pragma omp parallel for num_threads(8)
        for (int i = 0; i < ofmap_elem_size; ++i) {
            output_ptr[i] = -1 * output_ptr[i];
        }
    }

    return 0;
}


int32_t fun_0(OPERAND_S ofmap_tensor, OPERAND_S small_tensor, OPERAND_S large_tensor, SUB_CONFIG_S* cfg) {

    float *input0_ptr = (float *) (small_tensor.p_data);
    float *input1_ptr = (float *) (large_tensor.p_data);
    float *output_ptr = (float *) (ofmap_tensor.p_data);

    int32_t ofmap_elem_size = operand_elem_size(&ofmap_tensor);
#pragma omp parallel for num_threads(8)
    for (int i = 0; i < ofmap_elem_size; ++i) {
        output_ptr[i] = input1_ptr[i] - input0_ptr[i];
    }

    return 0;
}


int32_t fun_1(OPERAND_S ofmap_tensor, OPERAND_S small_tensor, OPERAND_S large_tensor, SUB_CONFIG_S* cfg) {

    float *small_ptr = (float *) (small_tensor.p_data);
    float *large_ptr = (float *) (large_tensor.p_data);
    float *output_ptr = (float *) (ofmap_tensor.p_data);

    int32_t ofmap_elem_size = operand_elem_size(&ofmap_tensor);
#pragma omp parallel for num_threads(8)
    for (int i = 0; i < ofmap_elem_size; ++i) {
        output_ptr[i] = large_ptr[i] - small_ptr[0];
    }

    return 0;
}

int32_t fun_2_dim3(OPERAND_S ofmap_tensor, OPERAND_S small_tensor, OPERAND_S large_tensor, SUB_CONFIG_S* cfg);
int32_t fun_2_dim4(OPERAND_S ofmap_tensor, OPERAND_S small_tensor, OPERAND_S large_tensor, SUB_CONFIG_S* cfg);
int32_t fun_2_dim5(OPERAND_S ofmap_tensor, OPERAND_S small_tensor, OPERAND_S large_tensor, SUB_CONFIG_S* cfg);
int32_t fun_2_dim6(OPERAND_S ofmap_tensor, OPERAND_S small_tensor, OPERAND_S large_tensor, SUB_CONFIG_S* cfg);

int32_t fun_2(OPERAND_S ofmap_tensor, OPERAND_S small_tensor, OPERAND_S large_tensor, SUB_CONFIG_S* cfg) {

    int32_t ofmap_dims_num = ofmap_tensor.dim_num_of_shapes;
    if (ofmap_dims_num <= 2) {
        LOG_ERR("在 sub op 中，输出的 shape 维度 <= 2\n");
    } else if (ofmap_dims_num <= 3) {
        fun_2_dim3(ofmap_tensor, small_tensor, large_tensor, cfg);
    } else if (ofmap_dims_num <= 4) {
        fun_2_dim4(ofmap_tensor, small_tensor, large_tensor, cfg);
    } else if (ofmap_dims_num <= 5) {
        fun_2_dim5(ofmap_tensor, small_tensor, large_tensor, cfg);
    } else if (ofmap_dims_num <= 6) {
        fun_2_dim6(ofmap_tensor, small_tensor, large_tensor, cfg);
    } else {
        LOG_ERR("在 sub op 中，输出的 shape 维度 > 6\n");
    }

    return 0;
}

int32_t fun_2_dim3(OPERAND_S ofmap_tensor, OPERAND_S small_tensor, OPERAND_S large_tensor, SUB_CONFIG_S* cfg){

    float *small_ptr = (float *) (small_tensor.p_data);
    float *large_ptr = (float *) (large_tensor.p_data);
    float *ofmap_ptr = (float *) (ofmap_tensor.p_data);

    int32_t large_dim0 = large_tensor.shapes[0];
    int32_t large_dim1 = large_tensor.shapes[1];
    int32_t large_dim2 = large_tensor.shapes[2];


    int32_t large_stride0 = large_dim1 * large_dim2;
    int32_t large_stride1 = large_dim2;
    int32_t large_stride2 = 1;

    /*
     * 注意：这里的 small_stride 的计算和普通的 stride 的计算逻辑有所不同。如果 small_dimX 为 1，则 small_strideX == 0，这是为了保
     * 证在下面的 dimX_i 随着 large_dimX 增加时，不会导致本来需要广播的 small 随着 large_dimX 增加而增加
     */
    int32_t small_dim0 = small_tensor.shapes[0];
    int32_t small_dim1 = small_tensor.shapes[1];
    int32_t small_dim2 = small_tensor.shapes[2];

    int32_t small_stride0 = (small_dim0 == 1) ? 0 : small_dim1 * small_dim2;
    int32_t small_stride1 = (small_dim1 == 1) ? 0 : small_dim2;
    int32_t small_stride2 = (small_dim2 == 1) ? 0 : 1;

    // 开始做减法
#pragma omp parallel for num_threads(8)
    for (int dim0_i = 0; dim0_i < large_dim0; ++dim0_i) {
        float *cur_ofmap_ptr, *cur_large_ptr, *cur_small_ptr;
        for (int dim1_i = 0; dim1_i < large_dim1; ++dim1_i) {
            cur_ofmap_ptr = ofmap_ptr + dim0_i * large_stride0 + dim1_i * large_stride1;
            cur_large_ptr = large_ptr + dim0_i * large_stride0 + dim1_i * large_stride1;
            cur_small_ptr = small_ptr + dim0_i * small_stride0 + dim1_i * small_stride1;
#pragma unroll 2
            for (int dim2_i = 0; dim2_i < large_dim2; ++dim2_i) {
                cur_ofmap_ptr[dim2_i] = cur_large_ptr[dim2_i] - cur_small_ptr[dim2_i * small_stride2];
            }
        }
    }

    return 0;
}

int32_t fun_2_dim4(OPERAND_S ofmap_tensor, OPERAND_S small_tensor, OPERAND_S large_tensor, SUB_CONFIG_S* cfg){

    float *small_ptr = (float *) (small_tensor.p_data);
    float *large_ptr = (float *) (large_tensor.p_data);
    float *ofmap_ptr = (float *) (ofmap_tensor.p_data);

    int32_t large_dim0 = large_tensor.shapes[0];
    int32_t large_dim1 = large_tensor.shapes[1];
    int32_t large_dim2 = large_tensor.shapes[2];
    int32_t large_dim3 = large_tensor.shapes[3];

    int32_t large_stride0 = large_dim1 * large_dim2 * large_dim3;
    int32_t large_stride1 = large_dim2 * large_dim3;
    int32_t large_stride2 = large_dim3;
    int32_t large_stride3 = 1;

    /*
     * 注意：这里的 small_stride 的计算和普通的 stride 的计算逻辑有所不同。如果 small_dimX 为 1，则 small_strideX == 0，这是为了保
     * 证在下面的 dimX_i 随着 large_dimX 增加时，不会导致本来需要广播的 small 随着 large_dimX 增加而增加
     */
    int32_t small_dim0 = small_tensor.shapes[0];
    int32_t small_dim1 = small_tensor.shapes[1];
    int32_t small_dim2 = small_tensor.shapes[2];
    int32_t small_dim3 = small_tensor.shapes[3];

    int32_t small_stride0 = (small_dim0 == 1) ? 0 : small_dim1 * small_dim2 * small_dim3;
    int32_t small_stride1 = (small_dim1 == 1) ? 0 : small_dim2 * small_dim3;
    int32_t small_stride2 = (small_dim2 == 1) ? 0 : small_dim3;
    int32_t small_stride3 = (small_dim3 == 1) ? 0 : 1;

    // 开始做减法
#pragma omp parallel for num_threads(8)
    for (int dim0_i = 0; dim0_i < large_dim0; ++dim0_i) {
        float *cur_ofmap_ptr, *cur_large_ptr, *cur_small_ptr;
        for (int dim1_i = 0; dim1_i < large_dim1; ++dim1_i) {
            for (int dim2_i = 0; dim2_i < large_dim2; ++dim2_i) {
                cur_ofmap_ptr = ofmap_ptr + dim0_i * large_stride0 + dim1_i * large_stride1 + dim2_i * large_stride2;
                cur_large_ptr = large_ptr + dim0_i * large_stride0 + dim1_i * large_stride1 + dim2_i * large_stride2;
                cur_small_ptr = small_ptr + dim0_i * small_stride0 + dim1_i * small_stride1 + dim2_i * small_stride2;
#pragma unroll 2
                for (int dim3_i = 0; dim3_i < large_dim3; ++dim3_i) {
                    cur_ofmap_ptr[dim3_i] = cur_large_ptr[dim3_i] - cur_small_ptr[dim3_i * small_stride3];
                }
            }
        }
    }

    return 0;
}

int32_t fun_2_dim5(OPERAND_S ofmap_tensor, OPERAND_S small_tensor, OPERAND_S large_tensor, SUB_CONFIG_S* cfg){

    float *small_ptr = (float *) (small_tensor.p_data);
    float *large_ptr = (float *) (large_tensor.p_data);
    float *ofmap_ptr = (float *) (ofmap_tensor.p_data);

    int32_t large_dim0 = large_tensor.shapes[0];
    int32_t large_dim1 = large_tensor.shapes[1];
    int32_t large_dim2 = large_tensor.shapes[2];
    int32_t large_dim3 = large_tensor.shapes[3];
    int32_t large_dim4 = large_tensor.shapes[4];

    int32_t large_stride0 = large_dim1 * large_dim2 * large_dim3 * large_dim4;
    int32_t large_stride1 = large_dim2 * large_dim3 * large_dim4;
    int32_t large_stride2 = large_dim3 * large_dim4;
    int32_t large_stride3 = large_dim4;
    int32_t large_stride4 = 1;

    /*
     * 注意：这里的 small_stride 的计算和普通的 stride 的计算逻辑有所不同。如果 small_dimX 为 1，则 small_strideX == 0，这是为了保
     * 证在下面的 dimX_i 随着 large_dimX 增加时，不会导致本来需要广播的 small 随着 large_dimX 增加而增加
     */
    int32_t small_dim0 = small_tensor.shapes[0];
    int32_t small_dim1 = small_tensor.shapes[1];
    int32_t small_dim2 = small_tensor.shapes[2];
    int32_t small_dim3 = small_tensor.shapes[3];
    int32_t small_dim4 = small_tensor.shapes[4];

    int32_t small_stride0 = (small_dim0 == 1) ? 0 : small_dim1 * small_dim2 * small_dim3 * small_dim4;
    int32_t small_stride1 = (small_dim1 == 1) ? 0 : small_dim2 * small_dim3 * small_dim4;
    int32_t small_stride2 = (small_dim2 == 1) ? 0 : small_dim3 * small_dim4;
    int32_t small_stride3 = (small_dim3 == 1) ? 0 : small_dim4;
    int32_t small_stride4 = (small_dim4 == 1) ? 0 : 1;

    // 开始做减法
#pragma omp parallel for num_threads(8)
    for (int dim0_i = 0; dim0_i < large_dim0; ++dim0_i) {
        float *cur_ofmap_ptr, *cur_large_ptr, *cur_small_ptr;
        for (int dim1_i = 0; dim1_i < large_dim1; ++dim1_i) {
            for (int dim2_i = 0; dim2_i < large_dim2; ++dim2_i) {
                for (int dim3_i = 0; dim3_i < large_dim3; ++dim3_i) {
                    cur_ofmap_ptr = ofmap_ptr + dim0_i * large_stride0 +
                                    dim1_i * large_stride1 + dim2_i * large_stride2 + dim3_i * large_stride3;
                    cur_large_ptr = large_ptr + dim0_i * large_stride0 +
                                    dim1_i * large_stride1 + dim2_i * large_stride2 + dim3_i * large_stride3;
                    cur_small_ptr = small_ptr + dim0_i * small_stride0 +
                                    dim1_i * small_stride1 + dim2_i * small_stride2 + dim3_i * small_stride3;
#pragma unroll 2
                    for (int dim4_i = 0; dim4_i < large_dim4; ++dim4_i) {
                        cur_ofmap_ptr[dim4_i] = cur_large_ptr[dim4_i] - cur_small_ptr[dim4_i * small_stride4];
                    }
                }
            }
        }
    }

    return 0;
}

int32_t fun_2_dim6(OPERAND_S ofmap_tensor, OPERAND_S small_tensor, OPERAND_S large_tensor, SUB_CONFIG_S* cfg){

    float *small_ptr = (float *) (small_tensor.p_data);
    float *large_ptr = (float *) (large_tensor.p_data);
    float *ofmap_ptr = (float *) (ofmap_tensor.p_data);

    int32_t large_dim0 = large_tensor.shapes[0];
    int32_t large_dim1 = large_tensor.shapes[1];
    int32_t large_dim2 = large_tensor.shapes[2];
    int32_t large_dim3 = large_tensor.shapes[3];
    int32_t large_dim4 = large_tensor.shapes[4];
    int32_t large_dim5 = large_tensor.shapes[5];

    int32_t large_stride0 = large_dim1 * large_dim2 * large_dim3 * large_dim4 * large_dim5;
    int32_t large_stride1 = large_dim2 * large_dim3 * large_dim4 * large_dim5;
    int32_t large_stride2 = large_dim3 * large_dim4 * large_dim5;
    int32_t large_stride3 = large_dim4 * large_dim5;
    int32_t large_stride4 = large_dim5;
    int32_t large_stride5 = 1;

    /*
     * 注意：这里的 small_stride 的计算和普通的 stride 的计算逻辑有所不同。如果 small_dimX 为 1，则 small_strideX == 0，这是为了保
     * 证在下面的 dimX_i 随着 large_dimX 增加时，不会导致本来需要广播的 small 随着 large_dimX 增加而增加
     */
    int32_t small_dim0 = small_tensor.shapes[0];
    int32_t small_dim1 = small_tensor.shapes[1];
    int32_t small_dim2 = small_tensor.shapes[2];
    int32_t small_dim3 = small_tensor.shapes[3];
    int32_t small_dim4 = small_tensor.shapes[4];
    int32_t small_dim5 = small_tensor.shapes[5];

    int32_t small_stride0 = (small_dim0 == 1) ? 0 : small_dim1 * small_dim2 * small_dim3 * small_dim4 * small_dim5;
    int32_t small_stride1 = (small_dim1 == 1) ? 0 : small_dim2 * small_dim3 * small_dim4 * small_dim5;
    int32_t small_stride2 = (small_dim2 == 1) ? 0 : small_dim3 * small_dim4 * small_dim5;
    int32_t small_stride3 = (small_dim3 == 1) ? 0 : small_dim4 * small_dim5;
    int32_t small_stride4 = (small_dim4 == 1) ? 0 : small_dim5;
    int32_t small_stride5 = (small_dim5 == 1) ? 0 : 1;

    // 开始做减法
#pragma omp parallel for num_threads(8)
    for (int dim0_i = 0; dim0_i < large_dim0; ++dim0_i) {
        float *cur_ofmap_ptr, *cur_large_ptr, *cur_small_ptr;
        for (int dim1_i = 0; dim1_i < large_dim1; ++dim1_i) {
            for (int dim2_i = 0; dim2_i < large_dim2; ++dim2_i) {
                for (int dim3_i = 0; dim3_i < large_dim3; ++dim3_i) {
                    cur_ofmap_ptr = ofmap_ptr + dim0_i * large_stride0 +
                                    dim1_i * large_stride1 + dim2_i * large_stride2 + dim3_i * large_stride3;
                    cur_large_ptr = large_ptr + dim0_i * large_stride0 +
                                    dim1_i * large_stride1 + dim2_i * large_stride2 + dim3_i * large_stride3;
                    cur_small_ptr = small_ptr + dim0_i * small_stride0 +
                                    dim1_i * small_stride1 + dim2_i * small_stride2 + dim3_i * small_stride3;
#pragma unroll 2
                    for (int dim4_i = 0; dim4_i < large_dim4; ++dim4_i) {
                        for (int dim5_i = 0; dim5_i < large_dim5; ++dim5_i) {
                            cur_ofmap_ptr[dim4_i * large_stride4 + dim5_i] =
                                    cur_large_ptr[dim4_i * large_stride4 + dim5_i]
                                    - cur_small_ptr[dim4_i * small_stride4 + dim5_i * small_stride5];
                        }
                    }
                }
            }
        }
    }

    return 0;
}

