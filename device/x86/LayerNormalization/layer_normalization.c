#include "layer_normalization.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include "stdint.h"
#include <omp.h>
#include "math.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs)
{
//    show_dev_input(params);

    LAYERNORMALIZATION_CONFIG_S * cfg = (LAYERNORMALIZATION_CONFIG_S*)(params[0].addr);

    float* input_ptr = (float*)(inputs[0].addr);
    float* weight_ptr = (float*)(inputs[1].addr);
    float* bias_ptr = (float*)(inputs[2].addr);
    float* output_ptr = (float*)(outputs[0].addr);

    OPERAND_S* in_tensor = (OPERAND_S*)(params[1].addr);
    int32_t elem_size = 1;
    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
        elem_size *= in_tensor->shapes[dim_i];
    }

    if (cfg->axes[0] == 1) {
        // 这里实际上不是正规的 layer norm 的计算，但是在 mobile sam encoder 这个模型结尾，有 axis = 1 的计算，除了是在 axis = 1 上做计算，其余和 layer norm 一模一样，所以把它也融合为了 layer norm
        int32_t reduce_size = in_tensor->shapes[1];
        int32_t loop_size = elem_size / reduce_size;

        float inv_reduce_size = 1.0f / reduce_size;
        for (int loop_i = 0; loop_i < loop_size; ++loop_i) {
            // setp 1: cale psum_val and psum_square_val
            float *cur_ifmap = input_ptr + loop_i;
            float *cur_ofmap = output_ptr + loop_i;
            float psum_val = 0;
            float psum_square_val = 0;
            for (int reduce_i = 0; reduce_i < reduce_size; ++reduce_i) {
                psum_val += cur_ifmap[reduce_i * loop_size];
                psum_square_val += cur_ifmap[reduce_i * loop_size] * cur_ifmap[reduce_i * loop_size];
            }

            // setp 2: calc mean and val
            float mean_val = psum_val * inv_reduce_size;
            float var_val = inv_reduce_size * psum_square_val - mean_val * mean_val;
            var_val += EPSILON;

            float std_val = sqrtf(var_val);
            float inv_std_val = 1.0f / std_val;

            // setp 3: save to ofmp
            float tmp;
            for (int reduce_i = 0; reduce_i < reduce_size; ++reduce_i) {
                tmp = (cur_ifmap[reduce_i * loop_size] - mean_val) * inv_std_val;
                cur_ofmap[reduce_i * loop_size] = tmp * weight_ptr[reduce_i] + bias_ptr[reduce_i];
            }
        }
    } else {
        int32_t reduce_size;
        for (int dim_i = SHAPE_LEN - 1; dim_i >= 0; --dim_i) {
            if (in_tensor->shapes[dim_i] != 1) {
                reduce_size = in_tensor->shapes[dim_i];
                break;
            }
        }

        int32_t loop_size = elem_size / reduce_size;

        float inv_reduce_size = 1.0f / reduce_size;
        for (int loop_i = 0; loop_i < loop_size; ++loop_i) {
            // setp 1: cale psum_val and psum_square_val
            float *cur_ifmap = input_ptr + loop_i * reduce_size;
            float *cur_ofmap = output_ptr + loop_i * reduce_size;
            float psum_val = 0;
            float psum_square_val = 0;
            for (int reduce_i = 0; reduce_i < reduce_size; ++reduce_i) {
                psum_val += cur_ifmap[reduce_i];
                psum_square_val += cur_ifmap[reduce_i] * cur_ifmap[reduce_i];
            }

            // setp 2: calc mean and val
            float mean_val = psum_val * inv_reduce_size;
            float var_val = inv_reduce_size * psum_square_val - mean_val * mean_val;
            var_val += EPSILON;

            float std_val = sqrtf(var_val);
            float inv_std_val = 1.0f / std_val;

            // setp 3: save to ofmp
            float tmp;
            for (int reduce_i = 0; reduce_i < reduce_size; ++reduce_i) {
                tmp = (cur_ifmap[reduce_i] - mean_val) * inv_std_val;
                cur_ofmap[reduce_i] = tmp * weight_ptr[reduce_i] + bias_ptr[reduce_i];
            }

        }
    }
    return 0;
}

