#include "resize.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include "math.h"

int eval_nearest(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);
int eval_linear(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs)
{
    OPERAND_S* in_tensor = (OPERAND_S*)(params[1].addr);
    OPERAND_S* out_tensor = (OPERAND_S *) (params[2].addr);
    if (out_tensor->shapes[2] / in_tensor->shapes[2] == 2) {
        eval_nearest(params, inputs, outputs);
    } else {
        eval_linear(params, inputs, outputs);
    }
    return 0;
}

int eval_nearest(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs)
{
    RESIZE_CONFIG_S* cfg = (RESIZE_CONFIG_S*)(params[0].addr);

    float* input_ptr = (float*)(inputs[0].addr);
    float* output_ptr = (float*)(outputs[0].addr);

    OPERAND_S* in_tensor = (OPERAND_S*)(params[1].addr);
    OPERAND_S* out_tensor = (OPERAND_S *) (params[2].addr);

    float h_scale;
    float w_scale;
    if (cfg->scales[0] == 0) {
        h_scale = out_tensor->shapes[2] * 1.0f / in_tensor->shapes[2];
        w_scale = out_tensor->shapes[3] * 1.0f / in_tensor->shapes[3];
    } else {
        h_scale = cfg->scales[2];
        w_scale = cfg->scales[3];
    }

    int32_t in_elem_size = 1;
    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
        in_elem_size *= in_tensor->shapes[dim_i];
    }

    int32_t in_n = in_tensor->shapes[0];
    int32_t in_c = in_tensor->shapes[1];
    int32_t in_h = in_tensor->shapes[2];
    int32_t in_w = in_tensor->shapes[3];

    int32_t out_n = out_tensor->shapes[0];
    int32_t out_c = out_tensor->shapes[1];
    int32_t out_h = out_tensor->shapes[2];
    int32_t out_w = out_tensor->shapes[3];

    float* cur_in_ptr;
    float* cur_out_ptr;
    for (int nc_i = 0; nc_i < out_n * out_c; nc_i++)
    {
        cur_in_ptr = input_ptr + nc_i * in_h * in_w;
        cur_out_ptr = output_ptr + nc_i * out_h * out_w;
        for (int outh_i = 0; outh_i < out_h; outh_i++)
        {
            for (int outw_i = 0; outw_i < out_w; outw_i++)
            {
                int inh_i = (int)(outh_i / h_scale + 0.0);
                int inw_i = (int)(outw_i / w_scale + 0.0);
                cur_out_ptr[outh_i * out_w + outw_i] = cur_in_ptr[inh_i * in_w + inw_i];
            }
        }
    }

    return 0;
}

int eval_linear(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs)
{
    RESIZE_CONFIG_S* cfg = (RESIZE_CONFIG_S*)(params[0].addr);

    float* input_ptr = (float*)(inputs[0].addr);
    float* output_ptr = (float*)(outputs[0].addr);

    OPERAND_S* in_tensor = (OPERAND_S*)(params[1].addr);
    OPERAND_S* out_tensor = (OPERAND_S *) (params[2].addr);

    int32_t in_n = in_tensor->shapes[0];
    int32_t in_c = in_tensor->shapes[1];
    int32_t in_h = in_tensor->shapes[2];
    int32_t in_w = in_tensor->shapes[3];

    int32_t out_n = out_tensor->shapes[0];
    int32_t out_c = out_tensor->shapes[1];
    int32_t out_h = out_tensor->shapes[2];
    int32_t out_w = out_tensor->shapes[3];

    float scale_h = (in_h * 1.0f) / (out_h * 1.0f);
    float scale_w = (in_w * 1.0f) / (out_w * 1.0f);

    float* cur_in_ptr;
    float* cur_out_ptr;
    for (int nc_i = 0; nc_i < out_n * out_c; nc_i++)
    {
        cur_in_ptr = input_ptr + nc_i * in_h * in_w;
        cur_out_ptr = output_ptr + nc_i * out_h * out_w;
        for (int outh_i = 0; outh_i < out_h; outh_i++)
        {
            for (int outw_i = 0; outw_i < out_w; outw_i++)
            {
                float cur_src_h = (outh_i + 0.5) * scale_h - 0.5;
                float cur_src_w = (outw_i + 0.5) * scale_w - 0.5;

                int cur_src_h_0 = (int)floorf(cur_src_h) > 0 ? (int)floorf(cur_src_h) : 0;
                int cur_src_w_0 = (int)floorf(cur_src_w) > 0 ? (int)floorf(cur_src_w) : 0;
                int cur_src_h_1 = (int)ceilf(cur_src_h) > (in_h - 1) ? in_h - 1 : (int)ceilf(cur_src_h);
                int cur_src_w_1 = (int)ceilf(cur_src_w) > (in_w - 1) ? in_w - 1 : (int)ceilf(cur_src_w);

                float w_val_a0 = cur_in_ptr[cur_src_h_0 * in_w + cur_src_w_0];
                float w_val_a1 = cur_in_ptr[cur_src_h_0 * in_w + cur_src_w_1];

                float w_val_b0 = cur_in_ptr[cur_src_h_1 * in_w + cur_src_w_0];
                float w_val_b1 = cur_in_ptr[cur_src_h_1 * in_w + cur_src_w_1];

                float val_0, val_1;
                if (cur_src_w_1 == cur_src_w_0) {
                    val_0 = w_val_a0;
                    val_1 = w_val_b0;
                } else {
                    val_0 = (cur_src_w_1 - cur_src_w) * w_val_a0 + (cur_src_w - cur_src_w_0) * w_val_a1;
                    val_1 = (cur_src_w_1 - cur_src_w) * w_val_b0 + (cur_src_w - cur_src_w_0) * w_val_b1;
                }

                float val;
                if (cur_src_h_1 == cur_src_h_0) {
                    val = val_0;
                } else {
                    val = (cur_src_h_1 - cur_src_h) * val_0 + (cur_src_h - cur_src_h_0) * val_1;
                }

                cur_out_ptr[outh_i * out_w + outw_i] = val;

            }
        }
    }

    return 0;
}


