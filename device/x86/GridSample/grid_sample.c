#include "grid_sample.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "math.h"
#include "stdint.h"

#include "float.h"
// 按照 https://blog.csdn.net/weixin_45377629/article/details/132218227 这个逻辑来实现
// 有一点需要注意，上面这个链接是 align_corners 为 true 的写法，我也是按照上面这个链接做的，但是 rt detr 是 align_corners 为 false 的
// 做法，所以有一些误差，具体可见 https://blog.csdn.net/weixin_42973210/article/details/132315563
int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
//    show_dev_input(params);
//    printf("this is x86 mul start\n");
    GRID_SAMPLE_CONFIG_S *cfg = (GRID_SAMPLE_CONFIG_S *) (params[0].addr);
//    printf("this is device, the op type is mul\n");

    float *input0_ptr = (float *) (inputs[0].addr);
    float *input1_ptr = (float *) (inputs[1].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *in1_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

    int32_t in0_elem_size = operand_elem_size(in0_tensor);
    int32_t in1_elem_size = operand_elem_size(in1_tensor);

    int32_t n = in0_tensor->shapes[0];
    int32_t c = in0_tensor->shapes[1];
    int32_t in_h = in0_tensor->shapes[2];
    int32_t in_w = in0_tensor->shapes[3];

    int32_t out_h = in1_tensor->shapes[1];
    int32_t out_w = in1_tensor->shapes[2];

    for (int n_i = 0; n_i < n; ++n_i) {
        for (int c_i = 0; c_i < c; ++c_i) {
            for (int outh_i = 0; outh_i < out_h; ++outh_i) {
                for (int outw_i = 0; outw_i < out_w; ++outw_i) {
                    float args[2] = {0.0f, 0.0f};
                    args[0] = (in_w - 1) * (input1_ptr[(n_i * out_h * out_w + outh_i * out_w + outw_i) * 2 + 0] + 1) / 2;
                    args[1] = (in_h - 1) * (input1_ptr[(n_i * out_h * out_w + outh_i * out_w + outw_i) * 2 + 1] + 1) / 2;
                    int x1 = (int) (args[0] + 1);
                    int x0 = x1 - 1;
                    int y1 = (int) (args[1] + 1);
                    int y0 = y1 - 1;
                    args[0] = fabs(args[0] - x0);
                    args[1] = fabs(args[1] - y0);

                    float left_top_pad = 0.0f, right_top_pad = 0.0f, left_bottom_pad = 0.0f, right_bottom_pad = 0.0f;
                    if (x0 >= 0.0f && x0 < in_w && y0 >= 0.0f && y0 < in_h) {
                        left_top_pad = input0_ptr[n_i * c * in_h * in_w + c_i * in_h * in_w + y0 * in_w + x0];
                    }
                    if (x1 >= 0.0f && x1 < in_w && y0 >= 0.0f && y0 < in_h) {
                        right_top_pad = input0_ptr[n_i * c * in_h * in_w + c_i * in_h * in_w + y0 * in_w + x1];
                    }
                    if (x0 >= 0.0f && x0 < in_w && y1 >= 0.0f && y1 < in_h) {
                        left_bottom_pad = input0_ptr[n_i * c * in_h * in_w + c_i * in_h * in_w + y1 * in_w + x0];
                    }
                    if (x1 >= 0.0f && x1 < in_w && y1 >= 0.0f && y1 < in_h) {
                        right_bottom_pad = input0_ptr[n_i * c * in_h * in_w + c_i * in_h * in_w + y1 * in_w + x1];
                    }

                    // 双线性插值
                    float left_top = left_top_pad * (1 - args[0]) * (1 - args[1]);
                    float left_bottom = left_bottom_pad * (1 - args[0]) * args[1];
                    float right_top = right_top_pad * args[0] * (1 - args[1]);
                    float right_bottom = right_bottom_pad * args[0] * args[1];
                    float res = left_top + left_bottom + right_top + right_bottom;
                    output_ptr[n_i * c * out_h * out_w + c_i * out_h * out_w + outh_i * out_w + outw_i] = res;
                }
            }
        }
    }

//    LOG_ERR("对不起，现在还没实现 grid sample 的计算过程");
    return 0;
}