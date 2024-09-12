#include "pad.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
//#include "pad_avgpool.h"
#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

//    show_dev_input(params);
    PAD_CONFIG_S *cfg = (PAD_CONFIG_S *) (params[0].addr);
//    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    int pad_c_st = cfg->pads[1];
    int pad_c_ed = cfg->pads[4 + 1];

    int pad_top = cfg->pads[2];
    int pad_bottom = cfg->pads[4 + 2];
    int pad_left = cfg->pads[3];
    int pad_right = cfg->pads[4 + 3];

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

    int32_t in_n = in_tensor->shapes[0];
    int32_t in_c = in_tensor->shapes[1];
    int32_t in_h = in_tensor->shapes[2];
    int32_t in_w = in_tensor->shapes[3];

    float pad_value = 0.0f;
    float *dst_f32 = (float *) output_ptr;
    float *src_f32 = (float *) input_ptr;

    int32_t src_c = in_c;
    int32_t src_h = in_h;
    int32_t src_w = in_w;

    int32_t dst_c = src_c + pad_c_st + pad_c_ed;
    int32_t dst_h = src_h + pad_top + pad_bottom;
    int32_t dst_w = src_w + pad_left + pad_right;
    for (int i = 0; i < dst_c * dst_h * dst_w; ++i) {
        dst_f32[i] = pad_value;
    }

//    LOG_DBG("%d %d %d %d %d %d", pad_c_st, pad_c_ed, pad_top, pad_bottom, pad_left, pad_right);
//    LOG_DBG("dst_c %d  dst_h %d dst_w %d", dst_c, dst_h, dst_w);

    for (int c_i = pad_c_st; c_i < dst_c - pad_c_ed; ++c_i) {
        for (int h_i = pad_top; h_i < dst_h - pad_bottom; ++h_i) {
            for (int w_i = pad_left; w_i < dst_w - pad_right; ++w_i) {
                dst_f32[c_i * dst_h * dst_w + h_i * dst_w + w_i]
                = src_f32[(c_i - pad_c_st) * src_h * src_w + (h_i - pad_top) * src_w + (w_i - pad_left)];
            }

        }
    }

    return 0;
}