#include "pad.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
//#include "pad_avgpool.h"
#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    PAD_CONFIG_S *cfg = (PAD_CONFIG_S *) (params[0].addr);
//    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    int pad_h = cfg->pads[2];
    int pad_w = cfg->pads[3];

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *in1_tensor = (OPERAND_S *) (params[2].addr); // is the pad shape, don't be used
    OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

    int32_t in_n = in_tensor->shapes[0];
    int32_t in_c = in_tensor->shapes[1];
    int32_t in_h = in_tensor->shapes[2];
    int32_t in_w = in_tensor->shapes[3];

    int32_t out_n = out_tensor->shapes[0];
    int32_t out_c = out_tensor->shapes[1];
    int32_t out_h = out_tensor->shapes[2];
    int32_t out_w = out_tensor->shapes[3];

    float pad_value = 0.0f;
    float *dst_f32 = (float *) output_ptr;
    float *src_f32 = (float *) input_ptr;

    int32_t src_c = in_c;
    int32_t src_h = in_h;
    int32_t src_w = in_w;

    int32_t dst_c = out_c;
    int32_t dst_h = src_h + 2 * pad_h;
    int32_t dst_w = src_w + 2 * pad_w;
    for (int i = 0; i < dst_c * dst_h * dst_w; ++i) {
        dst_f32[i] = pad_value;
    }

    for (int c_i = 0; c_i < dst_c; ++c_i) {
        for (int h_i = pad_h; h_i < dst_h - pad_h; ++h_i) {
            for (int w_i = pad_w; w_i < dst_w - pad_w; ++w_i) {
                dst_f32[c_i * dst_h * dst_w + h_i * dst_w + w_i] 
                = src_f32[c_i * src_h * src_w + (h_i - pad_h) * src_w + (w_i - pad_w)];
            }

        }
    }

    return 0;
}