//
// Created by wanzai on 24-2-8.
//

#ifndef ONENEW_PAD_MAXPOOL_H
#define ONENEW_PAD_MAXPOOL_H

#include "../../../common/nn_common.h"
//#include "memory.h"

int do_pad_maxpool(char *dst_ptr, char *src_ptr, OPERAND_S *src_data_desc, PAD_INNER_CONFIG_S *cfg) {
    float pad_value = -32768000;
    float *dst_f32 = (float *) dst_ptr;
    float *src_f32 = (float *) src_ptr;

    int32_t src_c = src_data_desc->shapes[1];
    int32_t src_h = src_data_desc->shapes[2];
    int32_t src_w = src_data_desc->shapes[3];

    int32_t dst_c = src_data_desc->shapes[1];
    int32_t dst_h = src_h + 2 * cfg->h;
    int32_t dst_w = src_w + 2 * cfg->w;
    for (int i = 0; i < dst_c * dst_h * dst_w; ++i) {
        dst_f32[i] = pad_value;
    }
//    memset(dst_ptr, dst_c * dst_h * dst_w * sizeof(float), pad_value);

    for (int c_i = 0; c_i < dst_c; ++c_i) {
        for (int h_i = cfg->h; h_i < dst_h - cfg->h; ++h_i) {
            for (int w_i = cfg->w; w_i < dst_w - cfg->w; ++w_i) {
                dst_f32[c_i * dst_h * dst_w + h_i * dst_w + w_i] = src_f32[c_i * src_h * src_w +
                                                                           (h_i - cfg->h) * src_w + (w_i - cfg->w)];
            }

        }
    }


    return 0;
}


#endif //ONENEW_PAD_MAXPOOL_H
