#include "global_avgpool.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>


int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    GLOBAL_AVGPOOL_CONFIG_S *cfg = (GLOBAL_AVGPOOL_CONFIG_S *) (params[0].addr);

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

    int32_t in_n = in_tensor->shapes[0];
    int32_t in_c = in_tensor->shapes[1];
    int32_t in_h = in_tensor->shapes[2];
    int32_t in_w = in_tensor->shapes[3];

    int32_t out_n = out_tensor->shapes[0];
    int32_t out_c = out_tensor->shapes[1];
    int32_t out_h = out_tensor->shapes[2];
    int32_t out_w = out_tensor->shapes[3];

    float coeff = 1.0f / (in_h * in_w);
    // loop params
    float *cur_input_ptr;
    float *cur_output_ptr;
    for (int n_i = 0; n_i < in_n; ++n_i) {
        for (int c_i = 0; c_i < in_c; ++c_i) {
            cur_input_ptr = input_ptr + n_i * in_c * in_h * in_w + c_i * in_h * in_w;
            cur_output_ptr = output_ptr + n_i * in_c + c_i;
            float psum = 0;
            for (int h_i = 0; h_i < in_h; ++h_i) {
                for (int w_i = 0; w_i < in_w; ++w_i) {
                    psum += cur_input_ptr[h_i * in_w + w_i];
                }
            }
            cur_output_ptr[0] = psum * coeff;
        }
    }

    return 0;
}