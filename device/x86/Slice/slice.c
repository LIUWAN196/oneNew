#include "slice.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "string.h"

int eval_equal5dims(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);
int eval_less5dims(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    SLICE_CONFIG_S *cfg = (SLICE_CONFIG_S *) (params[0].addr);
    if (cfg->axes[0] == 5) {
        eval_equal5dims(params, inputs, outputs);
    } else {
        eval_less5dims(params, inputs, outputs);
    }

    return 0;
}

int eval_less5dims(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    SLICE_CONFIG_S *cfg = (SLICE_CONFIG_S *) (params[0].addr);

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

    int32_t st1 = 0, step1 = 1;
    int32_t st2 = 0, step2 = 1;

    if (cfg->axes[0] == 1) {
        int32_t starts = cfg->starts[0];
        if (starts < 0){
            starts = in_tensor->shapes[1] + starts;
        }
        st1 = starts;
        step1 = cfg->steps[0];
    } else {
        st1 = 0;
        step1 = 1;
        if (cfg->axes[0] == 2) {
            int32_t starts = cfg->starts[0];
            if (starts < 0){
                starts = in_tensor->shapes[1] + starts;
            }
            st2 = starts;
            step2 = cfg->steps[0];
        }
    }

    if (cfg->slice_axes_num == 2 && cfg->axes[1] == 2) {
        int32_t starts = cfg->starts[1];
        if (starts < 0){
            starts = in_tensor->shapes[2] + starts;
        }
        st2 = starts;
        step2 = cfg->steps[1];
    }

    int32_t inner_elem_size = 1;
    for (int i = 3; i < SHAPE_LEN; ++i) {
        inner_elem_size *= in_tensor->shapes[i];
    }

    for (int out0_i = 0; out0_i < out_tensor->shapes[0]; ++out0_i) {
        for (int out1_i = 0; out1_i < out_tensor->shapes[1]; ++out1_i) {
            for (int out2_i = 0; out2_i < out_tensor->shapes[2]; ++out2_i) {
                float *cur_ofmap_ptr = output_ptr + out0_i * out_tensor->shapes[1] * out_tensor->shapes[2] * inner_elem_size + out1_i * out_tensor->shapes[2] * inner_elem_size + out2_i * inner_elem_size;
                float *cur_ifmap_ptr = input_ptr + out0_i * in_tensor->shapes[1] * in_tensor->shapes[2] * inner_elem_size + (out1_i + st1) * step1 * in_tensor->shapes[2] * inner_elem_size
                                       + (out2_i + st2) * step2 * inner_elem_size;
                memcpy(cur_ofmap_ptr, cur_ifmap_ptr, inner_elem_size * sizeof(float));

            }
        }
    }

    return 0;
}

int eval_equal5dims(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    SLICE_CONFIG_S *cfg = (SLICE_CONFIG_S *) (params[0].addr);

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

    int32_t in_elem_size = 1;
    for (int i = 0; i < SHAPE_LEN; ++i) {
        in_elem_size *= in_tensor->shapes[i];
    }

    if (cfg->starts[0] == 0) {
        for (int i = 0; i < in_elem_size / 4; ++i) {
            output_ptr[2 * i + 0] = input_ptr[4 * i + 0];
            output_ptr[2 * i + 1] = input_ptr[4 * i + 1];
        }
    } else {
        for (int i = 0; i < in_elem_size / 4; ++i) {
            output_ptr[2 * i + 0] = input_ptr[4 * i + 2];
            output_ptr[2 * i + 1] = input_ptr[4 * i + 3];
        }
    }

    return 0;
}


