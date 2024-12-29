#include "hard_sigmoid.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "math.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    HARD_SIGMOID_CONFIG_S *cfg = (HARD_SIGMOID_CONFIG_S *) (params[0].addr);

    float *input0_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

    int32_t in_elem_size = 1;
    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
        in_elem_size *= in0_tensor->shapes[dim_i];
    }

    float tmp;
    float alpha = cfg->alpha;
    float beta = cfg->beta;
    // follow min and max value is define of hard sigmoid, you can see https://onnx.ai/onnx/operators/onnx__HardSigmoid.html
    float min = 0, max = 1;
    for (int i = 0; i < in_elem_size; ++i) {
        tmp = alpha * input0_ptr[i] + beta;
        tmp = (tmp > min) ? tmp : min;
        output_ptr[i] = (tmp < max) ? tmp : max;
    }

    return 0;
}