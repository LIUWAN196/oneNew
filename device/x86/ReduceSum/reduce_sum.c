#include "reduce_sum.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    REDUCE_SUM_CONFIG_S *cfg = (REDUCE_SUM_CONFIG_S *) (params[0].addr);

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    int32_t in_elem_size = 1;
    for (int dim_i = 0; dim_i < in_tensor->dim_num_of_shapes; ++dim_i) {
        in_elem_size *= in_tensor->shapes[dim_i];
    }

    float psum = 0;
    for (int in_i = 0; in_i < in_elem_size; ++in_i) {
        psum += input_ptr[in_i];
//        LOG_DBG("in_i is %d, input_ptr[in_i] is %f, psum is %f", in_i, input_ptr[in_i], psum);
    }
    output_ptr[0] = psum;

    return 0;
}