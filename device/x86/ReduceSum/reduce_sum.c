#include "reduce_sum.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

//    show_dev_input(params);
    REDUCE_SUM_CONFIG_S *cfg = (REDUCE_SUM_CONFIG_S *) (params[0].addr);

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);
    int32_t in_elem_size = 1;
    for (int dim_i = 0; dim_i < in_tensor->dim_num_of_shapes; ++dim_i) {
        in_elem_size *= in_tensor->shapes[dim_i];
    }

    int32_t out_elem_size = 1;
    for (int dim_i = 0; dim_i < out_tensor->dim_num_of_shapes; ++dim_i) {
        out_elem_size *= out_tensor->shapes[dim_i];
    }
    int32_t reduce_size = in_elem_size / out_elem_size;

    for (int out_i = 0; out_i < out_elem_size; ++out_i) {
        float psum = 0;
        for (int reduce_i = 0; reduce_i < reduce_size; ++reduce_i) {
            psum += input_ptr[out_i * reduce_size + reduce_i];
//        LOG_DBG("in_i is %d, input_ptr[in_i] is %f, psum is %f", in_i, input_ptr[in_i], psum);
        }
        output_ptr[out_i] = psum;
    }

//    float psum = 0;
//    for (int in_i = 0; in_i < in_elem_size; ++in_i) {
//        psum += input_ptr[in_i];
////        LOG_DBG("in_i is %d, input_ptr[in_i] is %f, psum is %f", in_i, input_ptr[in_i], psum);
//    }
//    output_ptr[0] = psum;

    return 0;
}