#include "reduce_max.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    REDUCE_MAX_CONFIG_S *cfg = (REDUCE_MAX_CONFIG_S *) (params[0].addr);

    if (cfg->axes_num != 1 || cfg->axes[0] != -1) {
        LOG_ERR("cur, reduce max op just support axes == -1");
    }

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

    int32_t outer_elem_size = 1;
    for (int dim_i = 0; dim_i < in_tensor->dim_num_of_shapes - 1; ++dim_i) {
        outer_elem_size *= in_tensor->shapes[dim_i];
    }
    int32_t inner_elem_size = in_tensor->shapes[in_tensor->dim_num_of_shapes - 1];

    for (int outer_i = 0; outer_i < outer_elem_size; ++outer_i) {
        float *cur_ifmap_ptr = input_ptr + outer_i * inner_elem_size;
        float *cur_ofmap_ptr = output_ptr;
        float max_val = -32768.0f;
        for (int inner_i = 0; inner_i < inner_elem_size; ++inner_i) {
            max_val = cur_ifmap_ptr[inner_i] > max_val ? cur_ifmap_ptr[inner_i] : max_val;
        }
        cur_ofmap_ptr[outer_i] = max_val;
    }

    return 0;
}