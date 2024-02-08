#include "softmax.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "math.h"
#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    SOFTMAX_CONFIG_S *cfg = (SOFTMAX_CONFIG_S *) (params[0].addr);
//    printf("yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

    if (cfg->axis != -1 && cfg->axis != in0_tensor->dim_num_of_shapes - 1) {
        printf("error, cur only support axis = -1 or in_tensor->dim_num_of_shapes - 1\n");
    }

    int32_t dim_num = in0_tensor->dim_num_of_shapes;

    int32_t out_loop = 1;
    for (int i = 0; i < dim_num - 1; ++i) {
        out_loop *= in0_tensor->shapes[i];
    }

    int32_t in_loop = in0_tensor->shapes[dim_num - 1];
    int32_t inner_elem_size = in_loop;

    for (int out_loopi = 0; out_loopi < out_loop; ++out_loopi) {
        // step 1: find max
        float max_elem = -32768.0f;
        for (int i = 0; i < inner_elem_size; ++i) {
            max_elem = (input_ptr[i] > max_elem) ? input_ptr[i] : max_elem;
        }

        // step 2: calc exp(x - max(x))
        float total_exp = 0;
        for (int i = 0; i < inner_elem_size; ++i) {
            output_ptr[i] = expf(input_ptr[i] - max_elem);
            total_exp += output_ptr[i];
        }

        // step 3: calc exp(x) / sum(exp(x))
        float total_exp_inv = 1.0f / total_exp;
        for (int i = 0; i < inner_elem_size; ++i) {
            output_ptr[i] = output_ptr[i] * total_exp_inv;
        }

        input_ptr += inner_elem_size;
        output_ptr += inner_elem_size;
    }



    return 0;
}