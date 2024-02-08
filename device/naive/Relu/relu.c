#include "relu.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs)
{
    RELU_CONFIG_S* cfg = (RELU_CONFIG_S*)(params[0].addr);
//    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    float* input_ptr = (float*)(inputs[0].addr);
    float* output_ptr = (float*)(outputs[0].addr);

    OPERAND_S* in_tensor = (OPERAND_S*)(params[1].addr);
    int32_t in_elem_size = in_tensor->shape.N * in_tensor->shape.C * in_tensor->shape.H * in_tensor->shape.W;

//    int32_t in_elem_size = operand_elem_size(in_tensor);

//    for (int i = 0; i < in_elem_size; ++i) {
//        input_ptr[i] = i / 45 - 56 + i % 32 + i;
//    }


    for (int i = 0; i < in_elem_size; ++i) {
        output_ptr[i] = (input_ptr[i] > 0.f) ? input_ptr[i] : 0.f;
    }

//    // write_bin(replace_char(cfg->out_operand_name[0]), in_elem_size * sizeof(float), output_ptr);


//    for (int i = 0; i < in_elem_size; ++i) {
//        if (i % 32 == 0){
//            printf("\n");
//        }
//        printf("%f  ", input_ptr[i]);
//    }
//
//    printf("===========================\n");
//    for (int i = 0; i < in_elem_size; ++i) {
//        if (i % 32 == 0){
//            printf("\n");
//        }
//        printf("%f  ", output_ptr[i]);
//    }

    return 0;
}