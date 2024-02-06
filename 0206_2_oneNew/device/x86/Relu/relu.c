#include "relu.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs)
{
    RELU_CONFIG_S* cfg = (RELU_CONFIG_S*)(params[0].addr);
//    printf("yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    float* input_ptr = (float*)(inputs[0].addr);
    float* output_ptr = (float*)(outputs->addr);

    OPERAND_S* in_tensor = (OPERAND_S*)(params[1].addr);
    int32_t in_elem_size = in_tensor->shape.N * in_tensor->shape.C * in_tensor->shape.H * in_tensor->shape.W;

//    for (int i = 0; i < in_elem_size; ++i) {
//        input_ptr[i] = i / 45 - 56 + i % 32 + i;
//    }
//
//
    for (int i = 0; i < in_elem_size; ++i) {
        output_ptr[i] = (input_ptr[i] > 0.f) ? input_ptr[i] : 0.f;
    }


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


//    for (int j = 0; j < 1000000; ++j) {
//        for (int i = 0; i < 5; ++i) {
//            output_ptr[i] = input_ptr[i] + 100;
////        printf("the i is %d , input_ptr[i] is %d, output_ptr[i] is %d\n", i, input_ptr[i], output_ptr[i]);
//        }
//    }


//    printf("\n start of device\n");
//
//    RELU_CONFIG_S *relu_cfg = (RELU_CONFIG_S *)params->buf_info[0].addr;
//    OPERAND_S *in_operand = (OPERAND_S *)params->buf_info[1].addr;
//    OPERAND_S *out_operand = (OPERAND_S *)params->buf_info[2].addr;
//
//    int8_t *inptr = (int8_t *)inputs->buf_info[0].addr;
//    int8_t *outptr = (int8_t *)outputs->buf_info[0].addr;
//
//    for (int j = 0; j < 100; ++j)
//    {
//        if (inptr[j] > 0)
//        {
//            outptr[j] = inptr[j];
//            printf("j is %d, the in is %d, the out is %d\n", j, inptr[j], outptr[j]);
//        }
//        else
//        {
//            outptr[j] = 0;
//            printf("j is %d, the in is %d, the out is %d\n", j, inptr[j], outptr[j]);
//        }
//    }
//
//    printf("\n end of device\n");

    return 0;
}