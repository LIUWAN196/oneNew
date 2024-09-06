#include "grid_sample.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
//    show_dev_input(params);
//    printf("this is x86 mul start\n");
    GRID_SAMPLE_CONFIG_S *cfg = (GRID_SAMPLE_CONFIG_S *) (params[0].addr);
//    printf("this is device, the op type is mul\n");

    float *input0_ptr = (float *) (inputs[0].addr);
    float *input1_ptr = (float *) (inputs[1].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *in1_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

    int32_t in0_elem_size = operand_elem_size(in0_tensor);
    int32_t in1_elem_size = operand_elem_size(in1_tensor);

    LOG_ERR("对不起，现在还没实现 grid sample 的计算过程");
    return 0;
}