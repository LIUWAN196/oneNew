#include "reshape.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"
#include "string.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs)
{
//    LOG_DBG("this is reshape op");
//    fflush(stdout);
//        show_dev_input(params);
    RESHAPE_CONFIG_S * cfg = (RESHAPE_CONFIG_S*)(params[0].addr);
//    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

//    if (strcmp(cfg->op_base_cfg.op_name, "/model.28/decoder/layers.4/cross_attn/Reshape_7") == 0) {
//        show_dev_input(params);
//    }
    float* input_ptr = (float*)(inputs[0].addr);
    float* output_ptr = (float*)(outputs[0].addr);
    int32_t * aa = (int32_t*)(inputs[0].addr);
//    for (int i = 0; i < 10; ++i) {
//        int32_t * aa = (int32_t*)(inputs[0].addr);
//        LOG_DBG("%d", aa[i]);
//    }

    OPERAND_S* in_tensor = (OPERAND_S*)(params[1].addr);
    int32_t in_elem_size = 1;
    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
        in_elem_size *= in_tensor->shapes[dim_i];
    }

    for (int i = 0; i < in_elem_size; ++i) {
        output_ptr[i] = input_ptr[i];
    }
//    LOG_DBG("end of this reshape, the op name is %s", cfg->op_base_cfg.op_name);

    return 0;
}