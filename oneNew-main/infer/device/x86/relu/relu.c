#include "relu.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval(BUFFER_GROUP_S *params, BUFFER_GROUP_S *inputs, BUFFER_GROUP_S *outputs)
{
    printf("\n start of device\n");

    RELU_CONFIG_S *relu_cfg = (RELU_CONFIG_S *)params->buf_info[0].addr;
    OPERAND_S *in_operand = (OPERAND_S *)params->buf_info[1].addr;
    OPERAND_S *out_operand = (OPERAND_S *)params->buf_info[2].addr;

    int8_t *inptr = (int8_t *)inputs->buf_info[0].addr;
    int8_t *outptr = (int8_t *)outputs->buf_info[0].addr;

    for (int j = 0; j < 100; ++j)
    {
        if (inptr[j] > 0)
        {
            outptr[j] = inptr[j];
            printf("j is %d, the in is %d, the out is %d\n", j, inptr[j], outptr[j]);
        }
        else
        {
            outptr[j] = 0;
            printf("j is %d, the in is %d, the out is %d\n", j, inptr[j], outptr[j]);
        }
    }

    printf("\n end of device\n");

    return 0;
}