#include "maxpool.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval(BUFFER_GROUP_S *pramas, BUFFER_GROUP_S *inputs, BUFFER_GROUP_S *outputs) {
    int32_t buf_size = 12;
    int i;
    printf("\n start of device\n");

    RELU6_CONFIG_S *relu6_cfg = (RELU6_CONFIG_S *)pramas->buf_info[0].addr;

    BUFFER_INFO_S in = inputs->buf_info[0];
    BUFFER_INFO_S out = outputs->buf_info[0];

    for (int j = 0; j < 100; ++j) {
        int8_t* inptr = (int8_t*)in.addr;
        int8_t* outptr = (int8_t*)out.addr;
        if(inptr[j] > relu6_cfg->threshold){

            outptr[j] = inptr[j];
            printf("j is %d, the in is %d, the out is %d\n", j, inptr[j], outptr[j]);
        } else {
            outptr[j] = 0;
            printf("j is %d, the in is %d, the out is %d\n", j, inptr[j], outptr[j]);
        }
    }

    printf("\n end of device\n");

    return 0;
}