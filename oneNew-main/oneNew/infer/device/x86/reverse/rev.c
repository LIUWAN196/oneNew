#include "rev.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

int reverse(const char* in_buf, const char* out_buf){
    int32_t buf_size = 12;
    int i;
    printf("\n start of device\n");

    int8_t* in_i8 = (int8_t*)in_buf;
    int8_t* out_i8 = (int8_t*)out_buf;

    for (i = 0; i < buf_size; ++i) {
        out_i8[i] = in_i8[buf_size - 1 - i];
    }
    printf("\n end of device\n");

    return 0;
}




