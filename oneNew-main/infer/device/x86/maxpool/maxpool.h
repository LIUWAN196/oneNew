#ifndef __RELU6_H__
#define __RELU6_H__

#include "../../../common/nn_common.h"


typedef struct {
    TENSOR_S input;
    TENSOR_S output;

    int8_t kernel_h;
    int8_t kernel_w;
    int8_t stride_x;
    int8_t stride_y;

    float threshold;
} MAXPOOL_CONFIG_S;

int eval(BUFFER_GROUP_S *, BUFFER_GROUP_S *, BUFFER_GROUP_S *);

#endif