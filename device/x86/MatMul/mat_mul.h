#ifndef __GEMM_H__
#define __GEMM_H__

#include "../../../common/nn_common.h"


// typedef struct {
//     OPERAND_S input;
//     OPERAND_S output;

//     float threshold;
// } RELU6_CONFIG_S;

int eval(BUFFER_INFO_S *, BUFFER_INFO_S *, BUFFER_INFO_S *);

#endif