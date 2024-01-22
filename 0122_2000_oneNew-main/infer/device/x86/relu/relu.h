#ifndef __RELU6_H__
#define __RELU6_H__

#include "../../../common/nn_common.h"


// typedef struct {
//     OPERAND_S input;
//     OPERAND_S output;

//     float threshold;
// } RELU6_CONFIG_S;

int eval(BUFFER_GROUP_S *, BUFFER_GROUP_S *, BUFFER_GROUP_S *);

#endif