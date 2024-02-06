#ifndef __NN_COMMON_CPP_H__
#define __NN_COMMON_CPP_H__

#include "nn_common.h"

#include <unordered_map>
static std::unordered_map<std::string, int32_t> cfg_size_map = {
    {"io", sizeof(IO_CONFIG_S)}, {"Relu", sizeof(RELU_CONFIG_S)},
    {"MaxPool", sizeof(MAX_POOL_CONFIG_S)}, {"Conv", sizeof(CONV_CONFIG_S)},
    {"Add", sizeof(ADD_CONFIG_S)}, {"GlobalAveragePool", sizeof(GLOBAL_AVGPOOL_CONFIG_S)},
    {"Flatten", sizeof(FLATTEN_CONFIG_S)}, {"Gemm", sizeof(GEMM_CONFIG_S)},
};


#endif // __NN_COMMON_CPP_H__
