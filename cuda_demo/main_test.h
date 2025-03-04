#ifndef CUDA_DEMO_MAIN_TEST_H
#define CUDA_DEMO_MAIN_TEST_H

#include <iostream>
#include "cstdio"
#include <cuda_runtime.h>
#include <stdint.h>

int reduce_test();
int transpose_test();
int gemm_test();

#endif //CUDA_DEMO_MAIN_TEST_H
