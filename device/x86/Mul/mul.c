#include "mul.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
//    show_dev_input(params);
//    printf("this is x86 mul start\n");
    MUL_CONFIG_S *cfg = (MUL_CONFIG_S *) (params[0].addr);
//    printf("this is device, the op type is mul\n");

    float *input0_ptr = (float *) (inputs[0].addr);
    float *input1_ptr = (float *) (inputs[1].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    OPERAND_S ifmap0_tensor = *(OPERAND_S *) (params[1].addr);
    OPERAND_S ifmap1_tensor = *(OPERAND_S *) (params[2].addr);

    OPERAND_S *in0_tensor = &ifmap0_tensor;
    OPERAND_S *in1_tensor = &ifmap1_tensor;
    OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

    // 将两个输入 tensor 的 shape 维度对齐到一样
    if (in0_tensor->dim_num_of_shapes == 1) {
        int32_t dims_size = in0_tensor->shapes[0];
        in0_tensor->shapes[0] = 1;
        in0_tensor->shapes[in1_tensor->dim_num_of_shapes - 1] = dims_size;
        in0_tensor->dim_num_of_shapes = in1_tensor->dim_num_of_shapes;
    }

    if (in1_tensor->dim_num_of_shapes == 1) {
        int32_t dims_size = in1_tensor->shapes[0];
        in1_tensor->shapes[0] = 1;
        in1_tensor->shapes[in0_tensor->dim_num_of_shapes - 1] = dims_size;
        in1_tensor->dim_num_of_shapes = in0_tensor->dim_num_of_shapes;
    }

    int32_t in0_elem_size = operand_elem_size(in0_tensor);
    int32_t in1_elem_size = operand_elem_size(in1_tensor);

    if (in0_elem_size == in1_elem_size) {
        // 两个输入 tensor 一一对应，无需广播
        for (int i = 0; i < in0_elem_size; ++i) {
            output_ptr[i] = input0_ptr[i] * input1_ptr[i];
        }
    } else if (in0_elem_size == 1) {
        for (int i = 0; i < in1_elem_size; ++i) {
            output_ptr[i] = input0_ptr[0] * input1_ptr[i];
        }
    } else if (in1_elem_size == 1) {
        for (int i = 0; i < in0_elem_size; ++i) {
            output_ptr[i] = input0_ptr[i] * input1_ptr[0];
        }
    } else {
        OPERAND_S *broadcast_tensor, *another_tensor;
        float *broadcast_ptr, *another_ptr;
        if (in0_elem_size < in1_elem_size) {
            // in0 tensor 的一个或者多个维度需要做广播
            broadcast_tensor = in0_tensor;
            another_tensor = in1_tensor;
            broadcast_ptr = input0_ptr;
            another_ptr = input1_ptr;
        } else if (in1_elem_size < in0_elem_size) {
            // in1 tensor 的一个或者多个维度需要做广播
            another_tensor = in0_tensor;
            broadcast_tensor = in1_tensor;
            another_ptr = input0_ptr;
            broadcast_ptr = input1_ptr;
        }

        int32_t broadcast_dims[8];
        int32_t broadcast_dims_num = 0;
        for (int dims_i = 0; dims_i < SHAPE_LEN; ++dims_i) {
            if (broadcast_tensor->shapes[dims_i] != another_tensor->shapes[dims_i]) {
                broadcast_dims[broadcast_dims_num] = dims_i;
                broadcast_dims_num++;
            }
        }

        int32_t outter_elem_size = 1, broadcast_elem_size = 1, inner_elem_size = 1;
        for (int dims_i = 0; dims_i < SHAPE_LEN; ++dims_i) {
            if (dims_i < broadcast_dims[0]) {
                outter_elem_size *= another_tensor->shapes[dims_i];
            } else if (dims_i > broadcast_dims[broadcast_dims_num - 1]) {
                inner_elem_size *= another_tensor->shapes[dims_i];
            } else {
                broadcast_elem_size *= another_tensor->shapes[dims_i];
            }
        }

        // 开始做乘法
        for (int outter_i = 0; outter_i < outter_elem_size; ++outter_i) {
            float *cur_broadcast_ptr = broadcast_ptr + outter_i * inner_elem_size;
            float *cur_another_ptr = another_ptr + outter_i * broadcast_elem_size * inner_elem_size;
            for (int broadcast_i = 0; broadcast_i < broadcast_elem_size; ++broadcast_i) {
                float *cur_ofmap_ptr = output_ptr + outter_i * broadcast_elem_size * inner_elem_size
                        + broadcast_i * inner_elem_size;
                for (int inner_i = 0; inner_i < inner_elem_size; ++inner_i) {
                    cur_ofmap_ptr[inner_i] = cur_another_ptr[broadcast_i * inner_elem_size + inner_i]
                            * cur_broadcast_ptr[inner_i];
                }
            }
        }

    }

//    LOG_DBG("end this op");

    return 0;
}