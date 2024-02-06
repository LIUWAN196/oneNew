#include "conv.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    CONV_CONFIG_S *cfg = (CONV_CONFIG_S *) (params[0].addr);
//    printf("yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs->addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *weight_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *bias_tensor = (OPERAND_S *) (params[3].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[4].addr);

    weight_tensor->shape.N = 4;
    weight_tensor->shape.C = 3;
    weight_tensor->shape.H = 2;
    weight_tensor->shape.W = 2;

    int32_t kernel_c = weight_tensor->shape.C;
    int32_t kernel_h = weight_tensor->shape.H;
    int32_t kernel_w = weight_tensor->shape.W;

    int32_t weight_n = weight_tensor->shape.N;
    int32_t weight_c = weight_tensor->shape.C;
    int32_t weight_h = weight_tensor->shape.H;
    int32_t weight_w = weight_tensor->shape.W;

    int32_t in_n = in_tensor->shape.N;
    int32_t in_c = in_tensor->shape.C;
    int32_t in_h = in_tensor->shape.H;
    int32_t in_w = in_tensor->shape.W;

    int32_t out_n = out_tensor->shape.N;
    int32_t out_c = out_tensor->shape.C;
    int32_t out_h = out_tensor->shape.H;
    int32_t out_w = out_tensor->shape.W;

//    float weight[48] = {0.24524809420108795, -0.13305999338626862,
//                        0.09786609560251236, -0.16296595335006714,
//                        -0.13327960669994354, 0.271098792552948,
//                        0.13574828207492828, -0.11812946200370789,
//                        0.04207799211144447, 0.04197176173329353,
//                        -0.21444804966449738, -0.1448458731174469,
//                        -0.1997447907924652, 0.2610602378845215,
//                        0.2525959014892578, -0.20367427170276642,
//                        0.01795695535838604, -0.0976543202996254,
//                        0.12799863517284393, 0.09930282831192017,
//                        0.1372622400522232, 0.1081470474600792,
//                        0.23163588345050812, -0.0664968341588974,
//                        -0.222599059343338, 0.004883721005171537,
//                        0.15702278912067413, -0.08567554503679276,
//                        -0.003808666253462434, -0.2750321328639984,
//                        -0.1864507645368576, 0.12595069408416748,
//                        0.004001687280833721, -0.23386947810649872,
//                        0.06617480516433716, -0.25665804743766785,
//                        0.023018762469291687, -0.09311627596616745,
//                        0.16506680846214294, -0.10667421668767929,
//                        -0.007906192913651466, -0.030921272933483124,
//                        -0.14069487154483795, -0.11744974553585052,
//                        0.18241900205612183, -0.06519744545221329,
//                        0.26238542795181274, -0.2695627212524414};

    float weight[48];
    for (int i = 0; i < 48; ++i) {
        weight[i] = 1 + i / 12;
    }
//
//    printf("hhhhhh=======================hhh\n");
//    for (int i = 0; i < 3 * 4 * 4; ++i) {
//        input_ptr[i] = i;
//    }
//
//    for (int i = 0; i < 3 * 4 * 4; ++i) {
//        if (i % 4 == 0){
//            printf("\n");
//        }
//        printf("%f  ", input_ptr[i]);
//    }


    // loop params
    float *tmp_input_ptr;
    float *tmp_output_ptr;
    float *tmp_weight_ptr;
    float *cur_input_ptr;
    float *cur_output_ptr;
    for (int n_i = 0; n_i < out_n; ++n_i) {
        for (int c_i = 0; c_i < out_c; ++c_i) {
            for (int h_i = 0; h_i < out_h; ++h_i) {
                for (int w_i = 0; w_i < out_w; ++w_i) {
                    tmp_output_ptr = output_ptr + n_i * out_c * out_h * out_w + c_i * out_h * out_w + h_i * out_w + w_i;
                    tmp_input_ptr = input_ptr + h_i * out_w * stride_x * stride_y + w_i * stride_x;
                    tmp_weight_ptr = weight + c_i * kernel_c * kernel_h * kernel_w;
                    float psum = 0;
                    for (int k_c = 0; k_c < kernel_c; ++k_c) {
                        cur_input_ptr = tmp_input_ptr + k_c * in_h * in_w;
                        for (int k_h = 0; k_h < kernel_h; ++k_h) {
                            for (int k_w = 0; k_w < kernel_w; ++k_w) {
                                psum += cur_input_ptr[k_h * in_w + k_w] * tmp_weight_ptr[k_c * kernel_h * kernel_w + k_h * kernel_w + k_w];
                            }
                        }
                    }
                    tmp_output_ptr[0] = psum;
                }
            }
        }
    }


//    printf("===========================\n");
//    for (int i = 0; i < 4 * 2 * 2; ++i) {
//        if (i % 2 == 0){
//            printf("\n");
//        }
//        printf("%f  ", output_ptr[i]);
//    }

    int c = 101;
    return 0;
}