#include "maxpool.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "pad_maxpool.h"
#include "stdint.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    MAX_POOL_CONFIG_S *cfg = (MAX_POOL_CONFIG_S *) (params[0].addr);
//    printf("\n yes this is device, the op type is %s, the op name is %s\n", cfg->op_type, cfg->op_name);

    int32_t kernel_h = cfg->kernel_shape[0];
    int32_t kernel_w = cfg->kernel_shape[1];

    int32_t stride_x = cfg->strides[0];
    int32_t stride_y = cfg->strides[1];

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs->addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[2].addr);

    int32_t in_n = in_tensor->shape.N;
    int32_t in_c = in_tensor->shape.C;
    int32_t in_h = in_tensor->shape.H;
    int32_t in_w = in_tensor->shape.W;

    int32_t out_n = out_tensor->shape.N;
    int32_t out_c = out_tensor->shape.C;
    int32_t out_h = out_tensor->shape.H;
    int32_t out_w = out_tensor->shape.W;


    void *src_pad_ptr;
    if (cfg->pads[0] != 0){
        // do pad
        src_pad_ptr = malloc(in_n * in_c * (in_h + 2 * cfg->pads[0]) * (in_w + 2 * cfg->pads[0]) * sizeof(float));
        PAD_INNER_CONFIG_S pad_cfg;
        pad_cfg.h = cfg->pads[0];
        pad_cfg.w = cfg->pads[0];

        in_h = in_h + 2 * cfg->pads[0];
        in_w = in_w + 2 * cfg->pads[0];

        do_pad_maxpool(src_pad_ptr, input_ptr, in_tensor, &pad_cfg);
        input_ptr = (float *)src_pad_ptr;
    }

    // loop params
    float *tmp_input_ptr;
    float *tmp_output_ptr;
    float *cur_input_ptr;
    float *cur_output_ptr;
    for (int n_i = 0; n_i < out_n; ++n_i) {
        for (int c_i = 0; c_i < out_c; ++c_i) {
            tmp_output_ptr = output_ptr + n_i * out_c * out_h * out_w + c_i * out_h * out_w;
            tmp_input_ptr = input_ptr + n_i * in_c * in_h * in_w + c_i * in_h * in_w;
            for (int h_i = 0; h_i < out_h; ++h_i) {
                for (int w_i = 0; w_i < out_w; ++w_i) {
                    cur_output_ptr = tmp_output_ptr + h_i * out_w + w_i;
                    cur_input_ptr = tmp_input_ptr + h_i * in_w * stride_y + w_i * stride_x;
                    float max_num = 0;
                    for (int k_h = 0; k_h < kernel_h; ++k_h) {
                        for (int k_w = 0; k_w < kernel_w; ++k_w) {
                            max_num = (cur_input_ptr[k_h * in_w + k_w] > max_num) ? cur_input_ptr[k_h * in_w + k_w]
                                                                                  : max_num;

                        }
                    }
                    cur_output_ptr[0] = max_num;
                }
            }
        }
    }


//    // write_bin(replace_char(cfg->out_operand_name[0]), out_n * out_c * out_h * out_w * sizeof(float), output_ptr);



//    for (int i = 0; i < 3 * 10 * 10; ++i) {
//        if (i % 10 == 0){
//            printf("\n");
//        }
//        printf("%f  ", input_ptr[i]);
//    }
//
//    printf("===========================\n");
//    for (int i = 0; i < 3 * 5 * 5; ++i) {
//        if (i % 5 == 0){
//            printf("\n");
//        }
//        printf("%f  ", output_ptr[i]);
//    }

    int a = 101;


    return 0;
}