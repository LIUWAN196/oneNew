#ifndef __CONV_H__
#define __CONV_H__

#include "../../../common/nn_common.h"

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);

int eval_1x1j1(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);
int im2col(float *input_col_ptr, float *input_ptr, OPERAND_S *in_tensor, OPERAND_S *out_tensor,
           OPERAND_S *weight_tensor, CONV_CONFIG_S *cfg);

int im2col_s8(int8_t *input_col_ptr, int8_t *input_ptr, OPERAND_S *in_tensor, OPERAND_S *out_tensor,
              OPERAND_S *weight_tensor, CONV_CONFIG_S *cfg);

int eval_mxn_img2col(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);

int eval_mxn_img2col_no_bias(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);

int eval_depthwise_conv_mxn_img2col(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);

int eval_depthwise_conv_mxn_img2col_W8A32(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);

int eval_depthwise_conv_3x3_pad1(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);

int eval_mxn_img2col_W8A32_with_input_scale(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);

int eval_mxn_img2col_W8A32_with_input_scale_no_bias(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);

int do_act(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);

#endif