#include "segment.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "math.h"
#include "stdint.h"
#include "stdlib.h"
#include "string.h"
#include "../../x86_utils/nms_impl.h"

int do_yolo_v8_segment(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
    // 输入 tensor 为两个，第一个为 [boxes_info,boxes_num]，例如 [116,8400]  （其中 116 = 32 + 4 + 80）;第二个为 [mask, w, h]，例如 [32, 160,160]
    do_yolo_v8_segment(params, inputs, outputs);

    return 0;
}

int do_yolo_v8_segment(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
    SEGMENT_CONFIG_S *cfg = (SEGMENT_CONFIG_S *) (params[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);

    float *input0_ptr = (float *) (inputs[0].addr);
    float *input1_ptr = (float *) (inputs[1].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    const int32_t cls_num = cfg->cls_num;
    const float score_threshold = cfg->score_threshold;

    const int32_t out_w = 160;
    const int32_t out_h = 160;
    const int32_t boxes_num = in_tensor->shapes[1];       // such as 8400
    const int32_t mask_len = 32;
    float *input0_mask_ptr = input0_ptr + (4 + cls_num) * boxes_num;

    // 将临时的中间数据存放到临时数组中
    int32_t tmp_cls_id_ptr[POTENTIAL_BOXES_NUM];
    float tmp_score_ptr[POTENTIAL_BOXES_NUM];
    float tmp_box_ptr[POTENTIAL_BOXES_NUM];
    float tmp_mask_ptr[POTENTIAL_BOXES_NUM * 32];

    int32_t potential_boxes_cnt = 0;
    // step 1: 通过 boxes 的最大得分类别，滤除掉大部分的 boxes
    for (int box_i = 0; box_i < boxes_num; ++box_i) {
        // 寻找该 box 的最大得分类别和分数
        float cls_score = -32768.0f;
        int cls_idx = 0;
        for (int cls_i = 0; cls_i < cls_num; ++cls_i) {
            if (input0_ptr[(4 + cls_i) * boxes_num + box_i] > cls_score) {
                cls_score = input0_ptr[(4 + cls_i) * boxes_num + box_i];
                cls_idx = cls_i;
            }
        }
        if (cls_score >= score_threshold)   // 滤除，只保留真实 > score_threshold 的 boxes
        {
            tmp_cls_id_ptr[potential_boxes_cnt] = cls_idx;
            tmp_score_ptr[potential_boxes_cnt] = cls_score;
            // boxes 坐标为 x_min、y_min、x_max、y_max
            tmp_box_ptr[potential_boxes_cnt * 4 + 0] = input0_ptr[0 * boxes_num + box_i];
            tmp_box_ptr[potential_boxes_cnt * 4 + 1] = input0_ptr[1 * boxes_num + box_i];
            tmp_box_ptr[potential_boxes_cnt * 4 + 2] = input0_ptr[2 * boxes_num + box_i];
            tmp_box_ptr[potential_boxes_cnt * 4 + 3] = input0_ptr[3 * boxes_num + box_i];

            for (int point_i = 0; point_i < mask_len; ++point_i) {
                tmp_mask_ptr[potential_boxes_cnt * mask_len + point_i] = input0_mask_ptr[point_i * boxes_num + box_i];
            }
            potential_boxes_cnt++;
        }
        if (potential_boxes_cnt >= POTENTIAL_BOXES_NUM) {
            break;
        }
    }

    // step 2: 做 nms，获取到待保留的 boxes 的索引，并从 tmp_***_ptr 中获取到最终的输出信息
    SEGMENT_OUT_INFO_S *out_info_ptr = (SEGMENT_OUT_INFO_S *) output_ptr;
    int32_t keep_box_idx_ptr[POTENTIAL_BOXES_NUM];

    int32_t keep_boxes_cnt;
    NMS_CONFIG_S nms_cfg;
    nms_cfg.img_h = cfg->img_h;
    nms_cfg.img_w = cfg->img_w;
    nms_cfg.score_threshold = cfg->score_threshold;
    nms_cfg.iou_threshold = cfg->iou_threshold;
    keep_boxes_cnt = nms(keep_box_idx_ptr, potential_boxes_cnt, tmp_cls_id_ptr, tmp_score_ptr, tmp_box_ptr, &nms_cfg);

    // step 3: 通过 step 2 获取到的索引，从 tmp_cls_id_ptr, tmp_score_ptr, tmp_box_ptr 中获取到最终的输出信息
    for (int i = 0; i < keep_boxes_cnt; ++i) {
        int32_t cur_keep_idx = keep_box_idx_ptr[i];
        out_info_ptr[i].cls_id = tmp_cls_id_ptr[cur_keep_idx];
        out_info_ptr[i].score = tmp_score_ptr[cur_keep_idx];
        out_info_ptr[i].box_info.x_min = tmp_box_ptr[cur_keep_idx * 4 + 0];
        out_info_ptr[i].box_info.y_min = tmp_box_ptr[cur_keep_idx * 4 + 1];
        out_info_ptr[i].box_info.x_max = tmp_box_ptr[cur_keep_idx * 4 + 2];
        out_info_ptr[i].box_info.y_max = tmp_box_ptr[cur_keep_idx * 4 + 3];
        // 为每个 boxes，保留 [out_w,out_h] 这样的 mask
        for (int j = 0; j < out_w * out_h; ++j) {
            float psum = 0;
            for (int k = 0; k < mask_len; ++k) {
                psum += tmp_mask_ptr[cur_keep_idx * mask_len + k] * input1_ptr[k * out_w * out_h + j];
            }
            out_info_ptr[i].mask[j] = psum;
        }
    }
    out_info_ptr[keep_boxes_cnt].cls_id = -1;   // 作为终止符

    return 0;
}


