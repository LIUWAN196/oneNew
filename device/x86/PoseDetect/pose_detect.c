#include "pose_detect.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "math.h"
#include "stdint.h"
#include "stdlib.h"
#include "string.h"

#define POTENTIAL_BOXES_NUM 2048

int do_yolo_v8_pose_detect(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);

int get_pose_detect_output_info(POSE_DETECT_OUT_INFO_S *out_info_ptr, int32_t potential_boxes_cnt, int32_t *tmp_cls_id_ptr,
                    float *tmp_score_ptr, float *tmp_box_ptr, float *tmp_key_points_ptr, POSE_DETECT_CONFIG_S *cfg);

int nms(int32_t *keep_box_idx_ptr, int32_t potential_boxes_cnt, int32_t *tmp_cls_id_ptr, float *tmp_score_ptr,
        float *tmp_box_ptr, POSE_DETECT_CONFIG_S *cfg);

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
    // 输入 shape 为 [boxes_info,boxes_num]，例如 [56,8400]
    do_yolo_v8_pose_detect(params, inputs, outputs);

    return 0;
}

int do_yolo_v8_pose_detect(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
    POSE_DETECT_CONFIG_S *cfg = (POSE_DETECT_CONFIG_S *) (params[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    const float score_threshold = cfg->score_threshold;

    int32_t boxes_info_len = in_tensor->shapes[0];  // such as 56
    int32_t boxes_num = in_tensor->shapes[1];       // such as 8400

    // 将临时的中间数据存放到临时数组中
    int32_t tmp_cls_id_ptr[POTENTIAL_BOXES_NUM];
    float tmp_score_ptr[POTENTIAL_BOXES_NUM];
    float tmp_box_ptr[POTENTIAL_BOXES_NUM];
    const int32_t key_points_len = 51;      // 51 = 17 * （2 + 1）    2 是关键点的 x、y 坐标，1 是该关键点是否被遮挡
    float tmp_key_points_ptr[POTENTIAL_BOXES_NUM * 51];

    int32_t potential_boxes_cnt = 0;
    // step 1: 通过 boxes 的最大得分类别，滤除掉大部分的 boxes
    for (int box_i = 0; box_i < boxes_num; ++box_i) {
        // 寻找该 box 的得分
        float cls_score = input_ptr[4 * boxes_num + box_i];
        int cls_idx = 0;    // 这里是姿态检测的后处理，所以只有一个类别 0  --> person

        if (cls_score >= score_threshold)   // 滤除，只保留真实 > score_threshold 的 boxes
        {
            tmp_cls_id_ptr[potential_boxes_cnt] = cls_idx;
            tmp_score_ptr[potential_boxes_cnt] = cls_score;

            tmp_box_ptr[potential_boxes_cnt * 4 + 0] = input_ptr[0 * boxes_num + box_i];
            tmp_box_ptr[potential_boxes_cnt * 4 + 1] = input_ptr[1 * boxes_num + box_i];
            tmp_box_ptr[potential_boxes_cnt * 4 + 2] = input_ptr[2 * boxes_num + box_i];
            tmp_box_ptr[potential_boxes_cnt * 4 + 3] = input_ptr[3 * boxes_num + box_i];

            for (int point_i = 0; point_i < key_points_len; ++point_i) {
                tmp_key_points_ptr[potential_boxes_cnt * key_points_len + point_i] =
                        input_ptr[(4 + point_i) * boxes_num + box_i];
            }

            potential_boxes_cnt++;
        }
        if (potential_boxes_cnt >= POTENTIAL_BOXES_NUM) {
            break;
        }
    }

    // step 2: 做 nms，获取到待保留的 boxes 的索引，并从 tmp_***_ptr 中获取到最终的输出信息
    POSE_DETECT_OUT_INFO_S *out_info_ptr = (POSE_DETECT_OUT_INFO_S *) output_ptr;
    get_pose_detect_output_info(out_info_ptr, potential_boxes_cnt, tmp_cls_id_ptr, tmp_score_ptr, tmp_box_ptr, tmp_key_points_ptr, cfg);

    return 0;
}

void q_sort(float *value, int *idx, int len) {
    int32_t i, j, tmp_idx;
    float p, tmp_value;
    if (len < 2) return;

    // get the middle element as pivot
    p = value[len / 2];

    for (i = 0, j = len - 1;; i++, j--) {
        while (value[i] > p) i++;
        while (p > value[j]) j--;
        if (i >= j) break;
        // swap value
        tmp_value = value[i];
        value[i] = value[j];
        value[j] = tmp_value;
        // swap index
        tmp_idx = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp_idx;
    }

    q_sort(value, idx, i);
    q_sort(value + i, idx + i, len - i);
}

float iou(BOX_INFO_S *box_a, BOX_INFO_S *box_b) {
    float x1 = box_a->x_min > box_b->x_min ? box_a->x_min : box_b->x_min;  // std::max
    float y1 = box_a->y_min > box_b->y_min ? box_a->y_min : box_b->y_min;  // std::max
    float x2 = box_a->x_max > box_b->x_max ? box_b->x_max : box_a->x_max;  // std::min
    float y2 = box_a->y_max > box_b->y_max ? box_b->y_max : box_a->y_max;  // std::min

    // 没有重叠面积
    if (x2 < x1 || y2 < y1) {
        return 0;
    }

    float a_width = box_a->x_max - box_a->x_min;
    float a_height = box_a->y_max - box_a->y_min;
    float b_width = box_b->x_max - box_b->x_min;
    float b_heihgt = box_b->y_max - box_b->y_min;

    float inter_area = (x2 - x1) * (y2 - y1);                                           // 交集
    float iou = inter_area / ((a_width * a_height) + b_width * b_heihgt - inter_area);  // 并集

    return iou;
}

int nms(int32_t *keep_box_idx_ptr, int32_t potential_boxes_cnt, int32_t *tmp_cls_id_ptr, float *tmp_score_ptr,
        float *tmp_box_ptr, POSE_DETECT_CONFIG_S *cfg) {
    int32_t keep_boxes_cnt = 0;

    float score_sorted_ptr[POTENTIAL_BOXES_NUM];
    int32_t idx_sorted_ptr[POTENTIAL_BOXES_NUM];

    // step 0: sort score
    for (int box_i = 0; box_i < potential_boxes_cnt; box_i++) {
        score_sorted_ptr[box_i] = tmp_score_ptr[box_i];
        idx_sorted_ptr[box_i] = (int32_t) box_i;
    }
    q_sort(score_sorted_ptr, idx_sorted_ptr, potential_boxes_cnt);

    // step 1: separate coord of each category
    int32_t img_size = cfg->img_w > cfg->img_h ? cfg->img_w : cfg->img_h;
    for (int box_i = 0; box_i < potential_boxes_cnt; box_i++) {
        int32_t cls_idx = tmp_cls_id_ptr[box_i];

        tmp_box_ptr[box_i * 4 + 0] += (float) (cls_idx * img_size);
        tmp_box_ptr[box_i * 4 + 1] += (float) (cls_idx * img_size);
        tmp_box_ptr[box_i * 4 + 2] += (float) (cls_idx * img_size);
        tmp_box_ptr[box_i * 4 + 3] += (float) (cls_idx * img_size);
    }

    // step 2: do nms
    for (int32_t i = 0; i < potential_boxes_cnt; ++i) {
        if (score_sorted_ptr[i] <= 0.0f) {
            continue;
        }
        // step 2.1: add idx info to keep_box_idx_ptr
        keep_box_idx_ptr[keep_boxes_cnt++] = idx_sorted_ptr[i];

        for (int32_t j = i + 1; j < potential_boxes_cnt; ++j) {
            if (score_sorted_ptr[j] <= 0.0f) {
                continue;
            }

            BOX_INFO_S *keep_box = (BOX_INFO_S *) &tmp_box_ptr[idx_sorted_ptr[i] * 4];
            BOX_INFO_S *check_box = (BOX_INFO_S *) &tmp_box_ptr[idx_sorted_ptr[j] * 4];
            // step 2.2: compute iou
            float iou_val = iou(keep_box, check_box);
            if (iou_val > cfg->iou_threshold) {
                // step 2.3: suppress boxes with high overlap
                score_sorted_ptr[j] = 0.0f;
            }
        }
    }

    // step 3: restore box coord
    for (int box_i = 0; box_i < potential_boxes_cnt; box_i++) {
        int32_t cls_idx = tmp_cls_id_ptr[box_i];

        tmp_box_ptr[box_i * 4 + 0] -= (float) (cls_idx * img_size);
        tmp_box_ptr[box_i * 4 + 1] -= (float) (cls_idx * img_size);
        tmp_box_ptr[box_i * 4 + 2] -= (float) (cls_idx * img_size);
        tmp_box_ptr[box_i * 4 + 3] -= (float) (cls_idx * img_size);
    }

    return keep_boxes_cnt;
}

int get_pose_detect_output_info(POSE_DETECT_OUT_INFO_S *out_info_ptr, int32_t potential_boxes_cnt, int32_t *tmp_cls_id_ptr,
                    float *tmp_score_ptr, float *tmp_box_ptr, float *tmp_key_points_ptr, POSE_DETECT_CONFIG_S *cfg) {
    int32_t keep_box_idx_ptr[POTENTIAL_BOXES_NUM];

    // step 1: 做 nms
    int32_t keep_boxes_cnt;
    keep_boxes_cnt = nms(keep_box_idx_ptr, potential_boxes_cnt, tmp_cls_id_ptr, tmp_score_ptr, tmp_box_ptr, cfg);

    // step 2: 通过 step 2 获取到的索引，从 tmp_cls_id_ptr, tmp_score_ptr, tmp_box_ptr 中获取到最终的输出信息
    for (int i = 0; i < keep_boxes_cnt; ++i) {
        out_info_ptr[i].cls_id = tmp_cls_id_ptr[keep_box_idx_ptr[i]];
        out_info_ptr[i].score = tmp_score_ptr[keep_box_idx_ptr[i]];
        out_info_ptr[i].box_info.x_min = tmp_box_ptr[keep_box_idx_ptr[i] * 4 + 0];
        out_info_ptr[i].box_info.y_min = tmp_box_ptr[keep_box_idx_ptr[i] * 4 + 1];
        out_info_ptr[i].box_info.x_max = tmp_box_ptr[keep_box_idx_ptr[i] * 4 + 2];
        out_info_ptr[i].box_info.y_max = tmp_box_ptr[keep_box_idx_ptr[i] * 4 + 3];
        memcpy(out_info_ptr[i].keypoints, tmp_key_points_ptr + 51 * keep_box_idx_ptr[i], 51 * sizeof(float));
    }
    out_info_ptr[keep_boxes_cnt].cls_id = -1;   // 作为终止符

    return 0;
}


