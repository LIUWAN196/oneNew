#include "object_detect.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "math.h"
#include "stdint.h"
#include "stdlib.h"
#include "string.h"

int do_yolov5_obj_detect(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);

int nms(int32_t* keep_box_idx_ptr, int32_t potential_boxes_cnt, int32_t* tmp_cls_id_ptr, float * tmp_score_ptr,
        float* tmp_box_ptr, OBJECT_DETECT_CONFIG_S *cfg, int64_t rem_buf_ptr, int64_t rem_buf_size);

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    OBJECT_DETECT_CONFIG_S *cfg = (OBJECT_DETECT_CONFIG_S *) (params[0].addr);

    if (cfg->net_type == YOLO_V5) {
        OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
        int32_t shape_1 = in0_tensor->shapes[1];

        const int32_t boxes_coord_len = 4;      // x_center、y_center、w、h
        const int32_t bg_conf_len = 1;          // background confidence
        const int32_t cls_num = cfg->cls_num;
        if (shape_1 != boxes_coord_len + bg_conf_len + cls_num) {
            LOG_ERR("sorry, 传入到 object detect 算子 (cfg->net_type == YOLO_V5) 的输入 shape[1] = %d 和 cls_num = %d "
                    "无法对应，他们的关系应该是 shape[1] = 4 + 1 + cls_num。\n", shape_1, cls_num);
        }
        do_yolov5_obj_detect(params, inputs, outputs);
    } else {
        LOG_ERR("sorry, current, just supoort YOLO_V5 obj detect, the enum you set is %d\n", cfg->net_type);
    }

    return 0;
}

int do_yolov5_obj_detect(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs){
    OBJECT_DETECT_CONFIG_S *cfg = (OBJECT_DETECT_CONFIG_S *) (params[0].addr);

    USEFUL_INFO_S* useful_info = (USEFUL_INFO_S *) (params[BUF_MAXNUM - 1].addr);
    int64_t public_buf_size = useful_info->public_buf_info.public_buf_size;
    int64_t public_buf_ptr = useful_info->public_buf_info.public_buf_ptr;
    int64_t rem_buf_size = public_buf_size;
    int64_t rem_buf_ptr = public_buf_ptr;

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    const int32_t cls_num = cfg->cls_num;
    const float score_threshold = cfg->score_threshold;

    int32_t shape_1 = in_tensor->shapes[1];

    // 将临时的中间数据存放到 public_buf_ptr 地址上
    const int32_t potential_boxes_num = 8192 * 2;
    int32_t *tmp_cls_id_ptr;
    float *tmp_score_ptr;
    float *tmp_box_ptr;
    int32_t *keep_box_idx_ptr;
    int32_t tmp_need_buf_size = potential_boxes_num * (sizeof(int32_t) + sizeof(float) + 4 * sizeof(float) + sizeof(int32_t));
    if (tmp_need_buf_size < rem_buf_size) {
        tmp_cls_id_ptr = (void *)rem_buf_ptr;
        rem_buf_ptr += (potential_boxes_num * sizeof(int32_t) + 31) & (~32);

        tmp_score_ptr = (void *)rem_buf_ptr;
        rem_buf_ptr += (potential_boxes_num * sizeof(float) + 31) & (~32);

        tmp_box_ptr = (void *)rem_buf_ptr;
        rem_buf_ptr += (potential_boxes_num * 4 * sizeof(float) + 31) & (~32);

        keep_box_idx_ptr = (void *)rem_buf_ptr;
        rem_buf_ptr += (potential_boxes_num * sizeof(int32_t) + 31) & (~32);

        rem_buf_size -= (tmp_need_buf_size + 31) & (~32);
    } else {
        LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)tmp_need_buf_size);
    }

    int32_t potential_boxes_cnt = 0;
    // step 1: 通过背景置信度来滤除大部分的 boxes，并且再统计保留下来的 boxes 的最大得分类别，再次滤除
    int32_t boxes_num = in_tensor->shapes[0];
    for (int box_i = 0; box_i < boxes_num; ++box_i) {
        float bg_conf = input_ptr[box_i * shape_1 + 4];
        if (bg_conf >= score_threshold) {   // 使用背景置信度来滤除大部分的 boxes
            // 寻找该 box 的最大得分类别和分数
            float cls_score = -32768.0f;
            int cls_idx = 0;
            for (int cls_i = 0; cls_i < cls_num; ++cls_i) {
                if (input_ptr[box_i * shape_1 + 4 + 1 + cls_i] > cls_score) {
                    cls_score = input_ptr[box_i * shape_1 + 4 + 1 + cls_i];
                    cls_idx = cls_i;
                }
            }
            if (bg_conf * cls_score >= score_threshold)   // 再次滤除，只保留真实 > score_threshold 的 boxes
            {
                tmp_cls_id_ptr[potential_boxes_cnt] = cls_idx;
                tmp_score_ptr[potential_boxes_cnt] = bg_conf * cls_score;
                // 把 boxes 从原始的 x_center、y_center、w、h 转为 x_min、y_min、x_max、y_max 来保留
                tmp_box_ptr[potential_boxes_cnt * 4 + 0] = input_ptr[box_i * shape_1 + 0] - input_ptr[box_i * shape_1 + 2] / 2;
                tmp_box_ptr[potential_boxes_cnt * 4 + 1] = input_ptr[box_i * shape_1 + 1] - input_ptr[box_i * shape_1 + 3] / 2;
                tmp_box_ptr[potential_boxes_cnt * 4 + 2] = input_ptr[box_i * shape_1 + 0] + input_ptr[box_i * shape_1 + 2] / 2;
                tmp_box_ptr[potential_boxes_cnt * 4 + 3] = input_ptr[box_i * shape_1 + 1] + input_ptr[box_i * shape_1 + 3] / 2;
                potential_boxes_cnt ++;
            }
        }
        if (potential_boxes_cnt >= potential_boxes_num) {
            break;
        }
    }

    // step 2: 做 nms
    int32_t keep_boxes_cnt = nms(keep_box_idx_ptr, potential_boxes_cnt, tmp_cls_id_ptr, tmp_score_ptr, tmp_box_ptr, cfg, rem_buf_ptr, rem_buf_size);

    // step 3: 通过 step 2 获取到的索引，从 tmp_cls_id_ptr, tmp_score_ptr, tmp_box_ptr 中获取到最终的输出信息
    OBJ_DETECT_OUT_INFO_S* out_info_ptr = (OBJ_DETECT_OUT_INFO_S*)output_ptr;
    for (int i = 0; i < keep_boxes_cnt; ++i) {
        out_info_ptr[i].cls_id = tmp_cls_id_ptr[keep_box_idx_ptr[i]];
        out_info_ptr[i].score = tmp_score_ptr[keep_box_idx_ptr[i]];
        out_info_ptr[i].box_info.x_min = tmp_box_ptr[keep_box_idx_ptr[i] * 4 + 0];
        out_info_ptr[i].box_info.y_min = tmp_box_ptr[keep_box_idx_ptr[i] * 4 + 1];
        out_info_ptr[i].box_info.x_max = tmp_box_ptr[keep_box_idx_ptr[i] * 4 + 2];
        out_info_ptr[i].box_info.y_max = tmp_box_ptr[keep_box_idx_ptr[i] * 4 + 3];
    }
    out_info_ptr[keep_boxes_cnt].cls_id = -1;   // 作为终止符

    int box_i = 0;
    while (out_info_ptr[box_i].cls_id != -1) {
        BOX_INFO_S *cur_box = &out_info_ptr[box_i].box_info;
        printf("box_i is %d, cls id is %d, score is %f, coord is %f  %f  %f  %f\n",
               box_i, out_info_ptr[box_i].cls_id, out_info_ptr[box_i].score,
               cur_box->x_min, cur_box->y_min, cur_box->x_max, cur_box->y_max);
        box_i++;
    }

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

int nms(int32_t* keep_box_idx_ptr, int32_t potential_boxes_cnt, int32_t* tmp_cls_id_ptr, float * tmp_score_ptr,
        float* tmp_box_ptr, OBJECT_DETECT_CONFIG_S *cfg, int64_t rem_buf_ptr, int64_t rem_buf_size) {

    int32_t keep_boxes_cnt = 0;

    const int32_t potential_boxes_num = 8192 * 2;
    float *score_sorted_ptr;
    int32_t *idx_sorted_ptr;
    int32_t tmp_need_buf_size = potential_boxes_num * (sizeof(float) + sizeof(int32_t));
    if (tmp_need_buf_size < rem_buf_size) {
        score_sorted_ptr = (void *)rem_buf_ptr;
        rem_buf_ptr += (potential_boxes_num * sizeof(float) + 31) & (~32);

        idx_sorted_ptr = (void *)rem_buf_ptr;
    } else {
        LOG_ERR("remaining buf size is %d, but need buf size is %d", (int32_t)rem_buf_size, (int32_t)tmp_need_buf_size);
    }

    // step 0: sort score
    for (int box_i = 0; box_i < potential_boxes_cnt; box_i++) {
        score_sorted_ptr[box_i] = tmp_score_ptr[box_i];
        idx_sorted_ptr[box_i] = box_i;
    }
    q_sort(score_sorted_ptr, idx_sorted_ptr, potential_boxes_cnt);

    // step 1: separate coord of each category
    int32_t img_size = cfg->img_w > cfg->img_h ? cfg->img_w : cfg->img_h;
    for (int box_i = 0; box_i < potential_boxes_cnt; box_i++) {
        int32_t cls_idx = tmp_cls_id_ptr[box_i];

        tmp_box_ptr[box_i * 4 + 0] += (float)(cls_idx * img_size);
        tmp_box_ptr[box_i * 4 + 1] += (float)(cls_idx * img_size);
        tmp_box_ptr[box_i * 4 + 2] += (float)(cls_idx * img_size);
        tmp_box_ptr[box_i * 4 + 3] += (float)(cls_idx * img_size);
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

            BOX_INFO_S *keep_box = (BOX_INFO_S *)&tmp_box_ptr[idx_sorted_ptr[i] * 4];
            BOX_INFO_S *check_box = (BOX_INFO_S *)&tmp_box_ptr[idx_sorted_ptr[j] * 4];
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

        tmp_box_ptr[box_i * 4 + 0] -= (float)(cls_idx * img_size);
        tmp_box_ptr[box_i * 4 + 1] -= (float)(cls_idx * img_size);
        tmp_box_ptr[box_i * 4 + 2] -= (float)(cls_idx * img_size);
        tmp_box_ptr[box_i * 4 + 3] -= (float)(cls_idx * img_size);
    }

    return keep_boxes_cnt;
}




