//
// Created by wanzai on 24-11-14.
//

#ifndef ONENEW_NMS_IMPL_H
#define ONENEW_NMS_IMPL_H

#define POTENTIAL_BOXES_NUM 2048

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
//    int32_t cls_num;              // total num of object categories in this network
    int32_t img_w;
    int32_t img_h;
    float score_threshold;
    float iou_threshold;
} NMS_CONFIG_S;

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
        float *tmp_box_ptr, NMS_CONFIG_S *cfg) {
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

#endif //ONENEW_NMS_IMPL_H
