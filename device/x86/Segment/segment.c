#include "segment.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "math.h"
#include "stdint.h"
#include "stdlib.h"
#include "string.h"

int segement_impl(char **out_ptr, char **in_ptr, char *cfg);

int yolov8_seg_box_decode(int *box_num, SEGMENT_OFMAP0_S *alternative_box_ptr, char *in_ptr,
                          char *in_mask_ptr, SEGMENT_CONFIG_S *segment_cfg, int ifmap_i);

int yolov8_seg_nms(char *out_ptr, SEGMENT_OFMAP0_S *alternative_box_ptr, SEGMENT_CONFIG_S *segment_cfg, int box_num);

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    SEGMENT_CONFIG_S *cfg = (SEGMENT_CONFIG_S *) (params[0].addr);

    SEGMENT_CONFIG_S segment_cfg;
    segment_cfg.ifmap_num = 6;

    segment_cfg.img_h = 640;
    segment_cfg.img_w = 640;

    segment_cfg.ifmap_tensor[0].N = 1, segment_cfg.ifmap_tensor[0].C = 144, segment_cfg.ifmap_tensor[0].H = 20,
    segment_cfg.ifmap_tensor[0].W = 20;
    segment_cfg.ifmap_tensor[1].N = 1, segment_cfg.ifmap_tensor[1].C = 144, segment_cfg.ifmap_tensor[1].H = 40,
    segment_cfg.ifmap_tensor[1].W = 40;
    segment_cfg.ifmap_tensor[2].N = 1, segment_cfg.ifmap_tensor[2].C = 144, segment_cfg.ifmap_tensor[2].H = 80,
    segment_cfg.ifmap_tensor[2].W = 80;

    segment_cfg.cls_num = 80;
    segment_cfg.max_boxes_per_class = 100;
    segment_cfg.max_boxes_per_batch = 200;
    segment_cfg.score_threshold = 0.45f;
    segment_cfg.iou_threshold = 0.6f;

    float *input0_ptr = (float *) (inputs[0].addr);
    float *input1_ptr = (float *) (inputs[1].addr);
    float *input2_ptr = (float *) (inputs[2].addr);

    float *in_mask0_ptr = (float *) (inputs[3].addr);
    float *in_mask1_ptr = (float *) (inputs[4].addr);
    float *in_mask2_ptr = (float *) (inputs[5].addr);

//    float *output_ptr = (float *) (outputs[0].addr);
    float *output_ptr = (float *) malloc(1024 * 1024);

    OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *in1_tensor = (OPERAND_S *) (params[2].addr);
    OPERAND_S *out_tensor = (OPERAND_S *) (params[3].addr);

    char *out_ptr[8];
    char *in_ptr[8];

    in_ptr[0] = (char *) input0_ptr;
    in_ptr[1] = (char *) input1_ptr;
    in_ptr[2] = (char *) input2_ptr;
    in_ptr[3] = (char *) in_mask0_ptr;
    in_ptr[4] = (char *) in_mask1_ptr;
    in_ptr[5] = (char *) in_mask2_ptr;
    out_ptr[0] = (char *) output_ptr;

    segement_impl(out_ptr, in_ptr, &segment_cfg);

    // show the output
    SEGMENT_OFMAP0_S *boxes_info_ptr = (SEGMENT_OFMAP0_S *) out_ptr[0];

    write_bin("segment_ofmap.bin", 100 * sizeof(SEGMENT_OFMAP0_S), output_ptr);

//    BOX_INFO_S *cur_box = &boxes_info_ptr[0];
    int box_i = 0;
    while (boxes_info_ptr[box_i].box_info.batch_id != -1.0f) {
        BOX_INFO_S *cur_box = &boxes_info_ptr[box_i].box_info;
        printf("th batch id is %f, cls id is %f, score is %f, coord is %f  %f  %f  %f\n", cur_box->batch_id,
               cur_box->cls_id, cur_box->score, cur_box->x_min * 640, cur_box->y_min * 640, cur_box->x_max * 640,
               cur_box->y_max * 640);
        box_i++;
    }


    return 0;
}

int segement_impl(char **out_ptr, char **in_ptr, char *cfg) {
    SEGMENT_CONFIG_S *segment_cfg = (SEGMENT_CONFIG_S *) cfg;
    int ifmap_num = segment_cfg->ifmap_num;

    SEGMENT_OFMAP0_S *alternative_box_ptr = (SEGMENT_OFMAP0_S *) malloc(20 * 1024 * sizeof(SEGMENT_OFMAP0_S));

    float *cur_ifmap_nhwc_ptr = (float *) malloc(8 * 1024 * 1024 * sizeof(float));
    float *cur_ifmap_mask_nhwc_ptr = (float *) malloc(8 * 1024 * 1024 * sizeof(float));
    int box_num = 0;
    for (int ifmap_i = 0; ifmap_i < ifmap_num / 2; ifmap_i++) {
        float *cur_ifmap_nchw_ptr = (float *) in_ptr[ifmap_i];
        float *cur_ifmap_mask_nchw_ptr = (float *) in_ptr[ifmap_i + 3];
        // step 0: trans n * c * h * w --> n * h * w * c,
        // for example: 1 x 144 x 80 x 80 --> 1 x 80 x 80 x 144; 1 x 32 x 80 x 80 --> 1 x 80 x 80 x 32
        int ifmap_in_c = 144;
        int ifmap_mask_in_c = 32;
        int in_h = segment_cfg->ifmap_tensor[ifmap_i].H, in_w = segment_cfg->ifmap_tensor[ifmap_i].W;

        for (int hw_i = 0; hw_i < in_h * in_w; hw_i++) {
            // 将 144 = 16 × 4 + 80 做转置
            for (int c_i = 0; c_i < ifmap_in_c; c_i++) {
                cur_ifmap_nhwc_ptr[hw_i * ifmap_in_c + c_i] = cur_ifmap_nchw_ptr[c_i * in_h * in_w + hw_i];
            }
            // 将 32 做转置
            for (int c_i = 0; c_i < ifmap_mask_in_c; c_i++) {
                cur_ifmap_mask_nhwc_ptr[hw_i * ifmap_mask_in_c + c_i] = cur_ifmap_mask_nchw_ptr[c_i * in_h * in_w + hw_i];
            }
        }

        // step 1: do yolo v8 seg decode
        yolov8_seg_box_decode(&box_num, alternative_box_ptr, cur_ifmap_nhwc_ptr, cur_ifmap_mask_nhwc_ptr, segment_cfg,
                              ifmap_i);
    }
    // step 2: do nms
    yolov8_seg_nms(out_ptr[0], alternative_box_ptr, segment_cfg, box_num);

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

int yolov8_seg_nms(char *out_ptr, SEGMENT_OFMAP0_S *alternative_box_ptr, SEGMENT_CONFIG_S *segment_cfg, int box_num) {
    SEGMENT_OFMAP0_S *final_info_ptr = (SEGMENT_OFMAP0_S *) out_ptr;

    // step 0: sort score
    float *score_sorted_ptr = (float *) malloc(box_num * sizeof(float));
    int *idx_sorted_ptr = (int *) malloc(box_num * sizeof(int));
    for (int box_i = 0; box_i < box_num; box_i++) {
        score_sorted_ptr[box_i] = alternative_box_ptr[box_i].box_info.score;
        idx_sorted_ptr[box_i] = box_i;
    }
    q_sort(score_sorted_ptr, idx_sorted_ptr, box_num);

    // step 1: separate coord of each category
    for (int box_i = 0; box_i < box_num; box_i++) {
        float cls_idx = alternative_box_ptr[box_i].box_info.cls_id;

        alternative_box_ptr[box_i].box_info.x_min += cls_idx;
        alternative_box_ptr[box_i].box_info.y_min += cls_idx;
        alternative_box_ptr[box_i].box_info.x_max += cls_idx;
        alternative_box_ptr[box_i].box_info.y_max += cls_idx;
    }

    // step 2: do nms
    int reslut_box_num = 0;
    for (uint16_t i = 0; i < box_num; ++i) {
        if (score_sorted_ptr[i] <= 0.0f) {
            continue;
        }
        // step 2.1: add boxes info to result
        final_info_ptr[reslut_box_num] = alternative_box_ptr[idx_sorted_ptr[i]];
        reslut_box_num++;

        uint16_t index = 1;
        for (uint16_t j = i + 1; j < box_num; ++j) {
            if (score_sorted_ptr[j] <= 0.0f) {
                continue;
            }

            BOX_INFO_S *keep_box = &alternative_box_ptr[idx_sorted_ptr[i]];
            BOX_INFO_S *check_box = &alternative_box_ptr[idx_sorted_ptr[j]];
            // step 2.2: compute iou
            if (iou(keep_box, check_box) > segment_cfg->iou_threshold) {
                score_sorted_ptr[j] = 0.0f;  // step 2.3: suppress boxes with high overlap
            }
        }
    }

    // step 3: restore box coord
    int box_i = 0;
    for (; box_i < reslut_box_num; box_i++) {
        float cls_idx = final_info_ptr[box_i].box_info.cls_id;

        final_info_ptr[box_i].box_info.x_min -= cls_idx;
        final_info_ptr[box_i].box_info.y_min -= cls_idx;
        final_info_ptr[box_i].box_info.x_max -= cls_idx;
        final_info_ptr[box_i].box_info.y_max -= cls_idx;
    }

    final_info_ptr[box_i].box_info.batch_id = -1.0f;  // add terminator

    return 0;
}

#define sigmoid(x) (1.0 / (1.0 + exp(-(x))))

int yolov8_seg_box_decode(int *box_num, SEGMENT_OFMAP0_S *alternative_box_ptr, char *in_ptr,
                          char *in_mask_ptr, SEGMENT_CONFIG_S *segment_cfg, int ifmap_i) {
    // the tmp_buf_ptr layout is: batch id / cls id / score / x_min / y_min / x_max / y_max
//    const int anchors_num = segment_cfg->anchors_num;
    const int cls_num = segment_cfg->cls_num;
    const int img_h = segment_cfg->img_h;
    const int img_w = segment_cfg->img_w;
    const int coord_info_size = 64;  //  layout is x / y / h / w * 16
    const int per_point_info_size = coord_info_size + cls_num;

    int cur_fmap_h = segment_cfg->ifmap_tensor[ifmap_i].H;
    int cur_fmap_w = segment_cfg->ifmap_tensor[ifmap_i].W;

    float scale_w = img_w / cur_fmap_w;
    float scale_h = img_h / cur_fmap_h;

    float score_threshold_before_loc = -log(1 / segment_cfg->score_threshold - 1);
    int num_potential_boxes = *box_num;

    float *ifmap_ptr = (float *) in_ptr;

    // step 1: decode potential box and calc score
    float *cur_ifmap_ptr;

    for (int h_i = 0; h_i < cur_fmap_h; h_i++) {
        cur_ifmap_ptr = ifmap_ptr + h_i * cur_fmap_w * per_point_info_size;
        for (int w_i = 0; w_i < cur_fmap_w; w_i++) {
            // step 1.1: find max score and this class
            float cls_score = -32768.0f;
            int cls_idx = 0;
            for (int cls_i = 0; cls_i < cls_num; cls_i++) {
                float tmp = cur_ifmap_ptr[w_i * per_point_info_size + coord_info_size + cls_i];
                if (tmp > cls_score) {
                    cls_score = tmp;
                    cls_idx = cls_i;
                }
            }

            float score_ = sigmoid(cls_score);

            if (score_ < segment_cfg->score_threshold) {
                continue;
            }

            // step 1.2: decode potential box
            const int per_coord_point_info_size = coord_info_size / 4; // =  64 / 4 = 16
            float* tx1 = cur_ifmap_ptr + w_i * per_point_info_size + per_coord_point_info_size * 0;
            float* ty1 = cur_ifmap_ptr + w_i * per_point_info_size + per_coord_point_info_size * 1;
            float* tx2 = cur_ifmap_ptr + w_i * per_point_info_size + per_coord_point_info_size * 2;
            float* ty2 = cur_ifmap_ptr + w_i * per_point_info_size + per_coord_point_info_size * 3;

            // find the max
            float max_tx1 = -32768.0f;
            float max_ty1 = -32768.0f;
            float max_tx2 = -32768.0f;
            float max_ty2 = -32768.0f;
            for (int i = 0; i < per_coord_point_info_size; ++i) {
                max_tx1 = tx1[i] > max_tx1 ? tx1[i] : max_tx1;
                max_ty1 = ty1[i] > max_ty1 ? ty1[i] : max_ty1;
                max_tx2 = tx2[i] > max_tx2 ? tx2[i] : max_tx2;
                max_ty2 = ty2[i] > max_ty2 ? ty2[i] : max_ty2;
            }

            // 做 exp（-x）求和
            float psum[4] = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < per_coord_point_info_size; ++i) {
                psum[0] += expf(-1 * (max_tx1 - tx1[i]));
                psum[1] += expf(-1 * (max_ty1 - ty1[i]));
                psum[2] += expf(-1 * (max_tx2 - tx2[i]));
                psum[3] += expf(-1 * (max_ty2 - ty2[i]));
            }

            // 做 softmax
            float coord_softmax[4][16];
            for (int i = 0; i < per_coord_point_info_size; ++i) {
                coord_softmax[0][i] = expf(-1 * (max_tx1 - tx1[i])) / psum[0];
                coord_softmax[1][i] = expf(-1 * (max_ty1 - ty1[i])) / psum[1];
                coord_softmax[2][i] = expf(-1 * (max_tx2 - tx2[i])) / psum[2];
                coord_softmax[3][i] = expf(-1 * (max_ty2 - ty2[i])) / psum[3];
            }

            // 和 16 个数做累乘
            float bx1 = 0, by1 = 0, bx2 = 0, by2 = 0;
            for (int i = 0; i < per_coord_point_info_size; ++i) {
                bx1 += coord_softmax[0][i] * i;
                by1 += coord_softmax[1][i] * i;
                bx2 += coord_softmax[2][i] * i;
                by2 += coord_softmax[3][i] * i;
            }

            // convert box from cxcywh to xyxy, and range in [0, 1]
            float x_min = (w_i + 0.5f - bx1) * scale_w;
            float y_min = (h_i + 0.5f - by1) * scale_h;
            float x_max = (w_i + 0.5f + bx2) * scale_w;
            float y_max = (h_i + 0.5f + by2) * scale_h;

            alternative_box_ptr[num_potential_boxes].box_info.batch_id = 0;
            alternative_box_ptr[num_potential_boxes].box_info.cls_id = (float) cls_idx;
            alternative_box_ptr[num_potential_boxes].box_info.score = score_;
            alternative_box_ptr[num_potential_boxes].box_info.x_min = x_min / img_w;
            alternative_box_ptr[num_potential_boxes].box_info.y_min = y_min / img_h;
            alternative_box_ptr[num_potential_boxes].box_info.x_max = x_max / img_w;
            alternative_box_ptr[num_potential_boxes].box_info.y_max = y_max / img_h;


            float *src_ptr = (float *)in_mask_ptr + h_i * cur_fmap_w * 32 + w_i * 32;
            float *dst_ptr = alternative_box_ptr[num_potential_boxes].mask;
            memcpy(dst_ptr, src_ptr, 32 * sizeof(float));

            num_potential_boxes++;
        }
    }

    *box_num = num_potential_boxes;

    return 0;
}


