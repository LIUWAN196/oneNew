#include "object_detect.h"
#include <stdlib.h>
#include <stdio.h>
#include "../../x86_utils/nms_impl.h"

int do_yolo_v3_v5_v7_obj_detect(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);
int do_yolo_v8_world_obj_detect(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);
int do_yolo_v10_obj_detect(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);
int do_rt_detr_obj_detect(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs);

int get_output_info(OBJ_DETECT_OUT_INFO_S *out_info_ptr, int32_t potential_boxes_cnt, int32_t *tmp_cls_id_ptr,
                    float *tmp_score_ptr, float *tmp_box_ptr, OBJECT_DETECT_CONFIG_S *cfg);

int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {

    OBJECT_DETECT_CONFIG_S *cfg = (OBJECT_DETECT_CONFIG_S *) (params[0].addr);

    if (cfg->net_type == YOLO_V3 || cfg->net_type == YOLO_V5 || cfg->net_type == YOLO_V7) {
        // 这个分支的输入 shape 为 [boxes_num,boxes_info]，例如 [25200，85],这里的 boxes_info 包括了背景置信度
        OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
        int32_t shape_1 = in0_tensor->shapes[1];
        const int32_t boxes_coord_len = 4;      // x_center、y_center、w、h
        const int32_t bg_conf_len = 1;          // background confidence
        const int32_t cls_num = cfg->cls_num;
        if (shape_1 != boxes_coord_len + bg_conf_len + cls_num) {
            LOG_ERR("sorry, 传入到 object detect 算子 (cfg->net_type == %d) 的输入 shape[1] = %d 和 cls_num = %d "
                    "无法对应，他们的关系必须是 shape[1] = 4 + 1 + cls_num。\n", cfg->net_type, shape_1, cls_num);
        }
        do_yolo_v3_v5_v7_obj_detect(params, inputs, outputs);
    } else if (cfg->net_type == YOLO_V8 || cfg->net_type == YOLO_WORLD) {
        // 这个分支的输入 shape 为 [boxes_info,boxes_num]，例如 [84,8400],这里的 boxes_info 不包括背景置信度
        OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
        int32_t shape_0 = in0_tensor->shapes[0];
        const int32_t boxes_coord_len = 4;
        const int32_t cls_num = cfg->cls_num;
        if (shape_0 != boxes_coord_len + cls_num) {
            LOG_ERR("sorry, 传入到 object detect 算子 (cfg->net_type == %d) 的输入 shape[0] = %d 和 cls_num = %d "
                    "无法对应，他们的关系必须是 shape[1] = 4 + cls_num。\n", cfg->net_type, shape_0, cls_num);
        }
        do_yolo_v8_world_obj_detect(params, inputs, outputs);
    }  else if (cfg->net_type == YOLO_V10) {
        // 这个分支的输入 shape 为 [boxes_num,boxes_info]，例如 [8400，84],这里的 boxes_info 不包括背景置信度
        OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
        int32_t shape_1 = in0_tensor->shapes[1];
        const int32_t boxes_coord_len = 4;
        const int32_t cls_num = cfg->cls_num;
        if (shape_1 != boxes_coord_len + cls_num) {
            LOG_ERR("sorry, 传入到 object detect 算子 (cfg->net_type == %d) 的输入 shape[1] = %d 和 cls_num = %d "
                    "无法对应，他们的关系必须是 shape[1] = 4 + cls_num。\n", cfg->net_type, shape_1, cls_num);
        }
        do_yolo_v10_obj_detect(params, inputs, outputs);
    } else if (cfg->net_type == RT_DETR) {
        // 这个分支的输入 shape 为 [boxes_num,boxes_info]，例如 [8400，84],这里的 boxes_info 不包括背景置信度
        OPERAND_S *in0_tensor = (OPERAND_S *) (params[1].addr);
        int32_t shape_1 = in0_tensor->shapes[1];
        const int32_t boxes_coord_len = 4;
        const int32_t cls_num = cfg->cls_num;
        if (shape_1 != boxes_coord_len + cls_num) {
            LOG_ERR("sorry, 传入到 object detect 算子 (cfg->net_type == %d) 的输入 shape[1] = %d 和 cls_num = %d "
                    "无法对应，他们的关系必须是 shape[1] = 4 + cls_num。\n", cfg->net_type, shape_1, cls_num);
        }
        do_rt_detr_obj_detect(params, inputs, outputs);
    } else {
        LOG_ERR("current, object_detect op support YOLO_V3、YOLO_V5、YOLO_V7、YOLO_V8、YOLO_V10、YOLO_WORLD、RT_DETR, "
                "the enum you set is %d\n", cfg->net_type);
    }

    return 0;
}

int do_yolo_v3_v5_v7_obj_detect(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
    OBJECT_DETECT_CONFIG_S *cfg = (OBJECT_DETECT_CONFIG_S *) (params[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    const int32_t cls_num = cfg->cls_num;
    const float score_threshold = cfg->score_threshold;

    int32_t boxes_num = in_tensor->shapes[0];
    int32_t shape_1 = in_tensor->shapes[1];

    // 将临时的中间数据存放到临时数组中
    int32_t tmp_cls_id_ptr[POTENTIAL_BOXES_NUM];
    float tmp_score_ptr[POTENTIAL_BOXES_NUM];
    float tmp_box_ptr[POTENTIAL_BOXES_NUM];

    int32_t potential_boxes_cnt = 0;
    // step 1: 通过背景置信度来滤除大部分的 boxes，并且再统计保留下来的 boxes 的最大得分类别，再次滤除
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
                tmp_box_ptr[potential_boxes_cnt * 4 + 0] =
                        input_ptr[box_i * shape_1 + 0] - input_ptr[box_i * shape_1 + 2] / 2;
                tmp_box_ptr[potential_boxes_cnt * 4 + 1] =
                        input_ptr[box_i * shape_1 + 1] - input_ptr[box_i * shape_1 + 3] / 2;
                tmp_box_ptr[potential_boxes_cnt * 4 + 2] =
                        input_ptr[box_i * shape_1 + 0] + input_ptr[box_i * shape_1 + 2] / 2;
                tmp_box_ptr[potential_boxes_cnt * 4 + 3] =
                        input_ptr[box_i * shape_1 + 1] + input_ptr[box_i * shape_1 + 3] / 2;
                potential_boxes_cnt++;
            }
        }
        if (potential_boxes_cnt >= POTENTIAL_BOXES_NUM) {
            break;
        }
    }

    // step 2: 做 nms，获取到待保留的 boxes 的索引，并从 tmp_***_ptr 中获取到最终的输出信息
    OBJ_DETECT_OUT_INFO_S *out_info_ptr = (OBJ_DETECT_OUT_INFO_S *) output_ptr;
    get_output_info(out_info_ptr, potential_boxes_cnt, tmp_cls_id_ptr, tmp_score_ptr, tmp_box_ptr, cfg);

    return 0;
}

int do_yolo_v8_world_obj_detect(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
    OBJECT_DETECT_CONFIG_S *cfg = (OBJECT_DETECT_CONFIG_S *) (params[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    const int32_t cls_num = cfg->cls_num;
    const float score_threshold = cfg->score_threshold;

    int32_t boxes_info_len = in_tensor->shapes[0];  // such as 84
    int32_t boxes_num = in_tensor->shapes[1];       // such as 8400

    // 将临时的中间数据存放到临时数组中
    int32_t tmp_cls_id_ptr[POTENTIAL_BOXES_NUM];
    float tmp_score_ptr[POTENTIAL_BOXES_NUM];
    float tmp_box_ptr[POTENTIAL_BOXES_NUM];

    int32_t potential_boxes_cnt = 0;
    // step 1: 通过 boxes 的最大得分类别，滤除掉大部分的 boxes
    for (int box_i = 0; box_i < boxes_num; ++box_i) {
        // 寻找该 box 的最大得分类别和分数
        float cls_score = -32768.0f;
        int cls_idx = 0;
        for (int cls_i = 0; cls_i < cls_num; ++cls_i) {
            if (input_ptr[(4 + cls_i) * boxes_num + box_i] > cls_score) {
                cls_score = input_ptr[(4 + cls_i) * boxes_num + box_i];
                cls_idx = cls_i;
            }
        }
        if (cls_score >= score_threshold)   // 滤除，只保留真实 > score_threshold 的 boxes
        {
            tmp_cls_id_ptr[potential_boxes_cnt] = cls_idx;
            tmp_score_ptr[potential_boxes_cnt] = cls_score;
            // boxes 坐标为 x_min、y_min、x_max、y_max
            tmp_box_ptr[potential_boxes_cnt * 4 + 0] = input_ptr[0 * boxes_num + box_i];
            tmp_box_ptr[potential_boxes_cnt * 4 + 1] = input_ptr[1 * boxes_num + box_i];
            tmp_box_ptr[potential_boxes_cnt * 4 + 2] = input_ptr[2 * boxes_num + box_i];
            tmp_box_ptr[potential_boxes_cnt * 4 + 3] = input_ptr[3 * boxes_num + box_i];
            potential_boxes_cnt++;
        }
        if (potential_boxes_cnt >= POTENTIAL_BOXES_NUM) {
            break;
        }
    }

    // step 2: 做 nms，获取到待保留的 boxes 的索引，并从 tmp_***_ptr 中获取到最终的输出信息
    OBJ_DETECT_OUT_INFO_S *out_info_ptr = (OBJ_DETECT_OUT_INFO_S *) output_ptr;
    get_output_info(out_info_ptr, potential_boxes_cnt, tmp_cls_id_ptr, tmp_score_ptr, tmp_box_ptr, cfg);

    return 0;
}

int do_yolo_v10_obj_detect(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
    OBJECT_DETECT_CONFIG_S *cfg = (OBJECT_DETECT_CONFIG_S *) (params[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    const int32_t cls_num = cfg->cls_num;
    const float score_threshold = cfg->score_threshold;

    int32_t shape_1 = in_tensor->shapes[1];

    // 将临时的中间数据存放到临时数组中
    int32_t tmp_cls_id_ptr[POTENTIAL_BOXES_NUM];
    float tmp_score_ptr[POTENTIAL_BOXES_NUM];
    float tmp_box_ptr[POTENTIAL_BOXES_NUM];

    int32_t potential_boxes_cnt = 0;
    // step 1: 通过 boxes 的最大得分类别，滤除掉大部分的 boxes
    int32_t boxes_num = in_tensor->shapes[0];
    for (int box_i = 0; box_i < boxes_num; ++box_i) {
        // 寻找该 box 的最大得分类别和分数
        float cls_score = -32768.0f;
        int cls_idx = 0;
        for (int cls_i = 0; cls_i < cls_num; ++cls_i) {
            if (input_ptr[box_i * shape_1 + 4 + cls_i] > cls_score) {
                cls_score = input_ptr[box_i * shape_1 + 4 + cls_i];
                cls_idx = cls_i;
            }
        }
        if (cls_score >= score_threshold)   // 滤除，只保留真实 > score_threshold 的 boxes
        {
            tmp_cls_id_ptr[potential_boxes_cnt] = cls_idx;
            tmp_score_ptr[potential_boxes_cnt] = cls_score;
            // boxes 坐标为 x_min、y_min、x_max、y_max
            tmp_box_ptr[potential_boxes_cnt * 4 + 0] = input_ptr[box_i * shape_1 + 0];
            tmp_box_ptr[potential_boxes_cnt * 4 + 1] = input_ptr[box_i * shape_1 + 1];
            tmp_box_ptr[potential_boxes_cnt * 4 + 2] = input_ptr[box_i * shape_1 + 2];
            tmp_box_ptr[potential_boxes_cnt * 4 + 3] = input_ptr[box_i * shape_1 + 3];
            potential_boxes_cnt++;
        }
        if (potential_boxes_cnt >= POTENTIAL_BOXES_NUM) {
            break;
        }
    }

    // step 2: 做 nms，获取到待保留的 boxes 的索引，并从 tmp_***_ptr 中获取到最终的输出信息
    OBJ_DETECT_OUT_INFO_S *out_info_ptr = (OBJ_DETECT_OUT_INFO_S *) output_ptr;
    get_output_info(out_info_ptr, potential_boxes_cnt, tmp_cls_id_ptr, tmp_score_ptr, tmp_box_ptr, cfg);

    return 0;
}

int do_rt_detr_obj_detect(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
    OBJECT_DETECT_CONFIG_S *cfg = (OBJECT_DETECT_CONFIG_S *) (params[0].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);

    float *input_ptr = (float *) (inputs[0].addr);
    float *output_ptr = (float *) (outputs[0].addr);

    const int32_t cls_num = cfg->cls_num;
    const float score_threshold = cfg->score_threshold;

    int32_t shape_1 = in_tensor->shapes[1];

    // 将临时的中间数据存放到临时数组中
    int32_t tmp_cls_id_ptr[POTENTIAL_BOXES_NUM];
    float tmp_score_ptr[POTENTIAL_BOXES_NUM];
    float tmp_box_ptr[POTENTIAL_BOXES_NUM];

    int32_t potential_boxes_cnt = 0;
    // step 1: 通过 boxes 的最大得分类别，滤除掉大部分的 boxes
    int32_t boxes_num = in_tensor->shapes[0];
    for (int box_i = 0; box_i < boxes_num; ++box_i) {
        // 寻找该 box 的最大得分类别和分数
        float cls_score = -32768.0f;
        int cls_idx = 0;
        for (int cls_i = 0; cls_i < cls_num; ++cls_i) {
            if (input_ptr[box_i * shape_1 + 4 + cls_i] > cls_score) {
                cls_score = input_ptr[box_i * shape_1 + 4 + cls_i];
                cls_idx = cls_i;
            }
        }
        if (cls_score >= score_threshold)   // 滤除，只保留真实 > score_threshold 的 boxes
        {
            tmp_cls_id_ptr[potential_boxes_cnt] = cls_idx;
            tmp_score_ptr[potential_boxes_cnt] = cls_score;

            tmp_box_ptr[potential_boxes_cnt * 4 + 0] = cfg->img_w * (input_ptr[box_i * shape_1 + 0] - input_ptr[box_i * shape_1 + 2] / 2);
            tmp_box_ptr[potential_boxes_cnt * 4 + 1] = cfg->img_h * (input_ptr[box_i * shape_1 + 1] - input_ptr[box_i * shape_1 + 3] / 2);
            tmp_box_ptr[potential_boxes_cnt * 4 + 2] = cfg->img_w * (input_ptr[box_i * shape_1 + 0] + input_ptr[box_i * shape_1 + 2] / 2);
            tmp_box_ptr[potential_boxes_cnt * 4 + 3] = cfg->img_h * (input_ptr[box_i * shape_1 + 1] + input_ptr[box_i * shape_1 + 3] / 2);
            potential_boxes_cnt++;
        }
        if (potential_boxes_cnt >= POTENTIAL_BOXES_NUM) {
            break;
        }
    }

    // step 2: 做 nms，获取到待保留的 boxes 的索引，并从 tmp_***_ptr 中获取到最终的输出信息
    OBJ_DETECT_OUT_INFO_S *out_info_ptr = (OBJ_DETECT_OUT_INFO_S *) output_ptr;
    get_output_info(out_info_ptr, potential_boxes_cnt, tmp_cls_id_ptr, tmp_score_ptr, tmp_box_ptr, cfg);

    return 0;
}

int get_output_info(OBJ_DETECT_OUT_INFO_S *out_info_ptr, int32_t potential_boxes_cnt, int32_t *tmp_cls_id_ptr,
                    float *tmp_score_ptr, float *tmp_box_ptr, OBJECT_DETECT_CONFIG_S *cfg) {
    int32_t keep_box_idx_ptr[POTENTIAL_BOXES_NUM];

    // step 1: 做 nms
    int32_t keep_boxes_cnt;
    NMS_CONFIG_S nms_cfg;
    nms_cfg.img_h = cfg->img_h;
    nms_cfg.img_w = cfg->img_w;
    nms_cfg.score_threshold = cfg->score_threshold;
    nms_cfg.iou_threshold = cfg->iou_threshold;
    keep_boxes_cnt = nms(keep_box_idx_ptr, potential_boxes_cnt, tmp_cls_id_ptr, tmp_score_ptr, tmp_box_ptr, &nms_cfg);

    // step 2: 通过 step 2 获取到的索引，从 tmp_cls_id_ptr, tmp_score_ptr, tmp_box_ptr 中获取到最终的输出信息
    for (int i = 0; i < keep_boxes_cnt; ++i) {
        out_info_ptr[i].cls_id = tmp_cls_id_ptr[keep_box_idx_ptr[i]];
        out_info_ptr[i].score = tmp_score_ptr[keep_box_idx_ptr[i]];
        out_info_ptr[i].box_info.x_min = tmp_box_ptr[keep_box_idx_ptr[i] * 4 + 0];
        out_info_ptr[i].box_info.y_min = tmp_box_ptr[keep_box_idx_ptr[i] * 4 + 1];
        out_info_ptr[i].box_info.x_max = tmp_box_ptr[keep_box_idx_ptr[i] * 4 + 2];
        out_info_ptr[i].box_info.y_max = tmp_box_ptr[keep_box_idx_ptr[i] * 4 + 3];
    }
    out_info_ptr[keep_boxes_cnt].cls_id = -1;   // 作为终止符

    return 0;
}


