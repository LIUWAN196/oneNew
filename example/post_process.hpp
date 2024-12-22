//
// Created by wanzai on 24-11-11.
//

#ifndef ONENEW_POSTPROCESS_H
#define ONENEW_POSTPROCESS_H

#include <cstring>
#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include "ops_head.h"
#include "net.h"
#include <dirent.h>

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <utility>
#include "math.h"

#include <utility>

class PostProcessPerform
{
public:
    std::unordered_map<std::string, std::string> cfg_info_map;
    std::unordered_map<std::string, BUFFER_INFO_S> io_buf_map;
    std::string net_in_operand;
    std::vector<OPERAND_S> in_tensor;
    std::vector<OPERAND_S> out_tensor;
    std::vector<BUFFER_INFO_S> params;
    std::vector<BUFFER_INFO_S> inputs;
    std::vector<BUFFER_INFO_S> outputs;
    PostProcessPerform(std::unordered_map<std::string, std::string> _cfg_info_map,
                       std::unordered_map<std::string, BUFFER_INFO_S> _io_buf_map, std::string _net_in_operand)
                : cfg_info_map(std::move(_cfg_info_map)), io_buf_map(std::move(_io_buf_map)),
                net_in_operand(std::move(_net_in_operand))
    {
        in_tensor.resize(BUF_MAXNUM);
        out_tensor.resize(BUF_MAXNUM);
    };

    virtual int fill_cfg() = 0;
    virtual int fill_tensor_desc() = 0;

    virtual int fill_inputs_buf() {
        BUFFER_INFO_S in_info;
        for (const auto& io_buf:io_buf_map) {
            if (io_buf.first != net_in_operand) {
                in_info.addr = (int64_t) (io_buf.second.addr);
                inputs.push_back(in_info);
            }
        }

        return 0;
    }

    virtual int prepare_outputs_buf() = 0;
    int launch_op() {
        // do object detect
        launch_post_process(params, inputs, outputs);
    }
    virtual int show_result() = 0;

    virtual int do_post_process() {
        fill_cfg();
        fill_tensor_desc();
        fill_inputs_buf();
        prepare_outputs_buf();
        launch_op();
        show_result();
    }

    virtual ~PostProcessPerform() = default;

};

class ClassifyPerform : public PostProcessPerform
{
public:
    int32_t cls_num{};
    ARGMAX_CONFIG_S argmax_cfg{};
    std::vector<int32_t> argmax_out_info;

    ClassifyPerform(std::unordered_map<std::string, std::string> _cfg_info_map,
                        std::unordered_map<std::string, BUFFER_INFO_S> _io_buf_map, std::string _net_in_operand)
            : PostProcessPerform(std::move(_cfg_info_map), std::move(_io_buf_map), std::move(_net_in_operand)) {
    };

    int fill_cfg() override {
        std::string op_type = "ArgMax";
        strcpy(argmax_cfg.op_base_cfg.op_type, op_type.c_str());
        argmax_cfg.topk = std::stoi(cfg_info_map["topk"]);

        // fill params
        BUFFER_INFO_S cfg;
        cfg.addr = (int64_t) (&argmax_cfg);
        params.push_back(cfg);

        return 0;
    }

    int fill_tensor_desc() override {
        // 确定该模型分类的类别数量
        for (const auto& io_buf:io_buf_map) {
            if (io_buf.first != net_in_operand) {
                cls_num = io_buf.second.elem_size;
                break;
            }
        }

        // 填充输入 tensor 的描述
        for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
            in_tensor[0].shapes[dim_i] = 1;
        }
        in_tensor[0].shapes[1] = cls_num;    // the dim1 is C

        // fill in tensor desc
        BUFFER_INFO_S in_tensor_desc;
        in_tensor_desc.addr = (int64_t) (&in_tensor[0]);
        params.push_back(in_tensor_desc);

        return 0;
    }

    int prepare_outputs_buf() override {
        argmax_out_info.resize(cls_num);
        BUFFER_INFO_S out_info;
        out_info.addr = (int64_t) (&argmax_out_info[0]);
        outputs.push_back(out_info);

        return 0;
    }

    int show_result() override {
        printf("top k cls_label is: ");
        for (int i = 0; i < argmax_cfg.topk; ++i) {
            printf("%d  ", argmax_out_info[i]);
        }
        printf("\n");

        return 0;
    }

    ~ClassifyPerform () override {
        ;
    }

};

class ObjectDetectPerform : public PostProcessPerform
{
public:
    OBJECT_DETECT_CONFIG_S object_detect_cfg{};
    std::vector<OBJ_DETECT_OUT_INFO_S> detect_out_info;

    ObjectDetectPerform(std::unordered_map<std::string, std::string> _cfg_info_map,
                        std::unordered_map<std::string, BUFFER_INFO_S> _io_buf_map, std::string _net_in_operand)
                    : PostProcessPerform(std::move(_cfg_info_map), std::move(_io_buf_map), std::move(_net_in_operand)) {
    };

    int fill_cfg() override {
        std::string op_type = "ObjectDetect";
        strcpy(object_detect_cfg.op_base_cfg.op_type, op_type.c_str());
        object_detect_cfg.net_type = NET_MAP[str2lower_str(cfg_info_map["net_type"])];
        object_detect_cfg.cls_num = std::stoi(cfg_info_map["cls_num"]);
        std::vector<int> crop_shapes = str2number<int>(cfg_info_map["crop_shapes"]);
        object_detect_cfg.img_w = crop_shapes[0];
        object_detect_cfg.img_h = crop_shapes[1];
        object_detect_cfg.score_threshold = std::stof(cfg_info_map["score_threshold"]);
        object_detect_cfg.iou_threshold = std::stof(cfg_info_map["iou_threshold"]);

        // fill params
        BUFFER_INFO_S cfg;
        cfg.addr = (int64_t) (&object_detect_cfg);
        params.push_back(cfg);

        return 0;
    }

    int fill_tensor_desc() override {
        const int32_t feature_map_num = (object_detect_cfg.img_w / 8 * object_detect_cfg.img_w / 8)
                                        + (object_detect_cfg.img_w / 16 * object_detect_cfg.img_w / 16)
                                        + (object_detect_cfg.img_w / 32 * object_detect_cfg.img_w / 32);
        if (object_detect_cfg.net_type == YOLO_V3 || object_detect_cfg.net_type == YOLO_V5 ||
            object_detect_cfg.net_type == YOLO_V7) {
            const int32_t anchor_num = 3;
            in_tensor[0].shapes[0] = anchor_num * feature_map_num;
            in_tensor[0].shapes[1] = 4 + 1 + object_detect_cfg.cls_num;
        } else if (object_detect_cfg.net_type == RT_DETR) {
            in_tensor[0].shapes[0] = 300;  // rt detr model have max 300 boxes output
            in_tensor[0].shapes[1] = 4 + object_detect_cfg.cls_num;
        } else if (object_detect_cfg.net_type == YOLO_V10) {
            in_tensor[0].shapes[0] = feature_map_num;
            in_tensor[0].shapes[1] = 4 + object_detect_cfg.cls_num;
        } else if (object_detect_cfg.net_type == YOLO_V8 || object_detect_cfg.net_type == YOLO_WORLD) {
            in_tensor[0].shapes[0] = 4 + object_detect_cfg.cls_num;
            in_tensor[0].shapes[1] = feature_map_num;
        }

        // fill in tensor desc
        BUFFER_INFO_S in_tensor_desc;
        in_tensor_desc.addr = (int64_t) (&in_tensor[0]);
        params.push_back(in_tensor_desc);

        return 0;
    }

    int prepare_outputs_buf() override {
        const int32_t max_keep_box_num = 256;
        detect_out_info.resize(max_keep_box_num);
        BUFFER_INFO_S out_info;
        out_info.addr = (int64_t) (&detect_out_info[0]);
        outputs.push_back(out_info);

        return 0;
    }

    int show_result() override {
        // show img
        cv::Mat image = cv::imread(cfg_info_map["input_data_path"]);
        if (image.empty()) {
            LOG_ERR("Could not open or find the image");
            return -1;
        }

        int real_w = image.cols;
        int real_h = image.rows;
        std::string img_path = cfg_info_map["input_data_path"];
        std::vector<int> crop_shapes = str2number<int>(cfg_info_map["crop_shapes"]);
        float w_ratio = real_w * 1.0f / crop_shapes[0];
        float h_ratio = real_h * 1.0f / crop_shapes[1];
        int box_i = 0;
        auto* detect_out_info = reinterpret_cast<OBJ_DETECT_OUT_INFO_S*>(outputs[0].addr);
        while (detect_out_info[box_i].cls_id != -1) {
            BOX_INFO_S *cur_box = &detect_out_info[box_i].box_info;

            int box_x_min = (int)(cur_box->x_min * w_ratio);
            int box_y_min = (int)(cur_box->y_min * h_ratio);
            int box_w = (int)((cur_box->x_max - cur_box->x_min) * w_ratio);
            int box_h = (int)((cur_box->y_max - cur_box->y_min) * h_ratio);

            // 定义矩形框的坐标（左上角和右下角）
            cv::Rect boundingBox(box_x_min, box_y_min, box_w, box_h);

            // 定义类别 ID 和得分
            int classId = detect_out_info[box_i].cls_id;
            float score = detect_out_info[box_i].score;

            cv::Scalar color((135 + classId * 67) % 225, (116 + classId * 23) % 225, (190 + classId * 44) % 225); // BGR

            // 绘制矩形框
            cv::rectangle(image, boundingBox, color, 2);

            // 准备显示的文本
            std::ostringstream label;
            label << "Class: " << classId << ", Score: " << score;

            // 获取文本大小，用于计算文本框的位置
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.6;
            int thickness = 1;
            cv::Size textSize = cv::getTextSize(label.str(), fontFace, fontScale, thickness, nullptr);

            // 在矩形框的左上角绘制文本背景色
            cv::rectangle(image, cv::Point(boundingBox.x, boundingBox.y),
                          cv::Point(boundingBox.x + textSize.width, boundingBox.y - textSize.height - 10),
                          color, -1); // 填充矩形

            // 在矩形框的左上角添加文本
            cv::putText(image, label.str(),
                        cv::Point(boundingBox.x, boundingBox.y - 6), // 文本位置
                        fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);

            box_i++;
        }

        // 显示图片
        cv::Mat img_show;

        const int show_len = 960;
        float ratio = show_len * 1.0f / (image.rows > image.cols ? image.rows : image.cols);
        cv::resize(image, img_show, cv::Size((int)(ratio * image.cols), (int)(ratio * image.rows)), 0, 0, cv::INTER_LINEAR);

        cv::imshow("Image with Bounding Box and Label", img_show);
        cv::waitKey(0); // 按任意键关闭窗口

        return 0;
    }

    ~ObjectDetectPerform() override {
        ;
    }

};

class PoseDetectPerform : public PostProcessPerform
{
public:
    POSE_DETECT_CONFIG_S pose_detect_cfg{};
    std::vector<POSE_DETECT_OUT_INFO_S> pose_detect_out_info;

    PoseDetectPerform(std::unordered_map<std::string, std::string> _cfg_info_map,
                        std::unordered_map<std::string, BUFFER_INFO_S> _io_buf_map, std::string _net_in_operand)
            : PostProcessPerform(std::move(_cfg_info_map), std::move(_io_buf_map), std::move(_net_in_operand)) {
    };

    int fill_cfg() override {
        std::string op_type = "PoseDetect";
        strcpy(pose_detect_cfg.op_base_cfg.op_type, op_type.c_str());
        std::vector<int> crop_shapes = str2number<int>(cfg_info_map["crop_shapes"]);
        pose_detect_cfg.img_w = crop_shapes[0];
        pose_detect_cfg.img_h = crop_shapes[1];
        pose_detect_cfg.score_threshold = std::stof(cfg_info_map["score_threshold"]);
        pose_detect_cfg.iou_threshold = std::stof(cfg_info_map["iou_threshold"]);

        // fill params
        BUFFER_INFO_S cfg;
        cfg.addr = (int64_t) (&pose_detect_cfg);
        params.push_back(cfg);

        return 0;
    }

    int fill_tensor_desc() override {
        const int32_t feature_map_num = (pose_detect_cfg.img_w / 8 * pose_detect_cfg.img_w / 8)
                                        + (pose_detect_cfg.img_w / 16 * pose_detect_cfg.img_w / 16)
                                        + (pose_detect_cfg.img_w / 32 * pose_detect_cfg.img_w / 32);
        in_tensor[0].shapes[0] = 32 + 4 + 80;
        in_tensor[0].shapes[1] = feature_map_num;

        // fill in tensor desc
        BUFFER_INFO_S in_tensor_desc;
        in_tensor_desc.addr = (int64_t) (&in_tensor[0]);
        params.push_back(in_tensor_desc);

        return 0;
    }

    int prepare_outputs_buf() override {
        const int32_t max_keep_box_num = 256;
        pose_detect_out_info.resize(max_keep_box_num);
        BUFFER_INFO_S out_info;
        out_info.addr = (int64_t) (&pose_detect_out_info[0]);
        outputs.push_back(out_info);

        return 0;
    }

    int show_result() override {
        // show img
        cv::Mat image = cv::imread(cfg_info_map["input_data_path"]);
        if (image.empty()) {
            LOG_ERR("Could not open or find the image");
            return -1;
        }

        int real_w = image.cols;
        int real_h = image.rows;
        std::string img_path = cfg_info_map["input_data_path"];
        std::vector<int> crop_shapes = str2number<int>(cfg_info_map["crop_shapes"]);
        float w_ratio = real_w * 1.0f / crop_shapes[0];
        float h_ratio = real_h * 1.0f / crop_shapes[1];
        int box_i = 0;
        auto* detect_out_info = reinterpret_cast<POSE_DETECT_OUT_INFO_S*>(outputs[0].addr);
        while (detect_out_info[box_i].cls_id != -1) {
            BOX_INFO_S *cur_box = &detect_out_info[box_i].box_info;

            int box_x_min = (int)(cur_box->x_min * w_ratio);
            int box_y_min = (int)(cur_box->y_min * h_ratio);
            int box_w = (int)((cur_box->x_max - cur_box->x_min) * w_ratio);
            int box_h = (int)((cur_box->y_max - cur_box->y_min) * h_ratio);

            // 定义矩形框的坐标（左上角和右下角）
            cv::Rect boundingBox(box_x_min, box_y_min, box_w, box_h);

            // 定义类别 ID 和得分
            int classId = detect_out_info[box_i].cls_id;
            float score = detect_out_info[box_i].score;

            cv::Scalar color((135 + box_i * 67) % 225, (116 + box_i * 23) % 225, (190 + box_i * 44) % 225); // BGR

            // 绘制矩形框
            cv::rectangle(image, boundingBox, color, 2);

            // 绘制 17 个关键点
            {
                int radius = 3; // 小点的半径
                int thickness = -1; // 如果厚度为-1，则填充圆圈

                float *keypoints_ptr = detect_out_info[box_i].keypoints;
                const float hide_threshold = 0.7;
                for (int point_i = 0; point_i < 17; ++point_i) {
                    if (keypoints_ptr[point_i * 3 + 0] > hide_threshold) {
                        cv::Point point((int) (keypoints_ptr[point_i * 3 + 1] * w_ratio),
                                        (int) (keypoints_ptr[point_i * 3 + 2] * w_ratio)); // 小点中心位置，这里设置在图像中心
                        cv::circle(image, point, radius, color, thickness);
                    }
                }
            }

            // 准备显示的文本
            std::ostringstream label;
            label << "Class: " << classId << ", Score: " << score;

            // 获取文本大小，用于计算文本框的位置
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.6;
            int thickness = 1;
            cv::Size textSize = cv::getTextSize(label.str(), fontFace, fontScale, thickness, nullptr);

            // 在矩形框的左上角绘制文本背景色
            cv::rectangle(image, cv::Point(boundingBox.x, boundingBox.y),
                          cv::Point(boundingBox.x + textSize.width, boundingBox.y - textSize.height - 10),
                          color, -1); // 填充矩形

            // 在矩形框的左上角添加文本
            cv::putText(image, label.str(),
                        cv::Point(boundingBox.x, boundingBox.y - 6), // 文本位置
                        fontFace, fontScale, cv::Scalar(0,255,0), thickness);

            box_i++;
        }

        // 显示图片
        cv::Mat img_show;

        const int show_len = 960;
        float ratio = show_len * 1.0f / (image.rows > image.cols ? image.rows : image.cols);
        cv::resize(image, img_show, cv::Size((int)(ratio * image.cols), (int)(ratio * image.rows)), 0, 0, cv::INTER_LINEAR);

        cv::imshow("Image with Bounding Box and Label", img_show);
        cv::waitKey(0); // 按任意键关闭窗口

        return 0;
    }

    ~PoseDetectPerform() override {
        ;
    }

};

class SegmentPerform : public PostProcessPerform
{
public:
    SEGMENT_CONFIG_S segment_cfg{};
    std::vector<SEGMENT_OUT_INFO_S> segment_out_info;

    SegmentPerform(std::unordered_map<std::string, std::string> _cfg_info_map,
                      std::unordered_map<std::string, BUFFER_INFO_S> _io_buf_map, std::string _net_in_operand)
            : PostProcessPerform(std::move(_cfg_info_map), std::move(_io_buf_map), std::move(_net_in_operand)) {
    };

    int fill_cfg() override {
        std::string op_type = "Segment";
        strcpy(segment_cfg.op_base_cfg.op_type, op_type.c_str());
        std::vector<int> crop_shapes = str2number<int>(cfg_info_map["crop_shapes"]);
        segment_cfg.img_w = crop_shapes[0];
        segment_cfg.img_h = crop_shapes[1];
        segment_cfg.cls_num = std::stoi(cfg_info_map["cls_num"]);
        segment_cfg.score_threshold = std::stof(cfg_info_map["score_threshold"]);
        segment_cfg.iou_threshold = std::stof(cfg_info_map["iou_threshold"]);

        // fill params
        BUFFER_INFO_S cfg;
        cfg.addr = (int64_t) (&segment_cfg);
        params.push_back(cfg);

        return 0;
    }

    int fill_tensor_desc() override {
        const int32_t feature_map_num = (segment_cfg.img_w / 8 * segment_cfg.img_w / 8)
                                        + (segment_cfg.img_w / 16 * segment_cfg.img_w / 16)
                                        + (segment_cfg.img_w / 32 * segment_cfg.img_w / 32);
        in_tensor[0].shapes[0] = 4 + segment_cfg.cls_num + 32;
        in_tensor[0].shapes[1] = feature_map_num;

        // fill in tensor desc
        BUFFER_INFO_S in_tensor_desc;
        in_tensor_desc.addr = (int64_t) (&in_tensor[0]);
        params.push_back(in_tensor_desc);

        return 0;
    }

    int fill_inputs_buf() override {
        inputs.resize(2);
        BUFFER_INFO_S in_info;
        for (const auto& io_buf:io_buf_map) {
            if (io_buf.first != net_in_operand) {
                in_info.addr = (int64_t) (io_buf.second.addr);
                if (io_buf.second.elem_size == in_tensor[0].shapes[0] * in_tensor[0].shapes[1]) {
                    inputs[0] = in_info;
                } else {
                    inputs[1] = in_info;
                }
            }
        }

        return 0;
    }

    int prepare_outputs_buf() override {
        const int32_t max_keep_box_num = 64;
        segment_out_info.resize(max_keep_box_num);
        BUFFER_INFO_S out_info;
        out_info.addr = (int64_t) (&segment_out_info[0]);
        outputs.push_back(out_info);

        return 0;
    }

    int show_result() override {
        // show img
        cv::Mat image = cv::imread(cfg_info_map["input_data_path"]);
        if (image.empty()) {
            LOG_ERR("Could not open or find the image");
            return -1;
        }

        int real_w = image.cols;
        int real_h = image.rows;
        std::string img_path = cfg_info_map["input_data_path"];
        std::vector<int> crop_shapes = str2number<int>(cfg_info_map["crop_shapes"]);
        float w_ratio = real_w * 1.0f / crop_shapes[0];
        float h_ratio = real_h * 1.0f / crop_shapes[1];
        int box_i = 0;
        auto* segment_info = reinterpret_cast<SEGMENT_OUT_INFO_S*>(outputs[0].addr);
        while (segment_info[box_i].cls_id != -1) {
            BOX_INFO_S *cur_box = &segment_info[box_i].box_info;

            int box_x_min = (int)(cur_box->x_min * w_ratio);
            int box_y_min = (int)(cur_box->y_min * h_ratio);
            int box_x_max= (int)(cur_box->x_max * w_ratio);
            int box_y_max = (int)(cur_box->y_max * h_ratio);
            int box_w = (int)((cur_box->x_max - cur_box->x_min) * w_ratio);
            int box_h = (int)((cur_box->y_max - cur_box->y_min) * h_ratio);

            // 定义矩形框的坐标（左上角和右下角）
            cv::Rect boundingBox(box_x_min, box_y_min, box_w, box_h);

            // 定义类别 ID 和得分
            int classId = segment_info[box_i].cls_id;
            float score = segment_info[box_i].score;

            cv::Scalar mask_color_1((135 + classId * 56) % 225, (116 + classId * 87) % 225, (190 + classId * 123) % 225); // BGR

            // 绘制矩形框
            cv::rectangle(image, boundingBox, mask_color_1, 2);

            // 准备显示的文本
            std::ostringstream label;
            label << "Class: " << classId << ", Score: " << score;

            // 获取文本大小，用于计算文本框的位置
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.6;
            int thickness = 1;
            cv::Size textSize = cv::getTextSize(label.str(), fontFace, fontScale, thickness, nullptr);

            // 在矩形框的左上角绘制文本背景色
            cv::rectangle(image, cv::Point(boundingBox.x, boundingBox.y),
                          cv::Point(boundingBox.x + textSize.width, boundingBox.y - textSize.height - 10),
                          mask_color_1, -1); // 填充矩形

            // 在矩形框的左上角添加文本
            cv::putText(image, label.str(),
                        cv::Point(boundingBox.x, boundingBox.y - 6), // 文本位置
                        fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);

            // 为图像加 mask
            const int32_t mask_w = 160;
            const int32_t mask_h = 160;
            std::vector<uint8_t> mask(real_w * real_h, 0);
            for (int h_i = 0; h_i < real_h; ++h_i) {
                for (int w_i = 0; w_i < real_w; ++w_i) {
                    if (h_i >= box_y_min && h_i <= box_y_max && w_i >= box_x_min && w_i <= box_x_max) {
                        int32_t mask_w_coord = (int32_t)(1.0f * w_i / real_w * mask_w);
                        int32_t mask_h_coord = (int32_t)(1.0f * h_i / real_h * mask_h);
                        mask[h_i * real_w + w_i] = segment_info[box_i].mask[mask_h_coord * mask_w + mask_w_coord] > 0 ? 255 : 0;
                    }
                }
            }


            image = get_masked_img(image, mask, mask_color_1);
            box_i++;
        }

        // 显示图片
        cv::Mat img_show;

        const int show_len = 960;
        float ratio = show_len * 1.0f / (image.rows > image.cols ? image.rows : image.cols);
        cv::resize(image, img_show, cv::Size((int)(ratio * image.cols), (int)(ratio * image.rows)), 0, 0, cv::INTER_LINEAR);

        cv::imshow("Image with Bounding Box and Label", img_show);
        cv::waitKey(0); // 按任意键关闭窗口

        return 0;
    }

    ~SegmentPerform() override {
           }

};
#endif //ONENEW_POSTPROCESS_H
