#ifndef __NN_COMMON_CPP_H__
#define __NN_COMMON_CPP_H__

#include "nn_common.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include "vector"
#include <unordered_map>
#include <cstring>
#include <functional>
#include "algorithm"
#include "cctype"

#include <functional>
#include <algorithm>

#include <iostream>
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"

int32_t load_bin(const char *filename, const int64_t size, char *buf) {
    FILE *file_p = NULL;

    file_p = fopen(filename, "r");
    if (file_p == NULL) {
        printf("cant open the input bin\n");
    }
    size_t bytes_read = fread(buf, sizeof(char), size, file_p);
    fclose(file_p);

    return 0;
}

std::vector<std::string> split(const std::string& s)
{
    std::vector<std::string> words;
    std::istringstream stream(s);

    std::string word;
    while (stream >> word)
    {
        words.push_back(word);
    }

    return words;
}

typedef enum {
    RGB = 0,
    BGR = 1,
    YUV_NV12 = 2,
    YUV_NV21 = 3,
    YUV420P = 4,
} COLOR_CODE_E;

typedef struct {
    int32_t resize_size[2];
    int32_t crop_size[2];
    COLOR_CODE_E out_color_code;
    float mean[3];
    float std[3];
} TRANSFORMS_CONFIG_S;

int transforms(std::vector<float> &rgb, std::string img_path, TRANSFORMS_CONFIG_S &trans_cfg) {
    // step 0 : read img
    cv::Mat ori_img = cv::imread(img_path);
    if (ori_img.empty()) {
        LOG_ERR("open %s is failed, please check the img path.", img_path.c_str());
        return -1;
    }

    int16_t resize_h = trans_cfg.resize_size[0];
    int16_t resize_w = trans_cfg.resize_size[1];

    int16_t crop_h = trans_cfg.crop_size[0];
    int16_t crop_w = trans_cfg.crop_size[1];

    // step 1 : resize img
    cv::Mat resized_img;
    // 注意，最后一个参数是 resize 参数，要选 1 = INTER_LINEAR
    cv::resize(ori_img, resized_img, cv::Size(resize_w, resize_h), 0, 0, 1);

    int16_t crop_x = (resize_w - crop_w) >> 1;
    int16_t crop_y = (resize_h - crop_h) >> 1;

    // step 2 : crop img
    cv::Rect roi(crop_x, crop_y, crop_w, crop_h);
    if (roi.x + roi.width > resized_img.cols || roi.y + roi.height > resized_img.rows) {
        std::cout << "cropping area beyond image boundaries." << std::endl;
        return -1;
    }
    cv::Mat cropped_img = resized_img(roi);

    cv::Mat rgb_img;
    cv::cvtColor(cropped_img, rgb_img, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> rgb_channels(3);
    cv::split(rgb_img, rgb_channels);
    for (auto i = 0; i < rgb_channels.size(); i++) {
        // 转换为浮点型
        rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1);
        // 减去均值
        rgb_channels[i] -= trans_cfg.mean[i] * 255.0f;
        // 除以标准差
        rgb_channels[i] /= trans_cfg.std[i];
        // 转换为 0 ~ 1
        rgb_channels[i] /= 255.0f;
    }

    // 将处理后的数据复制到一维数组中
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < crop_w * crop_h; ++j) {
            rgb[i * crop_w * crop_h + j] = rgb_channels[i].at<float>(j);
        }
    }

}

int launch_post_process(std::vector<BUFFER_INFO_S> &params, std::vector<BUFFER_INFO_S> &inputs, std::vector<BUFFER_INFO_S> &outputs){
    Manager &m = Manager::getInstance();

    char* op_cfg_ptr = (char*)(params[0].addr);

    creator_ creator_op_method = m.Opmap[std::string(op_cfg_ptr)];

    std::shared_ptr<op> op_ptr;

    creator_op_method(op_ptr, op_cfg_ptr);

    op_ptr->prepare(op_cfg_ptr);
    op_ptr->forward(&params[0], &inputs[0], &outputs[0]);


    return 0;
}

template <class T>
std::vector<T> str2number(const std::string& str)
{
    std::string tmp_string;
    for (auto& c : str) {
        if (c != '[' && c != ']') {
            tmp_string += c;
        }
    }

    std::vector<T> vec;
    std::stringstream str_s(tmp_string);
    std::string token;
    while (std::getline(str_s, token, ',') || std::getline(str_s, token, ';')) {
        std::stringstream val;
        val << token;
        T v;
        val >> v;
        vec.push_back(v);
    }
    return vec;
}

std::string str2lower_str(const std::string& str)
{
    std::string result = str;
    transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

int set_default_args(std::unordered_map<std::string, std::string>& cfg_info_map,
                     const std::string key, const std::string value){
    if (cfg_info_map[key].empty())
    {
        cfg_info_map[key] = value;
    }
    return 1;
}

std::vector<std::string> get_string_vec(std::string src_string) {
    // 以 “;” 分割字符串
    std::vector<std::string> tokens;
    std::istringstream iss(src_string);
    std::string token;
    while (std::getline(iss, token, ';')) {
        tokens.push_back(token);
    }

    return tokens;
}

int yml2map(std::unordered_map<std::string, std::string>& cfg_info_map, const std::string file_name)
{
    // step 1: open yml cfg
    std::ifstream file(file_name);

    if (!file.is_open()) {
        std::cerr << "Unable to open file";
        return 1;
    }

    // step 2: read yml cfg to cfg_info_map
    std::string line;
    while (std::getline(file, line)) {
        // 跳过以 # 或 - 开头的行
        if (line.empty() || line[0] == '#' || line[0] == '-') {
            continue;
        }

        // 使用 : 作为分隔符
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);  // 跳过 : 字符
            value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());

            // 存储到 unordered_map 中
            cfg_info_map[key] = value;
        }
    }

    // step 4: is the value of the corresponding key is empty, set default cfg
    set_default_args(cfg_info_map, "topk", "5");
    set_default_args(cfg_info_map, "ifmap_num", "3");
    set_default_args(cfg_info_map, "anchor_num", "3");
    set_default_args(cfg_info_map, "anchor_scale_table",
                     "[116, 90, 156, 198, 373, 326], [30, 61, 62, 5, 59, 119], [10, 13, 16, 30, 33, 23]");
    set_default_args(cfg_info_map, "cls_num", "80");
    set_default_args(cfg_info_map, "max_boxes_per_class", "60");
    set_default_args(cfg_info_map, "max_boxes_per_batch", "200");
    set_default_args(cfg_info_map, "score_threshold", "0.5f");
    set_default_args(cfg_info_map, "iou_threshold", "0.4f");

    file.close();
    return 0;
}


// 自定义比较函数，使用 op type 用于比较两个 NODE_INFO_S
bool compareNodeWithType(NODE_INFO_S node_a, NODE_INFO_S node_b) {
    return std::strcmp(node_a.op_type, node_b.op_type) < 0;
}


#endif // __NN_COMMON_CPP_H__
