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

//static int32_t fused_cfg_size_vec[] = {0, sizeof(CONV_CONFIG_S)};


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
    if (cfg_info_map["key"].empty())
    {
        cfg_info_map["key"] = value;
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

//    // step 3: parameter check
//    std::string a = str2lower_str(cfg_info_map["do_preprocess4img"]);
//    if (str2lower_str(cfg_info_map["do_preprocess4img"]) != "yes" &&
//        str2lower_str(cfg_info_map["do_preprocess4img"]) != "no") {
//        std::cerr << "the args: do_preprocess4img must be set, and the value must be yes or no" << std::endl;
//        return -1;
//    }

    if (str2lower_str(cfg_info_map["do_postprocess4net"]) != "yes" &&
        str2lower_str(cfg_info_map["do_postprocess4net"]) != "no") {
        std::cerr << "the args: do_postprocess4net must be set, and the value must be yes or no" << std::endl;
        return -1;

    }

//    if (str2lower_str(cfg_info_map["dump_output4each_node"]) != "yes" &&
//        str2lower_str(cfg_info_map["dump_output4each_node"]) != "no") {
//        std::cerr << "the args: dump_output4each_node must be set, and the value must be yes or no" << std::endl;
//        return -1;
//    }

//    if (cfg_info_map["input_path"].empty()) {
//        std::cerr << "the args: input_path must be set, and the value are the absolute path" << std::endl;
//        return -1;
//    }

    if (str2lower_str(cfg_info_map["do_postprocess4net"]) == "yes") {
        if (str2lower_str(cfg_info_map["postprocess_type"]) != "detection" &&
            str2lower_str(cfg_info_map["postprocess_type"]) != "classification"  &&
            str2lower_str(cfg_info_map["postprocess_type"]) != "pose_detection"  &&
            str2lower_str(cfg_info_map["postprocess_type"]) != "segmentation") {
            std::cerr << "you have set do_postprocess4net to yes, so the args: postprocess_type must be set, "
                         "and the value must be detection、 classification、 pose_detection、 segmentation" << std::endl;
            return -1;

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
