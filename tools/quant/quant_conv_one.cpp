#include <iostream>
#include <fstream>
#include "cstring"
#include <time.h>
#include <vector>
#include "cmath"
#include <iostream>
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include <dlfcn.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include <sys/time.h>
#include <thread>
#include <pthread.h>
#include <functional>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "../manager/manager.h"
#include "../../common/nn_common_cpp.h"

#include "ops_head.h"
#include "net.h"

void quant_weight_bias(CONV_CONFIG_S *conv_cfg, float * weight_ptr, float* bias_ptr, int32_t * weight_shapes){

    int kernel_num = weight_shapes[0];
    int kernel_c = weight_shapes[1];
    int kernel_h = weight_shapes[2];
    int kernel_w = weight_shapes[3];

    const int inner = kernel_c * kernel_w * kernel_h;
    int32_t * biase_s32_ptr = (int32_t *)bias_ptr;
    for (int k_ni = 0; k_ni < kernel_num; ++k_ni) {
        float* cur_weight_ptr = weight_ptr + k_ni * inner;
        int8_t * cur_weight_s8_ptr = (int8_t *)weight_ptr + k_ni * inner;
        float cur_max_abs = 0;
        for (int i = 0; i < inner; ++i) {
            cur_max_abs = fabs(cur_weight_ptr[i]) > cur_max_abs ? fabs(cur_weight_ptr[i]) : cur_max_abs;
        }
        for (int i = 0; i < inner; ++i) {
            cur_weight_s8_ptr[i] = (int8_t)(cur_weight_ptr[i] * 127 / cur_max_abs);
        }
        biase_s32_ptr[k_ni] = (int32_t)(bias_ptr[k_ni] * 127 / cur_max_abs);
        conv_cfg->weight_aux[k_ni] = cur_max_abs;
    }

}

void quant_conv_weight_bias(char *one_buf_ptr)
{
    Manager &m = Manager::getInstance();

    int32_t *head_ptr = (int32_t *)one_buf_ptr;
    int32_t node_cnt = head_ptr[1];
    char *cur_node_cfg_ptr = (char *)(one_buf_ptr + head_ptr[2]);

    std::vector<OPERAND_S *> weight_operand_ptr_vec;
    std::vector<OPERAND_S *> bias_operand_ptr_vec;
    for (int32_t node_i = 0; node_i < node_cnt; node_i++)
    {
        std::string op_type_str(cur_node_cfg_ptr);
        if (op_type_str == "Conv")
        {
            CONV_CONFIG_S *conv_cfg = (CONV_CONFIG_S *)cur_node_cfg_ptr;

            // quant weight and bias
            int32_t init_cnt = head_ptr[3];
            char *cur_init_info_ptr = (char *)(one_buf_ptr + head_ptr[4]);
            char *weight_ptr, *bias_ptr;
            int32_t * weight_shapes;
            int weight_n, weight_c, weight_h, weight_w;
            for (int32_t init_i = 0; init_i < init_cnt; init_i++) {
                OPERAND_S *operand_ptr = (OPERAND_S *) cur_init_info_ptr;
                if (strcmp(operand_ptr->operand_name, conv_cfg->op_base_cfg.in_operand_name[1]) == 0) {
                    // this operand is weight
                    weight_ptr = (char *)(cur_init_info_ptr + sizeof(OPERAND_S));
                    weight_shapes = &operand_ptr->shapes[0];
                    weight_operand_ptr_vec.push_back(operand_ptr);
                } else if (strcmp(operand_ptr->operand_name, conv_cfg->op_base_cfg.in_operand_name[2]) == 0) {
                    // this operand is bias
                    bias_ptr = (char *)(cur_init_info_ptr + sizeof(OPERAND_S));
                    bias_operand_ptr_vec.push_back(operand_ptr);
                }
                // update cur_init_info_ptr
                cur_init_info_ptr += align_buf_size(sizeof(OPERAND_S) + operand_buf_size(operand_ptr));
            }
            quant_weight_bias(conv_cfg, (float *)weight_ptr, (float *)bias_ptr, weight_shapes);
        }
        // update cur_node_cfg_ptr
        cur_node_cfg_ptr += align_buf_size(m.op_cfg_size_map[op_type_str]);
    }

    return;
}


typedef enum {
    RGB = 0,
    BGR = 1,
    YUV_NV12 = 2,
    YUV_NV21 = 3,
    YUV420P = 4,
} COLOR_CODE_E;
typedef enum {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4,
} RESIZE_METHOD_E;
typedef struct {
    int32_t resize_size[2];
    int32_t crop_size[2];
    COLOR_CODE_E out_color_code;
    RESIZE_METHOD_E resize_method;
    float mean[3];
    float std[3];
} TRANSFORMS_CONFIG_S;



int transforms(std::vector<float> &rgb, std::string img_path, TRANSFORMS_CONFIG_S &trans_cfg)
{
    // step 0 : read img
    cv::Mat ori_img = cv::imread(img_path);
    if (ori_img.empty())
    {
        std::cout << "error: open " << img_path
                  << " is failed, please check the img path." << std::endl;
        return -1;
    }

    int16_t resize_h = trans_cfg.resize_size[0];
    int16_t resize_w = trans_cfg.resize_size[1];

    int16_t crop_h = trans_cfg.crop_size[0];
    int16_t crop_w = trans_cfg.crop_size[1];

    // step 1 : resize img
    cv::Mat resized_img;
    cv::resize(ori_img, resized_img, cv::Size(resize_h, resize_w), 0, 0, 0);

    int16_t crop_x = (resize_w - crop_w) >> 1;
    int16_t crop_y = (resize_h - crop_h) >> 1;

    // step 2 : crop img
    cv::Rect roi(crop_x, crop_y, crop_w, crop_h);
    if (roi.x + roi.width > resized_img.cols || roi.y + roi.height > resized_img.rows)
    {
        std::cout << "cropping area beyond image boundaries." << std::endl;
        return -1;
    }
    cv::Mat cropped_img = resized_img(roi);

    cv::cvtColor(cropped_img, cropped_img, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> rgb_channels(3);
    cv::split(cropped_img, rgb_channels);
    for (auto i = 0; i < rgb_channels.size(); i++)
    {
        rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / (trans_cfg.std[i] * 255.f), (0.0 - trans_cfg.mean[i] * 255.0f) / (trans_cfg.std[i] * 255.f));
    }

    std::vector<float> r = rgb_channels[0].reshape(1, 1);
    std::vector<float> g = rgb_channels[1].reshape(1, 1);
    std::vector<float> b = rgb_channels[2].reshape(1, 1);

    memcpy(&rgb[0], &r[0], crop_w * crop_h * sizeof(float));
    memcpy(&rgb[0 + crop_w * crop_h], &g[0], crop_w * crop_h * sizeof(float));
    memcpy(&rgb[0 + 2 * crop_w * crop_h], &b[0], crop_w * crop_h * sizeof(float));
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

bool greater(float a, float b) {
    // 从大到小排序
    return abs(a) > abs(b);
}

int quant_percent(std::unordered_map<std::string, float> &threshold_map, net *net_1, std::unordered_map<std::string, std::string> cfg_info_map) {

    extractor* b1 = net_1->create_exe();

    std::unordered_map<std::string, std::vector<float>> io_buf_map;
    std::string in_operand_name = cfg_info_map["in_operand_name"];

    std::vector<int> crop_shapes = str2number<int>(cfg_info_map["crop_shapes"]);
    std::vector<int> resize_shapes = str2number<int>(cfg_info_map["resize_shapes"]);
    std::vector<float> normal_mean = str2number<float>(cfg_info_map["normal_mean"]);
    std::vector<float> normal_std = str2number<float>(cfg_info_map["normal_std"]);
    std::vector<int> img_num_vec = str2number<int>(cfg_info_map["img_num"]);
    int32_t img_num = img_num_vec[0];

    int in_elem_size = 3 * crop_shapes[0] * crop_shapes[1];

    std::vector<float> in_buf(in_elem_size);

    TRANSFORMS_CONFIG_S trans_cfg;
    trans_cfg.resize_size[0] = resize_shapes[0];
    trans_cfg.resize_size[1] = resize_shapes[1];
    trans_cfg.crop_size[0] = crop_shapes[0];
    trans_cfg.crop_size[1] = crop_shapes[1];

    trans_cfg.mean[0] = normal_mean[0];
    trans_cfg.mean[1] = normal_mean[1];
    trans_cfg.mean[2] = normal_mean[2];

    trans_cfg.std[0] = normal_std[0];
    trans_cfg.std[1] = normal_std[1];
    trans_cfg.std[2] = normal_std[2];

    float percent = 99.998 / 100;

    std::ifstream file(cfg_info_map["imagenet_label_txt_path"]);
    std::string line;

    /* 步骤 1：申请空间，用于存放所有 batch 的较大数值。为了保证尽量不遗落每个 batch 的较大值，如果 percent = 99.9%，我会保存每个 batch 的
     * (1 - 99.99%) * 10 = 1% 的较大值。然后等所有 batch 都计算完后，再对所有 batch 的较大值做一次排序，找出其中第 percent * img_num
     * 个数作为阈值来使用
     */
    const int32_t affluence_ratio = 10;
    std::unordered_map<std::string, std::vector<float>> large_elem_map;
    // 获得到所有 conv op 的输入 vector
    for (auto op : b1->net_ptr->op_exec_order) {
        if (strcmp(op.get()->op_type, "Conv") == 0)
        {
            std::string ifmap_name = op.get()->in_operands[0];
            int32_t ifmap_elem_size = b1->operand_buf_map[ifmap_name].size();
            int32_t mult_batch_large_elem_size = (int32_t)(img_num * ifmap_elem_size * (1 - percent) * affluence_ratio);
            std::vector<float> large_elem_vec(mult_batch_large_elem_size, 0);
            large_elem_map[ifmap_name] = large_elem_vec;
        }
    }

    // 步骤 2：遍历矫正集，获取每个 conv 的输入数据，并将每个 batch 的较大值保存到 large_elem_map 中
    int32_t img_i = 0;
    while (std::getline(file, line) && img_i < img_num) {
        std::vector<std::string> words = split(line);
        std::string path = cfg_info_map["calibrate_img_path"];
        std::string img = words[0];
        std::string img_path = path + img;

        transforms(in_buf, img_path, trans_cfg);
        io_buf_map[in_operand_name] = in_buf;
        std::unordered_map<std::string, std::string> cfg_info_map;
        b1->impl(io_buf_map, cfg_info_map);

        // 获得到所有 conv op 的输入 vector
        for (auto op : b1->net_ptr->op_exec_order) {
            if (strcmp(op.get()->op_type, "Conv") == 0)
            {
                std::string ifmap_name = op.get()->in_operands[0];
                std::vector<float>* cur_ifmap = &b1->operand_buf_map[ifmap_name];
                std::sort(cur_ifmap->begin(), cur_ifmap->end(), greater);   // 按照绝对值从大到小排序

                // 将本张图的本个 conv 的输入的 reserve_elem_size 个元素，保存到 large_elem_map 的对应位置
                int32_t ifmap_elem_size = b1->operand_buf_map[ifmap_name].size();
                int32_t reserve_elem_size = (int32_t)(ifmap_elem_size * (1 - percent) * affluence_ratio);
                std::copy(cur_ifmap->begin(), cur_ifmap->begin() + reserve_elem_size, large_elem_map[ifmap_name].begin() + img_i * reserve_elem_size);
            }
        }
        img_i ++;
    }

    // 步骤 3： 对 large_elem_map 的较大值再进行一次排序，找出其中第 percent * img_num 个数作为阈值来使用
    for (auto map_i : large_elem_map) {
        std::string ifmap_name = map_i.first;
        int32_t ori_ifmap_elem_size = b1->operand_buf_map[ifmap_name].size();   // 获取原始的本个 ifmap 的单 batch 数据量
        std::vector<float>* cur_ifmap = &map_i.second;
        std::sort(cur_ifmap->begin(), cur_ifmap->end(), greater);
        int32_t threshold_idx = (int32_t)((1 - percent) * img_num * ori_ifmap_elem_size);
        float threshold = map_i.second[threshold_idx];
        threshold_map[ifmap_name] = threshold;
        std::cout << "ifmap name is: " << ifmap_name << ", threshold is: " << threshold << std::endl;
    }


    return 0;
}

float calc_kl(std::vector<float> &p, std::vector<float> &q)
{
    float kl = 0;
    const int32_t len = p.size();
    for (size_t i = 0; i < len; i++) {
        kl += p[i] * log(p[i] / q[i]);
    }
    return kl;
}

int quant_kl(std::unordered_map<std::string, float> &threshold_map, net *net_1, std::unordered_map<std::string, std::string> cfg_info_map) {

    extractor* b1 = net_1->create_exe();

    std::unordered_map<std::string, std::vector<float>> io_buf_map;
    std::string in_operand_name = cfg_info_map["in_operand_name"];

    std::vector<int> crop_shapes = str2number<int>(cfg_info_map["crop_shapes"]);
    std::vector<int> resize_shapes = str2number<int>(cfg_info_map["resize_shapes"]);
    std::vector<float> normal_mean = str2number<float>(cfg_info_map["normal_mean"]);
    std::vector<float> normal_std = str2number<float>(cfg_info_map["normal_std"]);
    std::vector<int> img_num_vec = str2number<int>(cfg_info_map["img_num"]);
    int32_t img_num = img_num_vec[0];

    int in_elem_size = 3 * crop_shapes[0] * crop_shapes[1];

    std::vector<float> in_buf(in_elem_size);

    TRANSFORMS_CONFIG_S trans_cfg;
    trans_cfg.resize_size[0] = resize_shapes[0];
    trans_cfg.resize_size[1] = resize_shapes[1];
    trans_cfg.crop_size[0] = crop_shapes[0];
    trans_cfg.crop_size[1] = crop_shapes[1];

    trans_cfg.mean[0] = normal_mean[0];
    trans_cfg.mean[1] = normal_mean[1];
    trans_cfg.mean[2] = normal_mean[2];

    trans_cfg.std[0] = normal_std[0];
    trans_cfg.std[1] = normal_std[1];
    trans_cfg.std[2] = normal_std[2];

    std::ifstream file(cfg_info_map["imagenet_label_txt_path"]);
    std::string line;

    /*
     * 步骤 1：申请空间，max_elem_map 用于存放所有 batch 的最大值; his_map 用来存放直方图
     */
    const int32_t num_his_bins = 2048;
    std::unordered_map<std::string, float> psum_map;
    std::unordered_map<std::string, int32_t> elem_size_map;
    std::unordered_map<std::string, float> max_elem_map;
    std::unordered_map<std::string, std::vector<float>> his_map;

    for (auto op : b1->net_ptr->op_exec_order) {
        if (strcmp(op.get()->op_type, "Conv") == 0)
        {
            std::string ifmap_name = op.get()->in_operands[0];
            int32_t ifmap_elem_size = b1->operand_buf_map[ifmap_name].size();
            std::vector<float> his_elem_vec(num_his_bins, 0);
            his_map[ifmap_name] = his_elem_vec;
            max_elem_map[ifmap_name] = 0.0f;
            float psum = 0.0f;
            for (auto elem_i : b1->operand_buf_map[ifmap_name]) {
                psum += elem_i;
            }
            psum_map[ifmap_name] += psum;

            elem_size_map[ifmap_name] = ifmap_elem_size * img_num;
        }
    }

    // 步骤 2：遍历矫正集，获取每个 conv 的输入数据，并将每个 batch 的最大值保存到 max_elem_map 中
    int32_t img_i = 0;
    while (std::getline(file, line) && img_i < img_num) {
        std::vector<std::string> words = split(line);
        std::string path = cfg_info_map["calibrate_img_path"];
        std::string img = words[0];
        std::string img_path = path + img;

        transforms(in_buf, img_path, trans_cfg);
        io_buf_map[in_operand_name] = in_buf;
        std::unordered_map<std::string, std::string> cfg_info_map;
        b1->impl(io_buf_map, cfg_info_map);

        // 获得到所有 conv op 的输入 vector
        for (auto op : b1->net_ptr->op_exec_order) {
            if (strcmp(op.get()->op_type, "Conv") == 0)
            {
                std::string ifmap_name = op.get()->in_operands[0];
                std::vector<float>* cur_ifmap = &b1->operand_buf_map[ifmap_name];

                float max_abs = 0.0f;
                // 遍历 vector 以找到绝对值最大的元素
                for (size_t i = 1; i < cur_ifmap->size(); ++i) {
                    if (std::fabs((*cur_ifmap)[i]) > max_abs) {
                        max_abs = std::fabs((*cur_ifmap)[i]);
                    }
                }
                if (max_abs > max_elem_map[ifmap_name]) {
                    max_elem_map[ifmap_name] = max_abs;
                }
            }
        }
        img_i ++;
    }

    // 步骤 3： 初始化直方图
    img_i = 0;
    while (std::getline(file, line) && img_i < img_num) {
        std::vector<std::string> words = split(line);
        std::string path = cfg_info_map["calibrate_img_path"];
        std::string img = words[0];
        std::string img_path = path + img;

        transforms(in_buf, img_path, trans_cfg);
        io_buf_map[in_operand_name] = in_buf;
        std::unordered_map<std::string, std::string> cfg_info_map;
        b1->impl(io_buf_map, cfg_info_map);

        for (auto op : b1->net_ptr->op_exec_order) {
            if (strcmp(op.get()->op_type, "Conv") == 0)
            {
                std::string ifmap_name = op.get()->in_operands[0];
                float cur_ifmap_max = max_elem_map[ifmap_name];

                for (auto num : b1->operand_buf_map[ifmap_name]) {
                    if (num == 0.0f){
                        continue;
                    }
                    int32_t idx = std::min((int)(fabs(num) * num_his_bins * 1.0f / cur_ifmap_max), num_his_bins - 1);
                    his_map[ifmap_name][idx] += 1;
                }
            }
        }
        img_i ++;
    }


    // 步骤 4： 计算阈值
    for (auto m_i : his_map) {
        std::string key = m_i.first;
        std::vector<float> his_vec = m_i.second;
        int32_t elem_size = elem_size_map[key];

        const int32_t target_bin = 128;
        const int32_t min_num_bins = 128;
        const float kl_eps = 1e-7;  // 防止算kl散度时产生log(0)
        float best_kl = 32768.6f;

        int32_t best_bins = 0;
        for (int32_t bins_i = min_num_bins; bins_i < num_his_bins; bins_i++) {
            // step 3、生成 p 概率分布
            std::vector<float> p_histogram(bins_i, kl_eps);
            std::vector<float> acc_p_histogram(bins_i, kl_eps);  // 将大于阈值的值加到直方图最后一个
            {
                for (size_t i = 0; i < bins_i; i++) {
                    acc_p_histogram[i] += his_vec[i];
                    p_histogram[i] += his_vec[i];
                }
                for (size_t i = bins_i; i < num_his_bins; i++) {
                    acc_p_histogram[bins_i - 1] += his_vec[i];
                }

                // step 4、p 转为 0 到 1 的概率分布
                for (auto &p_his : acc_p_histogram) {
                    p_his = p_his / (double)elem_size;
                }

                for (auto &p_his : p_histogram) {
                    p_his = p_his / (double)elem_size;
                }
            }

            // step 5、生成 q 概率分布
            // 生成 q 直方图时，使用的是舍弃阈值之外的值，而不是将阈值之外的值加大最后一个的直方图
            std::vector<float> q_histogram(target_bin, 0.0f);
            {
                const float ratio = (float)bins_i * 1.0f / target_bin;
                for (size_t i = 0; i < target_bin; i++) {
                    const float st = i * ratio;
                    const float ed = (i + 1) * ratio;

                    if (floor(st) == floor(ed)) {
                        q_histogram[i] += (ed - st) * p_histogram[floor(st)];
                    } else {
                        q_histogram[i] += (ceil(st) - st) * p_histogram[floor(st)];
                        for (size_t j = ceil(st); j < floor(ed); j++) {
                            q_histogram[i] += p_histogram[j];
                        }
                        q_histogram[i] += (ed - floor(ed)) * p_histogram[floor(ed)];
                    }

                    q_histogram[i] /= (ed - st);
                }

//                {
//                    const float end = ratio;
//
//                    const int right_lower = (int)floor(end);
//                    const float right_scale = end - right_lower;
//
//                    if (right_scale > 0)
//                    {
//                        q_histogram[0] += right_scale * p_histogram[right_lower];
//                    }
//
//                    for (int k = 0; k < right_lower; k++)
//                    {
//                        q_histogram[0] += p_histogram[k];
//                    }
//
//                    q_histogram[0] /= right_lower + right_scale;
//                }
//                for (int j = 1; j < target_bin - 1; j++)
//                {
//                    const float start = j * ratio;
//                    const float end = (j + 1) * ratio;
//
//                    const int left_upper = (int)ceil(start);
//                    const float left_scale = left_upper - start;
//
//                    const int right_lower = (int)floor(end);
//                    const float right_scale = end - right_lower;
//
//                    if (left_scale > 0)
//                    {
//                        q_histogram[j] += left_scale * p_histogram[left_upper - 1];
//                    }
//
//                    if (right_scale > 0)
//                    {
//                        q_histogram[j] += right_scale * p_histogram[right_lower];
//                    }
//
//                    for (int k = left_upper; k < right_lower; k++)
//                    {
//                        q_histogram[j] += p_histogram[k];
//                    }
//
//                    q_histogram[j] /= right_lower - left_upper + left_scale + right_scale;
//                }
//                {
//                    const float start = bins_i - ratio;
//
//                    const int left_upper = (int)ceil(start);
//                    const float left_scale = left_upper - start;
//
//                    if (left_scale > 0)
//                    {
//                        q_histogram[target_bin - 1] += left_scale * p_histogram[left_upper - 1];
//                    }
//
//                    for (int k = left_upper; k < bins_i; k++)
//                    {
//                        q_histogram[target_bin - 1] += p_histogram[k];
//                    }
//
//                    q_histogram[target_bin - 1] /= bins_i - left_upper + left_scale;
//                }


            }

            // step 6、扩展 q 到 p 的长度
            std::vector<float> expand_q_histogram(bins_i, kl_eps);
            {
                const float ratio = (float)bins_i * 1.00f / target_bin;
                for (size_t i = 0; i < target_bin; i++) {
                    const float st = i * ratio;
                    const float ed = (i + 1) * ratio;

                    if (floor(st) == floor(ed)) {
                        expand_q_histogram[floor(st)] += (ed - st) * q_histogram[i];
                    } else {
                        expand_q_histogram[floor(st)] += (ceil(st) - st) * q_histogram[i];
                        for (size_t j = ceil(st); j < floor(ed); j++) {
                            expand_q_histogram[j] += 1.0f * q_histogram[i];
                        }
                        int32_t tail;
                        tail = floor(ed) >= bins_i ? bins_i - 1 : floor(ed);
                        expand_q_histogram[tail] += (ed - floor(ed)) / 1.0f * q_histogram[i];
                    }
                }

//                {
//                    const float end = ratio;
//
//                    const int right_lower = (int)floor(end);
//                    const float right_scale = end - right_lower;
//
//                    if (right_scale > 0)
//                    {
//                        expand_q_histogram[right_lower] += right_scale * q_histogram[0];
//                    }
//
//                    for (int k = 0; k < right_lower; k++)
//                    {
//                        expand_q_histogram[k] += q_histogram[0];
//                    }
//                }
//                for (int j = 1; j < target_bin - 1; j++)
//                {
//                    const float start = j * ratio;
//                    const float end = (j + 1) * ratio;
//
//                    const int left_upper = (int)ceil(start);
//                    const float left_scale = left_upper - start;
//
//                    const int right_lower = (int)floor(end);
//                    const float right_scale = end - right_lower;
//
//                    if (left_scale > 0)
//                    {
//                        expand_q_histogram[left_upper - 1] += left_scale * q_histogram[j];
//                    }
//
//                    if (right_scale > 0)
//                    {
//                        expand_q_histogram[right_lower] += right_scale * q_histogram[j];
//                    }
//
//                    for (int k = left_upper; k < right_lower; k++)
//                    {
//                        expand_q_histogram[k] += q_histogram[j];
//                    }
//                }
//                {
//                    const float start = bins_i - ratio;
//
//                    const int left_upper = (int)ceil(start);
//                    const float left_scale = left_upper - start;
//
//                    if (left_scale > 0)
//                    {
//                        expand_q_histogram[left_upper - 1] += left_scale * q_histogram[target_bin - 1];
//                    }
//
//                    for (int k = left_upper; k < bins_i; k++)
//                    {
//                        expand_q_histogram[k] += q_histogram[target_bin - 1];
//                    }
//                }


            }

            // step 7、计算 q 和 p 的 kl 散度
            float kl = calc_kl(acc_p_histogram, expand_q_histogram);

            // step 8、find the best bin
            if (kl < best_kl) {
                best_kl = kl;
                best_bins = bins_i;
            }
        }

        float threshold = (best_bins + 0.5f) * max_elem_map[key] / num_his_bins;
        threshold_map[key] = threshold;
        std::cout << "ifmap name is: " << key << ", threshold is: " << threshold << ", best_bins is: " << best_bins << std::endl;
    }



    return 0;
}

int quant_mse(std::unordered_map<std::string, float> &threshold_map, net *net_1, std::unordered_map<std::string, std::string> cfg_info_map) {
    Manager &m = Manager::getInstance();

    extractor* b1 = net_1->create_exe();

    std::unordered_map<std::string, std::vector<float>> io_buf_map;
    std::string in_operand_name = cfg_info_map["in_operand_name"];

    std::vector<int> crop_shapes = str2number<int>(cfg_info_map["crop_shapes"]);
    std::vector<int> resize_shapes = str2number<int>(cfg_info_map["resize_shapes"]);
    std::vector<float> normal_mean = str2number<float>(cfg_info_map["normal_mean"]);
    std::vector<float> normal_std = str2number<float>(cfg_info_map["normal_std"]);
    std::vector<int> img_num_vec = str2number<int>(cfg_info_map["img_num"]);
    int32_t img_num = img_num_vec[0];

    int in_elem_size = 3 * crop_shapes[0] * crop_shapes[1];

    std::vector<float> in_buf(in_elem_size);

    TRANSFORMS_CONFIG_S trans_cfg;
    trans_cfg.resize_size[0] = resize_shapes[0];
    trans_cfg.resize_size[1] = resize_shapes[1];
    trans_cfg.crop_size[0] = crop_shapes[0];
    trans_cfg.crop_size[1] = crop_shapes[1];

    trans_cfg.mean[0] = normal_mean[0];
    trans_cfg.mean[1] = normal_mean[1];
    trans_cfg.mean[2] = normal_mean[2];

    trans_cfg.std[0] = normal_std[0];
    trans_cfg.std[1] = normal_std[1];
    trans_cfg.std[2] = normal_std[2];

    std::ifstream file(cfg_info_map["imagenet_label_txt_path"]);
    std::string line;

    /*
     * 步骤 1：申请空间，max_elem_map 用于存放所有 batch 的最大值; his_map 用来存放直方图
     */
    const int32_t num_his_bins = 2048;
    std::unordered_map<std::string, int32_t> elem_size_map;
    std::unordered_map<std::string, float> psum_map;
    std::unordered_map<std::string, float> max_elem_map;
    std::unordered_map<std::string, std::vector<float>> his_map;

    for (auto op : b1->net_ptr->op_exec_order) {
        if (strcmp(op.get()->op_type, "Conv") == 0)
        {
            std::string ifmap_name = op.get()->in_operands[0];
            int32_t ifmap_elem_size = b1->operand_buf_map[ifmap_name].size();
            std::vector<float> his_elem_vec(num_his_bins, 0);
            his_map[ifmap_name] = his_elem_vec;
            max_elem_map[ifmap_name] = 0.0f;
            
            float psum = 0.0f;
            for (auto elem_i : b1->operand_buf_map[ifmap_name]) {
                psum += elem_i;
            }
            psum_map[ifmap_name] += psum;

            elem_size_map[ifmap_name] = ifmap_elem_size * img_num;
        }
    }

    // 步骤 2：遍历矫正集，获取每个 conv 的输入数据，并将每个 batch 的最大值保存到 max_elem_map 中
    int32_t img_i = 0;
    while (std::getline(file, line) && img_i < img_num) {
        std::vector<std::string> words = split(line);
        std::string path = cfg_info_map["calibrate_img_path"];
        std::string img = words[0];
        std::string img_path = path + img;

        transforms(in_buf, img_path, trans_cfg);
        io_buf_map[in_operand_name] = in_buf;
        std::unordered_map<std::string, std::string> cfg_info_map;
        b1->impl(io_buf_map, cfg_info_map);

        // 获得到所有 conv op 的输入 vector
        for (auto op : b1->net_ptr->op_exec_order) {
            if (strcmp(op.get()->op_type, "Conv") == 0)
            {
                std::string ifmap_name = op.get()->in_operands[0];
                std::vector<float>* cur_ifmap = &b1->operand_buf_map[ifmap_name];

                float max_abs = 0.0f;
                // 遍历 vector 以找到绝对值最大的元素
                for (size_t i = 1; i < cur_ifmap->size(); ++i) {
                    if (std::fabs((*cur_ifmap)[i]) > max_abs) {
                        max_abs = std::fabs((*cur_ifmap)[i]);
                    }
                }
                if (max_abs > max_elem_map[ifmap_name]) {
                    max_elem_map[ifmap_name] = max_abs;
                }
            }
        }
        img_i ++;
    }

    // 步骤 3： 初始化直方图
    img_i = 0;
    while (std::getline(file, line) && img_i < img_num) {
        std::vector<std::string> words = split(line);
        std::string path = cfg_info_map["calibrate_img_path"];
        std::string img = words[0];
        std::string img_path = path + img;

        transforms(in_buf, img_path, trans_cfg);
        io_buf_map[in_operand_name] = in_buf;
        std::unordered_map<std::string, std::string> cfg_info_map;
        b1->impl(io_buf_map, cfg_info_map);

        for (auto op : b1->net_ptr->op_exec_order) {
            if (strcmp(op.get()->op_type, "Conv") == 0)
            {
                std::string ifmap_name = op.get()->in_operands[0];
                float cur_ifmap_max = max_elem_map[ifmap_name];

                for (auto num : b1->operand_buf_map[ifmap_name]) {
                    if (num == 0.0f){
                        continue;
                    }
                    int32_t idx = std::min((int)(fabs(num) * num_his_bins * 1.0f / cur_ifmap_max), num_his_bins - 1);
                    his_map[ifmap_name][idx] += 1;
                }
            }
        }
        img_i ++;
    }


    // 步骤 4： 计算阈值
    for (auto m_i : his_map) {
        std::string key = m_i.first;
        std::vector<float> his_vec = m_i.second;
        int32_t elem_size = elem_size_map[key];

        const int32_t target_bin = 128;
        const int32_t min_num_bins = 128;
        const float kl_eps = 1e-7;  // 防止算kl散度时产生log(0)
        float best_kl = 32768.6f;

        int32_t best_bins = 0;
        float loss_min = 32768.0f;

        for (int32_t bins_i = min_num_bins; bins_i < num_his_bins; bins_i++) {
            float loss = 0.0f;
            float err;

            const float step = (float)bins_i / (target_bin - 1);

            for (int idx = 0; idx < his_vec.size(); ++idx) {
                float idx_quanted = std::round(((float)idx + 0.5f) / step);
                idx_quanted = idx_quanted > 127.0f ? 127.0f : idx_quanted;
                idx_quanted *= step;
                err = (float)idx + 0.5 - idx_quanted;
                loss += (his_vec[idx] * err * err) / elem_size;
            }

            if (loss < loss_min) {
                loss_min = loss;
                best_bins = bins_i;
            }
        }

        float threshold = (best_bins + 0.5f) * max_elem_map[key] / num_his_bins;
        threshold_map[key] = threshold;
        std::cout << "in mes quant type, ifmap name is: " << key << ", threshold is: " << threshold << ", best_bins is: " << best_bins << std::endl;
    }



    return 0;
}



int main(int argc, char **argv)
{

    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " [quant config.yml]" << std::endl;
        exit(-1);
    }
    const char *rt_cfg_txt = argv[1];
    std::string rt_cfg_txt_str(rt_cfg_txt);
    std::unordered_map<std::string, std::string> cfg_info_map;
    yml2map(cfg_info_map, rt_cfg_txt_str);

    std::cout << "start quant one model, the quant type is: " << cfg_info_map["quant_type"] << std::endl;

    const char* one_file_path = cfg_info_map["one_file_path"].c_str();
    net *net_1 = new net(one_file_path);

    // step 1: get one file size
    std::ifstream one_file(one_file_path, std::ios::ate | std::ios::binary);
    int32_t one_file_size = static_cast<int32_t>(one_file.tellg());
    one_file.close();

    // step 2: load one file
    char *one_buf_ptr = (char *)malloc(one_file_size);
    FILE *file_p = NULL;

    file_p = fopen(one_file_path, "r");
    if (file_p == NULL)
    {
        std::cout << "failed: can't open the one file" << std::endl;
        return 0;
    }
    fread(one_buf_ptr, sizeof(char), one_file_size, file_p);
    fclose(file_p);

    // step 3: quant ifmap of conv
    net_1->build_graph();

    std::unordered_map<std::string, float> threshold_map;
    double op_st = omp_get_wtime();

    if (cfg_info_map["quant_type"] == "percent") {
        quant_percent(threshold_map, net_1, cfg_info_map);
    } else if (cfg_info_map["quant_type"] == "kl") {
        quant_kl(threshold_map, net_1, cfg_info_map);
    } else if (cfg_info_map["quant_type"] == "mse") {
        quant_mse(threshold_map, net_1, cfg_info_map);
    } else {
        std::cerr << "you should choose a quantification method" << std::endl;
    }

    double op_ed = omp_get_wtime();
    double elapsed = op_ed - op_st;
    std::vector<int> img_num_vec = str2number<int>(cfg_info_map["img_num"]);
    int32_t img_num = img_num_vec[0];
    std::cout << "============= using img num: " << img_num << " to quant, using time is: " << elapsed << " s. =============="<< std::endl;

    int32_t *head_ptr = (int32_t *)one_buf_ptr;
    int32_t node_cnt = head_ptr[1];
    char *cur_node_cfg_ptr = (char *)(one_buf_ptr + head_ptr[2]);

    Manager &m = Manager::getInstance();

    for (int32_t node_i = 0; node_i < node_cnt; node_i++)
    {
        std::string op_type_str(cur_node_cfg_ptr);
        if (op_type_str == "Conv")
        {
            CONV_CONFIG_S *conv_cfg = (CONV_CONFIG_S *)cur_node_cfg_ptr;
            char* ifmap_name = conv_cfg->op_base_cfg.in_operand_name[0];
            std::string ifmap_name_str(ifmap_name);
            float ifmap_threshold = threshold_map[ifmap_name_str];
            conv_cfg->input_scale = ifmap_threshold / 127;
            conv_cfg->ifmap_quant2 = TYPE_INT8;
        }
        // update cur_node_cfg_ptr
        cur_node_cfg_ptr += align_buf_size(m.op_cfg_size_map[op_type_str]);
    }

    // step 4: quant initial data such as weight and bias of conv op
    quant_conv_weight_bias(one_buf_ptr);

    // step 5: dump quant .one file
    FILE *quant_file_p = fopen(cfg_info_map["quant_one_file_path"].c_str(), "w");
    fwrite((void *)one_buf_ptr, 1, one_file_size, quant_file_p);
    fclose(quant_file_p);


    return 0;
}