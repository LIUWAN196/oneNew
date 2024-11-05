//
// Created by wanzai on 24-8-22.
//

#ifndef ONENEW_MODEL_QUANT_HPP
#define ONENEW_MODEL_QUANT_HPP

//#include <iostream>
//#include <fstream>
//#include "cstring"
//#include <time.h>
//#include <vector>
//#include "cmath"
//#include <iostream>
//#include "opencv/cv.h"
//#include "opencv2/opencv.hpp"
//#include "opencv/cv.h"
//#include "opencv2/opencv.hpp"
//#include <dlfcn.h>
//#include <cstring>
//#include <iostream>
//#include <memory>
//#include <vector>
//#include <sys/time.h>
//#include <thread>
//#include <pthread.h>
//#include <functional>
//#include <vector>
//#include <algorithm>
//#include <omp.h>
//#include "../manager/manager.h"

//#include "ops_head.h"
//#include "net.h"
#include "optimize.hpp"
//#include "../../common/utils_cpp.hpp"


//#include <iostream>
//#include <fstream>
//#include "cstring"
//#include <time.h>
//#include <vector>
//#include "cmath"
//#include <iostream>
//#include "opencv/cv.h"
//#include "opencv2/opencv.hpp"
//#include "opencv/cv.h"
//#include "opencv2/opencv.hpp"
//#include <dlfcn.h>
//#include <cstring>
//#include <iostream>
//#include <memory>
//#include <vector>
//#include <sys/time.h>
//#include <thread>
//#include <pthread.h>
//#include <functional>
//#include <vector>
//#include <algorithm>
//#include <omp.h>
//#include "../manager/manager.h"
//#include "../../common/utils_cpp.hpp"
//
//#include "ops_head.h"
//#include "net.h"



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

    ONE_MODEL_DESC_S *one_model_info_ptr = (ONE_MODEL_DESC_S *)one_buf_ptr;
    int32_t node_cnt = one_model_info_ptr->node_cnt;
    char *cur_node_cfg_ptr = (char *) (one_buf_ptr + one_model_info_ptr->node_cfg_offset);

    std::vector<OPERAND_S *> weight_operand_ptr_vec;
    std::vector<OPERAND_S *> bias_operand_ptr_vec;
    for (int32_t node_i = 0; node_i < node_cnt; node_i++)
    {
        std::string op_type_str(cur_node_cfg_ptr);
        if (op_type_str == "Conv")
        {
            CONV_CONFIG_S *conv_cfg = (CONV_CONFIG_S *)cur_node_cfg_ptr;

            // quant weight and bias
            int32_t init_cnt = one_model_info_ptr->init_cnt;
            char *cur_init_info_ptr = (char *)(one_buf_ptr + one_model_info_ptr->init_info_offset);
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

bool greater(float a, float b) {
    // 从大到小排序
    return abs(a) > abs(b);
}

int quant_percent(std::unordered_map<std::string, float> &threshold_map, net *net_1, std::unordered_map<std::string, std::string> cfg_info_map) {

    extractor* exe_net = net_1->create_exe();

    std::unordered_map<std::string, BUF_INFO_S> io_buf_map;
    auto* ifmap_of_model = (io*)exe_net->net_ptr->op_exec_order[0].get();
    std::string in_operand_name = ifmap_of_model->io_cfg.operand.operand_name;

    std::vector<int> crop_shapes = str2number<int>(cfg_info_map["crop_shapes"]);
    std::vector<int> resize_shapes = str2number<int>(cfg_info_map["resize_shapes"]);
    std::vector<float> normal_mean = str2number<float>(cfg_info_map["normal_mean"]);
    std::vector<float> normal_std = str2number<float>(cfg_info_map["normal_std"]);
    std::vector<int> img_num_vec = str2number<int>(cfg_info_map["calibrate_img_num"]);
    int32_t calibrate_img_num = img_num_vec[0];

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

    std::ifstream file(cfg_info_map["calibrate_img_name_txt_path"]);
    std::string line;

    /* 步骤 1：申请空间，用于存放所有 batch 的较大数值。为了保证尽量不遗落每个 batch 的较大值，如果 percent = 99.9%，我会保存每个 batch 的
     * (1 - 99.99%) * 10 = 1% 的较大值。然后等所有 batch 都计算完后，再对所有 batch 的较大值做一次排序，找出其中第 percent * calibrate_img_num
     * 个数作为阈值来使用
     */
    const int32_t affluence_ratio = 10;
    std::unordered_map<std::string, std::vector<float>> large_elem_map;
    // 获得到所有 conv op 的输入 vector
    for (auto op : exe_net->net_ptr->op_exec_order) {
        if (strcmp(op.get()->op_type, "Conv") == 0)
        {
            std::string ifmap_name = op.get()->in_operands[0];
            int32_t ifmap_elem_size = exe_net->operand_buf_map[ifmap_name].elem_size;
            int32_t mult_batch_large_elem_size = (int32_t)(calibrate_img_num * ifmap_elem_size * (1 - percent) * affluence_ratio);
            std::vector<float> large_elem_vec(mult_batch_large_elem_size, 0);
            large_elem_map[ifmap_name] = large_elem_vec;
        }
    }

    char buffer[101] = {0};//存储进度条字符
    char arr[5] = {"-/|\\"};//存储基本的变化字幕
    printf("quant [%.2f%%] [%-100s][%c]\r", 0.0f, buffer, arr[1]);
    fflush(stdout);

    int32_t elem_size = in_buf.size();
    int32_t buf_size = elem_size * sizeof(float);
    int64_t cur_operand_ptr = (int64_t)&in_buf[0];
    io_buf_map[in_operand_name] = {cur_operand_ptr, elem_size, buf_size};
    exe_net->prepare_for_op(io_buf_map);

    // 步骤 2：遍历矫正集，获取每个 conv 的输入数据，并将每个 batch 的较大值保存到 large_elem_map 中
    int32_t img_i = 0;
    while (std::getline(file, line) && img_i < calibrate_img_num) {
        std::vector<std::string> words = split(line);
        std::string path = cfg_info_map["calibrate_img_folder"];
        std::string img = words[0];
        std::string img_path = path + img;

        transforms(in_buf, img_path, trans_cfg);
//        int32_t elem_size = in_buf.size();
//        int32_t buf_size = elem_size * sizeof(float);
//        int64_t cur_operand_ptr = (int64_t)&in_buf[0];
//        io_buf_map[in_operand_name] = {cur_operand_ptr, elem_size, buf_size};
//        exe_net->prepare_for_op(io_buf_map);


        std::unordered_map<std::string, std::string> cfg_info_map;

        exe_net->impl(io_buf_map, cfg_info_map);

        float schedule = (img_i + 1.0f) / calibrate_img_num / 2.0f * 100;
        int schedule_int = (int)schedule;
        buffer[schedule_int] = '#';
        printf("quant [%.2f%%] [%-100s][%c]\r", schedule, buffer, arr[schedule_int % 4]);
        fflush(stdout);

        // 获得到所有 conv op 的输入 vector
        for (auto op : exe_net->net_ptr->op_exec_order) {
            if (strcmp(op.get()->op_type, "Conv") == 0)
            {
                std::string ifmap_name = op.get()->in_operands[0];
                float *ifmap_ptr = (float *)exe_net->operand_buf_map[ifmap_name].st_ptr;
                int32_t ifmap_elem_size = exe_net->operand_buf_map[ifmap_name].elem_size;
                std::vector<float> cur_ifmap(ifmap_elem_size, 0);
                memcpy(&cur_ifmap[0], ifmap_ptr, ifmap_elem_size * sizeof(float));
                std::sort(cur_ifmap.begin(), cur_ifmap.end(), greater);   // 按照绝对值从大到小排序

                // 将本张图的本个 conv 的输入的 reserve_elem_size 个元素，保存到 large_elem_map 的对应位置
                int32_t reserve_elem_size = (int32_t)(ifmap_elem_size * (1 - percent) * affluence_ratio);
                std::copy(cur_ifmap.begin(), cur_ifmap.begin() + reserve_elem_size, large_elem_map[ifmap_name].begin() + img_i * reserve_elem_size);
            }
        }
        img_i ++;
    }

    // 步骤 3： 对 large_elem_map 的较大值再进行一次排序，找出其中第 percent * calibrate_img_num 个数作为阈值来使用
    for (auto map_i : large_elem_map) {
        std::string ifmap_name = map_i.first;
        int32_t ori_ifmap_elem_size = exe_net->operand_buf_map[ifmap_name].elem_size;   // 获取原始的本个 ifmap 的单 batch 数据量
        std::vector<float>* cur_ifmap = &map_i.second;
        std::sort(cur_ifmap->begin(), cur_ifmap->end(), greater);
        int32_t threshold_idx = (int32_t)((1 - percent) * calibrate_img_num * ori_ifmap_elem_size);
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

    extractor* exe_net = net_1->create_exe();

    std::unordered_map<std::string, BUF_INFO_S> io_buf_map;
    auto* ifmap_of_model = (io*)exe_net->net_ptr->op_exec_order[0].get();
    std::string in_operand_name = ifmap_of_model->io_cfg.operand.operand_name;

    std::vector<int> crop_shapes = str2number<int>(cfg_info_map["crop_shapes"]);
    std::vector<int> resize_shapes = str2number<int>(cfg_info_map["resize_shapes"]);
    std::vector<float> normal_mean = str2number<float>(cfg_info_map["normal_mean"]);
    std::vector<float> normal_std = str2number<float>(cfg_info_map["normal_std"]);
    std::vector<int> img_num_vec = str2number<int>(cfg_info_map["calibrate_img_num"]);
    int32_t calibrate_img_num = img_num_vec[0];

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

    std::ifstream file(cfg_info_map["calibrate_img_name_txt_path"]);
    std::string line;

    /*
     * 步骤 1：申请空间，max_elem_map 用于存放所有 batch 的最大值; his_map 用来存放直方图
     */
    const int32_t num_his_bins = 2048;
    std::unordered_map<std::string, float> psum_map;
    std::unordered_map<std::string, int32_t> elem_size_map;
    std::unordered_map<std::string, float> max_elem_map;
    std::unordered_map<std::string, std::vector<float>> his_map;

    for (auto op : exe_net->net_ptr->op_exec_order) {
        if (strcmp(op.get()->op_type, "Conv") == 0)
        {
            std::string ifmap_name = op.get()->in_operands[0];
            int32_t ifmap_elem_size = exe_net->operand_buf_map[ifmap_name].elem_size;
            std::vector<float> his_elem_vec(num_his_bins, 0);
            his_map[ifmap_name] = his_elem_vec;
            max_elem_map[ifmap_name] = 0.0f;
            float psum = 0.0f;

            float * fmap_ptr = (float *)exe_net->operand_buf_map[ifmap_name].st_ptr;
            int64_t elem_size = exe_net->operand_buf_map[ifmap_name].elem_size;
            for (int i = 0; i < elem_size; ++i) {
                psum += fmap_ptr[i];
            }
            psum_map[ifmap_name] += psum;

            elem_size_map[ifmap_name] = ifmap_elem_size * calibrate_img_num;
        }
    }

    char buffer[101] = {0};//存储进度条字符
    char arr[5] = {"-/|\\"};//存储基本的变化字幕
    printf("quant [%.2f%%] [%-100s][%c]\r", 0.0f, buffer, arr[1]);
    fflush(stdout);

    int32_t elem_size = in_buf.size();
    int32_t buf_size = elem_size * sizeof(float);
    int64_t cur_operand_ptr = (int64_t)&in_buf[0];
    io_buf_map[in_operand_name] = {cur_operand_ptr, elem_size, buf_size};
    exe_net->prepare_for_op(io_buf_map);

    // 步骤 2：遍历矫正集，获取每个 conv 的输入数据，并将每个 batch 的最大值保存到 max_elem_map 中
    int32_t img_i = 0;
    while (std::getline(file, line) && img_i < calibrate_img_num) {
        std::vector<std::string> words = split(line);
        std::string path = cfg_info_map["calibrate_img_folder"];
        std::string img = words[0];
        std::string img_path = path + img;

        transforms(in_buf, img_path, trans_cfg);

        std::unordered_map<std::string, std::string> cfg_info_map;

        exe_net->impl(io_buf_map, cfg_info_map);

        float schedule = (img_i + 1.0f) / calibrate_img_num / 2.0f * 100;
        int schedule_int = (int)schedule;
        buffer[schedule_int] = '#';
        printf("quant [%.2f%%] [%-100s][%c]\r", schedule, buffer, arr[schedule_int % 4]);
        fflush(stdout);

        // 获得到所有 conv op 的输入 vector
        for (auto op : exe_net->net_ptr->op_exec_order) {
            if (strcmp(op.get()->op_type, "Conv") == 0)
            {
                std::string ifmap_name = op.get()->in_operands[0];

                float * fmap_ptr = (float *)exe_net->operand_buf_map[ifmap_name].st_ptr;
                int64_t elem_size = exe_net->operand_buf_map[ifmap_name].elem_size;
                float max_abs = 0.0f;
                for (int i = 0; i < elem_size; ++i) {
                    max_abs = max_abs > std::fabs(fmap_ptr[i]) ? max_abs : std::fabs(fmap_ptr[i]);
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
    while (std::getline(file, line) && img_i < calibrate_img_num) {
        std::vector<std::string> words = split(line);
        std::string path = cfg_info_map["calibrate_img_folder"];
        std::string img = words[0];
        std::string img_path = path + img;

        transforms(in_buf, img_path, trans_cfg);
//        int32_t elem_size = in_buf.size();
//        int32_t buf_size = elem_size * sizeof(float);
//        int64_t cur_operand_ptr = (int64_t)&in_buf[0];
//        io_buf_map[in_operand_name] = {cur_operand_ptr, elem_size, buf_size};

        std::unordered_map<std::string, std::string> cfg_info_map;
//        exe_net->prepare_for_op(io_buf_map);
        exe_net->impl(io_buf_map, cfg_info_map);

        float schedule = 50.0f + (img_i + 1.0f) / calibrate_img_num / 2.0f * 100;
        int schedule_int = (int)schedule;
        buffer[schedule_int] = '#';
        printf("quant [%.2f%%] [%-100s][%c]\r", schedule, buffer, arr[schedule_int % 4]);
        fflush(stdout);

        for (auto op : exe_net->net_ptr->op_exec_order) {
            if (strcmp(op.get()->op_type, "Conv") == 0)
            {
                std::string ifmap_name = op.get()->in_operands[0];
                float cur_ifmap_max = max_elem_map[ifmap_name];

                float * ofmap_ptr = (float *)exe_net->operand_buf_map[ifmap_name].st_ptr;
                int64_t elem_size = exe_net->operand_buf_map[ifmap_name].elem_size;
                for (int i = 0; i < elem_size; ++i) {
                    if (ofmap_ptr[i] == 0.0f){
                        continue;
                    }
                    int32_t idx = std::min((int)(fabs(ofmap_ptr[i]) * num_his_bins * 1.0f / cur_ifmap_max), num_his_bins - 1);
                    his_map[ifmap_name][idx] += 1;
                }
            }
        }
        img_i ++;
    }

    printf("quant done [%.2f%%] [%-100s][%c]\r", 100.0f, buffer, arr[3]);
    fflush(stdout);

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
//        std::cout << "ifmap name is: " << key << ", threshold is: " << threshold << ", best_bins is: " << best_bins << std::endl;
    }



    return 0;
}

int quant_mse(std::unordered_map<std::string, float> &threshold_map, net *net_1, std::unordered_map<std::string, std::string> cfg_info_map) {
    Manager &m = Manager::getInstance();

    extractor* exe_net = net_1->create_exe();

    std::unordered_map<std::string, BUF_INFO_S> io_buf_map;
    auto* ifmap_of_model = (io*)exe_net->net_ptr->op_exec_order[0].get();
    std::string in_operand_name = ifmap_of_model->io_cfg.operand.operand_name;


    std::vector<int> crop_shapes = str2number<int>(cfg_info_map["crop_shapes"]);
    std::vector<int> resize_shapes = str2number<int>(cfg_info_map["resize_shapes"]);
    std::vector<float> normal_mean = str2number<float>(cfg_info_map["normal_mean"]);
    std::vector<float> normal_std = str2number<float>(cfg_info_map["normal_std"]);
    std::vector<int> img_num_vec = str2number<int>(cfg_info_map["calibrate_img_num"]);
    int32_t calibrate_img_num = img_num_vec[0];

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

    std::ifstream file(cfg_info_map["calibrate_img_name_txt_path"]);
    std::string line;

    /*
     * 步骤 1：申请空间，max_elem_map 用于存放所有 batch 的最大值; his_map 用来存放直方图
     */
    const int32_t num_his_bins = 2048;
    std::unordered_map<std::string, int32_t> elem_size_map;
    std::unordered_map<std::string, float> psum_map;
    std::unordered_map<std::string, float> max_elem_map;
    std::unordered_map<std::string, std::vector<float>> his_map;

    for (auto op : exe_net->net_ptr->op_exec_order) {
        if (strcmp(op.get()->op_type, "Conv") == 0)
        {
            std::string ifmap_name = op.get()->in_operands[0];
            int32_t ifmap_elem_size = exe_net->operand_buf_map[ifmap_name].elem_size;
            std::vector<float> his_elem_vec(num_his_bins, 0);
            his_map[ifmap_name] = his_elem_vec;
            max_elem_map[ifmap_name] = 0.0f;

            float psum = 0.0f;
            float * fmap_ptr = (float *)exe_net->operand_buf_map[ifmap_name].st_ptr;
            int64_t elem_size = exe_net->operand_buf_map[ifmap_name].elem_size;
            for (int i = 0; i < elem_size; ++i) {
                psum += fmap_ptr[i];
            }
            psum_map[ifmap_name] += psum;

            elem_size_map[ifmap_name] = ifmap_elem_size * calibrate_img_num;
        }
    }

    char buffer[101] = {0};//存储进度条字符
    char arr[5] = {"-/|\\"};//存储基本的变化字幕
    printf("quant [%.2f%%] [%-100s][%c]\r", 0.0f, buffer, arr[1]);
    fflush(stdout);

    int32_t elem_size = in_buf.size();
    int32_t buf_size = elem_size * sizeof(float);
    int64_t cur_operand_ptr = (int64_t)&in_buf[0];
    io_buf_map[in_operand_name] = {cur_operand_ptr, elem_size, buf_size};

    exe_net->prepare_for_op(io_buf_map);

    // 步骤 2：遍历矫正集，获取每个 conv 的输入数据，并将每个 batch 的最大值保存到 max_elem_map 中
    int32_t img_i = 0;
    while (std::getline(file, line) && img_i < calibrate_img_num) {
        std::vector<std::string> words = split(line);
        std::string path = cfg_info_map["calibrate_img_folder"];
        std::string img = words[0];
        std::string img_path = path + img;

        transforms(in_buf, img_path, trans_cfg);
//        int32_t elem_size = in_buf.size();
//        int32_t buf_size = elem_size * sizeof(float);
//        int64_t cur_operand_ptr = (int64_t)&in_buf[0];
//        io_buf_map[in_operand_name] = {cur_operand_ptr, elem_size, buf_size};
//
//        exe_net->prepare_for_op(io_buf_map);

        std::unordered_map<std::string, std::string> cfg_info_map;

        exe_net->impl(io_buf_map, cfg_info_map);

        float schedule = (img_i + 1.0f) / calibrate_img_num / 2.0f * 100;
        int schedule_int = (int)schedule;
        buffer[schedule_int] = '#';
        printf("quant [%.2f%%] [%-100s][%c]\r", schedule, buffer, arr[schedule_int % 4]);
        fflush(stdout);

        // 获得到所有 conv op 的输入 vector
        for (auto op : exe_net->net_ptr->op_exec_order) {
            if (strcmp(op.get()->op_type, "Conv") == 0)
            {
                std::string ifmap_name = op.get()->in_operands[0];

                float * fmap_ptr = (float *)exe_net->operand_buf_map[ifmap_name].st_ptr;
                int64_t elem_size = exe_net->operand_buf_map[ifmap_name].elem_size;
                float max_abs = 0.0f;
                for (int i = 0; i < elem_size; ++i) {
                    max_abs = max_abs > std::fabs(fmap_ptr[i]) ? max_abs : std::fabs(fmap_ptr[i]);
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
    while (std::getline(file, line) && img_i < calibrate_img_num) {
        std::vector<std::string> words = split(line);
        std::string path = cfg_info_map["calibrate_img_folder"];
        std::string img = words[0];
        std::string img_path = path + img;

        transforms(in_buf, img_path, trans_cfg);
        int32_t elem_size = in_buf.size();
        int32_t buf_size = elem_size * sizeof(float);
        int64_t cur_operand_ptr = (int64_t)&in_buf[0];
        io_buf_map[in_operand_name] = {cur_operand_ptr, elem_size, buf_size};

        std::unordered_map<std::string, std::string> cfg_info_map;
        exe_net->prepare_for_op(io_buf_map);
        exe_net->impl(io_buf_map, cfg_info_map);

        float schedule = 50.0f + (img_i + 1.0f) / calibrate_img_num / 2.0f * 100;
        int schedule_int = (int)schedule;
        buffer[schedule_int] = '#';
        printf("quant [%.2f%%] [%-100s][%c]\r", schedule, buffer, arr[schedule_int % 4]);
        fflush(stdout);

        for (auto op : exe_net->net_ptr->op_exec_order) {
            if (strcmp(op.get()->op_type, "Conv") == 0)
            {
                std::string ifmap_name = op.get()->in_operands[0];
                float cur_ifmap_max = max_elem_map[ifmap_name];

                float * ofmap_ptr = (float *)exe_net->operand_buf_map[ifmap_name].st_ptr;
                int64_t elem_size = exe_net->operand_buf_map[ifmap_name].elem_size;
                for (int i = 0; i < elem_size; ++i) {
                    if (ofmap_ptr[i] == 0.0f){
                        continue;
                    }
                    int32_t idx = std::min((int)(fabs(ofmap_ptr[i]) * num_his_bins * 1.0f / cur_ifmap_max), num_his_bins - 1);
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
//        std::cout << "in mes quant type, ifmap name is: " << key << ", threshold is: " << threshold << ", best_bins is: " << best_bins << std::endl;
    }



    return 0;
}

int32_t model_quant(char *quant_one_buf_ptr, char *one_buf_ptr, CFG_MAP cfg_info_map) {

    int32_t one_file_size = std::stoi(cfg_info_map["one_file_size"]);
    memcpy(quant_one_buf_ptr, one_buf_ptr, one_file_size);

    net *net_1 = new net((void*)quant_one_buf_ptr);

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
        LOG_ERR("you should choose a quantification method, for exmaple: kl / mse / percent");
    }

    double op_ed = omp_get_wtime();
    double elapsed = op_ed - op_st;
    std::vector<int> img_num_vec = str2number<int>(cfg_info_map["calibrate_img_num"]);
    int32_t calibrate_img_num = img_num_vec[0];
//    std::cout << "============= using img num: " << calibrate_img_num << " to quant, using time is: " << elapsed << " s. =============="<< std::endl;

    ONE_MODEL_DESC_S *one_model_info_ptr = (ONE_MODEL_DESC_S *)quant_one_buf_ptr;
    int32_t node_cnt = one_model_info_ptr->node_cnt;
    char *cur_node_cfg_ptr = (char *) (quant_one_buf_ptr + one_model_info_ptr->node_cfg_offset);

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
    quant_conv_weight_bias(quant_one_buf_ptr);

    return 0;
}

#endif //ONENEW_MODEL_QUANT_HPP
