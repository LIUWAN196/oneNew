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

#include "ops_head.h"
#include "net.h"

int main(int argc, char **argv) {
    if (argc != 2)
    {
        LOG_ERR("Usage: %s [metrics.yml]", argv[0]);
    }
    const char *rt_cfg_txt = argv[1];
    std::string rt_cfg_txt_str(rt_cfg_txt);
    std::unordered_map<std::string, std::string> cfg_info_map;
    yml2map(cfg_info_map, rt_cfg_txt_str);

    const char *one_file_path = cfg_info_map["one_file_path"].c_str();
    net *model = new net(one_file_path);
    model->build_graph();
    extractor* exe_net = model->create_exe();

    auto* ifmap_of_model = (io*)exe_net->net_ptr->op_exec_order[0].get();
    std::string in_operand_name = ifmap_of_model->io_cfg.operand.operand_name;

    std::unordered_map<std::string, BUF_INFO_S> io_buf_map;

    std::vector<int> resize_shapes = str2number<int>(cfg_info_map["resize_shapes"]);
    std::vector<int> crop_shapes = str2number<int>(cfg_info_map["crop_shapes"]);
    std::vector<float> normal_mean = str2number<float>(cfg_info_map["normal_mean"]);
    std::vector<float> normal_std = str2number<float>(cfg_info_map["normal_std"]);

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

    int32_t metrics_img_num = std::stoi(cfg_info_map["metrics_img_num"]);

    std::ifstream file(cfg_info_map["metrics_img_label_txt_path"]);
    std::string line;

    char buffer[101] = {0};//存储进度条字符
    char arr[5] = {"-/|\\"};//存储基本的变化字幕
    printf("metrics [%.2f%%] [%-100s][%c]    img_cnt: %d/%d, top1: %.4f, top5: %.4f, time(s): %.2f\r",
           0.0f, buffer, arr[1], 0, metrics_img_num, 0.0f, 0.0f, 0.0f);
    fflush(stdout);

    double omp_st = omp_get_wtime();

    int32_t img_cnt = 0, top1_cnt = 0, top5_cnt = 0;
    while (std::getline(file, line) && img_cnt < metrics_img_num) {
        std::vector<std::string> words = split(line);
        std::string path = cfg_info_map["metrics_img_path"];
        std::string img = words[0];
        int32_t label = std::stoi(words[1]);
        std::string img_path = path + img;

        transforms(in_buf, img_path, trans_cfg);
        int32_t elem_size = in_buf.size();
        int32_t buf_size = elem_size * sizeof(float);
        int64_t cur_operand_ptr = (int64_t)&in_buf[0];
        io_buf_map[in_operand_name] = {cur_operand_ptr, elem_size, buf_size};

        exe_net->impl(io_buf_map, cfg_info_map);

        std::vector<int> argmax_topk = str2number<int>(cfg_info_map["topk"]);
        int topk = argmax_topk[0];

        for (auto io_buf: io_buf_map) {
            if (io_buf.first != in_operand_name) {
                // do softmax
                std::vector<float> softmax_out_info(1000, 0);
                {
                    SOFTMAX_CONFIG_S soft_max_cfg;
                    std::string op_type = "Softmax";
                    soft_max_cfg.axis = -1;
                    strcpy(soft_max_cfg.op_base_cfg.op_type, op_type.c_str());

                    std::vector<BUFFER_INFO_S> params;
                    std::vector<BUFFER_INFO_S> inputs;
                    std::vector<BUFFER_INFO_S> outputs;

                    BUFFER_INFO_S cfg;
                    cfg.addr = (int64_t) (&soft_max_cfg);
                    params.push_back(cfg);

                    OPERAND_S in;
                    in.dim_num_of_shapes = 2;
                    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
                        in.shapes[dim_i] = 1;
                    }
                    in.shapes[1] = 1000;    // the dim1 is C

                    BUFFER_INFO_S in_desc;
                    in_desc.addr = (int64_t) (&in);
                    params.push_back(in_desc);

                    OPERAND_S out;
                    BUFFER_INFO_S out_desc;
                    out_desc.addr = (int64_t) (&out);
                    params.push_back(out_desc);

                    BUFFER_INFO_S in_info;
                    in_info.addr = (int64_t) (io_buf.second.st_ptr);
                    inputs.push_back(in_info);

                    BUFFER_INFO_S out_info;
                    out_info.addr = (int64_t) (&softmax_out_info[0]);
                    outputs.push_back(out_info);

                    launch_post_process(params, inputs, outputs);
                }

                // do argmax
                std::vector<int32_t> argmax_out_info(topk, 0);

                {
                    ARGMAX_CONFIG_S argmax_cfg;
                    std::string op_type = "ArgMax";
                    argmax_cfg.axis = 1;
                    argmax_cfg.topk = topk;
                    strcpy(argmax_cfg.op_base_cfg.op_type, op_type.c_str());

                    std::vector<BUFFER_INFO_S> params;
                    std::vector<BUFFER_INFO_S> inputs;
                    std::vector<BUFFER_INFO_S> outputs;

                    BUFFER_INFO_S cfg;
                    cfg.addr = (int64_t) (&argmax_cfg);
                    params.push_back(cfg);

                    OPERAND_S in;
                    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
                        in.shapes[dim_i] = 1;
                    }
                    in.shapes[1] = 1000;    // the dim1 is C

                    BUFFER_INFO_S in_desc;
                    in_desc.addr = (int64_t) (&in);
                    params.push_back(in_desc);

                    OPERAND_S out;
                    BUFFER_INFO_S out_desc;
                    out_desc.addr = (int64_t) (&out);
                    params.push_back(out_desc);

                    BUFFER_INFO_S in_info;
                    in_info.addr = (int64_t) (&softmax_out_info[0]);
                    inputs.push_back(in_info);

                    BUFFER_INFO_S out_info;
                    out_info.addr = (int64_t) (&argmax_out_info[0]);
                    outputs.push_back(out_info);

                    launch_post_process(params, inputs, outputs);

                    if (label == argmax_out_info[0]) {
                        top1_cnt ++;
                        top5_cnt ++;
                    } else {
                        for (int i = 1; i < argmax_cfg.topk; ++i) {
                            if (label == argmax_out_info[i]) {
                                top5_cnt ++;
                            }
                        }
                    }
                    img_cnt ++;
                }   // end argmax

            }
        }

        if (img_cnt % 20 == 0 || metrics_img_num < 2000) {
            double omp_ed = omp_get_wtime();
            double elapsed = omp_ed - omp_st;
            double guess_using_time = elapsed / img_cnt * metrics_img_num;
            float schedule = (img_cnt + 1.0f) / metrics_img_num * 100;
            float top1_ratio = top1_cnt * 1.0f / img_cnt;
            float top5_ratio = top5_cnt * 1.0f / img_cnt;
            int schedule_int = (int)schedule;
            buffer[schedule_int] = '#';
            printf("metrics [%.2f%%] [%-100s][%c]    img_cnt: %d/%d, time(s): %.2f/%.2f, top1: %.4f, top5: %.4f\r",
                   schedule, buffer, arr[schedule_int % 4], img_cnt, metrics_img_num, elapsed, guess_using_time, top1_ratio, top5_ratio);
            fflush(stdout);
        }
    }

    return 0;

}

