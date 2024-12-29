#ifndef ONENEW_NORMAL_MODEL_H
#define ONENEW_NORMAL_MODEL_H

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
#include <memory>

#include "post_process.hpp"

int do_normal_model(std::unordered_map<std::string, std::string> cfg_info_map) {

    const char* one_file_path = cfg_info_map["one_file_path"].c_str();
    net *model = new net(one_file_path);
    model->build_graph();

    extractor* exe_net = model->create_exe();

    auto* ifmap_of_model = (io*)exe_net->net_ptr->op_exec_order[0].get();
    std::string in_operand_name = ifmap_of_model->io_cfg.operand.operand_name;

    std::unordered_map<std::string, BUFFER_INFO_S> io_buf_map;

    TRANSFORMS_CONFIG_S trans_cfg = cfg_info_map2preprocess_params(cfg_info_map);
    int in_elem_size = 3 * trans_cfg.crop_size[0] * trans_cfg.crop_size[1];

    std::vector<float> in_buf(in_elem_size);

    std::string img_path = cfg_info_map["input_data_path"];

    transforms(in_buf, img_path, trans_cfg);

    int32_t elem_size = in_buf.size();
    int32_t buf_size = elem_size * sizeof(float);
    int64_t cur_operand_ptr = (int64_t)&in_buf[0];
    io_buf_map[in_operand_name] = {cur_operand_ptr, elem_size, buf_size};


    std::string ifmap_folder = cfg_info_map["ofmap_folder"];
    std::string ifmap_name("model_ifmap.bin");
    std::string ifmap_path = ifmap_folder + ifmap_name;

    exe_net->prepare_for_op(io_buf_map);

    if (cfg_info_map["model_exc_type"] == "ofmap_dumping") {
        write_bin(ifmap_path.c_str(), in_elem_size * sizeof(float), (char *)&in_buf[0]);
        exe_net->impl_dump_ofmap(io_buf_map, cfg_info_map);
    } else if (cfg_info_map["model_exc_type"] == "perf_profiling") {
        const int32_t repeat_cnt = 10;  // 前 9 次预热，最后保留下来的是第 10 次的耗时
        for (int cnt_i = 0; cnt_i < repeat_cnt; ++cnt_i) {
            exe_net->impl_tracing(io_buf_map, cfg_info_map);
        }
    } else {
        exe_net->impl(io_buf_map, cfg_info_map);
    }

    if (cfg_info_map["dump_ifmap&ofmap"] == "true") {
        write_bin(ifmap_path.c_str(), in_elem_size * sizeof(float), (char *)&in_buf[0]);

        for (auto io_buf:io_buf_map) {
            if (io_buf.first != in_operand_name) {
                std::string ofmap_name = io_buf.first;
                char* omap_name_c = (char*)ofmap_name.c_str();
                std::string ofmap_name_replace_char(replace_char(omap_name_c));

                std::string ofmap_path = ifmap_folder + ofmap_name_replace_char;
                write_bin(ofmap_path.c_str(), io_buf.second.buf_size, (char *) io_buf.second.addr);
            }
        }
    }

    if (cfg_info_map["do_postprocess"] == "false") {
        return 0;
    }

    std::unique_ptr<PostProcessPerform> post_process_cls;
    if (cfg_info_map["postprocess_type"] == "classify") {
        post_process_cls = std::make_unique<ClassifyPerform>(cfg_info_map, io_buf_map, in_operand_name);
    } else if (cfg_info_map["postprocess_type"] == "object_detect") {
        post_process_cls = std::make_unique<ObjectDetectPerform>(cfg_info_map, io_buf_map, in_operand_name);
    } else if (cfg_info_map["postprocess_type"] == "pose_detect") {
        post_process_cls = std::make_unique<PoseDetectPerform>(cfg_info_map, io_buf_map, in_operand_name);
    } else if (cfg_info_map["postprocess_type"] == "segment") {
        post_process_cls = std::make_unique<SegmentPerform>(cfg_info_map, io_buf_map, in_operand_name);
    }

    if (!post_process_cls) {
        LOG_ERR("post_process_cls == nullptr, please check post process setting");
        return 0;
    }
    post_process_cls->do_post_process();

    return 0;
}
#endif //ONENEW_NORMAL_MODEL_H
