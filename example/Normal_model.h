//
// Created by wanzai on 24-11-9.
//

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

#include "post_process.hpp"

int do_normal_model(std::unordered_map<std::string, std::string> cfg_info_map) {

    const char* one_file_path = cfg_info_map["one_file_path"].c_str();
    net *model = new net(one_file_path);
    model->build_graph();

//    LOG_DBG("开始开辟空间");
    extractor* exe_net = model->create_exe();

//    LOG_ERR("先看看 shape infer 有问题没有");

    auto* ifmap_of_model = (io*)exe_net->net_ptr->op_exec_order[0].get();
    std::string in_operand_name = ifmap_of_model->io_cfg.operand.operand_name;

    std::unordered_map<std::string, BUF_INFO_S> io_buf_map;

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
        const int32_t repeat_cnt = 5;  // 前 4 次预热，最后保留下来的是第五次的耗时
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
                write_bin(ofmap_path.c_str(), io_buf.second.buf_size, (char *) io_buf.second.st_ptr);
            }
        }
    }

    if (cfg_info_map["do_postprocess"] == "false") {
        return 0;
    }

    if (cfg_info_map["postprocess_type"] == "classify") {
        ClassifyPerform classify_cls(cfg_info_map, io_buf_map, in_operand_name);
        classify_cls.do_post_process();
    } else if (cfg_info_map["postprocess_type"] == "object_detect") {
        ObjectDetectPerform obj_detect_cls(cfg_info_map, io_buf_map, in_operand_name);
        obj_detect_cls.do_post_process();
    } else if (cfg_info_map["postprocess_type"] == "pose_detect") {
        PoseDetectPerform pose_detect_cls(cfg_info_map, io_buf_map, in_operand_name);
        pose_detect_cls.do_post_process();
    } else if (cfg_info_map["postprocess_type"] == "segment") {
        SegmentPerform segment_cls(cfg_info_map, io_buf_map, in_operand_name);
        segment_cls.do_post_process();
    }

//
//    if (cfg_info_map["postprocess_type"] == "segment") {
//        // do segment op
//        const int32_t max_keep_box_num = 30;
//        std::vector<float> detect_out0_info(max_keep_box_num * sizeof(SEGMENT_OFMAP0_S), 0);
//        std::vector<float> detect_out1_info(max_keep_box_num * (32 * 160 * 160), 0);
//        SEGMENT_CONFIG_S segment_cfg;
//        std::string op_type = "Segment";
//        strcpy(segment_cfg.op_base_cfg.op_type, op_type.c_str());
//
//        std::vector<BUFFER_INFO_S> params;
//        std::vector<BUFFER_INFO_S> inputs;
//        std::vector<BUFFER_INFO_S> outputs;
//
//        // fill params
//        BUFFER_INFO_S cfg;
//        cfg.addr = (int64_t) (&segment_cfg);
//        params.push_back(cfg);
//
//        OPERAND_S out;
//        BUFFER_INFO_S out_desc;
//        out_desc.addr = (int64_t) (&out);
//        params.push_back(out_desc);
//
//        // fill inputs
//        // find the three ofmap of backbone, they are the ifmap of detct op
//        BUFFER_INFO_S in0_info;
//        std::string in0_name = get_string_vec(cfg_info_map["ofmap_name"])[0];
//        in0_info.addr = (int64_t) (&io_buf_map[in0_name].st_ptr);
//        inputs.push_back(in0_info);
//
//        BUFFER_INFO_S in1_info;
//        std::string in1_name = get_string_vec(cfg_info_map["ofmap_name"])[1];
//        in1_info.addr = (int64_t) (&io_buf_map[in1_name].st_ptr);
//        inputs.push_back(in1_info);
//
//        BUFFER_INFO_S in2_info;
//        std::string in2_name = get_string_vec(cfg_info_map["ofmap_name"])[2];
//        in2_info.addr = (int64_t) (&io_buf_map[in2_name].st_ptr);
//        inputs.push_back(in2_info);
//
//        BUFFER_INFO_S in3_info;
//        std::string in3_name = get_string_vec(cfg_info_map["ofmap_name"])[3];
//        in3_info.addr = (int64_t) (&io_buf_map[in3_name].st_ptr);
//        inputs.push_back(in3_info);
//
//        BUFFER_INFO_S in4_info;
//        std::string in4_name = get_string_vec(cfg_info_map["ofmap_name"])[4];
//        in4_info.addr = (int64_t) (&io_buf_map[in4_name].st_ptr);
//        inputs.push_back(in4_info);
//
//        BUFFER_INFO_S in5_info;
//        std::string in5_name = get_string_vec(cfg_info_map["ofmap_name"])[5];
//        in5_info.addr = (int64_t) (&io_buf_map[in5_name].st_ptr);
//        inputs.push_back(in5_info);
//
//        // fill outputs
//        BUFFER_INFO_S out0_info;
//        out0_info.addr = (int64_t) (&detect_out0_info[0]);
//        outputs.push_back(out0_info);
//
//        BUFFER_INFO_S out1_info;
//        out1_info.addr = (int64_t) (&detect_out1_info[0]);
//        outputs.push_back(out1_info);
//
//        // do segment
//        launch_post_process(params, inputs, outputs);
//    }
//


}
#endif //ONENEW_NORMAL_MODEL_H
