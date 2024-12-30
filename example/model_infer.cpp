#include <cstring>
#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>

#ifdef USING_GPU
    #include "cuda_runtime.h"
#endif

#include "ops_head.h"
#include "net.h"
#include <dirent.h>

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <utility>
#include "math.h"

#include "CLIP_model.h"
#include "MobileSAM_model.h"
#include "Normal_model.h"

int32_t rt_cfg_check(std::unordered_map<std::string, std::string> &cfg_info_map) {

    std::string do_preprocess = str2lower_str(cfg_info_map["do_preprocess"]);
    if (do_preprocess.empty()) {    // default args: true
        set_default_args(cfg_info_map, "do_preprocess", "true");
    }
    if (str2lower_str(cfg_info_map["do_preprocess"]) != "true" &&
        str2lower_str(cfg_info_map["do_preprocess"]) != "false") {
        LOG_ERR("the args: do_preprocess must be set: true or false");
        return -1;
    }

    std::string do_postprocess = str2lower_str(cfg_info_map["do_postprocess"]);
    if (do_postprocess.empty()) {    // default args: false
        set_default_args(cfg_info_map, "do_postprocess", "false");
    }
    if (str2lower_str(cfg_info_map["do_postprocess"]) != "true" &&
        str2lower_str(cfg_info_map["do_postprocess"]) != "false") {
        LOG_ERR("the args: do_postprocess must be set: true or false");
        return -1;
    }

    std::string model_type = str2lower_str(cfg_info_map["model_type"]);
    if (model_type.empty()) {    // default args: normal
        set_default_args(cfg_info_map, "model_type", "normal");
    }
    if (str2lower_str(cfg_info_map["model_type"]) != "normal" &&
        str2lower_str(cfg_info_map["model_type"]) != "mobile_sam" &&
        str2lower_str(cfg_info_map["model_type"]) != "clip") {
        LOG_ERR("the args: model_type must be set: normal or mobile_sam or clip");
        return -1;
    }

    // 根据 model_type 读取对应的 one 文件的路径
    if (str2lower_str(cfg_info_map["model_type"]) == "normal") {
        std::string one_file_path = cfg_info_map["one_file_path"];
        if (one_file_path.empty()) {
            LOG_ERR("model_type is normal，the args: one_file_path must be set");
            return -1;
        }
    } else if (str2lower_str(cfg_info_map["model_type"]) == "mobile_sam") {
        std::string sam_encoder_one_file_path = cfg_info_map["sam_encoder_one_file_path"];
        std::string sam_decoder_one_file_path = cfg_info_map["sam_decoder_one_file_path"];
        if (sam_encoder_one_file_path.empty()) {
            LOG_ERR("model_type is mobile_sam，the args: sam_encoder_one_file_path must be set");
            return -1;
        }
        if (sam_decoder_one_file_path.empty()) {
            LOG_ERR("model_type is mobile_sam，the args: sam_decoder_one_file_path must be set");
            return -1;
        }
    } else if (str2lower_str(cfg_info_map["model_type"]) == "clip") {
        std::string clip_img_one_file_path = cfg_info_map["clip_img_one_file_path"];
        std::string clip_txt_one_file_path = cfg_info_map["clip_txt_one_file_path"];
        if (clip_img_one_file_path.empty()) {
            LOG_ERR("model_type is clip，the args: clip_img_one_file_path must be set");
            return -1;
        }
        if (clip_txt_one_file_path.empty()) {
            LOG_ERR("model_type is clip，the args: clip_txt_one_file_path must be set");
            return -1;
        }
    }

    std::string dump_fmap = str2lower_str(cfg_info_map["dump_ifmap&ofmap"]);
    if (dump_fmap.empty()) {    // default args: false
        set_default_args(cfg_info_map, "dump_ifmap&ofmap", "false");
    }
    if (str2lower_str(cfg_info_map["dump_ifmap&ofmap"]) != "true" &&
        str2lower_str(cfg_info_map["dump_ifmap&ofmap"]) != "false") {
        LOG_ERR("the args: dump_ifmap&ofmap must be set: true or false");
        return -1;
    }

    std::string model_exc_type = str2lower_str(cfg_info_map["model_exc_type"]);
    if (model_exc_type.empty()) {    // default args: efficient_exc
        set_default_args(cfg_info_map, "model_exc_type", "efficient_exc");
    }
    if (str2lower_str(cfg_info_map["model_exc_type"]) != "efficient_exc" &&
        str2lower_str(cfg_info_map["model_exc_type"]) != "ofmap_dumping" &&
        str2lower_str(cfg_info_map["model_exc_type"]) != "perf_profiling") {
        LOG_ERR("the args: model_exc_type must be set: efficient_exc or ofmap_dumping or perf_profiling");
        return -1;
    }

    if (str2lower_str(cfg_info_map["model_exc_type"]) == "ofmap_dumping" ||
        str2lower_str(cfg_info_map["dump_ifmap&ofmap"]) == "true") {
        std::string ofmap_folder = cfg_info_map["ofmap_folder"];
        if (ofmap_folder.empty()) {
            LOG_ERR("the args: ofmap_folder must be set, when model_exc_type is ofmap_dumping or dump_ifmap&ofmap is true");
            return -1;
        }
    }

    if (str2lower_str(cfg_info_map["do_postprocess"]) == "true") {
        std::string postprocess_type = cfg_info_map["postprocess_type"];
        if (postprocess_type.empty()) {
            LOG_ERR("the args: postprocess_type must set, when do_postprocess is true");
            return -1;
        }
        if (postprocess_type != "classify" && postprocess_type != "object_detect" &&
            postprocess_type != "pose_detect" && postprocess_type != "segment") {
            LOG_ERR("the args: postprocess_type must be set: classify or object_detect "
                    "or pose_detect or segment, when do_postprocess is true");
            return -1;
        }
    }

    std::string resize_shapes = cfg_info_map["resize_shapes"];
    if (resize_shapes.empty()) {
        LOG_ERR("the args: resize_shapes must be set, for example: [256, 256]");
        return -1;
    }

    std::string crop_shapes = cfg_info_map["crop_shapes"];
    if (crop_shapes.empty()) {
        LOG_ERR("the args: crop_shapes must be set, for example: [224, 224]");
        return -1;
    }

    std::string normal_mean = cfg_info_map["normal_mean"];
    if (normal_mean.empty()) {
        LOG_ERR("the args: normal_mean must be set, for example: [0.485ff, 0.456ff, 0.406ff]");
        return -1;
    }

    std::string normal_std = cfg_info_map["normal_std"];
    if (normal_mean.empty()) {
        LOG_ERR("the args: normal_std must be set, for example: [0.229f, 0.224f, 0.225f]");
        return -1;
    }

    if (str2lower_str(cfg_info_map["do_postprocess"]) == "true") {
        std::string postprocess_type = cfg_info_map["postprocess_type"];
        if (postprocess_type == "classify") {
            std::string topk = str2lower_str(cfg_info_map["topk"]);
            if (topk.empty()) {    // default args: 5
                set_default_args(cfg_info_map, "topk", "5");
            }
        } else {
            std::string score_threshold = str2lower_str(cfg_info_map["score_threshold"]);
            if (score_threshold.empty()) {    // default args: 0.5
                set_default_args(cfg_info_map, "score_threshold", "0.5");
            }
            std::string iou_threshold = str2lower_str(cfg_info_map["iou_threshold"]);
            if (iou_threshold.empty()) {    // default args: 0.7
                set_default_args(cfg_info_map, "iou_threshold", "0.7");
            }
        }

        if (postprocess_type == "object_detect" || postprocess_type == "segment") {
            std::string cls_num = str2lower_str(cfg_info_map["cls_num"]);
            if (cls_num.empty()) {
                LOG_ERR("the args: cls_num must be set, for example: 80");
                return -1;
            }
        }

        if (postprocess_type == "object_detect") {
            std::string net_type = str2lower_str(cfg_info_map["net_type"]);
            if (net_type.empty()) {
                LOG_ERR("the args: net_type must be set: "
                        "yolo_v3、yolo_v5、yolo_v7、yolo_v8、yolo_v10、yolo_world、rt_detr\");");
                return -1;
            }
        }
    }

    if (str2lower_str(cfg_info_map["model_exc_type"]) != "perf_profiling") {
        return 0;
    }

    // into there, the model_exc_type must be perf_profiling
    std::string hw_power = cfg_info_map["hw_computing_power (GOPS)"];
    if (hw_power.empty()) {
        LOG_ERR("the args: hw_computing_power (GOPS) must be set, for example: 3200");
        return -1;
    }

    std::string tracing_csv_path = cfg_info_map["tracing_csv_path"];
    if (tracing_csv_path.empty()) {
        LOG_ERR("the args: tracing_csv_path must be set");
        return -1;
    }

    return 0;
}

void printHelp(const std::string& programName) {
    std::cout << "Usage: " << programName << " [runtime.yml]，for example: configs/samples/runtime_sample.yml]\n";
}

int main(int argc, char **argv) {
    if (argc != 2) {
        LOG_ERR("Usage: %s [runtime.yml]，for example: configs/samples/runtime_sample.yml]", argv[0]);
    }

    std::vector<std::string> args(argv, argv + argc);
    for (const auto& arg : args) {
        if (arg == "-h" || arg == "--h" || arg == "-help" || arg == "--help") {
            printHelp(args[0]);
            return 0;
        }
    }

    const char *rt_cfg_txt = argv[1];
    std::string rt_cfg_txt_str(rt_cfg_txt);
    std::unordered_map<std::string, std::string> cfg_info_map;
    yml2map(cfg_info_map, rt_cfg_txt_str);

    rt_cfg_check(cfg_info_map);

    std::string model_type = str2lower_str(cfg_info_map["model_type"]);
    if (model_type == "normal") {    // do normal model, such as resnet、yolo、vit
        do_normal_model(cfg_info_map);
    } else if (model_type == "mobile_sam") {    // do mobile_sam
        do_mobile_sam(cfg_info_map);
    } else if (model_type == "clip") {    // do clip
        do_clip(cfg_info_map);
    }

    return 0;
}

