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


#include <iostream>
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"

int transforms(std::vector<float> &rgb, std::string img_path, TRANSFORMS_CONFIG_S &trans_cfg) {
    // step 0 : read img
    cv::Mat ori_img = cv::imread(img_path);
    if (ori_img.empty()) {
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


int launch_post_process(std::vector<BUFFER_INFO_S> &params, std::vector<BUFFER_INFO_S> &inputs,
                        std::vector<BUFFER_INFO_S> &outputs) {
    Manager &m = Manager::getInstance();

    char *op_cfg_ptr = (char *) (params[0].addr);

    creator_ creator_op_method = m.Opmap[std::string(op_cfg_ptr)];

    std::shared_ptr<op> op_ptr;

    creator_op_method(op_ptr, op_cfg_ptr);

    op_ptr->prepare(op_cfg_ptr);
    op_ptr->forward(&params[0], &inputs[0], &outputs[0]);


    return 0;
}

std::vector<std::string> split(const std::string &s) {
    std::vector<std::string> words;
    std::istringstream stream(s);

    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }

    return words;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " [runtime config.txt]" << std::endl;
        exit(-1);
    }
    const char *rt_cfg_txt = argv[1];
    std::string rt_cfg_txt_str(rt_cfg_txt);
    std::unordered_map<std::string, std::string> cfg_info_map;
    yml2map(cfg_info_map, rt_cfg_txt_str);

    const char *one_file_path = cfg_info_map["one_file_path"].c_str();
    net *model = new net(one_file_path);
    std::string in_operand_name = cfg_info_map["in_operand_name"];

    model->build_graph();
    extractor *exe_net = model->create_exe();

    std::unordered_map<std::string, std::vector<float>> io_buf_map;

    std::vector<int> crop_shapes = str2number<int>(cfg_info_map["crop_shapes"]);
    std::vector<int> resize_shapes = str2number<int>(cfg_info_map["resize_shapes"]);
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

    std::ifstream file(cfg_info_map["imagenet_label_txt_path"]);
    std::string line;

    double omp_st = omp_get_wtime();

    int32_t img_cnt = 0, top1_cnt = 0, top5_cnt = 0;
    while (std::getline(file, line)) {
        std::vector<std::string> words = split(line);
        std::string path = cfg_info_map["calibrate_img_path"];
        std::string img = words[0];
        int32_t label = std::stoi(words[1]);
        std::string img_path = path + img;

//        printf("tha tar label is %d\n", label);

        transforms(in_buf, img_path, trans_cfg);
        io_buf_map[in_operand_name] = in_buf;

        std::string ifmap_folder = cfg_info_map["ofmap_folder"];
        std::string ifmap_name("model_ifmap.bin");
        std::string ifmap_path = ifmap_folder + ifmap_name;
//        write_bin(ifmap_path.c_str(), in_elem_size * sizeof(float), (char *) &in_buf[0]);

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
                    in_info.addr = (int64_t) (&io_buf.second[0]);
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
                    std::string op_type = "Argmax";
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


        if (img_cnt % 100 == 0) {
            double omp_ed = omp_get_wtime();
            double elapsed = omp_ed - omp_st;
            printf("img cnt is %d, top1 ratio is %f, top5 ratio is %f, using time: %fs\n",
                   img_cnt, top1_cnt * 1.0f / img_cnt, top5_cnt * 1.0f / img_cnt, elapsed);
        }
    }

    return 0;

}

