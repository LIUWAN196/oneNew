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

int32_t rt_cfg_check(std::unordered_map<std::string, std::string>& cfg_info_map){

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

    std::string one_file_path = cfg_info_map["one_file_path"];
    if (one_file_path.empty()) {    // default args: false
        LOG_ERR("the args: one_file_path must be set");
        return -1;
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
        if (postprocess_type != "classification" && postprocess_type != "segmentation" &&
            postprocess_type != "pose_detection") {
            LOG_ERR("the args: postprocess_type must be set: classification or segmentation "
                    "or pose_detection, when do_postprocess is true");
            return -1;
        }
    }

    if (str2lower_str(cfg_info_map["do_postprocess"]) == "true") {
        std::string postprocess_type = cfg_info_map["postprocess_type"];
        if (postprocess_type == "segmentation" || postprocess_type == "pose_detection") {
            std::string ofmap_name = cfg_info_map["ofmap_name"];
            if (ofmap_name.empty()) {
                LOG_ERR("the args: ofmap_name must set, when postprocess_type is %s", postprocess_type.c_str());
                return -1;
            }
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

    std::string topk = str2lower_str(cfg_info_map["topk"]);
    if (topk.empty()) {    // default args: 5
        set_default_args(cfg_info_map, "topk", "5");
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

int show_img(std::vector<std::pair<std::string, float>> img_ratio, std::string img_folder_path) {
    const float abs_probability_threshold = 0.01f;
    const float ratio_with_last_probability = 4.0f;
    const int total_w = 832;

    std::vector<std::pair<std::string, float>> to_show_img;
    to_show_img.push_back(img_ratio[0]);
    for (int i = 1; i < img_ratio.size(); ++i) {
        float last_probability = img_ratio[i - 1].second;
        float cur_probability = img_ratio[i].second;
        if (cur_probability > abs_probability_threshold && last_probability / cur_probability < ratio_with_last_probability) {
            to_show_img.push_back(img_ratio[i]);
        }
    }

    // resize img
    const int to_show_img_cnt = to_show_img.size();
    const int img_cnt_w = ceil(sqrt((float)to_show_img_cnt));
    const int img_cnt_h = ceil(sqrt(to_show_img_cnt * 1.0f / img_cnt_w));
    const int pixel_per_img = total_w / img_cnt_w;

    std::vector<cv::Mat> img_resized;
    for (int i = 0; i < to_show_img.size(); ++i) {
        std::string img_path = img_folder_path + to_show_img[i].first;
        cv::Mat src = cv::imread(img_path);
        cv::Mat dst_0;
        int src_h = src.rows;
        int src_w = src.cols;
        int src_len_max = src_h > src_w ? src_h : src_w;
        float ratio = pixel_per_img * 1.0f / src_len_max;

        cv::resize(src, dst_0, cv::Size((int)(ratio * src_w), (int)(ratio * src_h)), 0, 0, cv::INTER_LINEAR);

        // 计算需要添加的边框宽度
        int top = (pixel_per_img - dst_0.rows) / 2;
        int bottom = pixel_per_img - dst_0.rows - top;
        int left = (pixel_per_img - dst_0.cols) / 2;
        int right = pixel_per_img - dst_0.cols - left;

        cv::Mat dst_1;
        cv::copyMakeBorder(dst_0, dst_1, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        img_resized.push_back(dst_1);
    }

    cv::Mat blackImage = cv::Mat::zeros(pixel_per_img, pixel_per_img, CV_8UC3);
    for (int i = to_show_img.size(); i < img_cnt_h * img_cnt_w; ++i) {
        img_resized.push_back(blackImage);
    }

    // concat img
    cv::Mat total_concat_img;
    std::vector<cv::Mat> v_img_vec(img_cnt_h);
    for (int i = 0; i < img_cnt_h; ++i) {
        std::vector<cv::Mat> w_img_vec;
        for (int j = i * img_cnt_w; j < (i + 1) * img_cnt_w; ++j) {

            if (j < to_show_img_cnt) {
                int fontFace = cv::FONT_HERSHEY_COMPLEX;
                double fontScale = 0.75;
                int thickness = 1;
                cv::Scalar lineType = cv::LINE_AA;
                // show img name
                {
                    std::string txt = to_show_img[j].first;
                    int baseline;
                    cv::Size textSize = cv::getTextSize(txt, fontFace, fontScale, thickness, &baseline);
                    cv::Point textOrg(10, textSize.height + 6);

                    // 绘制底色矩形（黑色）
                    cv::Point topLeft(textOrg.x, textOrg.y - textSize.height - 2);
                    cv::Point bottomRight(textOrg.x + textSize.width, textOrg.y + 5 + thickness);
                    cv::rectangle(img_resized[j], topLeft, bottomRight, cv::Scalar(0, 0, 0), cv::FILLED);

                    cv::putText(img_resized[j], txt, textOrg, fontFace, fontScale, cv::Scalar(17, 90, 197), thickness, cv::LINE_AA);

                }
                // show score
                {
                    char buffer[64]; // 确保缓冲区足够大以存储格式化后的字符串
                    sprintf(buffer, "%.3f", to_show_img[j].second); // 设置精度为3
                    std::string score(buffer);
                    std::string txt = "score: " + score;
                    int baseline;
                    cv::Size textSize = cv::getTextSize(txt, fontFace, fontScale, thickness, &baseline);
                    cv::Point textOrg(10, textSize.height + 32); // 调整为左上角

                    // 绘制底色矩形（黑色）
                    cv::Point topLeft(textOrg.x, textOrg.y - textSize.height - 2);
                    cv::Point bottomRight(textOrg.x + textSize.width, textOrg.y + 5 + thickness);
                    cv::rectangle(img_resized[j], topLeft, bottomRight, cv::Scalar(0, 0, 0), cv::FILLED);

                    cv::putText(img_resized[j], txt, textOrg, fontFace, fontScale, cv::Scalar(17, 90, 197),
                                thickness);
                }
            }
            w_img_vec.push_back(img_resized[j]);
        }
        cv::hconcat(w_img_vec, v_img_vec[i]);
    }

    // show img
    cv::vconcat(v_img_vec, total_concat_img);
    cv::imshow("Image", total_concat_img);
    cv::waitKey(0);

    return 0;
}

int do_clip(std::unordered_map<std::string, std::string> cfg_info_map) {
    DIR *dir;
    struct dirent *ent;

    std::vector<std::pair<std::string, std::vector<float>>> img_ofmp;
    std::vector<std::string> img_name_vec;

    std::string img_folder_path = cfg_info_map["img_folder_path"];
    if ((dir = opendir(img_folder_path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string img_name(ent->d_name);
            if (img_name.front() != '.') {
                img_name_vec.push_back(img_name);
            }
        }
        closedir(dir);
    } else {
        // 打开目录失败
        LOG_ERR("open img_folder_path failed");
        return -1;
    }

    // do clip img
    const char* one_file_path = cfg_info_map["one_file_path"].c_str();
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

    const int img_num = img_name_vec.size();
    char buffer[101] = {0};//存储进度条字符
    char arr[5] = {"-/|\\"};//存储基本的变化字幕
    printf("img encoding [%.2f%%] [%-100s][%c]    img_cnt: %d/%d, time(s): %.2f\r",
           0.0f, buffer, arr[1], 0, img_num, 0.0f);
    buffer[0] = '#';
    fflush(stdout);
    double omp_st = omp_get_wtime();

    for (int img_i = 0; img_i < img_num; ++img_i) {
        std::string img_path = img_folder_path + img_name_vec[img_i];

        transforms(in_buf, img_path, trans_cfg);
        int64_t st_ptr = (int64_t)(&in_buf[0]);
        int32_t elem_size = (int64_t)(in_buf.size());
        int32_t buf_size = (int64_t)(elem_size * sizeof(float));
        io_buf_map[in_operand_name] = {st_ptr, elem_size, buf_size};

        std::string ifmap_folder = cfg_info_map["ofmap_folder"];
        std::string ifmap_name("model_ifmap.bin");
        std::string ifmap_path = ifmap_folder + ifmap_name;

        exe_net->impl(io_buf_map, cfg_info_map);

        std::string ofmap = "image_features";

        BUF_INFO_S ofmap_info = io_buf_map[ofmap];
        std::vector<float> clip_img_ofmap(ofmap_info.elem_size);
        memcpy(&clip_img_ofmap[0], (void *)ofmap_info.st_ptr, ofmap_info.buf_size);

        // 处理 img 的输出
        double psum = 0;
        for (int i = 0; i < clip_img_ofmap.size(); ++i) {
            psum += clip_img_ofmap[i] * clip_img_ofmap[i];
        }
        double psum_sqrt = sqrt(psum);
        for (int i = 0; i < clip_img_ofmap.size(); ++i) {
            clip_img_ofmap[i] = clip_img_ofmap[i] * 100.0f / psum_sqrt;
        }
        img_ofmp.push_back({img_name_vec[img_i], clip_img_ofmap});


        double omp_ed = omp_get_wtime();
        double elapsed = omp_ed - omp_st;
        double guess_using_time = elapsed / img_i * img_num;
        float schedule = img_i * 1.0f / img_num * 100;
        int schedule_int = (int)schedule;
        float last_schedule = ((img_i - 1) > 0 ? img_i - 1 : 0) * 1.0f / img_num * 100;
        int last_schedule_int = (int)last_schedule;
        for (int sche_i = last_schedule_int + 1; sche_i < schedule_int + 1; ++sche_i) {
            buffer[sche_i] = '#';
        }
        printf("img encoding [%.2f%%] [%-100s][%c]    img_cnt: %d/%d, time(s): %.2f/%.2f\r",
               schedule, buffer, arr[schedule_int % 4], img_i, img_num, elapsed, guess_using_time);
        fflush(stdout);
    }
    double omp_ed = omp_get_wtime();
    double elapsed = omp_ed - omp_st;
    for (int i = 0; i < 101; ++i) {
        buffer[i] = '#';
    }
    printf("img encoding done [%.2f%%] [%-100s][%c]    img_cnt: %d/%d, time(s): %.2f/%.2f\r",
           100.0, buffer, arr[100 % 4], img_num, img_num, elapsed, elapsed);
    fflush(stdout);
    printf("\n\n");
    fflush(stdout);

    // do clip txt
    std::string token_list;
    // 使用一个无限循环来持续读取输入
    while (true) {
        std::string yml_token = str2lower_str(cfg_info_map["token"]);
        if (yml_token.empty()) {    // default args: true
            std::cout << "please enter token（exit the program using 'exit'）>: ";
            std::getline(std::cin, token_list); // 读取一整行输入
        } else {
            token_list = yml_token;
        }

        // 检查是否输入了'exit'以退出循环
        if (token_list == "exit") {
            break;
        }

        const char* txt_one_file_path = cfg_info_map["clip_txt_one_file_path"].c_str();
        net *txt_model = new net(txt_one_file_path);
        txt_model->build_graph();
        extractor* txt_exe_net = txt_model->create_exe();

        auto* txt_ifmap_of_model = (io*)txt_exe_net->net_ptr->op_exec_order[0].get();
        std::string txt_in_operand_name = txt_ifmap_of_model->io_cfg.operand.operand_name;

        std::unordered_map<std::string, BUF_INFO_S> txt_io_buf_map;

        std::vector<int> texts = str2number<int>(token_list);
//        std::vector<int> texts = str2number<int>(cfg_info_map["token_list"]);
        std::vector<float> texts_float(texts.size());
        memcpy((char *)&texts_float[0], (char *)&texts[0], texts.size() * sizeof(float));

        int32_t elem_size = texts_float.size();
        int32_t buf_size = elem_size * sizeof(float);
        int64_t cur_operand_ptr = (int64_t)&texts_float[0];
        txt_io_buf_map[txt_in_operand_name] = {cur_operand_ptr, elem_size, buf_size};

        std::string txt_ifmap_folder = cfg_info_map["ofmap_folder"];
        std::string txt_ifmap_name("model_ifmap.bin");
        std::string txt_ifmap_path = txt_ifmap_folder + txt_ifmap_name;

        if (cfg_info_map["model_exc_type"] == "ofmap_dumping") {
            write_bin(txt_ifmap_path.c_str(), texts.size() * sizeof(float), (char *)&texts[0]);
            txt_exe_net->impl_dump_ofmap(txt_io_buf_map, cfg_info_map);
        } else if (cfg_info_map["model_exc_type"] == "perf_profiling") {
            const int32_t repeat_cnt = 5;  // 前 4 次预热，最后保留下来的是第五次的耗时
            for (int cnt_i = 0; cnt_i < repeat_cnt; ++cnt_i) {
                txt_exe_net->impl_tracing(txt_io_buf_map, cfg_info_map);
            }
        } else {
            txt_exe_net->impl(txt_io_buf_map, cfg_info_map);
        }

        std::string txt_ofmap = "text_features";

        BUF_INFO_S ofmap_info = txt_io_buf_map[txt_ofmap];
        std::vector<float> clip_txt_ofmap(ofmap_info.elem_size);
        memcpy(&clip_txt_ofmap[0], (void *)ofmap_info.st_ptr, ofmap_info.buf_size);

        // 计算图片和文字的匹配程度
        std::vector<std::pair<std::string, float>> img_sim;
        for (int img_i = 0; img_i < img_num; ++img_i) {
            std::vector<float> clip_img_ofmap = img_ofmp[img_i].second;
            float sim = 0;
            for (int i = 0; i < clip_img_ofmap.size(); ++i) {
                sim += clip_txt_ofmap[i] * clip_img_ofmap[i];
            }
            img_sim.push_back({img_ofmp[img_i].first, sim});
        }

        std::sort(img_sim.begin(), img_sim.end(),
                  [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
                      return a.second > b.second; // 降序排序
                  });


        // do softmax
        // find max
        float max_sim = -32768.0f;
        for (int img_i = 0; img_i < img_num; ++img_i) {
            max_sim = img_sim[img_i].second > max_sim ? img_sim[img_i].second : max_sim;
        }
        // calc sum(exp(x - max))
        float psum = 0;
        for (int img_i = 0; img_i < img_num; ++img_i) {
            img_sim[img_i].second = exp(img_sim[img_i].second - max_sim);
            psum += img_sim[img_i].second;
        }
        // do softmax
        for (int img_i = 0; img_i < img_num; ++img_i) {
            img_sim[img_i].second /= psum;
        }

        show_img(img_sim, img_folder_path);

    }
    return 0;
}



// 传入 resize 到 1024x1024 的图片，以及 std::vector<uint8_t> mask_vec 数据，还有 mask 的 BGR 值。返回一个带 mask 的图像
cv::Mat get_masked_img(cv::Mat src_img, std::vector<uint8_t> mask_vec, cv::Scalar& mask_color) {

    const int width = src_img.cols; // 图像宽度
    const int height = src_img.rows; // 图像高度

    // 读取mask图像，假设mask是一个单通道的灰度图像
    cv::Mat mask(height, width, CV_8UC1, mask_vec.data()); // CV_8UC1 表示单通道8位无符号整数

    // 创建一个 mask 的图像
    cv::Mat lightBlueImage = cv::Mat::zeros(src_img.size(), src_img.type());
    lightBlueImage.setTo(mask_color);

    // 按照权重混合原图和 mask
    double alpha = 0.75; // mask的权重
    double beta = 0.25;  // 原图的权重
    cv::Mat masked_img = src_img.clone();

    // 对于 mask 为 255 的部分应用淡蓝色
    for (int row = 0; row < mask.rows; ++row) {
        for (int col = 0; col < mask.cols; ++col) {
            if (mask.at<uchar>(row, col) == 255) // 如果 mask 值为 255，则混合
            {
                cv::Vec3b &pixel = masked_img.at<cv::Vec3b>(row, col);
                cv::Vec3b lightBluePixel = lightBlueImage.at<cv::Vec3b>(row, col);
                cv::Vec3b originalPixel = src_img.at<cv::Vec3b>(row, col);
                for (int c = 0; c < 3; ++c) {
                    pixel[c] = static_cast<uchar>((alpha * lightBluePixel[c]) + (beta * originalPixel[c]));
                }
            }
        }
    }

    return masked_img;
}

#include <opencv2/opencv.hpp>
#include <iostream>

// 上下文结构体
struct MouseCallbackContext {
    extractor* decoder_exe_net;
    cv::Mat image_resized;
    std::unordered_map<std::string, BUF_INFO_S> decoder_io_buf_map;
    std::unordered_map<std::string, std::string> cfg_info_map;
};

// 鼠标回调函数
void mouseCallback(int event, int x, int y, int, void* userdata) {
    MouseCallbackContext* context = static_cast<MouseCallbackContext*>(userdata);
    extractor* decoder_exe_net = context->decoder_exe_net;
    cv::Mat image_resized = context->image_resized;
    std::unordered_map<std::string, BUF_INFO_S> decoder_io_buf_map = context->decoder_io_buf_map;
    std::unordered_map<std::string, std::string> cfg_info_map = context->cfg_info_map;
    if (event == cv::EVENT_LBUTTONDOWN) {
        int32_t w_coord = cv::Point(x, y).x;
        int32_t h_coord = cv::Point(x, y).y;
//        printf("w_coord is %d, h_coord is %d\n", w_coord, h_coord);

        std::string point_coords = "point_coords";
        // 自己构造一个坐标，给到 decoder 模型作为输入
        std::vector<float> point_coords_buf = {w_coord * 1.0f, h_coord * 1.0f};
        int64_t point_coords_st_ptr = (int64_t)(&point_coords_buf[0]);
        int32_t point_coords_elem_size = (int64_t)(point_coords_buf.size());
        int32_t point_coords_buf_size = (int64_t)(point_coords_elem_size * sizeof(float));
        decoder_io_buf_map[point_coords] = {point_coords_st_ptr, point_coords_elem_size, point_coords_buf_size};

        decoder_exe_net->impl(decoder_io_buf_map, cfg_info_map);
        std::string sam_model_ofmap = "masks";

        BUF_INFO_S decoder_ofmap_info = decoder_io_buf_map[sam_model_ofmap];
        std::vector<float> decoder_ofmap(decoder_ofmap_info.elem_size);
        memcpy(&decoder_ofmap[0], (void *)decoder_ofmap_info.st_ptr, decoder_ofmap_info.buf_size);


        // 开始准备绘制带 mask 的图像
        std::vector<int> mask_level_vec = str2number<int>(cfg_info_map["sam_mask_level"]);
        int32_t mask_level = mask_level_vec[0];  // 0  1  2  3
        const int32_t mask_length = 1024 * 1024;
        std::vector<uint8_t> mask(mask_length);
        for (int st_i = 0; st_i < mask_length; ++st_i) {
            mask[st_i] = decoder_ofmap[mask_level * mask_length + st_i] > 0 ? 255 : 0;
        }

        // step 3: 为图片加 mask
        cv::Scalar mask_color_1(135, 116, 190); // BGR
//        cv::Scalar mask_color_1(190, 207, 229); // BGR
        image_resized = get_masked_img(image_resized, mask, mask_color_1);

        imshow("Image", image_resized);
        cv::waitKey(0);
    }
}

int do_mobile_sam(std::unordered_map<std::string, std::string> cfg_info_map) {

    // do sam encoder
    const char* one_file_path = cfg_info_map["one_file_path"].c_str();
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

    std::string img_path = cfg_info_map["input_data_path"];

    transforms(in_buf, img_path, trans_cfg);
    int64_t st_ptr = (int64_t)(&in_buf[0]);
    int32_t elem_size = (int64_t)(in_buf.size());
    int32_t buf_size = (int64_t)(elem_size * sizeof(float));
    io_buf_map[in_operand_name] = {st_ptr, elem_size, buf_size};

    std::string ifmap_folder = cfg_info_map["ofmap_folder"];
    std::string ifmap_name("model_ifmap.bin");
    std::string ifmap_path = ifmap_folder + ifmap_name;

    exe_net->impl(io_buf_map, cfg_info_map);

    std::string ofmap = "image_embeddings";
    BUF_INFO_S encoder_ofmap_info = io_buf_map[ofmap];


//    char *image_embeddings_buf = (char*) aligned_alloc(32, 256*64*64*sizeof(float));
//    std::string file_name = "/home/wanzai/桌面/mobile_sam_test_img/image_embeddings";
//    load_bin(file_name.c_str(), 256*64*64*sizeof(float), image_embeddings_buf);
//    BUF_INFO_S encoder_ofmap_info = {(int64_t)image_embeddings_buf, 256*64*64, 256*64*64*sizeof(float)};


    // do sam decoder
    const char* decoder_one_file_path = cfg_info_map["sam_decoder_one_file_path"].c_str();
    net *decoder_model = new net(decoder_one_file_path);
    decoder_model->build_graph();
    extractor* decoder_exe_net = decoder_model->create_exe();

    std::string image_embeddings = "image_embeddings";
    std::string point_coords = "point_coords";

    std::unordered_map<std::string, BUF_INFO_S> decoder_io_buf_map;

    // 将第一部分的 encoder 的结果给到 decoder 模型作为输入
    decoder_io_buf_map[image_embeddings] = encoder_ofmap_info;

    // step 1: 加载图片
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);

    // step 2: 设置新的尺寸并调整图像尺寸
    const int width = 1024; // 图像宽度
    const int height = 1024; // 图像高度
    cv::Size newSize(width, height);
    cv::Mat image_resized;
    cv::resize(image, image_resized, newSize);

    MouseCallbackContext context;
    context.decoder_exe_net = decoder_exe_net;
    context.image_resized = image_resized;
    context.decoder_io_buf_map = decoder_io_buf_map;
    context.cfg_info_map = cfg_info_map;

    // 创建窗口并设置鼠标回调
    namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Image", mouseCallback, &context);

    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    // 显示图片并等待用户操作
    while (true) {
        imshow("Image", image_resized);
        if (cv::waitKey(1000) == 27) // 按ESC键退出
        {
            break;
        }
    }

    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        LOG_ERR("Usage: %s [runtime.yml]，for example: model_and_cfg_zoo/configs/samples/runtime_sample.yml]", argv[0]);
    }
    const char *rt_cfg_txt = argv[1];
    std::string rt_cfg_txt_str(rt_cfg_txt);
    std::unordered_map<std::string, std::string> cfg_info_map;
    yml2map(cfg_info_map, rt_cfg_txt_str);

//    LOG_MSG("start abcded");

    rt_cfg_check(cfg_info_map);

    std::string clip_model = str2lower_str(cfg_info_map["clip_txt_one_file_path"]);
    std::string mobile_sam_model = str2lower_str(cfg_info_map["sam_decoder_one_file_path"]);
    if (!clip_model.empty()) {    // do clip
        do_clip(cfg_info_map);
        return 0;
    } else if (!mobile_sam_model.empty()) {    // do mobile_sam
        do_mobile_sam(cfg_info_map);
        return 0;
    }

    const char* one_file_path = cfg_info_map["one_file_path"].c_str();
    net *model = new net(one_file_path);
    model->build_graph();

//    LOG_DBG("开始开辟空间");
    extractor* exe_net = model->create_exe();

//    LOG_ERR("先看看 shape infer 有问题没有");

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

    std::string img_path = cfg_info_map["input_data_path"];

    transforms(in_buf, img_path, trans_cfg);
//    io_buf_map[in_operand_name] = in_buf;

    int32_t elem_size = in_buf.size();
    int32_t buf_size = elem_size * sizeof(float);
    int64_t cur_operand_ptr = (int64_t)&in_buf[0];
    io_buf_map[in_operand_name] = {cur_operand_ptr, elem_size, buf_size};


    std::string ifmap_folder = cfg_info_map["ofmap_folder"];
    std::string ifmap_name("model_ifmap.bin");
    std::string ifmap_path = ifmap_folder + ifmap_name;

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

    if (cfg_info_map["postprocess_type"] == "classification") {
        std::vector<int> argmax_topk = str2number<int>(cfg_info_map["topk"]);
        int topk = argmax_topk[0];

        for (auto io_buf:io_buf_map) {
            if (io_buf.first != in_operand_name){
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
                    argmax_cfg.axis = 1;
                    std::string op_type = "ArgMax";
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

                    printf("the topk cls_label is: ");
                    for (int i = 0; i < argmax_cfg.topk; ++i) {
                        printf("%d  ", argmax_out_info[i]);
                    }
                    printf("\n");
                }
            }
        }
    }

    if (cfg_info_map["postprocess_type"] == "segmentation") {
        // do segment op
        const int32_t max_keep_box_num = 30;
        std::vector<float> detect_out0_info(max_keep_box_num * sizeof(SEGMENT_OFMAP0_S), 0);
        std::vector<float> detect_out1_info(max_keep_box_num * (32 * 160 * 160), 0);
        SEGMENT_CONFIG_S segment_cfg;
        std::string op_type = "Segment";
        strcpy(segment_cfg.op_base_cfg.op_type, op_type.c_str());

        std::vector<BUFFER_INFO_S> params;
        std::vector<BUFFER_INFO_S> inputs;
        std::vector<BUFFER_INFO_S> outputs;

        // fill params
        BUFFER_INFO_S cfg;
        cfg.addr = (int64_t) (&segment_cfg);
        params.push_back(cfg);

        OPERAND_S out;
        BUFFER_INFO_S out_desc;
        out_desc.addr = (int64_t) (&out);
        params.push_back(out_desc);

        // fill inputs
        // find the three ofmap of backbone, they are the ifmap of detct op
        BUFFER_INFO_S in0_info;
        std::string in0_name = get_string_vec(cfg_info_map["ofmap_name"])[0];
        in0_info.addr = (int64_t) (&io_buf_map[in0_name].st_ptr);
        inputs.push_back(in0_info);

        BUFFER_INFO_S in1_info;
        std::string in1_name = get_string_vec(cfg_info_map["ofmap_name"])[1];
        in1_info.addr = (int64_t) (&io_buf_map[in1_name].st_ptr);
        inputs.push_back(in1_info);

        BUFFER_INFO_S in2_info;
        std::string in2_name = get_string_vec(cfg_info_map["ofmap_name"])[2];
        in2_info.addr = (int64_t) (&io_buf_map[in2_name].st_ptr);
        inputs.push_back(in2_info);

        BUFFER_INFO_S in3_info;
        std::string in3_name = get_string_vec(cfg_info_map["ofmap_name"])[3];
        in3_info.addr = (int64_t) (&io_buf_map[in3_name].st_ptr);
        inputs.push_back(in3_info);

        BUFFER_INFO_S in4_info;
        std::string in4_name = get_string_vec(cfg_info_map["ofmap_name"])[4];
        in4_info.addr = (int64_t) (&io_buf_map[in4_name].st_ptr);
        inputs.push_back(in4_info);

        BUFFER_INFO_S in5_info;
        std::string in5_name = get_string_vec(cfg_info_map["ofmap_name"])[5];
        in5_info.addr = (int64_t) (&io_buf_map[in5_name].st_ptr);
        inputs.push_back(in5_info);

        // fill outputs
        BUFFER_INFO_S out0_info;
        out0_info.addr = (int64_t) (&detect_out0_info[0]);
        outputs.push_back(out0_info);

        BUFFER_INFO_S out1_info;
        out1_info.addr = (int64_t) (&detect_out1_info[0]);
        outputs.push_back(out1_info);

        // do segment
        launch_post_process(params, inputs, outputs);
    }

    if (cfg_info_map["postprocess_type"] == "pose_detection") {
        // do segment op
        const int32_t max_keep_box_num = 30;
        std::vector<float> detect_out_info(max_keep_box_num * sizeof(POSE_DETECTION_CONFIG_S), 0);
        POSE_DETECTION_CONFIG_S pose_detection_cfg;
        std::string op_type = "PoseDetection";
        strcpy(pose_detection_cfg.op_base_cfg.op_type, op_type.c_str());

        std::vector<BUFFER_INFO_S> params;
        std::vector<BUFFER_INFO_S> inputs;
        std::vector<BUFFER_INFO_S> outputs;

        // fill params
        BUFFER_INFO_S cfg;
        cfg.addr = (int64_t) (&pose_detection_cfg);
        params.push_back(cfg);

        OPERAND_S out;
        BUFFER_INFO_S out_desc;
        out_desc.addr = (int64_t) (&out);
        params.push_back(out_desc);

        // fill inputs
        // find the three ofmap of backbone, they are the ifmap of detct op
        BUFFER_INFO_S in0_info;
        std::string in0_name = get_string_vec(cfg_info_map["ofmap_name"])[0];
        in0_info.addr = (int64_t) (&io_buf_map[in0_name].st_ptr);
        inputs.push_back(in0_info);

        BUFFER_INFO_S in1_info;
        std::string in1_name = get_string_vec(cfg_info_map["ofmap_name"])[1];
        in1_info.addr = (int64_t) (&io_buf_map[in1_name].st_ptr);
        inputs.push_back(in1_info);

        BUFFER_INFO_S in2_info;
        std::string in2_name = get_string_vec(cfg_info_map["ofmap_name"])[2];
        in2_info.addr = (int64_t) (&io_buf_map[in2_name].st_ptr);
        inputs.push_back(in2_info);

        BUFFER_INFO_S in3_info;
        std::string in3_name = get_string_vec(cfg_info_map["ofmap_name"])[3];
        in3_info.addr = (int64_t) (&io_buf_map[in3_name].st_ptr);
        inputs.push_back(in3_info);

        BUFFER_INFO_S in4_info;
        std::string in4_name = get_string_vec(cfg_info_map["ofmap_name"])[4];
        in4_info.addr = (int64_t) (&io_buf_map[in4_name].st_ptr);
        inputs.push_back(in4_info);

        BUFFER_INFO_S in5_info;
        std::string in5_name = get_string_vec(cfg_info_map["ofmap_name"])[5];
        in5_info.addr = (int64_t) (&io_buf_map[in5_name].st_ptr);
        inputs.push_back(in5_info);

        // fill outputs
        BUFFER_INFO_S out_info;
        out_info.addr = (int64_t) (&detect_out_info[0]);
        outputs.push_back(out_info);

        // do segment
        launch_post_process(params, inputs, outputs);
    }

    return 0;



    return 0;

}

