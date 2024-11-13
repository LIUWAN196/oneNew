//
// Created by wanzai on 24-11-9.
//

#ifndef ONENEW_CLIP_MODEL_H
#define ONENEW_CLIP_MODEL_H

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
    const char* img_one_file_path = cfg_info_map["clip_img_one_file_path"].c_str();
    net *img_model = new net(img_one_file_path);
    img_model->build_graph();
    extractor* exe_net = img_model->create_exe();

    auto* ifmap_of_model = (io*)exe_net->net_ptr->op_exec_order[0].get();
    std::string in_operand_name = ifmap_of_model->io_cfg.operand.operand_name;

    std::unordered_map<std::string, BUF_INFO_S> io_buf_map;



//    std::vector<int> resize_shapes = str2number<int>(cfg_info_map["resize_shapes"]);
//    std::vector<int> crop_shapes = str2number<int>(cfg_info_map["crop_shapes"]);
//    std::vector<float> normal_mean = str2number<float>(cfg_info_map["normal_mean"]);
//    std::vector<float> normal_std = str2number<float>(cfg_info_map["normal_std"]);
//
//    int in_elem_size = 3 * crop_shapes[0] * crop_shapes[1];
//
//    std::vector<float> in_buf(in_elem_size);
//
//    TRANSFORMS_CONFIG_S trans_cfg;
//    trans_cfg.resize_size[0] = resize_shapes[0];
//    trans_cfg.resize_size[1] = resize_shapes[1];
//    trans_cfg.crop_size[0] = crop_shapes[0];
//    trans_cfg.crop_size[1] = crop_shapes[1];
//
//    trans_cfg.mean[0] = normal_mean[0];
//    trans_cfg.mean[1] = normal_mean[1];
//    trans_cfg.mean[2] = normal_mean[2];
//
//    trans_cfg.std[0] = normal_std[0];
//    trans_cfg.std[1] = normal_std[1];
//    trans_cfg.std[2] = normal_std[2];

    TRANSFORMS_CONFIG_S trans_cfg = cfg_info_map2preprocess_params(cfg_info_map);
    int in_elem_size = 3 * trans_cfg.crop_size[0] * trans_cfg.crop_size[1];
    std::vector<float> in_buf(in_elem_size);

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

        exe_net->prepare_for_op(io_buf_map);

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

        txt_exe_net->prepare_for_op(txt_io_buf_map);

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


#endif //ONENEW_CLIP_MODEL_H
