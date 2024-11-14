//
// Created by wanzai on 24-11-9.
//

#ifndef ONENEW_MOBILESAM_MODEL_H
#define ONENEW_MOBILESAM_MODEL_H


// 上下文结构体
struct MouseCallbackContext {
    extractor* decoder_exe_net;
    cv::Mat image_resized;
    std::unordered_map<std::string, BUFFER_INFO_S> decoder_io_buf_map;
    std::unordered_map<std::string, std::string> cfg_info_map;
};

// 鼠标回调函数
void mouseCallback(int event, int x, int y, int, void* userdata) {
    MouseCallbackContext* context = static_cast<MouseCallbackContext*>(userdata);
    extractor* decoder_exe_net = context->decoder_exe_net;
    cv::Mat image_resized = context->image_resized;
    std::unordered_map<std::string, BUFFER_INFO_S> decoder_io_buf_map = context->decoder_io_buf_map;
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

        decoder_exe_net->prepare_for_op(decoder_io_buf_map);

        decoder_exe_net->impl(decoder_io_buf_map, cfg_info_map);
        std::string sam_model_ofmap = "masks";

        BUFFER_INFO_S decoder_ofmap_info = decoder_io_buf_map[sam_model_ofmap];
        std::vector<float> decoder_ofmap(decoder_ofmap_info.elem_size);
        memcpy(&decoder_ofmap[0], (void *)decoder_ofmap_info.addr, decoder_ofmap_info.buf_size);


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
    const char* encoder_one_file_path = cfg_info_map["sam_encoder_one_file_path"].c_str();
    net *encoder_model = new net(encoder_one_file_path);
    encoder_model->build_graph();
    extractor* exe_net = encoder_model->create_exe();

    auto* ifmap_of_model = (io*)exe_net->net_ptr->op_exec_order[0].get();
    std::string in_operand_name = ifmap_of_model->io_cfg.operand.operand_name;

    std::unordered_map<std::string, BUFFER_INFO_S> io_buf_map;

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

    std::string img_path = cfg_info_map["input_data_path"];

    transforms(in_buf, img_path, trans_cfg);
    int64_t addr = (int64_t)(&in_buf[0]);
    int32_t elem_size = (int64_t)(in_buf.size());
    int32_t buf_size = (int64_t)(elem_size * sizeof(float));
    io_buf_map[in_operand_name] = {addr, elem_size, buf_size};

    std::string ifmap_folder = cfg_info_map["ofmap_folder"];
    std::string ifmap_name("model_ifmap.bin");
    std::string ifmap_path = ifmap_folder + ifmap_name;

    exe_net->prepare_for_op(io_buf_map);

    exe_net->impl(io_buf_map, cfg_info_map);

    std::string ofmap = "image_embeddings";
    BUFFER_INFO_S encoder_ofmap_info = io_buf_map[ofmap];


//    char *image_embeddings_buf = (char*) aligned_alloc(32, 256*64*64*sizeof(float));
//    std::string file_name = "/home/wanzai/桌面/mobile_sam_test_img/image_embeddings";
//    load_bin(file_name.c_str(), 256*64*64*sizeof(float), image_embeddings_buf);
//    BUFFER_INFO_S encoder_ofmap_info = {(int64_t)image_embeddings_buf, 256*64*64, 256*64*64*sizeof(float)};


    // do sam decoder
    const char* decoder_one_file_path = cfg_info_map["sam_decoder_one_file_path"].c_str();
    net *decoder_model = new net(decoder_one_file_path);
    decoder_model->build_graph();
    extractor* decoder_exe_net = decoder_model->create_exe();

    std::string image_embeddings = "image_embeddings";
    std::string point_coords = "point_coords";

    std::unordered_map<std::string, BUFFER_INFO_S> decoder_io_buf_map;

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

#endif //ONENEW_MOBILESAM_MODEL_H
