// # 加载 imageNet 的时候需要进行 resize 和 normalize 操作
// transform_imgnet = transforms.Compose([
//     transforms.Resize(256),
//     transforms.CenterCrop(224),
//     transforms.ToTensor(),
//     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
// ])

#include "stdint.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath> // 用于std::sqrt函数

typedef enum
{
    RGB = 0,
    BGR = 1,
    YUV_NV12 = 2,
    YUV_NV21 = 3,
    YUV420P = 4,
} COLOR_CODE_E;

typedef enum
{
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4,
} RESIZE_METHOD_E;

typedef struct
{
    int32_t resize_size[2];
    int32_t crop_size[2];

    COLOR_CODE_E out_color_code;
    RESIZE_METHOD_E resize_method;
    float mean[3];
    float std[3];

} TRANSFORMS_CONFIG_S;

int transforms(std::string img_path, TRANSFORMS_CONFIG_S &trans_cfg)
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

    // cv::imshow("cropped image", cropped_img);
    // std::cout << "success to resize and crop img " << std::endl;
    // cv::waitKey(0);

    // step 2 : normlize img
    // std::vector<float> mean_value{0, 0, 0};
    // std::vector<float> std_value{1, 1, 1};

    // // std::vector<float> mean_value{0.406, 0.456, 0.485};
    // // std::vector<float> std_value{0.225, 0.224, 0.229};

    cv::cvtColor(cropped_img, cropped_img, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> rgb_channels(3);
    cv::split(cropped_img, rgb_channels);
    for (auto i = 0; i < rgb_channels.size(); i++)
    {
        rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / (trans_cfg.std[i] * 255.0f), (0.0 - trans_cfg.mean[i]) / (trans_cfg.std[i] * 255.0f));
        // rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / std_value[i], (0.0 - mean_value[i]) / std_value[i]);
    }

    std::vector<float> r = rgb_channels[0].reshape(1, 1);
    std::vector<float> g = rgb_channels[1].reshape(1, 1);
    std::vector<float> b = rgb_channels[2].reshape(1, 1);


    std::cout << "----------------------the r is:" << std::endl;
    int i = 0;
    for (auto r_ : r)
    {
        if (i > 4)
        {
            break;
        }
        i++;

        std::cout << r_ << std::endl;
    }

    i = 0;
    std::cout << "-----------------------the g is:" << std::endl;
    for (auto g_ : g)
    {
        if (i > 4)
        {
            break;
        }
        i++;
        std::cout << g_ << std::endl;
    }

    i = 0;
    std::cout << "-------------------the b is:" << std::endl;
    for (auto b_ : b)
    {
        if (i > 4)
        {
            break;
        }
        i++;
        std::cout << b_ << std::endl;
    }
}

int main()
{

    TRANSFORMS_CONFIG_S trans_cfg;
    trans_cfg.resize_size[0] = 640;
    trans_cfg.resize_size[1] = 640;
    trans_cfg.crop_size[0] = 144;
    trans_cfg.crop_size[1] = 144;

    // trans_cfg.mean[0] = 0.0f;
    // trans_cfg.mean[1] = 0.0f;
    // trans_cfg.mean[2] = 0.0f;

    // trans_cfg.std[0] = 1.0f;
    // trans_cfg.std[1] = 1.0f;
    // trans_cfg.std[2] = 1.0f;

    trans_cfg.mean[0] = 0.406f;
    trans_cfg.mean[1] = 0.456f;
    trans_cfg.mean[2] = 0.485f;

    trans_cfg.std[0] = 0.225f;
    trans_cfg.std[1] = 0.224f;
    trans_cfg.std[2] = 0.229f;

    std::string img_path = "/IBU_8T/IBU_SOFTWARE/e0006369/qiangguo/yolov3/img/giraffe.jpg";

    transforms(img_path, trans_cfg);

    return 0;
}
