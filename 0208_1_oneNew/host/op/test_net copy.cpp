#include "../../host/op/relu.h"
#include "../../host/op/maxpool.h"
#include "../../host/op/conv.h"
#include "../../host/op/io.h"
#include "../../host/op/add.h"
#include "../../host/op/global_avgpool.h"
#include "../../host/op/flatten.h"
#include "../../host/op/gemm.h"
#include "net.h"

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

#include "stdint.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath> // 用于std::sqrt函数


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

    cv::cvtColor(cropped_img, cropped_img, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> rgb_channels(3);
    cv::split(cropped_img, rgb_channels);
    for (auto i = 0; i < rgb_channels.size(); i++)
    {
        rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / (trans_cfg.std[i] * 255.f), (0.0 - trans_cfg.mean[i] * 255.0f) / (trans_cfg.std[i] * 255.f));
    }

    // std::vector<float> r = rgb_channels[0].reshape(1, 1);
    // std::vector<float> g = rgb_channels[1].reshape(1, 1);
    // std::vector<float> b = rgb_channels[2].reshape(1, 1);

    std::vector<float> r = rgb_channels[0].reshape(1, 1);
    std::vector<float> g = rgb_channels[1].reshape(1, 1);
    std::vector<float> b = rgb_channels[2].reshape(1, 1);
    
    // std::vector<float> rgb;

    // rgb.insert(rgb.end(), r.begin(), r.end());
    // rgb.insert(rgb.end(), g.begin(), g.end());
    // rgb.insert(rgb.end(), b.begin(), b.end());

// std::cout << "----------------------the rgb  is:" << std::endl;
// for (auto v_:rgb)
// {
//     std::cout << v_ << std::endl;
// }

    // int32_t total = 3;
    // std::cout << "----------------------the r is:" << std::endl;
    // int i = 0;
    // for (auto r_ : r)
    // {
    //     if (i > total)
    //     {
    //         break;
    //     }
    //     i++;

    //     std::cout << r_ << std::endl;
    // }

    // i = 0;
    // std::cout << "-----------------------the g is:" << std::endl;
    // for (auto g_ : g)
    // {
    //     if (i > total)
    //     {
    //         break;
    //     }
    //     i++;
    //     std::cout << g_ << std::endl;
    // }

    // i = 0;
    // std::cout << "-------------------the b is:" << std::endl;
    // for (auto b_ : b)
    // {
    //     if (i > total)
    //     {
    //         break;
    //     }
    //     i++;
    //     std::cout << b_ << std::endl;
    // }
}


int main() {

    TRANSFORMS_CONFIG_S trans_cfg;
    trans_cfg.resize_size[0] = 256;
    trans_cfg.resize_size[1] = 256;
    trans_cfg.crop_size[0] = 224;
    trans_cfg.crop_size[1] = 224;

    trans_cfg.mean[0] = 0.0f;
    trans_cfg.mean[1] = 0.0f;
    trans_cfg.mean[2] = 0.0f;

    trans_cfg.std[0] = 1.0f;
    trans_cfg.std[1] = 1.0f;
    trans_cfg.std[2] = 1.0f;

    trans_cfg.mean[0] = 0.485f;
    trans_cfg.mean[1] = 0.456f;
    trans_cfg.mean[2] = 0.406f;

    trans_cfg.std[0] = 0.229f;
    trans_cfg.std[1] = 0.224f;
    trans_cfg.std[2] = 0.225f;

    std::string img_path = "/IBU_8T/IBU_SOFTWARE/e0006369/qiangguo/yolov3/img/giraffe.jpg";
    // std::string img_path = "/home/e0006809/Desktop/0207_onenew/0206_2_oneNew/host/op/bb.jpg";



    struct timeval begin, end;
    gettimeofday(&begin, 0);

    transforms(img_path, trans_cfg);

    gettimeofday(&end, 0);

    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds * 1e-6;
    printf("time measured: %.3f seconds.\n", elapsed);
  
    int a = 101;
    return 0;
}
