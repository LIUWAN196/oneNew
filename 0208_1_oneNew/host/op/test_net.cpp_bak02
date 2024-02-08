//#include <iostream>
//#include "opencv/cv.h"
//#include "opencv2/opencv.hpp"
//
//
//void ConvertOpencvMatToArray()
//{
//    cv::Mat img = cv::imread("/home/wanzai/桌面/oneNew/host/op/bus.jpg");
//
//    int img_length = img.total() * img.channels();
//    unsigned char* image_array_ptr = new unsigned char[img_length]();
//
//    std::memcpy(image_array_ptr, img.ptr<unsigned char>(0), img_length * sizeof(unsigned char));
//
//    // 从unsigned char* 再转换为cv::Mat，验证转换是否正确
//    cv::Mat a = cv::Mat(img.rows, img.cols, img.type(), image_array_ptr);
//    cv::imwrite("a.jpg", a);
//
//    delete[] image_array_ptr;
//}


//int main()
//{
//    ConvertOpencvMatToArray();
//
//
//    getchar();
//
//    return 0;
//}















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

template<typename _Tp>
std::vector<_Tp> convert_mat_to_vector(cv::Mat& mat)
{
    //通道数不变，按行转为一行
    return (std::vector<_Tp>)(mat.reshape(1, 1));
}

template<typename _Tp>
cv::Mat convert_vector_to_mat(std::vector<_Tp> v, int channels, int rows)
{
    //将vector变成单列的mat，这里需要clone(),因为这里的赋值操作是浅拷贝
    cv::Mat mat = cv::Mat(v).clone();
    cv::Mat dest = mat.reshape(channels, rows);
    return dest;
}



using namespace cv;
using namespace std;

/***************** Mat转vector **********************/
template<typename _Tp>
vector<_Tp> convertMat2Vector(const Mat &mat)
{
    return (vector<_Tp>)(mat.reshape(1, 1));//通道数不变，按行转为一行
}

/****************** vector转Mat *********************/
template<typename _Tp>
cv::Mat convertVector2Mat(vector<_Tp> v, int channels, int rows)
{
    cv::Mat mat = cv::Mat(v).clone();//将vector变成单列的mat，这里需要clone(),因为这里的赋值操作是浅拷贝
    cv::Mat dest = mat.reshape(channels, rows);
    return dest;


}



int transforms(std::string img_path, TRANSFORMS_CONFIG_S &trans_cfg) {


    uchar arr[4][3] = { { 1, 1,1 },{ 2, 2,2 },{ 3, 3,3 },{ 4,4, 4 } };
    cv::Mat srcData(4, 3, CV_8UC1, arr);
    cout << "srcData=\n" << srcData << endl;

    vector<uchar> v = convertMat2Vector<uchar>(srcData);
    cv::Mat dest = convertVector2Mat<uchar>(v, 1, 4);//把数据转为1通道，4行的Mat数据
    cout << "dest=\n" << dest << endl;

    system("pause");
    waitKey();








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

    float* mean_ptr = trans_cfg.mean;
    float* std_ptr = trans_cfg.std;


    // step 1 : resize img
    cv::Mat resized_img;
    cv::resize(ori_img, resized_img, cv::Size(resize_h, resize_w));

    int16_t crop_x = (resize_w - crop_w) >> 1;
    int16_t crop_y = (resize_h - crop_h) >> 1;

    // step 2 : crop img
    cv::Rect roi(crop_x, crop_y, crop_w, crop_h);
    if (roi.x + roi.width > resized_img.cols || roi.y + roi.height > resized_img.rows) {
        std::cout << "cropping area beyond image boundaries." << std::endl;
        return -1;
    }
    cv::Mat cropped_img = resized_img(roi);

//    cv::namedWindow("Image");
////    cv::namedWindow("Image",WINDOW_AUTOSIZE);
//    cv::imshow("Image", cropped_img);
//    cv::waitKey(0);

//    std::vector<std::vector<uchar>> vecaa;
//    for (int x = 0; x < cropped_img.rows; x++) {
//        for (int y = 0; y < cropped_img.cols; y++) {
//            vecaa[x][y] = cropped_img.at<uchar>(x, y);
//            //std::cout << mat.at<float>(x, y) << std::endl;
//        }
//    }

//    cv::Mat img = cv::imread("a.jpg");
//    std::vector<float> v = convert_mat_to_vector<float>(cropped_img);
//

    // step 4 : trans img to vector
    std::vector<std::vector<uint8_t>> u8_img;
    for (int i = 0; i < cropped_img.rows; ++i) {
        std::vector<uint8_t> row;
        for (int j = 0; j < cropped_img.cols; ++j) {
            row.push_back(cropped_img.at<uint8_t>(i, j));
        }
        u8_img.push_back(row);
    }


    // step 3 : normlize
// https://blog.csdn.net/guyuealian/article/details/80253066
    std::vector<float> mean_value{0.406, 0.456, 0.485};
    std::vector<float> std_value{0.225, 0.224, 0.229};
    cv::Mat norm_img;
    std::vector<cv::Mat> bgr_channels(3);
    cv::split(cropped_img, bgr_channels);
    for (auto i = 0; i < bgr_channels.size(); i++) {
        bgr_channels[i].convertTo(bgr_channels[i], CV_32FC1, 1.0f / std_ptr[i], (0.0 - mean_ptr[i]) / std_ptr[i]);
    }
    cv::merge(bgr_channels, norm_img);


//    // step 4 : trans img to vector
//    std::vector<std::vector<float>> f32_img;
//    for (int i = 0; i < norm_img.rows; ++i) {
//        std::vector<float> row;
//        for (int j = 0; j < norm_img.cols; ++j) {
//            row.push_back(norm_img.at<uchar>(i, j));
//        }
//        f32_img.push_back(row);
//    }

    int c = 101;
//// c++ - openCV:将 Mat 保存到 vector<vector<int>>
//// 我有一个名为 myImage 的 cv::Mat，我想将其保存到名为 savedVec 的 std::vector >。这是我写的代码:
//    for (int i = 0; i < myImage.rows; ++i) {
//        std::vector<int> row;
//        for (int j = 0; j < myImage.cols; ++j) {
//            row.push_back(Play::getInstance()->getFinder()->getImage().at<uchar>(i, j));
//        }
//        savedVec.push_back(row);
//    }

//// 显示裁剪后的图像
//    cv::imshow("cropped image", cropped_img);
//    std::cout << "success to resize and crop img " << std::endl;
//    cv::waitKey(0);

}

int main() {
    TRANSFORMS_CONFIG_S trans_cfg;
//    trans_cfg.resize_size[0] = 256;
//    trans_cfg.resize_size[1] = 256;
//    trans_cfg.crop_size[0] = 256;
//    trans_cfg.crop_size[1] = 256;

    trans_cfg.resize_size[0] = 700;
    trans_cfg.resize_size[1] = 700;
    trans_cfg.crop_size[0] = 640;
    trans_cfg.crop_size[1] = 640;

    trans_cfg.mean[0] = 0.406f;
    trans_cfg.mean[1] = 0.456f;
    trans_cfg.mean[2] = 0.485f;

    trans_cfg.std[0] = 0.225f;
    trans_cfg.std[1] = 0.224f;
    trans_cfg.std[2] = 0.229f;

    std::string img_path = "/home/wanzai/桌面/oneNew/host/op/bus.jpg";


    struct timeval begin, end;
    gettimeofday(&begin, 0);
    transforms(img_path, trans_cfg);
    gettimeofday(&end, 0);

    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds * 1e-6;
    printf("time measured: %.3f seconds.\n", elapsed);

//    net *net_1 = new net("/home/wanzai/桌面/oneNew/host/op/relu_conv.one");
//
//    net_1->build_graph();
//
//    extractor* bb = net_1->create_exe();
//
//    int ret;
//    struct timeval begin, end;
//    gettimeofday(&begin, 0);
//    bb->impl();
//    gettimeofday(&end, 0);
//
//    long seconds = end.tv_sec - begin.tv_sec;
//    long microseconds = end.tv_usec - begin.tv_usec;
//    double elapsed = seconds + microseconds * 1e-6;
//    printf("one thread, time measured: %.3f seconds.\n", elapsed);


    int a = 101;
    return 0;
}









//#include <dlfcn.h>
//#include <cstring>
//#include <iostream>
//#include <memory>
//#include <vector>
//#include <sys/time.h>
//#include <thread>
//#include <pthread.h>
//#include <functional>
//#include <vector>
//#include <algorithm>
//typedef int (*eval_dev)(void *, int32_t num_size, int32_t thread_id);
//eval_dev find_handle()
//{
//    char op_lib_path[256] = {0};
//    snprintf(op_lib_path, sizeof(op_lib_path), "/home/e0006809/Desktop/cpp_0129/device/builb/libAdd.so");
//    void *handle = dlopen(op_lib_path, RTLD_LAZY);
//
//    eval_dev evla_impl = (int (*)(void *, int32_t, int32_t))dlsym(handle, "eval");
//
//    return evla_impl;
//
//};

//int fun_(eval_dev eval_fun)
//{
//    eval_fun(ptr, num_size, 1);
//    eval_fun(ptr, num_size, 2);
//    eval_fun(ptr, num_size, 3);
//    eval_fun(ptr, num_size, 4);
//    eval_fun(ptr, num_size, 5);
//    eval_fun(ptr, num_size, 6);
//    eval_fun(ptr, num_size, 7);
//    eval_fun(ptr, num_size, 8);
//    eval_fun(ptr, num_size, 9);
//    eval_fun(ptr, num_size, 10);
//    return 0;
//
//}


//int main()
//{
//
//    net *net_1 = new net("/home/wanzai/桌面/oneNew/host/op/relu_conv.one");
//    net_1->build_graph();
//    extractor* b1 = net_1->create_exe();
//
//    net *net_2 = new net("/home/wanzai/桌面/oneNew/host/op/relu_conv.one");
//    net_2->build_graph();
//    extractor* b2 = net_2->create_exe();
//
//    net *net_3 = new net("/home/wanzai/桌面/oneNew/host/op/relu_conv.one");
//    net_3->build_graph();
//    extractor* b3 = net_3->create_exe();
//
//    net *net_4 = new net("/home/wanzai/桌面/oneNew/host/op/relu_conv.one");
//    net_4->build_graph();
//    extractor* b4 = net_4->create_exe();
//
//    net *net_5 = new net("/home/wanzai/桌面/oneNew/host/op/relu_conv.one");
//    net_5->build_graph();
//    extractor* b5 = net_5->create_exe();
//
//    net *net_6 = new net("/home/wanzai/桌面/oneNew/host/op/relu_conv.one");
//    net_6->build_graph();
//    extractor* b6 = net_6->create_exe();
//
////    bb->impl();
//
//    int a = 101;
//
//    unsigned int thread_count = std::thread::hardware_concurrency();
//    std::cout << "硬件支持的线程数: " << thread_count << std::endl;
//
//    eval_dev eval_fun = find_handle();
//    int32_t num_vec = 6;
//    int32_t num_size = 1 * 1024 * 1024;
//    std::vector<std::vector<int>> vec(num_vec, std::vector<int>(num_size));
//
//    int ret;
//    struct timeval begin, end;
//    struct timeval begin_, end_;
//
//    // one thread
//    gettimeofday(&begin, 0);
////    for (size_t i = 0; i < num_vec; i++)
////    {
////        bb->impl();
////    }
//    b1->impl();
//    b2->impl();
//    b3->impl();
//    b4->impl();
//    b5->impl();
//    b6->impl();
//
//    gettimeofday(&end, 0);
//    long seconds = end.tv_sec - begin.tv_sec;
//    long microseconds = end.tv_usec - begin.tv_usec;
//    double elapsed = seconds + microseconds * 1e-6;
////    printf("one thread, time measured: %.3f seconds.\n", elapsed);
//
//    // 4 thread
//    std::vector<std::thread> threadList;
//    gettimeofday(&begin_, 0);
//
////    threadList.push_back(std::thread(&extractor::impl, bb));
////    threadList.push_back(std::thread(&extractor::impl, cc));
////    threadList.push_back(std::thread(&extractor::impl, dd));
////    threadList.push_back(std::thread(&extractor::impl, ee));
//
//    threadList.push_back(std::thread(&extractor::impl, b1));
//    threadList.push_back(std::thread(&extractor::impl, b2));
//    threadList.push_back(std::thread(&extractor::impl, b3));
//    threadList.push_back(std::thread(&extractor::impl, b4));
//    threadList.push_back(std::thread(&extractor::impl, b5));
//    threadList.push_back(std::thread(&extractor::impl, b6));
//
////    gettimeofday(&end_, 0);
////    for (int i = 0; i < num_vec; ++i)
////    {
//////        std::thread t(&extractor::impl, bb);
////        threadList.push_back(std::thread(&extractor::impl, bb));
////    }
//    // 等待所有的 worker 执行完成。即，需要调用每个 thread 对象的 join() 函数
//    std::for_each(threadList.begin(), threadList.end(), std::mem_fn(&std::thread::join));
//
//    gettimeofday(&end_, 0);
//    long seconds1 = end_.tv_sec - begin_.tv_sec;
//    long microseconds1 = end_.tv_usec - begin_.tv_usec;
//    double elapsed1 = seconds1 + microseconds1 * 1e-6;
//    printf("one thread, time measured: %.3f seconds.\n", elapsed);
//    printf("4 thread, time measured: %.3f seconds.\n", elapsed1);
//
//    return 0;
//
//}
//
