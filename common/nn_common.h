#ifndef __NN_COMMON_H__
#define __NN_COMMON_H__

#include "stdint.h"
#include <stdlib.h>
#include <stdio.h>
#include "log.h"
//#define OP_LIB_DIR "/home/wanzai/桌面/"

//#define OP_X86_LIB_DIR "/home/wanzai/桌面/oneNew/cmake-build-debug/device/x86/"
//#define OP_CU_LIB_DIR "/home/wanzai/桌面/oneNew/cmake-build-debug/device/cuda/"

#define ONE_NODEL_MAGIC_NUM (123456)
#define USEFUL_INFO_MAGIC_NUM (45678)

/*
 * one magic number
 * 0x6F  0x6E  0x65
 *  o     n     e
 */
#define ONE_MAGIC_NUM (0x6F6E65)

#define EPSILON (1e-20)
#define BUF_MAXNUM 12
#define OPERAND_MAXNUM 8

#define SHAPE_LEN 8
#define OP_TYPE_LEN 256
#define OP_NAME_LEN 256
#define OPERAND_NAME_LEN 256

typedef struct
{
    int32_t x86_gemm_single_threads_tile_m;
    int32_t x86_gemm_single_threads_tile_n;
    int32_t x86_gemm_single_threads_tile_k;
    int32_t x86_gemm_multi_threads_tile_m;
    int32_t x86_gemm_multi_threads_tile_n;
    int32_t x86_gemm_multi_threads_tile_k;
    int32_t cuda_gemm_tile_m;
    int32_t cuda_gemm_tile_n;
    int32_t cuda_gemm_tile_k;
} BLOCK_INFO_S;

typedef struct
{
    int64_t public_buf_size;
    int64_t public_buf_ptr;
} PUBLIC_BUF_INFO_S;

typedef struct
{
    int64_t kv_cache_buf_size;
    int64_t kv_cache_buf_ptr;
    int64_t prompts_amount;
    int64_t max_token_supported;
    int64_t cur_token_index;
} KV_CACHE_INFO_S;

typedef struct
{
    int32_t useful_info_magic_num;
    BLOCK_INFO_S block_info;
    PUBLIC_BUF_INFO_S public_buf_info;
    KV_CACHE_INFO_S kv_cache_info;
} USEFUL_INFO_S;

typedef struct
{
    int32_t one_model_magic_num;
    int32_t version;
    int32_t node_cnt;
    int32_t node_cfg_offset;
    int32_t init_cnt;
    int32_t init_info_offset;
    int32_t io_cfg_cnt;
    int32_t io_cfg_offset;
    USEFUL_INFO_S useful_info;
} ONE_MODEL_DESC_S;

typedef struct
{
    int64_t addr;
    int64_t size;
} BUFFER_INFO_S;

typedef struct
{
    BUFFER_INFO_S buf_info[BUF_MAXNUM];
} BUFFER_GROUP_S;

typedef struct
{
    int32_t N;
    int32_t C;
    int32_t H;
    int32_t W;
} OPERAND_SHAPE_S;

// reference onnx      https://blog.csdn.net/weixin_43945848/article/details/122474749
typedef enum
{
    TYPE_UNKNOWN = 0,
    TYPE_FP32 = 1,
    TYPE_UINT8 = 2,
    TYPE_INT8 = 3,
    TYPE_UINT16 = 4,
    TYPE_INT16 = 5,
    TYPE_INT32 = 6,
    TYPE_INT64 = 7,
    TYPE_STRING = 8,
    TYPE_BOOLEAN = 9,
    TYPE_FP16 = 10,
    TYPE_FP64 = 11,
    TYPE_UINT32 = 12,
    TYPE_UINT64 = 14,
    TYPE_COMPLEX128 = 15,
    TYPE_BF16 = 16,
} ELEM_TYPE_E;

typedef enum
{
    FALSE = 0,
    TRUE = 1,
} BOOL;

typedef struct {
    float x_min;
    float y_min;
    float x_max;
    float y_max;
} BOX_INFO_S;

typedef struct {
    int32_t cls_id;
    float score;
    BOX_INFO_S box_info;
} OBJ_DETECT_OUT_INFO_S;

typedef enum
{
    NOT_ACT = 0,
    RELU = 1,
    RELU6 = 2,
    SILU = 3,
    CLIP = 4,
    HARDSILU = 5,
    LEAKYRELU = 6,
} ACT_TYPE_E;

typedef struct
{
    char op_type[OP_TYPE_LEN];
    char op_name[OP_NAME_LEN];
} NODE_INFO_S;

typedef struct
{
    char operand_name[OP_NAME_LEN];
    BOOL is_fixed_val; // such as: weight, bias
    BOOL not_need_buf;  // 用于 flatten 和 reshape 等算子的输出，此类输出不需要开辟 buffer
    int64_t p_data;
    int32_t dim_num_of_shapes;
    int32_t shapes[SHAPE_LEN];
    ELEM_TYPE_E data_type;
} OPERAND_S;

typedef struct
{
    char op_type[OP_TYPE_LEN]; //  io
    char op_name[OP_NAME_LEN]; //  "input" or "output"
    OPERAND_S operand;
} IO_CONFIG_S;

typedef struct
{
    int64_t st_ptr;
    int32_t elem_size;
    int32_t buf_size;
} BUF_INFO_S;

typedef struct
{
    char op_type[OP_TYPE_LEN];
    char op_name[OP_NAME_LEN];
    char in_operand_name[OPERAND_MAXNUM][OPERAND_NAME_LEN];
    char out_operand_name[OPERAND_MAXNUM][OPERAND_NAME_LEN];
    int32_t in_operand_num;
    int32_t out_operand_num;

    NODE_INFO_S producer[OPERAND_MAXNUM];
    NODE_INFO_S consumer[OPERAND_MAXNUM];
    int32_t producer_num;
    int32_t consumer_num;

    // used to mark whether fusion is required
    int32_t fusion_op_type;
    int64_t fusion_op_cnt_in_entire_net;    // 如果融合，那么融合后的融合算子这是整个网络的第几个融合算子
    int64_t cur_op_cnt_in_fusion_op;        // 如果融合，那么当前这个算子是融合算子的第几个子算子
} BASE_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} ADD_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
    int64_t topk;
    int64_t axis;
    int64_t keepdims;
    int64_t select_last_index;
} ARGMAX_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    int64_t ceil_mode;
    int64_t kernel_shape[2];
    int64_t pads[4];
    int64_t strides[2];
} AVG_POOL_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
    float min;
    float max;
} CLIP_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    int64_t axis;
} CONCAT_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    BOOL has_bias;
    int64_t dilations[2];
    int64_t group;
    int64_t kernel_shape[2];
    int64_t pads[4];
    int64_t strides[4];

    ACT_TYPE_E act_type;
    float clip_min;
    float clip_max;
    float leaky_relu_alpha;
    float hard_sigmoid_alpha;
    float hard_sigmoid_beta;
    ELEM_TYPE_E ifmap_quant2;   // the data type of ifmap (f32) / input_scale, generally, if quantified, it is TYPE_INT8, and occasionally it is TYPE_INT16
    float input_scale;          //  == ifmap_threshold / 127
    float output_scale;         //  == ofmap_threshold / 127
    float weight_aux[4096];
} CONV_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    BOOL has_bias;
    int64_t dilations[2];
    int64_t group;
    int64_t kernel_shape[2];
    int64_t pads[4];
    int64_t strides[4];

    ELEM_TYPE_E ifmap_quant2;   // the data type of ifmap (f32) / input_scale, generally, if quantified, it is TYPE_INT8, and occasionally it is TYPE_INT16
    float input_scale;          //  == ifmap_threshold / 127
    float output_scale;         //  == ofmap_threshold / 127
    float weight_aux[4096];
} CONV_TRANSPOSE_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} COS_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} DIV_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} ERF_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} EXP_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
    char equation[32];
} EINSUM_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    int64_t dst_shape[8];
    int64_t dst_shape_num;
} EXPAND_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
    int64_t axis;
} FLATTEN_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    int64_t axis;
    int64_t indices;

    BOOL indices_from_ifmap;
} GATHER_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    BOOL has_bias;
    float alpha;
    float beta;
    int64_t transB;
} GEMM_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} GELU_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} GLOBAL_AVGPOOL_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
    int64_t align_corners;
    char mode[32];
    char padding_mode[32];
} GRID_SAMPLE_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
    float alpha;
    float beta;
} HARD_SIGMOID_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
    float alpha;
} LEAKYRELU_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
    int32_t axes[8];
    int32_t axes_num;
    int32_t keepdims;
} LAYERNORMALIZATION_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} LOG_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} MATMUL_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    int64_t ceil_mode;
    int64_t kernel_shape[2];
    int64_t pads[4];
    int64_t strides[2];
} MAX_POOL_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} MUL_CONFIG_S;

#define MAX_IN_TENSOR_NUM 9
typedef enum
{
    NET_UNKNOWN = 0,
    YOLO_V3 = 1,
    YOLO_V5 = 2,
    YOLO_V7 = 3,
    YOLO_V8 = 4,
    YOLO_V10 = 5,
    YOLO_WORLD = 6,
    RT_DETR = 7,    // runtime detr model
} DETECT_NET_TYPE_E;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    DETECT_NET_TYPE_E net_type;

    int32_t cls_num;              // total num of object categories in this network
    int32_t img_w;
    int32_t img_h;
    int32_t max_boxes_per_class;  // the max num of keep boxes after nms per class
    int32_t max_boxes_per_batch;  // the max num of keep boxes after nms in single image
    float score_threshold;
    float iou_threshold;
} OBJECT_DETECT_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    int64_t pads[8];
} PAD_CONFIG_S;

typedef struct {
    int32_t cls_id;
    float score;
    BOX_INFO_S box_info;
    float keypoints[51]; // 51 = 17 * （2 + 1） 2 是关键点的 x、y 坐标，1 是该关键点是否被遮挡 (如果被遮挡就不要在图上画出来)
} POSE_DETECT_OUT_INFO_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    int32_t img_h;                                         // the input image_h size of the detect network
    int32_t img_w;                                         // the input image_w size of the detect network
    int32_t max_boxes_per_class;  // the max num of keep boxes after nms per class
    int32_t max_boxes_per_batch;  // the max num of keep boxes after nms in single image
    float score_threshold;
    float iou_threshold;
} POSE_DETECT_CONFIG_S;

typedef struct
{
    int64_t c;
    int64_t h;
    int64_t w;
    int64_t pads[4];
    int32_t left_pad;
    int32_t right_pad;
    int32_t top_pad;
    int32_t bottom_pad;
} PAD_INNER_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
    float power_num;
} POW_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
    int32_t axes[8];
    int32_t axes_num;
    int32_t keepdims;
    int32_t noop_with_empty_axes;
} REDUCE_SUM_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
    int32_t axes[8];
    int32_t axes_num;
    int32_t keepdims;
} REDUCE_MAX_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
    int32_t axes[8];
    int32_t axes_num;
    int32_t keepdims;
} REDUCE_MEAN_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} RELU_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    int64_t allowzero;
    int64_t dst_shape[8];
    int64_t dst_shape_num;
} RESHAPE_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    float cubic_coeff_a;
    float scales[8];    //  nchw
    int64_t sizes[8];    //  nchw
    int64_t scales_num;
} RESIZE_CONFIG_S;


typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    int32_t img_h;                                         // the input image_h size of the detect network
    int32_t img_w;                                         // the input image_w size of the detect network

    int32_t cls_num;              // total num of object categories in this network
    float score_threshold;
    float iou_threshold;
} SEGMENT_CONFIG_S;

typedef struct {
    int32_t cls_id;
    float score;
    BOX_INFO_S box_info;
    float mask[160 * 160];
} SEGMENT_OUT_INFO_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} SIGMOID_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} SIN_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    int64_t starts[8];
    int64_t ends[8];
    int64_t axes[8];
    int64_t steps[8];
    int64_t slice_axes_num;
} SLICE_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
    int64_t dims;
    int64_t axis;
} SOFTMAX_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    int64_t axis;
    int64_t split[8];
    int64_t split_num;
} SPLIT_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} SQRT_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    int64_t axes[8];
    int64_t axes_num;
} SQUEEZE_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} SUB_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;
} TILE_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    int64_t perm[8];
    int64_t perm_num;
} TRANSPOSE_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    int64_t axis;
    int64_t largest;
    int64_t sorted;
    int64_t topk_num;
} TOP_K_CONFIG_S;

typedef struct
{
    BASE_CONFIG_S op_base_cfg;

    int64_t axes[SHAPE_LEN];
    int64_t axes_num;
} UNSQUEEZE_CONFIG_S;

typedef struct
{
    ELEM_TYPE_E elem_type;
    char name[8];
    int size;
} ELEM_INFO;

extern const ELEM_INFO elem_info_map[];

const ELEM_INFO elem_info_map[]= {
        {TYPE_UNKNOWN, "unknown", 0},
        {TYPE_FP32, "fp32", 4},
        {TYPE_UINT8, "uint8", 1},
        {TYPE_INT8, "int8", 1},
        {TYPE_UINT16, "uint16", 2},
        {TYPE_INT16, "int16", 2},
        {TYPE_INT32, "int32", 4},
        {TYPE_INT64, "int64", 8},

        {TYPE_STRING, "string", 128},
        {TYPE_BOOLEAN, "boolean", 8},
        {TYPE_FP16, "float16", 2},
        {TYPE_FP64, "float64", 8},
        {TYPE_UINT32, "uint32", 4},
        {TYPE_UINT64, "uint64", 8},
        {TYPE_COMPLEX128, "complex", 128},
        {TYPE_BF16, "bf16", 2},
};


#include "utils_c.h"

#endif // _NN_COMMON_H__

