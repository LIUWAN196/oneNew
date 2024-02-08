#ifndef __NN_COMMON_H__
#define __NN_COMMON_H__

#include "stdint.h"

//#define OP_LIB_DIR "/home/wanzai/桌面/"

#define OP_LIB_DIR "/Project/e0006809/0206_2_oneNew/cmake-build-debug/device/x86/"

#define BUF_MAXNUM 8
#define NAME_LEN 256

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

typedef struct
{
    char operand_name[NAME_LEN];
    BOOL is_fixed_val; // such as: weight, bias
    int64_t p_data;
    OPERAND_SHAPE_S shape;
    ELEM_TYPE_E data_type;
} OPERAND_S;

typedef struct
{
    char op_type[NAME_LEN]; //  io
    char op_name[NAME_LEN]; //  input or output
    OPERAND_S operand;
} IO_CONFIG_S;

typedef struct
{
    char op_type[NAME_LEN];
    char op_name[NAME_LEN];
    char in_operand_name[1][NAME_LEN];
    char out_operand_name[1][NAME_LEN];
} RELU_CONFIG_S;

typedef struct
{
    char op_type[NAME_LEN];
    char op_name[NAME_LEN];
    char in_operand_name[2][NAME_LEN];
    char out_operand_name[1][NAME_LEN];
} ADD_CONFIG_S;

typedef struct
{
    char op_type[NAME_LEN];
    char op_name[NAME_LEN];
    char in_operand_name[2][NAME_LEN];
    char out_operand_name[1][NAME_LEN];
    int64_t dims;
} SOFTMAX_CONFIG_S;

typedef struct
{
    char op_type[NAME_LEN];
    char op_name[NAME_LEN];
    char in_operand_name[1][NAME_LEN];
    char out_operand_name[1][NAME_LEN];

    int64_t ceil_mode;
    int64_t kernel_shape[2];
    int64_t pads[4];
    int64_t strides[2];
} MAX_POOL_CONFIG_S;

typedef struct
{
    char op_type[NAME_LEN];
    char op_name[NAME_LEN];
    char in_operand_name[3][NAME_LEN]; // the order is: input, weight, and bias
    char out_operand_name[1][NAME_LEN];

    BOOL has_bias;

    int64_t dilations[2];
    int64_t group;
    int64_t kernel_shape[2];
    int64_t pads[4];
    int64_t strides[4];
} CONV_CONFIG_S;

typedef struct
{
    char op_type[NAME_LEN];
    char op_name[NAME_LEN];
    char in_operand_name[1][NAME_LEN];
    char out_operand_name[1][NAME_LEN];
} GLOBAL_AVGPOOL_CONFIG_S;

typedef struct
{
    char op_type[NAME_LEN];
    char op_name[NAME_LEN];
    char in_operand_name[1][NAME_LEN];
    char out_operand_name[1][NAME_LEN];
    int64_t axis;
} FLATTEN_CONFIG_S;

typedef struct
{
    char op_type[NAME_LEN];
    char op_name[NAME_LEN];
    char in_operand_name[3][NAME_LEN]; // the order is: input, fc.weight, and fc.bias
    char out_operand_name[1][NAME_LEN];

    BOOL has_bias;

    float alpha;
    float beta;
    int64_t transB;
} GEMM_CONFIG_S;


typedef struct
{
    ELEM_TYPE_E elem_type;
    char name[8];
    int size;
} ELEM_INFO;

static const ELEM_INFO elem_info_map[] = {
    {TYPE_UNKNOWN, "unknown", 0},
    {TYPE_FP32, "fp32", 4},
};

inline int32_t align_buf_size(int32_t ori_size)
{
    return (ori_size + 63) & (~63);
}

inline int32_t operand_elem_size(OPERAND_S *cur_operand)
{
    OPERAND_SHAPE_S *shape = &(cur_operand->shape);
    int32_t elem_size = shape->N * shape->C * shape->H * shape->W;
    return elem_size;
}

inline int32_t operand_buf_size(OPERAND_S *cur_operand)
{
    int32_t buf_size = operand_elem_size(cur_operand) * elem_info_map[cur_operand->data_type].size;
    return buf_size;
}

inline int32_t align_operand_buf_size(OPERAND_S *cur_operand)
{
    return (operand_buf_size(cur_operand) + 63) & (~63);
}


#endif // _NN_COMMON_H__
