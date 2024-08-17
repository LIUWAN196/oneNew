#include <iostream>
#include <fstream>
#include "cstring"
#include <time.h>
#include <vector>
#include "cmath"
#include <iostream>
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
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
#include <omp.h>
#include <numeric>
#include "../../common/nn_common_cpp.h"

#include "ops_head.h"
#include "net.h"

typedef struct
{
    BASE_CONFIG_S * base_cfg;
    int32_t be_fused;
//    std::string fusion_op_type;
//    char fusion_op_type[256];
//    FUSION_OP_TYPE_E fusion_op_type;
    int32_t fusion_op_cnt_in_entire_net;    // 如果融合，那么融合后的融合算子这是整个网络的第几个融合算子
    int32_t cur_op_cnt_in_fusion_op;        // 如果融合，那么当前这个算子是融合算子的第几个子算子
} FUSE_MARK_S;

typedef int (*FillFuseOpCfg)(std::vector<BASE_CONFIG_S *> &total_fuse_op_cfg_set, std::vector<BASE_CONFIG_S *> sub_op_cfg);

int fill_conv_relu_fuse_op_cfg(std::vector<BASE_CONFIG_S *> &total_fuse_op_cfg_set, std::vector<BASE_CONFIG_S *> sub_op_cfg){

    CONV_CONFIG_S* conv_relu_cfg = (CONV_CONFIG_S*) malloc(sizeof(CONV_CONFIG_S));
    memcpy(conv_relu_cfg, sub_op_cfg[0], sizeof(CONV_CONFIG_S));

    // 修改消费者 op 为 relu 的输出 op
    BASE_CONFIG_S *relu_cfg = sub_op_cfg[1];
    memcpy(&conv_relu_cfg->op_base_cfg.consumer[0], &relu_cfg->consumer[0],
           OPERAND_MAXNUM * OPERAND_NAME_LEN);

    // 修改输出操作数为 relu 的输出操作数
    memcpy(&conv_relu_cfg->op_base_cfg.out_operand_name[0], &relu_cfg->out_operand_name[0],
           OPERAND_MAXNUM * sizeof(NODE_INFO_S));

    // 修改激活函数类型
    conv_relu_cfg->act_type = RELU;

    conv_relu_cfg->op_base_cfg.in_operand_num = 3;
    conv_relu_cfg->op_base_cfg.out_operand_num = 1;
    conv_relu_cfg->op_base_cfg.producer_num = 1;
    conv_relu_cfg->op_base_cfg.consumer_num = 1;

    // 将 conv relu 这个融合算子的 cfg 挂载到 total_fuse_op_cfg_set 中
    total_fuse_op_cfg_set.push_back((BASE_CONFIG_S *)conv_relu_cfg);

    return 0;
}

int fill_conv_silu_fuse_op_cfg(std::vector<BASE_CONFIG_S *> &total_fuse_op_cfg_set, std::vector<BASE_CONFIG_S *> sub_op_cfg){

    CONV_CONFIG_S* conv_silu_cfg = (CONV_CONFIG_S*) malloc(sizeof(CONV_CONFIG_S));
    memcpy(conv_silu_cfg, sub_op_cfg[0], sizeof(CONV_CONFIG_S));

    // 修改消费者 op 为 mul 的输出 op
    BASE_CONFIG_S *mul_cfg = sub_op_cfg[2];
    memcpy(&conv_silu_cfg->op_base_cfg.consumer[0], &mul_cfg->consumer[0],
           OPERAND_MAXNUM * sizeof(NODE_INFO_S));

    // 修改输出操作数为 silu 的输出操作数
    memcpy(&conv_silu_cfg->op_base_cfg.out_operand_name[0], &mul_cfg->out_operand_name[0],
           OPERAND_MAXNUM * OPERAND_NAME_LEN);

    // 修改激活函数类型
    conv_silu_cfg->act_type = SILU;

    conv_silu_cfg->op_base_cfg.in_operand_num = 3;
    conv_silu_cfg->op_base_cfg.out_operand_num = 1;
    conv_silu_cfg->op_base_cfg.producer_num = 1;
    conv_silu_cfg->op_base_cfg.consumer_num = 1;

    // 将 conv silu 这个融合算子的 cfg 挂载到 total_fuse_op_cfg_set 中
    total_fuse_op_cfg_set.push_back((BASE_CONFIG_S *)conv_silu_cfg);

    return 0;
}

int fill_gelu_fuse_op_cfg(std::vector<BASE_CONFIG_S *> &total_fuse_op_cfg_set, std::vector<BASE_CONFIG_S *> sub_op_cfg){

    GELU_CONFIG_S* gelu_cfg = (GELU_CONFIG_S*) malloc(sizeof(GELU_CONFIG_S));
    memcpy(gelu_cfg, sub_op_cfg[1], sizeof(DIV_CONFIG_S));
    memset(gelu_cfg->op_base_cfg.op_type, 0, OP_TYPE_LEN);

    std::string op_type = "Gelu";
    strcpy(gelu_cfg->op_base_cfg.op_type, op_type.c_str());

    // 修改消费者 op 为最后一个 mul 的输出 op
    BASE_CONFIG_S *mul_cfg = sub_op_cfg[5];
    memcpy(&gelu_cfg->op_base_cfg.consumer[0], &mul_cfg->consumer[0],
           OPERAND_MAXNUM * sizeof(NODE_INFO_S));

    // 修改输出操作数为 silu 的输出操作数
    memcpy(&gelu_cfg->op_base_cfg.out_operand_name[0], &mul_cfg->out_operand_name[0],
           OPERAND_MAXNUM * OPERAND_NAME_LEN);

    gelu_cfg->op_base_cfg.in_operand_num = 1;
    gelu_cfg->op_base_cfg.out_operand_num = 1;
    gelu_cfg->op_base_cfg.producer_num = 1;
    gelu_cfg->op_base_cfg.consumer_num = 1;

    // 将 gelu 这个融合算子的 cfg 挂载到 total_fuse_op_cfg_set 中
    total_fuse_op_cfg_set.push_back((BASE_CONFIG_S *)gelu_cfg);

    return 0;
}

int fill_layer_norm_fuse_op_cfg(std::vector<BASE_CONFIG_S *> &total_fuse_op_cfg_set, std::vector<BASE_CONFIG_S *> sub_op_cfg){

    LAYERNORM_CONFIG_S * ln_cfg = (LAYERNORM_CONFIG_S*) malloc(sizeof(LAYERNORM_CONFIG_S));
    memcpy(ln_cfg, sub_op_cfg[1], sizeof(LAYERNORM_CONFIG_S));
    memset(ln_cfg->op_base_cfg.op_type, 0, OP_TYPE_LEN);

    std::string op_type = "LayerNorm";
    strcpy(ln_cfg->op_base_cfg.op_type, op_type.c_str());

    // 修改消费者 op 为最后一个 div 的输出 op
    BASE_CONFIG_S *div_cfg = sub_op_cfg[7];
    memcpy(&ln_cfg->op_base_cfg.consumer[0], &div_cfg->consumer[0],
           OPERAND_MAXNUM * sizeof(NODE_INFO_S));

    // 修改输出操作数为 silu 的输出操作数
    memcpy(&ln_cfg->op_base_cfg.out_operand_name[0], &div_cfg->out_operand_name[0],
           OPERAND_MAXNUM * OPERAND_NAME_LEN);

    ln_cfg->op_base_cfg.in_operand_num = 1;
    ln_cfg->op_base_cfg.out_operand_num = 1;
    ln_cfg->op_base_cfg.producer_num = 1;
    ln_cfg->op_base_cfg.consumer_num = 1;

    // 将 gelu 这个融合算子的 cfg 挂载到 total_fuse_op_cfg_set 中
    total_fuse_op_cfg_set.push_back((BASE_CONFIG_S *)ln_cfg);

    return 0;
}

int fuse_operator(std::string fuse_op_type, char* one_file_buf, std::vector<std::pair<NODE_INFO_S, std::vector<int32_t>> > op_and_their_consumer,
                  std::vector<BASE_CONFIG_S *> &total_fuse_op_cfg_set, FillFuseOpCfg fill_fuse_op_cfg) {
    Manager &m = Manager::getInstance();

    int fusion_op_cnt_st = total_fuse_op_cfg_set.size();

    ONE_MODEL_DESC_S *one_model_info_ptr = (ONE_MODEL_DESC_S *) one_file_buf;
    char *node_cfg_ptr = (char *) ((char *) one_model_info_ptr + one_model_info_ptr->node_cfg_offset);
    int32_t node_cnt = one_model_info_ptr->node_cnt;

    std::vector<BASE_CONFIG_S *> base_op_vec;

    // 步骤 0： 直接把把所有 op 的 cfg 的地址全部放到 vector<base_op_cfg*> 中
    for (int node_i = 0; node_i < node_cnt; ++node_i) {
        BASE_CONFIG_S *base_op_ptr = (BASE_CONFIG_S *) node_cfg_ptr;
        base_op_vec.push_back(base_op_ptr);
        // update cur_node_cfg_ptr
        node_cfg_ptr += align_buf_size(m.op_cfg_size_map[base_op_ptr->op_type]);
    }

    // 步骤 1: 开始匹配 op_and_their_consumer 子图
    for (int node_i = 0; node_i < base_op_vec.size(); ++node_i) {
        next_node_i:
        if (node_i >= base_op_vec.size()) {
            break;
        }

        // 先匹配 op_and_their_consumer 子图的第一个 node
        NODE_INFO_S sub_first_op = op_and_their_consumer[0].first;
        if (strcmp(base_op_vec[node_i]->op_type, sub_first_op.op_type) != 0 &&
                strcmp("Foo", sub_first_op.op_type) != 0) {
            // OP 类型没匹配上，并且不是 Foo 这个通配 op type，直接匹配整个网络模型的下一个 node_i
            continue;
        }

        BOOL is_foo_op_st = FALSE;
        if (strcmp("Foo", sub_first_op.op_type) == 0) {
            is_foo_op_st = TRUE;
        }

        // 拷贝一份 op_and_their_consumer，因为下面要做修改
        std::vector<std::pair<NODE_INFO_S, std::vector<int32_t>> > tmp_op_and_their_consumer = op_and_their_consumer;
        // 类型匹配上了，将 op 类型和名称改为 base_op_vec 中的类型和名称
        memcpy(tmp_op_and_their_consumer[0].first.op_type, base_op_vec[node_i]->op_type, OP_TYPE_LEN);
        memcpy(tmp_op_and_their_consumer[0].first.op_name, base_op_vec[node_i]->op_name, OP_NAME_LEN);

        std::vector<FUSE_MARK_S> tmp_sub_op_be_mark;
        std::vector<BASE_CONFIG_S *> sub_op_cfg;

        for (int sub_op_i = 0; sub_op_i < tmp_op_and_their_consumer.size(); ++sub_op_i) {
            NODE_INFO_S sub_op = tmp_op_and_their_consumer[sub_op_i].first;

            std::vector<int32_t> sub_consumer_op_idx = tmp_op_and_their_consumer[sub_op_i].second;

            // 这里需要遍历一遍 base_op_vec，从 base_op_vec 找到 op tpye 和 op name 都符合 node
            int32_t tmp_node_i = 0;
            BASE_CONFIG_S *tmp_node_cfg;
            for (;tmp_node_i < base_op_vec.size(); ++tmp_node_i) {
                if (strcmp(base_op_vec[tmp_node_i]->op_type, sub_op.op_type) != 0 ||
                    strcmp(base_op_vec[tmp_node_i]->op_name, sub_op.op_name) != 0) {
                    if (tmp_node_i == base_op_vec.size() - 1){
                        // OP 类型和 OP name 没匹配上，直接匹配整个网络模型的下一个 node_i
                        node_i++;
                        goto next_node_i;
                    }
                    continue;
                } else {
                    // 找到匹配的了, 跳出这个循环，开始下面的步骤
                    tmp_node_cfg = base_op_vec[tmp_node_i];
                    break;
                }
            }

            if (tmp_node_cfg->fusion_op_type == 0 &&
                sub_consumer_op_idx.size() == 0) {
                // 如果 sub_consumer_op_idx.size() 为 0, 则表明该 op 为待匹配子图最后一个 op 了，就不用去匹配它的输出消费者了
            } else if (tmp_node_cfg->fusion_op_type == 0 &&
                    (sub_consumer_op_idx.size() == tmp_node_cfg->consumer_num || (is_foo_op_st == TRUE && sub_op_i == 0))) {
                // 走到这里，说明本个 sub_op 的名称以及 consumer_num 都匹配上了。现在需要一个一个来匹配 consumer 的 op 类型
                // 如果匹配上了，需要把 tmp_op_and_their_consumer 的 op name 改为 base_op_vec 中的 op name
                for (int consumer_i = 0; consumer_i < tmp_node_cfg->consumer_num; ++consumer_i) {
                    NODE_INFO_S * cur_consumer_node = &tmp_node_cfg->consumer[consumer_i];
                    for (int sub_consumer_i = 0; sub_consumer_i < sub_consumer_op_idx.size(); ++sub_consumer_i) {
                        int32_t cur_consumer_idx = tmp_op_and_their_consumer[sub_op_i].second[sub_consumer_i];
                        NODE_INFO_S * sub_consumer_node = &tmp_op_and_their_consumer[cur_consumer_idx].first;
                        if (strcmp(cur_consumer_node->op_type, sub_consumer_node->op_type) == 0) {
                            // 传入进来的待匹配子图的 op type 是真实的，但是 op name 是假的，需要映射为真实的 op_name
                            memcpy(sub_consumer_node, cur_consumer_node, sizeof(NODE_INFO_S));
                        }
                    }
                }
            } else {
                // 待匹配的子图的第 sub_op_i 个 op tmp_node_cfg 已经被融合过了，或者它的的消费者个数没有和 tmp_node_cfg 匹配
                // 上，开始匹配整个网络模型的下一个 node_i
                node_i++;
                goto next_node_i;
            }

            // 下面这个 tmp_sub_op_be_mark 只是暂时的
            int32_t be_fused = 1;
            tmp_sub_op_be_mark.push_back({tmp_node_cfg, be_fused, fusion_op_cnt_st, sub_op_i});
            sub_op_cfg.push_back(tmp_node_cfg);
        }
        // 走到这里，说明全部都匹配上了, 开始做子图的标记
        for (int sub_op_i = 0; sub_op_i < tmp_sub_op_be_mark.size(); ++sub_op_i) {
            FUSE_MARK_S* cur_op = &tmp_sub_op_be_mark[sub_op_i];
            BASE_CONFIG_S * sub_op_cfg_ptr = sub_op_cfg[sub_op_i];
            if (is_foo_op_st == TRUE && sub_op_i == 0) {
                // 如果第一个 op 是 Foo 的话，不要将 one buf 的第一个位置的 op 标识为 fuse op
                continue;
            }
            sub_op_cfg_ptr->fusion_op_type = cur_op->be_fused;
            sub_op_cfg_ptr->fusion_op_cnt_in_entire_net = cur_op->fusion_op_cnt_in_entire_net;
            if (is_foo_op_st == TRUE) {
                // 如果第一个 op 是 Foo 的话，为了后续真正融合时，能识别成功， 需要把 cur_op_cnt_in_fusion_op 标识往前提一位
                sub_op_cfg_ptr->cur_op_cnt_in_fusion_op = cur_op->cur_op_cnt_in_fusion_op - 1;
            } else {
                sub_op_cfg_ptr->cur_op_cnt_in_fusion_op = cur_op->cur_op_cnt_in_fusion_op;
            }
        }

        fusion_op_cnt_st++;

        // 标记完后，需要生成新的 cfg
        fill_fuse_op_cfg(total_fuse_op_cfg_set, sub_op_cfg);

    }

    return 0;
}

int do_fuse(char* optimize_one_buf_ptr, char* one_buf_ptr,
            int32_t ori_one_file_size, std::vector<BASE_CONFIG_S*> total_fuse_op_cfg_set){
    Manager &m = Manager::getInstance();

    ONE_MODEL_DESC_S *ori_one_model_info_ptr = (ONE_MODEL_DESC_S *) one_buf_ptr;
    char *node_cfg_ptr = (char *) ((char *) ori_one_model_info_ptr + ori_one_model_info_ptr->node_cfg_offset);
    int32_t ori_node_cnt = ori_one_model_info_ptr->node_cnt;

    int32_t ori_one_model_head_buf_size = ori_one_model_info_ptr->node_cfg_offset - 0;
    int32_t ori_one_model_node_cfg_buf_size = ori_one_model_info_ptr->init_info_offset - ori_one_model_info_ptr->node_cfg_offset;
    int32_t ori_one_model_init_buf_size = ori_one_model_info_ptr->io_cfg_offset - ori_one_model_info_ptr->init_info_offset;
    int32_t ori_one_model_io_cfg_buf_size = ori_one_file_size - ori_one_model_info_ptr->io_cfg_offset;

    // copy the head of table to optimize_one_buf_ptr
    ONE_MODEL_DESC_S *optimize_one_model_info_ptr = (ONE_MODEL_DESC_S *) optimize_one_buf_ptr;
    memcpy(optimize_one_buf_ptr, one_buf_ptr, ori_one_model_info_ptr->node_cfg_offset);
    char *optimize_node_cfg_ptr = (char *) ((char *) optimize_one_buf_ptr + optimize_one_model_info_ptr->node_cfg_offset);

    std::vector<BASE_CONFIG_S *> ori_base_op_vec;
    // 步骤 0： 直接把把所有 op 的 cfg 的地址全部放到 vector<base_op_cfg*> 中
    for (int node_i = 0; node_i < ori_node_cnt; ++node_i) {
        BASE_CONFIG_S *base_op_ptr = (BASE_CONFIG_S *) node_cfg_ptr;
        ori_base_op_vec.push_back(base_op_ptr);
        // update cur_node_cfg_ptr
        node_cfg_ptr += align_buf_size(m.op_cfg_size_map[base_op_ptr->op_type]);
    }

    // fill fused op
    BASE_CONFIG_S *ori_base_op_ptr;
    int optimize_node_cnt = 0;
    for (int node_i = 0; node_i < ori_node_cnt; ++node_i) {
        ori_base_op_ptr = ori_base_op_vec[node_i];
        if (ori_base_op_ptr->fusion_op_type == 0) {
            // the op is don't be fuse, direct copy
            int32_t align_cfg_size = align_buf_size(m.op_cfg_size_map[ori_base_op_ptr->op_type]);
            memcpy(optimize_node_cfg_ptr, ori_base_op_ptr, align_cfg_size);
            optimize_node_cfg_ptr += align_cfg_size;
            optimize_node_cnt ++;
        } else if(ori_base_op_ptr->fusion_op_type != 0 && ori_base_op_ptr->cur_op_cnt_in_fusion_op != 0) {
            // the op is fused, but it is not the first op in fuse op, skip
        } else {
            // the op is fused, and it is the first op in fuse op, replace by total_fuse_op_cfg_set
            int32_t fused_cfg_size = m.op_cfg_size_map[ori_base_op_ptr->op_type];
            memcpy(optimize_node_cfg_ptr, total_fuse_op_cfg_set[ori_base_op_ptr->fusion_op_cnt_in_entire_net], fused_cfg_size);
            optimize_node_cfg_ptr += align_buf_size(fused_cfg_size);
            optimize_node_cnt ++;
        }
    }
    optimize_one_model_info_ptr->node_cnt = optimize_node_cnt;

    // fill init info, such as weight / bias
    optimize_one_model_info_ptr->init_info_offset = (int32_t)((int64_t)optimize_node_cfg_ptr - (int64_t)optimize_one_buf_ptr);
    char* ori_init_buf_ptr = (char *) (one_buf_ptr + ori_one_model_info_ptr->init_info_offset);
    memcpy(optimize_node_cfg_ptr, ori_init_buf_ptr, ori_one_model_init_buf_size);
    optimize_node_cfg_ptr += ori_one_model_init_buf_size;

    // fill io cfg
    optimize_one_model_info_ptr->io_cfg_offset = (int32_t)((int64_t)optimize_node_cfg_ptr - (int64_t)optimize_one_buf_ptr);
    char* ori_io_cfg_buf_ptr = (char *) (one_buf_ptr + ori_one_model_info_ptr->io_cfg_offset);
    memcpy(optimize_node_cfg_ptr, ori_io_cfg_buf_ptr, ori_one_model_io_cfg_buf_size);
    optimize_node_cfg_ptr += ori_one_model_io_cfg_buf_size;


    return 0;
}

void print_op_cfg(char *one_ptr) {
    Manager &m = Manager::getInstance();
    ONE_MODEL_DESC_S *one_desc = (ONE_MODEL_DESC_S *) (one_ptr);
    int32_t node_cnt = one_desc->node_cnt;
    char* node_cfg = one_ptr + one_desc->node_cfg_offset;
    for (int node_i = 0; node_i < node_cnt; ++node_i) {
        BASE_CONFIG_S *base_op = (BASE_CONFIG_S*)node_cfg;
        LOG_MSG("%dth node type is %s, name is %s", node_i, base_op->op_type, base_op->op_name);
        node_cfg += align_buf_size(m.op_cfg_size_map[base_op->op_type]);
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " [optimize config.yml]" << std::endl;
        exit(-1);
    }
    const char *rt_cfg_txt = argv[1];
    std::string rt_cfg_txt_str(rt_cfg_txt);
    std::unordered_map<std::string, std::string> cfg_info_map;
    yml2map(cfg_info_map, rt_cfg_txt_str);

    const char* one_file_path = cfg_info_map["one_file_path"].c_str();
    const char* quant_one_file_path = cfg_info_map["quant_one_file_path"].c_str();

    // step 1: get one file size
    std::ifstream one_file(one_file_path, std::ios::ate | std::ios::binary);
    int32_t one_file_size = static_cast<int32_t>(one_file.tellg());
    one_file.close();

    // step 2: load one file
    char *one_buf_ptr = (char *)malloc(one_file_size);
    FILE *file_p = NULL;

    file_p = fopen(one_file_path, "r");
    if (file_p == NULL)
    {
        LOG_ERR("failed: can't open the one file");
        return 0;
    }
    fread(one_buf_ptr, sizeof(char), one_file_size, file_p);
    fclose(file_p);

    auto src_one_desc = (ONE_MODEL_DESC_S*)(one_buf_ptr);
     LOG_MSG("src one have %d op", src_one_desc->node_cnt);

    // step 9: fuse_operator
    std::vector<BASE_CONFIG_S *> total_fuse_op_cfg_set;

    // 这个是做 conv + relu 的子图标记
    {
        std::vector<std::pair<NODE_INFO_S, std::vector<int32_t>> > op_and_their_consumer;

        std::pair<NODE_INFO_S, std::vector<int32_t> > conv_op = {(NODE_INFO_S){"Conv", "Conv123"}, {1}};
        std::pair<NODE_INFO_S, std::vector<int32_t> > relu_op = {(NODE_INFO_S){"Relu", "Relu123"}, {}};
        op_and_their_consumer.push_back(conv_op);
        op_and_their_consumer.push_back(relu_op);

        fuse_operator("Conv", one_buf_ptr, op_and_their_consumer,
                      total_fuse_op_cfg_set, fill_conv_relu_fuse_op_cfg);
    }

    // 这个是做 conv + silu 的子图标记
    {
        std::vector<std::pair<NODE_INFO_S, std::vector<int32_t>> > op_and_their_consumer;

        std::pair<NODE_INFO_S, std::vector<int32_t> > conv_op = {(NODE_INFO_S){"Conv", "Conv123"}, {1, 2}};
        std::pair<NODE_INFO_S, std::vector<int32_t> > sigmoid_op = {(NODE_INFO_S){"Sigmoid", "Sigmoid123"}, {2}};
        std::pair<NODE_INFO_S, std::vector<int32_t> > mul_op = {(NODE_INFO_S){"Mul", "Mul123"}, {}};
        op_and_their_consumer.push_back(conv_op);
        op_and_their_consumer.push_back(sigmoid_op);
        op_and_their_consumer.push_back(mul_op);

        fuse_operator("Conv", one_buf_ptr, op_and_their_consumer,
                      total_fuse_op_cfg_set, fill_conv_silu_fuse_op_cfg);
    }

    // 这个是做 gelu 的子图标记
    {
        std::vector<std::pair<NODE_INFO_S, std::vector<int32_t>> > op_and_their_consumer;

        std::pair<NODE_INFO_S, std::vector<int32_t> > foo_op = {(NODE_INFO_S){"Foo", "Foo123"}, {1, 4}};
        std::pair<NODE_INFO_S, std::vector<int32_t> > div_op = {(NODE_INFO_S){"Div", "Div123"}, {2}};
        std::pair<NODE_INFO_S, std::vector<int32_t> > erf_op = {(NODE_INFO_S){"Erf", "Erf123"}, {3}};
        std::pair<NODE_INFO_S, std::vector<int32_t> > add_op = {(NODE_INFO_S){"Add", "Add123"}, {4}};
        std::pair<NODE_INFO_S, std::vector<int32_t> > mul_op0 = {(NODE_INFO_S){"Mul", "Mul123"}, {5}};
        std::pair<NODE_INFO_S, std::vector<int32_t> > mul_op1 = {(NODE_INFO_S){"Mul", "Mul456"}, {}};
        op_and_their_consumer.push_back(foo_op);
        op_and_their_consumer.push_back(div_op);
        op_and_their_consumer.push_back(erf_op);
        op_and_their_consumer.push_back(add_op);
        op_and_their_consumer.push_back(mul_op0);
        op_and_their_consumer.push_back(mul_op1);

        fuse_operator("Gelu", one_buf_ptr, op_and_their_consumer,
                      total_fuse_op_cfg_set, fill_gelu_fuse_op_cfg);
    }

    // 这个是做 layer norm 的子图标记
    {
        std::vector<std::pair<NODE_INFO_S, std::vector<int32_t>> > op_and_their_consumer;

        std::pair<NODE_INFO_S, std::vector<int32_t> > foo_op = {(NODE_INFO_S){"Foo", "Foo123"}, {1, 2}};
        std::pair<NODE_INFO_S, std::vector<int32_t> > reduce_mean_op0 = {(NODE_INFO_S){"ReduceMean", "ReduceMean123"}, {2}};
        std::pair<NODE_INFO_S, std::vector<int32_t> > sub_op = {(NODE_INFO_S){"Sub", "Sub123"}, {3, 7}};
        std::pair<NODE_INFO_S, std::vector<int32_t> > pow_op = {(NODE_INFO_S){"Pow", "Pow123"}, {4}};
        std::pair<NODE_INFO_S, std::vector<int32_t> > reduce_mean_op1 = {(NODE_INFO_S){"ReduceMean", "ReduceMean456"}, {5}};
        std::pair<NODE_INFO_S, std::vector<int32_t> > add_op1 = {(NODE_INFO_S){"Add", "Add123"}, {6}};
        std::pair<NODE_INFO_S, std::vector<int32_t> > sqrt_op = {(NODE_INFO_S){"Sqrt", "Sqrt123"}, {7}};
        std::pair<NODE_INFO_S, std::vector<int32_t> > div_op = {(NODE_INFO_S){"Div", "Div123"}, {}};
        op_and_their_consumer.push_back(foo_op);
        op_and_their_consumer.push_back(reduce_mean_op0);
        op_and_their_consumer.push_back(sub_op);
        op_and_their_consumer.push_back(pow_op);
        op_and_their_consumer.push_back(reduce_mean_op1);
        op_and_their_consumer.push_back(add_op1);
        op_and_their_consumer.push_back(sqrt_op);
        op_and_their_consumer.push_back(div_op);

        fuse_operator("LayerNorm", one_buf_ptr, op_and_their_consumer,
                      total_fuse_op_cfg_set, fill_layer_norm_fuse_op_cfg);
    }

    // 开始做真正的融合
    char *opt_one_buf_ptr = (char *)malloc(one_file_size); // todo: maybe the optimize_one_buf is bigger than the old one buf
    memset(opt_one_buf_ptr, 0, one_file_size);
    do_fuse(opt_one_buf_ptr, one_buf_ptr, one_file_size, total_fuse_op_cfg_set);

    auto dst_one_desc = (ONE_MODEL_DESC_S*)(opt_one_buf_ptr);
    LOG_MSG("dst one have %d op", dst_one_desc->node_cnt);
//    print_op_cfg(opt_one_buf_ptr);
    // step 10: // dump the optimize_one_buf_ptr as .one
    FILE *optimize_file_p = fopen(quant_one_file_path, "w");
    fwrite((void *)opt_one_buf_ptr, 1, one_file_size, optimize_file_p);
    fclose(optimize_file_p);

    return 0;
}

