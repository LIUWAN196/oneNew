#ifndef OP_MAXPOOL_H
#define OP_MAXPOOL_H

#include "op.h"
#include "../manager/manager.h"

class MaxPool : public op
{
public:
    MAX_POOL_CONFIG_S max_pool_cfg;

    MaxPool()
    {
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *max_pool_cfg_ptr)
    {
        // new MaxPool op
        std::shared_ptr<MaxPool> max_pool_ptr = std::make_shared<MaxPool>();

        // fill op config
        memcpy(&(max_pool_ptr->max_pool_cfg), max_pool_cfg_ptr, sizeof(MAX_POOL_CONFIG_S));

        op_ptr = max_pool_ptr;

        return 0;
    }

    virtual int shape_infer(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];

        OPERAND_S* out = &operand_stu_map[out_operands[0]];
        // the out shape equal in shape
        memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
        out->dim_num_of_shapes = in->dim_num_of_shapes;

        out->shapes[0] = in->shapes[0];
        out->shapes[1] = in->shapes[1];
        out->shapes[2] =
                (in->shapes[2] + max_pool_cfg.pads[0] + max_pool_cfg.pads[2] - max_pool_cfg.kernel_shape[0]) / max_pool_cfg.strides[0] +
                1;
        out->shapes[3] =
                (in->shapes[3] + max_pool_cfg.pads[1] + max_pool_cfg.pads[3] - max_pool_cfg.kernel_shape[1]) / max_pool_cfg.strides[1] +
                1;


        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&max_pool_cfg);
        params_vec[0] = params;

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        ONE_MODEL_DESC_S *one_model_desc_ptr = (ONE_MODEL_DESC_S *) one_buf_ptr;

        USEFUL_INFO_S* useful_ptr =  &one_model_desc_ptr->useful_info;
        BUFFER_INFO_S useful_info;
        useful_info.addr = (int64_t) useful_ptr;
        params_vec[BUF_MAXNUM - 1] = useful_info;

        // fill op type and op name
        op_type = (char *)(&(this->max_pool_cfg));
        op_name = (char *)((int64_t)&(this->max_pool_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->max_pool_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->max_pool_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->max_pool_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }

    virtual double get_computation() override {
        int32_t out_elem_size = 1;
        OPERAND_S* ifmap = (OPERAND_S*)params_vec[1].addr;
        OPERAND_S* ofmap = (OPERAND_S*)params_vec[1 + this->in_operands.size()].addr;
        for (int i = 0; i < SHAPE_LEN; ++i) {
            out_elem_size *= ofmap->shapes[i];
        }

        return (double)(out_elem_size * max_pool_cfg.kernel_shape[0] * max_pool_cfg.kernel_shape[1] * 1e-6);
    };

};

OP_REGISTER_GLOBAL(MaxPool, MaxPool::create_instance, sizeof(MAX_POOL_CONFIG_S));

#endif // OP_MAXPOOL_H
