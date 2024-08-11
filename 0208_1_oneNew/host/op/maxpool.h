#ifndef OP_MAXPOOL_H
#define OP_MAXPOOL_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class MaxPool : public op
{
public:
    MAX_POOL_CONFIG_S max_pool_cfg;

    MaxPool()
    {
        printf("new a MaxPool\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *max_pool_cfg_ptr)
    {
        // new MaxPool op
        std::shared_ptr<MaxPool> max_pool_ptr = std::make_shared<MaxPool>();
//        max_pool_ptr.get()->find_handle((BUFFER_GROUP_S *)max_pool_cfg_ptr);

        // fill op config
        memcpy(&(max_pool_ptr->max_pool_cfg), max_pool_cfg_ptr, sizeof(MAX_POOL_CONFIG_S));

        // // fill op type and op name
        // op_type = max_pool_cfg_ptr;
        // op_name = max_pool_cfg_ptr + NAME_LEN;

        op_ptr = max_pool_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];

        OPERAND_S* out = &operand_stu_map[out_operands[0]];
        out->shape.N = in->shape.N;
        out->shape.C = in->shape.C;
        out->shape.H = (in->shape.H + max_pool_cfg.pads[0] + max_pool_cfg.pads[2] - max_pool_cfg.kernel_shape[0]) / max_pool_cfg.strides[0] + 1;
        out->shape.W = (in->shape.W + max_pool_cfg.pads[1] + max_pool_cfg.pads[3] - max_pool_cfg.kernel_shape[1]) / max_pool_cfg.strides[1] + 1;
        int b = 101;

        BUFFER_INFO_S params;
        params.addr = (int64_t)(&max_pool_cfg);
        params_vec.push_back(params);
        return  0;

    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char *)(&(this->max_pool_cfg));
        op_name = (char *)((int64_t)&(this->max_pool_cfg) + NAME_LEN);

        std::string in_operand(this->max_pool_cfg.in_operand_name[0]);
        this->in_operands.push_back(in_operand);

        std::string out_operand(this->max_pool_cfg.out_operand_name[0]);
        this->out_operands.push_back(out_operand);

        return 0;
    }
};

OP_REGISTER_GLOBAL(MaxPool, MaxPool::create_instance);

#endif // OP_MAXPOOL_H
