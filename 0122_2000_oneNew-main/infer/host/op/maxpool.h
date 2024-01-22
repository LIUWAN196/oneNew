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

        // fill op config
        memcpy(&(max_pool_ptr->max_pool_cfg), max_pool_cfg_ptr, sizeof(MAX_POOL_CONFIG_S));

        // // fill op type and op name
        // op_type = max_pool_cfg_ptr;
        // op_name = max_pool_cfg_ptr + NAME_LEN;

        op_ptr = max_pool_ptr;

        return 0;
    }

    int fill_operands() override
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
