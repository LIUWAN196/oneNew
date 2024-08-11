#ifndef OP_GLOBAL_AVGPOOL_H
#define OP_GLOBAL_AVGPOOL_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class GlobalAveragePool : public op
{
public:
    GLOBAL_AVGPOOL_CONFIG_S global_avgpool_cfg;

    GlobalAveragePool()
    {
        printf("new a Relu\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *global_avgpool_cfg_ptr)
    {
        // new GlobalAveragePool op
        std::shared_ptr<GlobalAveragePool> global_avgpool_ptr = std::make_shared<GlobalAveragePool>();

        // fill op config
        memcpy(&(global_avgpool_ptr->global_avgpool_cfg), global_avgpool_cfg_ptr, sizeof(GLOBAL_AVGPOOL_CONFIG_S));


        op_ptr = global_avgpool_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S in = operand_stu_map[in_operands[0]];

        OPERAND_S out = operand_stu_map[out_operands[0]];


        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->global_avgpool_cfg));
        op_name = (char*)((int64_t)&(this->global_avgpool_cfg) + NAME_LEN);

        std::string in_operand(this->global_avgpool_cfg.in_operand_name[0]);
        this->in_operands.push_back(in_operand);

        std::string out_operand(this->global_avgpool_cfg.out_operand_name[0]);
        this->out_operands.push_back(out_operand);

        return 0;
    }
};

OP_REGISTER_GLOBAL(GlobalAveragePool, GlobalAveragePool::create_instance);

#endif // OP_GLOBAL_AVGPOOL_H
