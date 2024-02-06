#ifndef OP_FLATTEN_H
#define OP_FLATTEN_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Flatten : public op
{
public:
    FLATTEN_CONFIG_S flatten_cfg;

    Flatten()
    {
        printf("new a Relu\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *flatten_cfg_ptr)
    {
        // new Flatten op
        std::shared_ptr<Flatten> flatten_ptr = std::make_shared<Flatten>();

        // fill op config
        memcpy(&(flatten_ptr->flatten_cfg), flatten_cfg_ptr, sizeof(FLATTEN_CONFIG_S));


        op_ptr = flatten_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S in = operand_stu_map[in_operands[0]];

        OPERAND_S out = operand_stu_map[out_operands[0]];


        return  0;
    };

    int fill_operands() override
    {
        // fill op type and op name
        op_type = (char*)(&(this->flatten_cfg));
        op_name = (char*)((int64_t)&(this->flatten_cfg) + NAME_LEN);

        std::string in_operand(this->flatten_cfg.in_operand_name[0]);
        this->in_operands.push_back(in_operand);

        std::string out_operand(this->flatten_cfg.out_operand_name[0]);
        this->out_operands.push_back(out_operand);

        return 0;
    }
};

OP_REGISTER_GLOBAL(Flatten, Flatten::create_instance);

#endif // OP_FLATTEN_H
