#ifndef OP_ADD_H
#define OP_ADD_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Add : public op
{
public:
    ADD_CONFIG_S add_cfg;

    Add()
    {
        printf("new a Add\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *relu_cfg_ptr)
    {
        // new Add op
        std::shared_ptr<Add> add_ptr = std::make_shared<Add>();

        // fill op config
        memcpy(&(add_ptr->add_cfg), relu_cfg_ptr, sizeof(ADD_CONFIG_S));

        // // fill op type and op name
        // op_type = relu_cfg_ptr;
        // op_name = relu_cfg_ptr + NAME_LEN;

        op_ptr = add_ptr;

        return 0;
    }

    int fill_operands() override
    {
        // fill op type and op name
        op_type = (char*)(&(this->add_cfg));
        op_name = (char*)((int64_t)&(this->add_cfg) + NAME_LEN);

        std::string in0_operand(this->add_cfg.in_operand_name[0]);
        this->in_operands.push_back(in0_operand);

        std::string in1_operand(this->add_cfg.in_operand_name[1]);
        this->in_operands.push_back(in1_operand);

        std::string out_operand(this->add_cfg.out_operand_name[0]);
        this->out_operands.push_back(out_operand);

        return 0;
    }
};

OP_REGISTER_GLOBAL(Add, Add::create_instance);

#endif // OP_ADD_H
