#ifndef OP_CONV_H
#define OP_CONV_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Conv : public op
{
public:
    CONV_CONFIG_S conv_cfg;

    Conv()
    {
        printf("new a Conv\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *conv_cfg_ptr)
    {
        // new Conv op
        std::shared_ptr<Conv> conv_ptr = std::make_shared<Conv>();

        // fill op config
        memcpy(&(conv_ptr->conv_cfg), conv_cfg_ptr, sizeof(CONV_CONFIG_S));

        // // fill op type and op name
        // op_type = conv_cfg_ptr;
        // op_name = conv_cfg_ptr + NAME_LEN;

        op_ptr = conv_ptr;

        return 0;
    }

    int fill_operands() override
    {
        // fill op type and op name
        op_type = (char *)(&(this->conv_cfg));
        op_name = (char *)((int64_t)&(this->conv_cfg) + NAME_LEN);

        int32_t in_operand_cnt = 3;
        for (size_t i = 0; i < in_operand_cnt; i++)
        {
            std::string in_operand(this->conv_cfg.in_operand_name[i]);
            this->in_operands.push_back(in_operand);
        }

        std::string out_operand(this->conv_cfg.out_operand_name[0]);
        this->out_operands.push_back(out_operand);

        return 0;
    }
};

OP_REGISTER_GLOBAL(Conv, Conv::create_instance);

#endif // OP_CONV_H
