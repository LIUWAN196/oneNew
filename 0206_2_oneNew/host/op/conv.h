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
//        conv_ptr.get()->find_handle((BUFFER_GROUP_S *)conv_cfg_ptr);

        // fill op config
        memcpy(&(conv_ptr->conv_cfg), conv_cfg_ptr, sizeof(CONV_CONFIG_S));

        // // fill op type and op name
        // op_type = conv_cfg_ptr;
        // op_name = conv_cfg_ptr + NAME_LEN;

        op_ptr = conv_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];

        OPERAND_S* out = &operand_stu_map[out_operands[0]];
        out->shape.N = in->shape.N;
        out->shape.C = in->shape.C;
        out->shape.H = (in->shape.H + conv_cfg.pads[0] + conv_cfg.pads[2] - conv_cfg.kernel_shape[0]) / conv_cfg.strides[0];
        out->shape.W = (in->shape.W + conv_cfg.pads[1] + conv_cfg.pads[3] - conv_cfg.kernel_shape[1]) / conv_cfg.strides[1];
        int b = 101;

        BUFFER_INFO_S params;
        params.addr = (int64_t)(&conv_cfg);
        params_vec.push_back(params);

        return  0;
    };

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
