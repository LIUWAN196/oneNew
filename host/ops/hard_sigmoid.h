#ifndef OP_HARD_SIGMOID_H
#define OP_HARD_SIGMOID_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class HardSigmoid : public op
{
public:
    HARD_SIGMOID_CONFIG_S hard_sigmoid_cfg;

    HardSigmoid()
    {
//        printf("new a HardSigmoid\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *relu_cfg_ptr)
    {
        // new HardSigmoid op
        std::shared_ptr<HardSigmoid> hard_sigmoid_ptr = std::make_shared<HardSigmoid>();

        // fill op config
        memcpy(&(hard_sigmoid_ptr->hard_sigmoid_cfg), relu_cfg_ptr, sizeof(HARD_SIGMOID_CONFIG_S));

        // // fill op type and op name
        // op_type = relu_cfg_ptr;
        // op_name = relu_cfg_ptr + OP_TYPE_LEN;

        op_ptr = hard_sigmoid_ptr;

        return 0;
    }

    virtual int shape_infer(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];

        OPERAND_S* out = &operand_stu_map[out_operands[0]];

        // the out shape equal in shape
        memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
        out->dim_num_of_shapes = in->dim_num_of_shapes;


inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&hard_sigmoid_cfg);
        params_vec[0] = params;
//        params_vec.push_back(params);

//        BUFFER_INFO_S params;
//        params.addr = (int64_t)(&hard_sigmoid_cfg);
//        params_vec.push_back(params);

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->hard_sigmoid_cfg));
        op_name = (char*)((int64_t)&(this->hard_sigmoid_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->hard_sigmoid_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->hard_sigmoid_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->hard_sigmoid_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(HardSigmoid, HardSigmoid::create_instance, sizeof(HARD_SIGMOID_CONFIG_S));

#endif // OP_HARD_SIGMOID_H
