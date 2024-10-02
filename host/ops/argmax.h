#ifndef OP_ARGMAX_H
#define OP_ARGMAX_H

#include "op.h"
#include "../manager/manager.h"
// namespace one_new {

class ArgMax : public op
{
public:
    ARGMAX_CONFIG_S argmax_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    ArgMax()
    {
//        printf("new a ArgMax\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *argmax_cfg_ptr)
    {
        // new ArgMax op
        std::shared_ptr<ArgMax> argmax_ptr = std::make_shared<ArgMax>();
//        argmax_ptr.get()->find_handle((BUFFER_GROUP_S *)argmax_cfg_ptr);

        // fill op config
        memcpy(&(argmax_ptr->argmax_cfg), argmax_cfg_ptr, sizeof(ARGMAX_CONFIG_S));

        // // fill op type and op name
        // op_type = argmax_cfg_ptr;
        // op_name = argmax_cfg_ptr + OP_TYPE_LEN;

        op_ptr = argmax_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];
        OPERAND_S* out = &operand_stu_map[out_operands[0]];

        memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
        if (argmax_cfg.axis == -1) {
            out->shapes[in->dim_num_of_shapes - 1] = 1;
        }

        if (argmax_cfg.keepdims == 0) {
            out->dim_num_of_shapes = in->dim_num_of_shapes - 1;
        }


        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&argmax_cfg);
        params_vec[0] = params;

        return  0;

    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->argmax_cfg));
        op_name = (char*)((int64_t)&(this->argmax_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->argmax_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->argmax_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->argmax_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }


};

OP_REGISTER_GLOBAL(ArgMax, ArgMax::create_instance, sizeof(ARGMAX_CONFIG_S));

#endif // OP_ARGMAX_H
