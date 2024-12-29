#ifndef OP_EXPAND_H
#define OP_EXPAND_H

#include "op.h"
#include "../manager/manager.h"

class Expand : public op
{
public:
    EXPAND_CONFIG_S expand_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    Expand()
    {
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *expand_cfg_ptr)
    {
        // new Expand op
        std::shared_ptr<Expand> relu_ptr = std::make_shared<Expand>();

        // fill op config
        memcpy(&(relu_ptr->expand_cfg), expand_cfg_ptr, sizeof(EXPAND_CONFIG_S));

        op_ptr = relu_ptr;

        return 0;
    }

    virtual int shape_infer(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];
        OPERAND_S* out = &operand_stu_map[out_operands[0]];

        for (int i = 0; i < SHAPE_LEN; ++i) {
            out->shapes[i] = 1;
        }

        int elem_size = 1;
        for (int i = 0; i < SHAPE_LEN; ++i) {
            elem_size *= in->shapes[i];
        }

        for (int i = 0; i < this->expand_cfg.dst_shape_num; ++i) {
            out->shapes[i] = this->expand_cfg.dst_shape[i];
        }

        int elem_size_no_doubt = 1;
        for (int i = 0; i < this->expand_cfg.dst_shape_num; ++i) {
            if (out->shapes[i] != -1) {
                elem_size_no_doubt *= out->shapes[i];
            }
        }

        for (int i = 0; i < this->expand_cfg.dst_shape_num; ++i) {
            if (out->shapes[i] == -1) {
                out->shapes[i] = elem_size / elem_size_no_doubt;
            }
        }

        out->dim_num_of_shapes = this->expand_cfg.dst_shape_num;

        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&expand_cfg);
        params_vec[0] = params;

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->expand_cfg));
        op_name = (char*)((int64_t)&(this->expand_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->expand_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->expand_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->expand_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(Expand, Expand::create_instance, sizeof(EXPAND_CONFIG_S));

#endif // OP_EXPAND_H
