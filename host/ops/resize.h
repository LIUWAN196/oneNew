#ifndef OP_RESIZE_H
#define OP_RESIZE_H

#include "op.h"
#include "../manager/manager.h"

class Resize : public op
{
public:
    RESIZE_CONFIG_S resize_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    Resize()
    {
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *resize_cfg_ptr)
    {
        // new Resize op
        std::shared_ptr<Resize> resize_ptr = std::make_shared<Resize>();

        // fill op config
        memcpy(&(resize_ptr->resize_cfg), resize_cfg_ptr, sizeof(RESIZE_CONFIG_S));

        op_ptr = resize_ptr;

        return 0;
    }

    virtual int shape_infer(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];

        OPERAND_S* out = &operand_stu_map[out_operands[0]];
        // the out shape equal in shape
        memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
        out->dim_num_of_shapes = in->dim_num_of_shapes;

        if (resize_cfg.scales[0] != 0) {
            out->shapes[2] = (int32_t)(in->shapes[2] * resize_cfg.scales[2]);       //   resize h
            out->shapes[3] = (int32_t)(in->shapes[3] * resize_cfg.scales[3]);       //   resize w
        } else {
            out->shapes[2] = (int32_t)(resize_cfg.sizes[2]);       //   resize h
            out->shapes[3] = (int32_t)(resize_cfg.sizes[3]);       //   resize w
        }

        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&resize_cfg);
        params_vec[0] = params;

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->resize_cfg));
        op_name = (char*)((int64_t)&(this->resize_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->resize_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->resize_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->resize_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(Resize, Resize::create_instance, sizeof(RESIZE_CONFIG_S));

#endif // OP_RESIZE_H
