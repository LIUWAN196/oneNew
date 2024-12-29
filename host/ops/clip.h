#ifndef OP_CLIP_H
#define OP_CLIP_H

#include "op.h"
#include "../manager/manager.h"

class Clip : public op
{
public:
    CLIP_CONFIG_S clip_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    Clip()
    {
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *clip_cfg_ptr)
    {
        // new Clip op
        std::shared_ptr<Clip> clip_ptr = std::make_shared<Clip>();

        // fill op config
        memcpy(&(clip_ptr->clip_cfg), clip_cfg_ptr, sizeof(CLIP_CONFIG_S));

        op_ptr = clip_ptr;

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
        params.addr = (int64_t) (&clip_cfg);
        params_vec[0] = params;

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->clip_cfg));
        op_name = (char*)((int64_t)&(this->clip_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->clip_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->clip_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->clip_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(Clip, Clip::create_instance, sizeof(CLIP_CONFIG_S));

#endif // OP_CLIP_H
