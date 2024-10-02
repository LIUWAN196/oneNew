#ifndef OP_PAD_H
#define OP_PAD_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Pad : public op
{
public:
    PAD_CONFIG_S pad_cfg;

    Pad()
    {
//        printf("new a Pad\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *pad_cfg_ptr)
    {
        // new Pad op
        std::shared_ptr<Pad> pad_ptr = std::make_shared<Pad>();
//        pad_ptr.get()->find_handle((BUFFER_GROUP_S *)pad_cfg_ptr);

        // fill op config
        memcpy(&(pad_ptr->pad_cfg), pad_cfg_ptr, sizeof(PAD_CONFIG_S));

        // // fill op type and op name
        // op_type = pad_cfg_ptr;
        // op_name = pad_cfg_ptr + OP_TYPE_LEN;

        op_ptr = pad_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];

        OPERAND_S* out = &operand_stu_map[out_operands[0]];
        // the out shape equal in shape
        memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
        out->dim_num_of_shapes = in->dim_num_of_shapes;

        out->shapes[0] = in->shapes[0] + pad_cfg.pads[0] + pad_cfg.pads[4 + 0];
        out->shapes[1] = in->shapes[1] + pad_cfg.pads[1] + pad_cfg.pads[4 + 1];
        out->shapes[2] = in->shapes[2] + pad_cfg.pads[2] + pad_cfg.pads[4 + 2];
        out->shapes[3] = in->shapes[3] + pad_cfg.pads[3] + pad_cfg.pads[4 + 3];


        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&pad_cfg);
        params_vec[0] = params;
//        params_vec.push_back(params);

//        BUFFER_INFO_S params;
//        params.addr = (int64_t)(&pad_cfg);
//        params_vec.push_back(params);
        return  0;

    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char *)(&(this->pad_cfg));
        op_name = (char *)((int64_t)&(this->pad_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->pad_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->pad_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->pad_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(Pad, Pad::create_instance, sizeof(PAD_CONFIG_S));

#endif // OP_PAD_H
