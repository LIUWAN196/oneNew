#ifndef OP_UNSQUEEZE_H
#define OP_UNSQUEEZE_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Unsqueeze : public op
{
public:
    UNSQUEEZE_CONFIG_S unsqueeze_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    Unsqueeze()
    {
//        printf("new a Unsqueeze\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *unsqueeze_cfg_ptr)
    {
        // new Unsqueeze op
        std::shared_ptr<Unsqueeze> relu_ptr = std::make_shared<Unsqueeze>();
//        relu_ptr.get()->find_handle((BUFFER_GROUP_S *)squeeze_cfg_ptr);

        // fill op config
        memcpy(&(relu_ptr->unsqueeze_cfg), unsqueeze_cfg_ptr, sizeof(UNSQUEEZE_CONFIG_S));

        op_ptr = relu_ptr;

        return 0;
    }


    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];
        OPERAND_S* out = &operand_stu_map[out_operands[0]];

        // the out shape equal in shape
//        memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
        for (int i = 0; i < SHAPE_LEN; ++i) {
            out->shapes[i] = 1;
        }
        out->dim_num_of_shapes = in->dim_num_of_shapes + unsqueeze_cfg.axes_num;

        if (unsqueeze_cfg.axes_num != 1) {
            LOG_ERR("sorry, cur unsqueeze op just surpport axes num is 1");
        }
        int reduce_dims = unsqueeze_cfg.axes[0];
        for (int i = 0; i < SHAPE_LEN; ++i) {
            if (i < reduce_dims) {
                out->shapes[i] = in->shapes[i];
            } else if (i == reduce_dims) {
                out->shapes[i] = 1;
            } else {
                out->shapes[i] = in->shapes[i - 1];
            }
        }

        params_vec.resize(1 + in_operands.size() + out_operands.size());
        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&unsqueeze_cfg);
        params_vec[0] = params;

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->unsqueeze_cfg));
        op_name = (char*)((int64_t)&(this->unsqueeze_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->unsqueeze_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->unsqueeze_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->unsqueeze_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(Unsqueeze, Unsqueeze::create_instance, sizeof(UNSQUEEZE_CONFIG_S));

#endif // OP_UNSQUEEZE_H