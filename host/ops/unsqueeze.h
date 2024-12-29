#ifndef OP_UNSQUEEZE_H
#define OP_UNSQUEEZE_H

#include "op.h"
#include "../manager/manager.h"

class Unsqueeze : public op {
public:
    UNSQUEEZE_CONFIG_S unsqueeze_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    Unsqueeze() {
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *unsqueeze_cfg_ptr) {
        // new Unsqueeze op
        std::shared_ptr<Unsqueeze> relu_ptr = std::make_shared<Unsqueeze>();

        // fill op config
        memcpy(&(relu_ptr->unsqueeze_cfg), unsqueeze_cfg_ptr, sizeof(UNSQUEEZE_CONFIG_S));

        op_ptr = relu_ptr;

        return 0;
    }

    virtual int shape_infer(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S *in = &operand_stu_map[in_operands[0]];
        OPERAND_S *out = &operand_stu_map[out_operands[0]];
        out->not_need_buf = TRUE;
        // the out shape equal in shape
        for (int i = 0; i < SHAPE_LEN; ++i) {
            out->shapes[i] = 1;
        }
        out->dim_num_of_shapes = in->dim_num_of_shapes + unsqueeze_cfg.axes_num;

        int32_t real_insert_dims[SHAPE_LEN];
        for (int i = 0; i < SHAPE_LEN; ++i) {
            real_insert_dims[i] = 1;
        }

        for (int i = 0; i < unsqueeze_cfg.axes_num; ++i) {
            if (unsqueeze_cfg.axes[i] >= 0) {
                real_insert_dims[i] = unsqueeze_cfg.axes[i];
            } else {
                real_insert_dims[i] = out->dim_num_of_shapes + unsqueeze_cfg.axes[i];
            }
        }

        int32_t have_using_in_dims = 0;
        for (int i = 0; i < real_insert_dims[0]; ++i) {
            out->shapes[i] = in->shapes[i];
            have_using_in_dims++;
        }

        for (int i = real_insert_dims[0] + unsqueeze_cfg.axes_num; i < out->dim_num_of_shapes; ++i) {
            out->shapes[i] = in->shapes[have_using_in_dims++];
        }

        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&unsqueeze_cfg);
        params_vec[0] = params;

        return 0;
    };

    int fill_operands(char *one_buf_ptr) override {
        // fill op type and op name
        op_type = (char *) (&(this->unsqueeze_cfg));
        op_name = (char *) ((int64_t) &(this->unsqueeze_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S *base_cfg = (BASE_CONFIG_S *) (&(this->unsqueeze_cfg));
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
