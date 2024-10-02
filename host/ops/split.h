#ifndef OP_SPLIT_H
#define OP_SPLIT_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Split : public op
{
public:
    SPLIT_CONFIG_S split_cfg;

    Split()
    {
//        printf("new a Split\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *cfg_ptr)
    {
        // new Split op
        std::shared_ptr<Split> split_ptr = std::make_shared<Split>();

        // fill op config
        memcpy(&(split_ptr->split_cfg), cfg_ptr, sizeof(SPLIT_CONFIG_S));

        // // fill op type and op name
        // op_type = relu_cfg_ptr;
        // op_name = relu_cfg_ptr + OP_TYPE_LEN;

        op_ptr = split_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];

        int32_t out_operand_num = split_cfg.op_base_cfg.out_operand_num;

        int32_t split_axis = split_cfg.axis;

        if (split_axis == -1) {
            for (int out_operand_i = 0; out_operand_i < out_operand_num; ++out_operand_i) {
                OPERAND_S* out = &operand_stu_map[out_operands[out_operand_i]];
                memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
                out->dim_num_of_shapes = in->dim_num_of_shapes;
                out->shapes[out->dim_num_of_shapes - 1] = split_cfg.split[out_operand_i];
            }
        } else {
            for (int out_operand_i = 0; out_operand_i < out_operand_num; ++out_operand_i) {
                OPERAND_S* out = &operand_stu_map[out_operands[out_operand_i]];
                out->dim_num_of_shapes = in->dim_num_of_shapes;
                memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
                out->shapes[split_axis] = split_cfg.split[out_operand_i];
            }
        }



inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&split_cfg);
        params_vec[0] = params;
//        params_vec.push_back(params);

//        BUFFER_INFO_S params;
//        params.addr = (int64_t)(&split_cfg);
//        params_vec.push_back(params);

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->split_cfg));
        op_name = (char*)((int64_t)&(this->split_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->split_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->split_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->split_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(Split, Split::create_instance, sizeof(SPLIT_CONFIG_S));

#endif // OP_SPLIT_H
