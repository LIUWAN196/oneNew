#ifndef OP_GATHER_H
#define OP_GATHER_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Gather : public op
{
public:
    GATHER_CONFIG_S gather_cfg;

    Gather()
    {
//        printf("new a Gather\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *cfg_ptr)
    {
        // new Gather op
        std::shared_ptr<Gather> gather_ptr = std::make_shared<Gather>();

        // fill op config
        memcpy(&(gather_ptr->gather_cfg), cfg_ptr, sizeof(GATHER_CONFIG_S));

        // // fill op type and op name
        // op_type = relu_cfg_ptr;
        // op_name = relu_cfg_ptr + OP_TYPE_LEN;

        op_ptr = gather_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];

        int32_t out_operand_num = gather_cfg.op_base_cfg.out_operand_num;

        int32_t gather_axis = gather_cfg.axis;

        OPERAND_S* out = &operand_stu_map[out_operands[0]];

        for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
            out->shapes[dim_i] = 1;
        }

        out->dim_num_of_shapes = in->dim_num_of_shapes - gather_axis;
        out->shapes[0] = 1;
        for (int dim_i = 1; dim_i < out->dim_num_of_shapes; ++dim_i) {
            out->shapes[dim_i] = in->shapes[dim_i + gather_axis];
        }

//        memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
//        out->shapes[gather_axis] = 1;

        params_vec.resize(1 + in_operands.size() + out_operands.size());
        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&gather_cfg);
        params_vec[0] = params;
//        params_vec.push_back(params);

//        BUFFER_INFO_S params;
//        params.addr = (int64_t)(&gather_cfg);
//        params_vec.push_back(params);

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->gather_cfg));
        op_name = (char*)((int64_t)&(this->gather_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->gather_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->gather_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->gather_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(Gather, Gather::create_instance, sizeof(GATHER_CONFIG_S));

#endif // OP_GATHER_H
