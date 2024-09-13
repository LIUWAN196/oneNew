#ifndef OP_COS_H
#define OP_COS_H

#include "op.h"
// #include "../../device/x86/cos6/cos6.h"
#include "../manager/manager.h"
// namespace one_new {

class Cos : public op
{
public:
    COS_CONFIG_S cos_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    Cos()
    {
//        printf("new a Cos\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *cos_cfg_ptr)
    {
        // new Cos op
        std::shared_ptr<Cos> cos_ptr = std::make_shared<Cos>();
//        cos_ptr.get()->find_handle((BUFFER_GROUP_S *)cos_cfg_ptr);

        // fill op config
        memcpy(&(cos_ptr->cos_cfg), cos_cfg_ptr, sizeof(COS_CONFIG_S));

        // // fill op type and op name
        // op_type = cos_cfg_ptr;
        // op_name = cos_cfg_ptr + OP_TYPE_LEN;

        op_ptr = cos_ptr;

        return 0;
    }


    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];
        OPERAND_S* out = &operand_stu_map[out_operands[0]];
        // the out shape equal in shape
        memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
        out->dim_num_of_shapes = in->dim_num_of_shapes;

        params_vec.resize(1 + in_operands.size() + out_operands.size());
        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&cos_cfg);
        params_vec[0] = params;
//
//        BUFFER_INFO_S params;
//        params.addr = (int64_t)(&cos_cfg);
//        params_vec.push_back(params);

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->cos_cfg));
        op_name = (char*)((int64_t)&(this->cos_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->cos_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->cos_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->cos_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }

    virtual double get_computation() override {
        int32_t out_elem_size = 1;
        OPERAND_S* ofmap = (OPERAND_S*)params_vec[1 + this->in_operands.size()].addr;
        for (int i = 0; i < SHAPE_LEN; ++i) {
            out_elem_size *= ofmap->shapes[i];
        }

        return (double)(out_elem_size * 1e-6);
    };
};

OP_REGISTER_GLOBAL(Cos, Cos::create_instance, sizeof(COS_CONFIG_S));

#endif // OP_COS_H
