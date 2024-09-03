#ifndef OP_POW_H
#define OP_POW_H

#include "op.h"
// #include "../../device/x86/pow6/pow6.h"
#include "../manager/manager.h"
// namespace one_new {

class Pow : public op
{
public:
    POW_CONFIG_S pow_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    Pow()
    {
//        printf("new a Pow\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *pow_cfg_ptr)
    {
        // new Pow op
        std::shared_ptr<Pow> pow_ptr = std::make_shared<Pow>();
//        pow_ptr.get()->find_handle((BUFFER_GROUP_S *)pow_cfg_ptr);

        // fill op config
        memcpy(&(pow_ptr->pow_cfg), pow_cfg_ptr, sizeof(POW_CONFIG_S));

        // // fill op type and op name
        // op_type = pow_cfg_ptr;
        // op_name = pow_cfg_ptr + OP_TYPE_LEN;

        op_ptr = pow_ptr;

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
        params.addr = (int64_t) (&pow_cfg);
        params_vec[0] = params;
//
//        BUFFER_INFO_S params;
//        params.addr = (int64_t)(&pow_cfg);
//        params_vec.push_back(params);

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->pow_cfg));
        op_name = (char*)((int64_t)&(this->pow_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->pow_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->pow_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->pow_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }

    virtual double get_computation() override {
        int32_t out_elem_size = 1;
        OPERAND_S* ifmap = (OPERAND_S*)params_vec[1].addr;
        OPERAND_S* ofmap = (OPERAND_S*)params_vec[1 + this->in_operands.size()].addr;
        for (int i = 0; i < SHAPE_LEN; ++i) {
            out_elem_size *= ofmap->shapes[i];
        }

        return (double)(2 * out_elem_size * 1e-6);
    };
};

OP_REGISTER_GLOBAL(Pow, Pow::create_instance, sizeof(POW_CONFIG_S));

#endif // OP_POW_H
