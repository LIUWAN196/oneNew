#ifndef OP_TRANSPOSE_H
#define OP_TRANSPOSE_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Transpose : public op
{
public:
    TRANSPOSE_CONFIG_S transpose_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    Transpose()
    {
//        printf("new a Transpose\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *transpose_cfg_ptr)
    {
        // new Transpose op
        std::shared_ptr<Transpose> transpose_ptr = std::make_shared<Transpose>();
//        transpose_ptr.get()->find_handle((BUFFER_GROUP_S *)transpose_cfg_ptr);

        // fill op config
        memcpy(&(transpose_ptr->transpose_cfg), transpose_cfg_ptr, sizeof(TRANSPOSE_CONFIG_S));

        // // fill op type and op name
        // op_type = transpose_cfg_ptr;
        // op_name = transpose_cfg_ptr + OP_TYPE_LEN;

        op_ptr = transpose_ptr;

        return 0;
    }


    virtual int shape_infer(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];
        OPERAND_S* out = &operand_stu_map[out_operands[0]];

        memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
        out->dim_num_of_shapes = in->dim_num_of_shapes;

        for (int i = 0; i < this->transpose_cfg.perm_num; ++i) {
            out->shapes[i] = in->shapes[this->transpose_cfg.perm[i]];
        }


        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&transpose_cfg);
        params_vec[0] = params;
//        params_vec.push_back(params);
//
//        BUFFER_INFO_S params;
//        params.addr = (int64_t)(&transpose_cfg);
//        params_vec.push_back(params);

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->transpose_cfg));
        op_name = (char*)((int64_t)&(this->transpose_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->transpose_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->transpose_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->transpose_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }

//    int prepare_init_operand_data() override {
//
//        int c = 101;
//        return 0;
//    }
};

OP_REGISTER_GLOBAL(Transpose, Transpose::create_instance, sizeof(TRANSPOSE_CONFIG_S));

#endif // OP_TRANSPOSE_H
