#ifndef OP_FLATTEN_H
#define OP_FLATTEN_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Flatten : public op
{
public:
    FLATTEN_CONFIG_S flatten_cfg;

    Flatten()
    {
//        printf("new a Flatten\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *flatten_cfg_ptr)
    {
        // new Flatten op
        std::shared_ptr<Flatten> flatten_ptr = std::make_shared<Flatten>();

        // fill op config
        memcpy(&(flatten_ptr->flatten_cfg), flatten_cfg_ptr, sizeof(FLATTEN_CONFIG_S));


        op_ptr = flatten_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S *in = &operand_stu_map[in_operands[0]];

        OPERAND_S *out = &operand_stu_map[out_operands[0]];

        // the out shape equal in shape
        for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
            out->shapes[dim_i] = 1;
        }
        out->dim_num_of_shapes = flatten_cfg.axis + 1;

//        memset(&out->shapes[0], 1, SHAPE_LEN * sizeof(int32_t));
        out->shapes[flatten_cfg.axis] = 1;     // NCHW, the 3 is W dims
        for (int dim_i = flatten_cfg.axis; dim_i < SHAPE_LEN; ++dim_i) {
            out->shapes[flatten_cfg.axis] *= in->shapes[dim_i];
        }

        if (out->shapes[0] == 1 && out->shapes[1] == 1 && out->shapes[2] == 512) {
            // todo: 这是为 clip txt 特别定制的，需要修改为通用的
            out->shapes[0] = 77;
            out->shapes[1] = 512;
            out->shapes[2] = 1;
            out->dim_num_of_shapes = 2;
        } else if (in->shapes[0] == 1 && in->shapes[1] == 8400 && in->shapes[2] == 256) {
            // todo: 这是为 rt detr 特别定制的，需要修改为通用的
            out->shapes[0] = 8400;
            out->shapes[1] = 256;
            out->shapes[2] = 1;
            out->dim_num_of_shapes = 2;
        }

//        LOG_DBG("flatten_cfg.axis is %d\n", flatten_cfg.axis);
        params_vec.resize(1 + in_operands.size() + out_operands.size());
        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&flatten_cfg);
        params_vec[0] = params;

//        BUFFER_INFO_S params;
//        params.addr = (int64_t)(&flatten_cfg);
//        params_vec.push_back(params);

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->flatten_cfg));
        op_name = (char*)((int64_t)&(this->flatten_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->flatten_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->flatten_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->flatten_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(Flatten, Flatten::create_instance, sizeof(FLATTEN_CONFIG_S));

#endif // OP_FLATTEN_H
