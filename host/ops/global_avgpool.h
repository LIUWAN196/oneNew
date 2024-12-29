#ifndef OP_GLOBAL_AVGPOOL_H
#define OP_GLOBAL_AVGPOOL_H

#include "op.h"
#include "../manager/manager.h"

class GlobalAveragePool : public op
{
public:
    GLOBAL_AVGPOOL_CONFIG_S global_avgpool_cfg;

    GlobalAveragePool()
    {
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *global_avgpool_cfg_ptr)
    {
        // new GlobalAveragePool op
        std::shared_ptr<GlobalAveragePool> global_avgpool_ptr = std::make_shared<GlobalAveragePool>();

        // fill op config
        memcpy(&(global_avgpool_ptr->global_avgpool_cfg), global_avgpool_cfg_ptr, sizeof(GLOBAL_AVGPOOL_CONFIG_S));

        op_ptr = global_avgpool_ptr;

        return 0;
    }

    virtual int shape_infer(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S *in = &operand_stu_map[in_operands[0]];

        OPERAND_S *out = &operand_stu_map[out_operands[0]];
        for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
            out->shapes[dim_i] = 1;
        }
        out->shapes[1] = in->shapes[1]; // out w equal in c
        out->dim_num_of_shapes = in->dim_num_of_shapes;

        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&global_avgpool_cfg);
        params_vec[0] = params;

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->global_avgpool_cfg));
        op_name = (char*)((int64_t)&(this->global_avgpool_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->global_avgpool_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->global_avgpool_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->global_avgpool_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(GlobalAveragePool, GlobalAveragePool::create_instance, sizeof(GLOBAL_AVGPOOL_CONFIG_S));

#endif // OP_GLOBAL_AVGPOOL_H
