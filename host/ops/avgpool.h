#ifndef OP_AVGPOOL_H
#define OP_AVGPOOL_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class AveragePool : public op
{
public:
    AVG_POOL_CONFIG_S avg_pool_cfg;

    AveragePool()
    {
//        printf("new a AveragePool\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *avg_pool_cfg_ptr)
    {
        // new AveragePool op
        std::shared_ptr<AveragePool> avg_pool_ptr = std::make_shared<AveragePool>();
//        avg_pool_ptr.get()->find_handle((BUFFER_GROUP_S *)avg_pool_cfg_ptr);

        // fill op config
        memcpy(&(avg_pool_ptr->avg_pool_cfg), avg_pool_cfg_ptr, sizeof(AVG_POOL_CONFIG_S));

        // // fill op type and op name
        // op_type = avg_pool_cfg_ptr;
        // op_name = avg_pool_cfg_ptr + OP_TYPE_LEN;

        op_ptr = avg_pool_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];

        OPERAND_S* out = &operand_stu_map[out_operands[0]];
        // the out shape equal in shape
        memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
        out->dim_num_of_shapes = in->dim_num_of_shapes;
        out->shapes[0] = in->shapes[0];
        out->shapes[1] = in->shapes[1];
        out->shapes[2] =
                (in->shapes[2] + avg_pool_cfg.pads[0] + avg_pool_cfg.pads[2] - avg_pool_cfg.kernel_shape[0]) / avg_pool_cfg.strides[0] +
                1;
        out->shapes[3] =
                (in->shapes[3] + avg_pool_cfg.pads[1] + avg_pool_cfg.pads[3] - avg_pool_cfg.kernel_shape[1]) / avg_pool_cfg.strides[1] +
                1;


inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&avg_pool_cfg);
        params_vec[0] = params;

//        BUFFER_INFO_S params;
//        params.addr = (int64_t)(&avg_pool_cfg);
//        params_vec.push_back(params);
        return  0;

    };

    int fill_operands(char *one_buf_ptr) override
    {
        ONE_MODEL_DESC_S *one_model_desc_ptr = (ONE_MODEL_DESC_S *) one_buf_ptr;

        USEFUL_INFO_S* useful_ptr =  &one_model_desc_ptr->useful_info;
        BUFFER_INFO_S useful_info;
        useful_info.addr = (int64_t) useful_ptr;
        params_vec[BUF_MAXNUM - 1] = useful_info;

        // fill op type and op name
        op_type = (char *)(&(this->avg_pool_cfg));
        op_name = (char *)((int64_t)&(this->avg_pool_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->avg_pool_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->avg_pool_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->avg_pool_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(AveragePool, AveragePool::create_instance, sizeof(AVG_POOL_CONFIG_S));

#endif // OP_AVGPOOL_H
