#ifndef OP_TOP_K_H
#define OP_TOP_K_H

#include "op.h"
// #include "../../device/x86/top_k6/top_k6.h"
#include "../manager/manager.h"
// namespace one_new {

class TopK : public op
{
public:
    TOP_K_CONFIG_S top_k_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    TopK()
    {
//        printf("new a TopK\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *top_k_cfg_ptr)
    {
        // new TopK op
        std::shared_ptr<TopK> top_k_ptr = std::make_shared<TopK>();
//        top_k_ptr.get()->find_handle((BUFFER_GROUP_S *)top_k_cfg_ptr);

        // fill op config
        memcpy(&(top_k_ptr->top_k_cfg), top_k_cfg_ptr, sizeof(TOP_K_CONFIG_S));

        // // fill op type and op name
        // op_type = top_k_cfg_ptr;
        // op_name = top_k_cfg_ptr + OP_TYPE_LEN;

        op_ptr = top_k_ptr;

        return 0;
    }


    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];

        OPERAND_S* out0 = &operand_stu_map[out_operands[0]];
        // topk 有两个输出，第一个是 topk val，第二个是 topk indices，这里只需要第二个
        OPERAND_S* out1 = &operand_stu_map[out_operands[1]];
        // the out shape equal in shape
        for (int i = 0; i < SHAPE_LEN; ++i) {
            out0->shapes[i] = 1;
            out1->shapes[i] = 1;
        }
        out0->dim_num_of_shapes = in->dim_num_of_shapes;
        out1->dim_num_of_shapes = in->dim_num_of_shapes;

        out0->shapes[out0->dim_num_of_shapes - 1] = top_k_cfg.topk_num;
        out1->shapes[out1->dim_num_of_shapes - 1] = top_k_cfg.topk_num;


        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&top_k_cfg);
        params_vec[0] = params;


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
        op_type = (char*)(&(this->top_k_cfg));
        op_name = (char*)((int64_t)&(this->top_k_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->top_k_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->top_k_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->top_k_cfg.op_base_cfg.out_operand_name[out_i]);
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

OP_REGISTER_GLOBAL(TopK, TopK::create_instance, sizeof(TOP_K_CONFIG_S));

#endif // OP_TOP_K_H
