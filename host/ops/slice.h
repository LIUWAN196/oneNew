#ifndef OP_SLICE_H
#define OP_SLICE_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Slice : public op
{
public:
    SLICE_CONFIG_S slice_cfg;

    Slice()
    {
//        printf("new a Slice\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *cfg_ptr)
    {
        // new Slice op
        std::shared_ptr<Slice> slice_ptr = std::make_shared<Slice>();

        // fill op config
        memcpy(&(slice_ptr->slice_cfg), cfg_ptr, sizeof(SLICE_CONFIG_S));

        // // fill op type and op name
        // op_type = relu_cfg_ptr;
        // op_name = relu_cfg_ptr + OP_TYPE_LEN;

        op_ptr = slice_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];

        OPERAND_S* out = &operand_stu_map[out_operands[0]];

        memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
        out->dim_num_of_shapes = in->dim_num_of_shapes;

        for (int axis_idx = 0; axis_idx < slice_cfg.slice_axes_num; ++axis_idx) {
            int32_t axis = slice_cfg.axes[axis_idx];
            int32_t starts = slice_cfg.starts[axis_idx];
            if (starts < 0){
                starts = in->shapes[axis] + starts;
            }
            int64_t ends = slice_cfg.ends[axis_idx];
            if (ends == 9223372036854775807) {
                // slice 算子用 9223372036854775807 来表示取上限
                ends = in->shapes[axis];
            } else if (ends < 0){
                ends = in->shapes[axis] + ends;
            }
            int32_t steps = slice_cfg.steps[axis_idx];
            out->shapes[axis] = (ends - starts) / steps;
        }

        params_vec.resize(1 + in_operands.size() + out_operands.size());
        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&slice_cfg);
        params_vec[0] = params;

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->slice_cfg));
        op_name = (char*)((int64_t)&(this->slice_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->slice_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->slice_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->slice_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(Slice, Slice::create_instance, sizeof(SLICE_CONFIG_S));

#endif // OP_SLICE_H
