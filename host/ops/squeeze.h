#ifndef OP_SQUEEZE_H
#define OP_SQUEEZE_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Squeeze : public op
{
public:
    SQUEEZE_CONFIG_S squeeze_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    Squeeze()
    {
//        printf("new a Squeeze\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *squeeze_cfg_ptr)
    {
        // new Squeeze op
        std::shared_ptr<Squeeze> relu_ptr = std::make_shared<Squeeze>();
//        relu_ptr.get()->find_handle((BUFFER_GROUP_S *)squeeze_cfg_ptr);

        // fill op config
        memcpy(&(relu_ptr->squeeze_cfg), squeeze_cfg_ptr, sizeof(SQUEEZE_CONFIG_S));

        // // fill op type and op name
        // op_type = squeeze_cfg_ptr;
        // op_name = squeeze_cfg_ptr + OP_TYPE_LEN;

        op_ptr = relu_ptr;

        return 0;
    }


    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];
        OPERAND_S* out = &operand_stu_map[out_operands[0]];

        // the out shape equal in shape
//        memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
        for (int i = 0; i < SHAPE_LEN; ++i) {
            out->shapes[i] = 1;
        }
        out->dim_num_of_shapes = in->dim_num_of_shapes - squeeze_cfg.axes_num;

        int32_t out_shape_i = 0;
        for (int i = 0; i < in->dim_num_of_shapes; ++i) {
            // 下面这个循环，是看 input 的这个维度是否在 squeeze axes 中
            int flag = 1;
            for (int reduce_axes_i = 0; reduce_axes_i < squeeze_cfg.axes_num; ++reduce_axes_i) {
                if (i == squeeze_cfg.axes[reduce_axes_i]) {
                    flag = 0;
                    break;
                }
            }
            if (flag == 1) {
                out->shapes[out_shape_i++] = in->shapes[i];
            }
        }

        params_vec.resize(1 + in_operands.size() + out_operands.size());
        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&squeeze_cfg);
        params_vec[0] = params;
//        params_vec.push_back(params);

//        BUFFER_INFO_S params;
//        params.addr = (int64_t)(&squeeze_cfg);
//        params_vec.push_back(params);

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->squeeze_cfg));
        op_name = (char*)((int64_t)&(this->squeeze_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->squeeze_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->squeeze_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->squeeze_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(Squeeze, Squeeze::create_instance, sizeof(SQUEEZE_CONFIG_S));

#endif // OP_SQUEEZE_H
