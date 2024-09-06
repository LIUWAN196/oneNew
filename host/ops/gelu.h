#ifndef OP_GELU_H
#define OP_GELU_H

#include "op.h"
#include "math.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Gelu : public op
{
public:
    GELU_CONFIG_S relu_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;
    std::vector<float> lut;
    Gelu()
    {

    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *relu_cfg_ptr)
    {
        // new Gelu op
        std::shared_ptr<Gelu> relu_ptr = std::make_shared<Gelu>();

        // fill op config
        memcpy(&(relu_ptr->relu_cfg), relu_cfg_ptr, sizeof(GELU_CONFIG_S));

        op_ptr = relu_ptr;

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
        params.addr = (int64_t) (&relu_cfg);
        params_vec[0] = params;

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->relu_cfg));
        op_name = (char*)((int64_t)&(this->relu_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->relu_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->relu_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->relu_cfg.op_base_cfg.out_operand_name[out_i]);
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

        return (double)(5 * out_elem_size * 1e-6);
    };

    int other_prepare() override {
        // 为 gelu 函数建立 lut 表
        float single_limit = 8.0f;
        float step = 1.0 / 512;
        int32_t lut_len = (int32_t)(2.0f * single_limit / step);
        lut.resize(lut_len);
        float inv_sqrt2 = 1.0f / 1.4142f;
        for (int lut_i = 0; lut_i < lut_len; ++lut_i) {
            float x_val = lut_i * step - single_limit;
            float x_silu_val = x_val * 0.5 * (1 + erf(x_val * inv_sqrt2));
            lut[lut_i] = x_silu_val;
        }

        BUFFER_INFO_S lut_buf;
        lut_buf.addr = (int64_t) (&lut[0]);
        inputs_vec[BUF_MAXNUM - 1] = lut_buf;
    }
};

OP_REGISTER_GLOBAL(Gelu, Gelu::create_instance, sizeof(GELU_CONFIG_S));

#endif // OP_GELU_H
