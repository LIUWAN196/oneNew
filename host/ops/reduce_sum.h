#ifndef OP_REDUCE_SUM_H
#define OP_REDUCE_SUM_H

#include "op.h"

#include "../manager/manager.h"

class ReduceSum : public op
{
public:
    REDUCE_SUM_CONFIG_S reduce_sum_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    ReduceSum()
    {
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *reduce_sum_cfg_ptr)
    {
        // new ReduceSum op
        std::shared_ptr<ReduceSum> reduce_sum_ptr = std::make_shared<ReduceSum>();

        // fill op config
        memcpy(&(reduce_sum_ptr->reduce_sum_cfg), reduce_sum_cfg_ptr, sizeof(REDUCE_SUM_CONFIG_S));

        op_ptr = reduce_sum_ptr;

        return 0;
    }

    virtual int shape_infer(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];
        OPERAND_S* out = &operand_stu_map[out_operands[0]];
        // the out shape equal in shape

        if (reduce_sum_cfg.keepdims == 1) {
            // reduce sum 后的维度保留
            out->dim_num_of_shapes = in->dim_num_of_shapes;

            // 目前只支持输入只有一个维度的 shape 不为 1
            int32_t shape_bigger_1 = 0;
            for (int dim_i = 0; dim_i < in->dim_num_of_shapes; ++dim_i){
                if (in->shapes[dim_i] > 1) {
                    shape_bigger_1 ++;
                }
            }
            if (shape_bigger_1 != 1) {
                LOG_ERR("抱歉，ReduceSum 算子，目前只支持 keepdims == 1 时，输入只有一个维度的 shape 不为 1");
            }

            for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
                out->shapes[dim_i] = 1;
            }
        } else
            if (reduce_sum_cfg.keepdims == 0) {
            memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));

            out->shapes[in->dim_num_of_shapes - 1] = 1;
            out->dim_num_of_shapes = in->dim_num_of_shapes - 1;
        } else {
            for (int i = 0; i < SHAPE_LEN; ++i) {
                out->shapes[i] = 1;
            }
            // reduce sum 后的维度保留，不保留
            out->dim_num_of_shapes = in->dim_num_of_shapes - reduce_sum_cfg.axes_num;
            int32_t out_shape_i = 0;
            for (int i = 0; i < in->dim_num_of_shapes; ++i) {
                // 下面这个循环，是看 input 的这个维度是否在 reduce axes 中
                int flag = 1;
                for (int reduce_axes_i = 0; reduce_axes_i < reduce_sum_cfg.axes_num; ++reduce_axes_i) {
                    int32_t real_dims;
                    if (reduce_sum_cfg.axes[reduce_axes_i] < 0) {
                        real_dims = in->dim_num_of_shapes + reduce_sum_cfg.axes[reduce_axes_i];
                    } else {
                        real_dims = reduce_sum_cfg.axes[reduce_axes_i];
                    }
                    if (i == real_dims) {
                        flag = 0;
                        break;
                    }
                }
                if (flag == 1) {
                    out->shapes[out_shape_i++] = in->shapes[i];
                }
            }
        }

        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&reduce_sum_cfg);
        params_vec[0] = params;

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->reduce_sum_cfg));
        op_name = (char*)((int64_t)&(this->reduce_sum_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->reduce_sum_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->reduce_sum_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->reduce_sum_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(ReduceSum, ReduceSum::create_instance, sizeof(REDUCE_SUM_CONFIG_S));

#endif // OP_REDUCE_SUM_H
