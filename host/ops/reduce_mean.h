#ifndef OP_REDUCE_MEAN_H
#define OP_REDUCE_MEAN_H

#include "op.h"
// #include "../../device/x86/reduce_mean6/reduce_mean6.h"
#include "../manager/manager.h"
// namespace one_new {

class ReduceMean : public op
{
public:
    REDUCE_MEAN_CONFIG_S reduce_mean_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    ReduceMean()
    {
//        printf("new a ReduceMean\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *reduce_mean_cfg_ptr)
    {
        // new ReduceMean op
        std::shared_ptr<ReduceMean> reduce_mean_ptr = std::make_shared<ReduceMean>();
//        reduce_mean_ptr.get()->find_handle((BUFFER_GROUP_S *)reduce_mean_cfg_ptr);

        // fill op config
        memcpy(&(reduce_mean_ptr->reduce_mean_cfg), reduce_mean_cfg_ptr, sizeof(REDUCE_MEAN_CONFIG_S));

        // // fill op type and op name
        // op_type = reduce_mean_cfg_ptr;
        // op_name = reduce_mean_cfg_ptr + OP_TYPE_LEN;

        op_ptr = reduce_mean_ptr;

        return 0;
    }


    virtual int shape_infer(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];
        OPERAND_S* out = &operand_stu_map[out_operands[0]];
        // the out shape equal in shape

        if (reduce_mean_cfg.keepdims == 1) {
            memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));

            // reduce mean 后的维度保留
            out->dim_num_of_shapes = in->dim_num_of_shapes;
            for (int axes_num_i = 0; axes_num_i < reduce_mean_cfg.axes_num; ++axes_num_i) {
                if (reduce_mean_cfg.axes[axes_num_i] == -1) {
                    out->shapes[out->dim_num_of_shapes - 1] = 1;
                } else {
                    out->shapes[reduce_mean_cfg.axes[axes_num_i]] = 1;
                }
            }
        } else {
            for (int i = 0; i < SHAPE_LEN; ++i) {
                out->shapes[i] = 1;
            }
            // reduce mean 后的维度保留，不保留
            out->dim_num_of_shapes = in->dim_num_of_shapes - reduce_mean_cfg.axes_num;
            int32_t out_shape_i = 0;
            for (int i = 0; i < in->dim_num_of_shapes; ++i) {
                // 下面这个循环，是看 input 的这个维度是否在 reduce axes 中
                int flag = 1;
                for (int reduce_axes_i = 0; reduce_axes_i < reduce_mean_cfg.axes_num; ++reduce_axes_i) {
                    int32_t real_dims;
                    if (reduce_mean_cfg.axes[reduce_axes_i] < 0) {
                        real_dims = in->dim_num_of_shapes + reduce_mean_cfg.axes[reduce_axes_i];
                    } else {
                        real_dims = reduce_mean_cfg.axes[reduce_axes_i];
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

//        std::cout << "out->operand_name is; " << out->operand_name << std::endl;


        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&reduce_mean_cfg);
        params_vec[0] = params;
//
//        BUFFER_INFO_S params;
//        params.addr = (int64_t)(&reduce_mean_cfg);
//        params_vec.push_back(params);

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->reduce_mean_cfg));
        op_name = (char*)((int64_t)&(this->reduce_mean_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->reduce_mean_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->reduce_mean_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->reduce_mean_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(ReduceMean, ReduceMean::create_instance, sizeof(REDUCE_MEAN_CONFIG_S));

#endif // OP_REDUCE_MEAN_H
