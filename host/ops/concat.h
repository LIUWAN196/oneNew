#ifndef OP_CONCAT_H
#define OP_CONCAT_H

#include "op.h"
#include "../manager/manager.h"

class Concat : public op
{
public:
    CONCAT_CONFIG_S concat_cfg;
    // 有可能将要被 concat 的输入操作数为 init
    std::vector<std::vector<float>> initial_datas;  // weight and bias
    std::vector<OPERAND_S> initial_operands;  // weight and bias
    int32_t init_ifmap_idx = -1;
    Concat()
    {
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *relu_cfg_ptr)
    {
        // new Concat op
        std::shared_ptr<Concat> concat_ptr = std::make_shared<Concat>();

        // fill op config
        memcpy(&(concat_ptr->concat_cfg), relu_cfg_ptr, sizeof(CONCAT_CONFIG_S));

        op_ptr = concat_ptr;

        return 0;
    }

    virtual int shape_infer(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];
        OPERAND_S* in1 = &operand_stu_map[in_operands[1]];

        OPERAND_S* out = &operand_stu_map[out_operands[0]];
        // the out shape equal in shape
        if (initial_operands.size() == 0) {
            memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
            out->dim_num_of_shapes = in->dim_num_of_shapes;
        } else {
            in = &initial_operands[0];
            memcpy(&out->shapes[0], &in1->shapes[0], SHAPE_LEN * sizeof(int32_t));
            out->dim_num_of_shapes = in1->dim_num_of_shapes;
        }

        int in_n_sum = 0, in_c_sum = 0, in_h_sum = 0, in_w_sum = 0;
        int32_t dim_num_possible_overlap[SHAPE_LEN] = {0};
        for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
            if (initial_operands.size() == 0) {
                dim_num_possible_overlap[dim_i] += operand_stu_map[in_operands[0]].shapes[dim_i];
            } else {
                dim_num_possible_overlap[dim_i] += initial_operands[0].shapes[dim_i];
            }
            for (int in_i = 1; in_i < concat_cfg.op_base_cfg.in_operand_num; ++in_i) {
                dim_num_possible_overlap[dim_i] += operand_stu_map[in_operands[in_i]].shapes[dim_i];
            }
        }
        if (concat_cfg.axis <= 0) {
            concat_cfg.axis = out->dim_num_of_shapes + concat_cfg.axis;
        }
        out->shapes[concat_cfg.axis] = dim_num_possible_overlap[concat_cfg.axis];

        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&concat_cfg);
        params_vec[0] = params;

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->concat_cfg));
        op_name = (char*)((int64_t)&(this->concat_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->concat_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->concat_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->concat_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        // 下面是输入操作数是否在 init 中
        ONE_MODEL_DESC_S *one_model_desc_ptr = (ONE_MODEL_DESC_S *) one_buf_ptr;
        int32_t init_cnt = one_model_desc_ptr->init_cnt;
        char *cur_init_info_ptr = (char *)(one_buf_ptr) + one_model_desc_ptr->init_info_offset;

        std::string first_oprand = this->in_operands[0];
        std::string second_oprand = this->in_operands[1];

        for (int32_t init_i = 0; init_i < init_cnt; init_i++) {
            OPERAND_S *operand_ptr = (OPERAND_S *) cur_init_info_ptr;
            std::string init_operands = std::string(operand_ptr->operand_name);

                if (init_operands == first_oprand) {
                    init_ifmap_idx = 0;
                    ifmap_st_idx = 1;
                    initial_operands.resize(1);
                    initial_datas.resize(1);
                    int32_t init_operand_elem_size = operand_elem_size(operand_ptr);
                    float *data_ptr = (float *) (cur_init_info_ptr + sizeof(OPERAND_S));
                    memcpy(&initial_operands[0], operand_ptr, sizeof(OPERAND_S));
                    initial_datas[0].assign(data_ptr, data_ptr + init_operand_elem_size);
            }

            // update cur_init_info_ptr
            int init_size = operand_buf_size(operand_ptr);
            cur_init_info_ptr += align_buf_size(sizeof(OPERAND_S) + init_size);
        }

        return 0;
    }

    int prepare_init_operand_data() override {
        // set desc struct
        if (initial_operands.size() != 0) {
            BUFFER_INFO_S first_operand_desc;
            first_operand_desc.addr = (int64_t) (&initial_operands[0]);
            params_vec[1] = first_operand_desc;

            // set buf
            BUFFER_INFO_S first_operand_buf;
            first_operand_buf.addr = (int64_t) (&(initial_datas[0][0]));
            inputs_vec[0] = first_operand_buf;
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(Concat, Concat::create_instance, sizeof(CONCAT_CONFIG_S));

#endif // OP_CONCAT_H
