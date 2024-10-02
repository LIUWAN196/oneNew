#ifndef OP_CONV_H
#define OP_CONV_H

#include "op.h"
#include "math.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Conv : public op {
public:
    CONV_CONFIG_S conv_cfg;
    std::vector<std::vector<float>> initial_datas;  // weight and bias
    std::vector<OPERAND_S> initial_operands;  // weight and bias
    std::vector<float> lut;
//    std::vector<float> weight;
//    std::vector<float> bias;
//    OPERAND_S weight_operand_desc;
//    OPERAND_S bias_operand_desc;

    Conv() {
//        printf("new a Conv\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *conv_cfg_ptr) {
        // new Conv op
        std::shared_ptr<Conv> conv_ptr = std::make_shared<Conv>();
//        conv_ptr.get()->find_handle((BUFFER_GROUP_S *)conv_cfg_ptr);

        // fill op config
        memcpy(&(conv_ptr->conv_cfg), conv_cfg_ptr, sizeof(CONV_CONFIG_S));

        op_ptr = conv_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S *in = &operand_stu_map[in_operands[0]];

        OPERAND_S *out = &operand_stu_map[out_operands[0]];
        // the out shape equal in shape
        memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));

        if (in->dim_num_of_shapes != 4) {
            printf("the conv imap an omap shape must be dim 4\n");
        }
        out->dim_num_of_shapes = in->dim_num_of_shapes;

        out->shapes[0] = 1;
        out->shapes[1] = initial_operands[0].shapes[0];
        out->shapes[2] =
                (in->shapes[2] + conv_cfg.pads[0] + conv_cfg.pads[2] - conv_cfg.kernel_shape[0]) / conv_cfg.strides[0] +
                1;
        out->shapes[3] =
                (in->shapes[3] + conv_cfg.pads[1] + conv_cfg.pads[3] - conv_cfg.kernel_shape[1]) / conv_cfg.strides[1] +
                1;


        inputs_vec.resize(BUF_MAXNUM);
//        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&conv_cfg);
        params_vec[0] = params;
//        params_vec.push_back(params);

        return 0;
    };

    int fill_operands(char *one_buf_ptr) override {
        ONE_MODEL_DESC_S *one_model_desc_ptr = (ONE_MODEL_DESC_S *) one_buf_ptr;
        int32_t init_cnt = one_model_desc_ptr->init_cnt;
        char *cur_init_info_ptr = (char *)(one_buf_ptr) + one_model_desc_ptr->init_info_offset;

        USEFUL_INFO_S* useful_ptr =  &one_model_desc_ptr->useful_info;
        BUFFER_INFO_S useful_info;
        useful_info.addr = (int64_t) useful_ptr;
        params_vec[BUF_MAXNUM - 1] = useful_info;

        // fill op type and op name
        op_type = (char *) (&(this->conv_cfg));
        op_name = (char *) ((int64_t) &(this->conv_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->conv_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->conv_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->conv_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        // set the weight and bias

        // 用于存放 weight bias 的描述和数据
        if (conv_cfg.has_bias == TRUE) {
            initial_operands.resize(2);
            initial_datas.resize(2);
        } else {
            initial_operands.resize(1);
            initial_datas.resize(1);
        }

        std::string weigth_oprand = this->in_operands[1];
        std::string bias_oprand;
        if (conv_cfg.has_bias == TRUE) {
            bias_oprand = this->in_operands[2];
        }

        for (int32_t init_i = 0; init_i < init_cnt; init_i++) {
            OPERAND_S *operand_ptr = (OPERAND_S *) cur_init_info_ptr;
            std::string init_operands = std::string(operand_ptr->operand_name);

            if (init_operands == weigth_oprand) {
                int32_t init_operand_elem_size = operand_elem_size(operand_ptr);
                float *data_ptr = (float *) (cur_init_info_ptr + sizeof(OPERAND_S));
//                std::cout << "the init operand is weight of " << this->op_type << "op." << std::endl;
                memcpy(&initial_operands[0], operand_ptr, sizeof(OPERAND_S));
                initial_datas[0].assign(data_ptr, data_ptr + init_operand_elem_size);
            }

            if (conv_cfg.has_bias == TRUE) {
                if (init_operands == bias_oprand){
                    int32_t init_operand_elem_size = operand_elem_size(operand_ptr);
                    float *data_ptr = (float *) (cur_init_info_ptr + sizeof(OPERAND_S));
//                std::cout << "the init operand is bias of " << this->op_type << " op." << std::endl;
                    memcpy(&initial_operands[1], operand_ptr, sizeof(OPERAND_S));
                    initial_datas[1].assign(data_ptr, data_ptr + init_operand_elem_size);
                }
            }

            // update cur_init_info_ptr
            int init_size = operand_buf_size(operand_ptr);
            cur_init_info_ptr += align_buf_size(sizeof(OPERAND_S) + init_size);
        }

        int b = 101;

        return 0;
    }

    int prepare_init_operand_data() override {
        // set desc struct
        // todo What is passed in here is not a real structure, and conv does not need to pass in a structure

//        params_vec: cfg / ifmap desc / weight desc / bias desc / ofmap desc
        BUFFER_INFO_S weight_desc;
        weight_desc.addr = (int64_t) (&initial_operands[0]);
        params_vec[2] = weight_desc;
//        this->params_vec.push_back(weight_desc);

        if (conv_cfg.has_bias == TRUE) {
            BUFFER_INFO_S bias_desc;
            bias_desc.addr = (int64_t) (&initial_operands[1]);
            params_vec[3] = bias_desc;
//        this->params_vec.push_back(bias_desc);
        }


        // set buf
        BUFFER_INFO_S weight_buf;
        weight_buf.addr = (int64_t) (&(initial_datas[0][0]));
        inputs_vec[1] = weight_buf;
//        this->inputs_vec.push_back(weight_buf);
        if (conv_cfg.has_bias == TRUE) {
            BUFFER_INFO_S bias_buf;
            bias_buf.addr = (int64_t) (&(initial_datas[1][0]));
            inputs_vec[2] = bias_buf;
//        this->inputs_vec.push_back(bias_buf);
        }

        int c = 101;
        return 0;
    }

    virtual double get_computation() override {
        int32_t out_elem_size = 1;
        OPERAND_S* ifmap = (OPERAND_S*)params_vec[1].addr;
        OPERAND_S* ofmap = (OPERAND_S*)params_vec[1 + this->in_operands.size()].addr;
        for (int i = 0; i < SHAPE_LEN; ++i) {
            out_elem_size *= ofmap->shapes[i];
        }

        int32_t each_ofmap_elem_computation = 1;
        int32_t in_c = ifmap->shapes[1];
        int32_t mac = 2; // mul and add, so is 2 computation
        each_ofmap_elem_computation = mac * in_c * conv_cfg.kernel_shape[0] * conv_cfg.kernel_shape[1];
        if (conv_cfg.group != 1) {
            each_ofmap_elem_computation = mac * conv_cfg.kernel_shape[0] * conv_cfg.kernel_shape[1];
        }

        return (double)(each_ofmap_elem_computation * 1e-6 * out_elem_size);
    };

    int other_prepare() override {
        if (conv_cfg.act_type == SILU) {
            float single_limit = 8.0f;
            float step = 1.0 / 512;
            int32_t lut_len = (int32_t)(2.0f * single_limit / step);
            lut.resize(lut_len);
            for (int lut_i = 0; lut_i < lut_len; ++lut_i) {
                float x_val = lut_i * step - single_limit;
                float x_silu_val = x_val / (1 + expf(-x_val));
                lut[lut_i] = x_silu_val;
            }

            BUFFER_INFO_S lut_buf;
            lut_buf.addr = (int64_t) (&lut[0]);
            inputs_vec[BUF_MAXNUM - 1] = lut_buf;
        }
    }
};

OP_REGISTER_GLOBAL(Conv, Conv::create_instance, sizeof(CONV_CONFIG_S));

#endif // OP_CONV_H
