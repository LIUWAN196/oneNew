#ifndef OP_CONV_TRANSPOSE_H
#define OP_CONV_TRANSPOSE_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class ConvTranspose : public op {
public:
    CONV_TRANSPOSE_CONFIG_S conv_transpose_cfg;
    std::vector<std::vector<float>> initial_datas;  // weight and bias
    std::vector<OPERAND_S> initial_operands;  // weight and bias
//    std::vector<float> weight;
//    std::vector<float> bias;
//    OPERAND_S weight_operand_desc;
//    OPERAND_S bias_operand_desc;

    ConvTranspose() {
//        printf("new a ConvTranspose\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *conv_transpose_cfg_ptr) {
        // new ConvTranspose op
        std::shared_ptr<ConvTranspose> conv_transpose_ptr = std::make_shared<ConvTranspose>();
//        conv_transpose_ptr.get()->find_handle((BUFFER_GROUP_S *)conv_transpose_cfg_ptr);

        // fill op config
        memcpy(&(conv_transpose_ptr->conv_transpose_cfg), conv_transpose_cfg_ptr, sizeof(CONV_TRANSPOSE_CONFIG_S));

        op_ptr = conv_transpose_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S *in = &operand_stu_map[in_operands[0]];

        OPERAND_S *out = &operand_stu_map[out_operands[0]];
        // the out shape equal in shape
        memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));

//        if (out->dim_num_of_shapes != 4 || in->dim_num_of_shapes != 4) {
//            printf("the conv_transpose imap an omap shape must be dim 4\n");
//        }

        out->shapes[0] = 1;
        out->shapes[1] = initial_operands[0].shapes[0];

        out->shapes[2] = in->shapes[2] * 2; // todo: using pad and kernel size and stride to calc
        out->shapes[3] = in->shapes[3] * 2; // todo: using pad and kernel size and stride to calc
//        out->shapes[2] =
//                (in->shapes[2] + conv_transpose_cfg.pads[0] + conv_transpose_cfg.pads[2] - conv_transpose_cfg.kernel_shape[0]) / conv_transpose_cfg.strides[0] +
//                1;
//        out->shapes[3] =
//                (in->shapes[3] + conv_transpose_cfg.pads[1] + conv_transpose_cfg.pads[3] - conv_transpose_cfg.kernel_shape[1]) / conv_transpose_cfg.strides[1] +
//                1;
        out->dim_num_of_shapes = in->dim_num_of_shapes;

        params_vec.resize(1 + in_operands.size() + out_operands.size());
inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&conv_transpose_cfg);
        params_vec[0] = params;

//        BUFFER_INFO_S params;
//        params.addr = (int64_t) (&conv_transpose_cfg);
//        params_vec.push_back(params);

        return 0;
    };

    int fill_operands(char *one_buf_ptr) override {
        // fill op type and op name
        op_type = (char *) (&(this->conv_transpose_cfg));
        op_name = (char *) ((int64_t) &(this->conv_transpose_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->conv_transpose_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->conv_transpose_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->conv_transpose_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        // set the weight and bias
        int32_t *head_ptr = (int32_t *) one_buf_ptr;
        int32_t init_cnt = head_ptr[3];
        char *cur_init_info_ptr = (char *) (one_buf_ptr + head_ptr[4]);

        // 用于存放 weight bias 的描述和数据
        initial_operands.resize(2);
        initial_datas.resize(2);

        std::string weigth_oprand = this->in_operands[1];
        std::string bias_oprand = this->in_operands[2];

        for (int32_t init_i = 0; init_i < init_cnt; init_i++) {
            OPERAND_S *operand_ptr = (OPERAND_S *) cur_init_info_ptr;
            std::string init_operands = std::string(operand_ptr->operand_name);

            if (init_operands == weigth_oprand) {
                int32_t init_operand_elem_size = operand_elem_size(operand_ptr);
                float *data_ptr = (float *) (cur_init_info_ptr + sizeof(OPERAND_S));
//                std::cout << "the init operand is weight of " << this->op_type << "op." << std::endl;
                memcpy(&initial_operands[0], operand_ptr, sizeof(OPERAND_S));
                initial_datas[0].assign(data_ptr, data_ptr + init_operand_elem_size);
            } else if (init_operands == bias_oprand) {
                int32_t init_operand_elem_size = operand_elem_size(operand_ptr);
                float *data_ptr = (float *) (cur_init_info_ptr + sizeof(OPERAND_S));
//                std::cout << "the init operand is bias of " << this->op_type << " op." << std::endl;
                memcpy(&initial_operands[1], operand_ptr, sizeof(OPERAND_S));
                initial_datas[1].assign(data_ptr, data_ptr + init_operand_elem_size);
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
        // todo What is passed in here is not a real structure, and conv_transpose does not need to pass in a structure
//        params_vec: cfg / ifmap desc / weight desc / bias desc / ofmap desc
        BUFFER_INFO_S weight_desc;
        weight_desc.addr = (int64_t) (&initial_operands[0]);
        params_vec[2] = weight_desc;
//        this->params_vec.push_back(weight_desc);
        BUFFER_INFO_S bias_desc;
        bias_desc.addr = (int64_t) (&initial_operands[1]);
        params_vec[3] = bias_desc;
//        this->params_vec.push_back(bias_desc);

        // set buf
        BUFFER_INFO_S weight_buf;
        weight_buf.addr = (int64_t) (&(initial_datas[0][0]));
        inputs_vec[1] = weight_buf;
//        this->inputs_vec.push_back(weight_buf);
        BUFFER_INFO_S bias_buf;
        bias_buf.addr = (int64_t) (&(initial_datas[1][0]));
        inputs_vec[2] = bias_buf;
//        this->inputs_vec.push_back(bias_buf);


        int c = 101;
        return 0;
    }
};

OP_REGISTER_GLOBAL(ConvTranspose, ConvTranspose::create_instance, sizeof(CONV_TRANSPOSE_CONFIG_S));

#endif // OP_CONV_TRANSPOSE_H
