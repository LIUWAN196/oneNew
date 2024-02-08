#ifndef OP_CONV_H
#define OP_CONV_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Conv : public op
{
public:
    CONV_CONFIG_S conv_cfg;
    std::vector<float> weight;
    std::vector<float> bias;
    OPERAND_S weight_operand_desc;
    OPERAND_S bias_operand_desc;

    Conv()
    {
        printf("new a Conv\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *conv_cfg_ptr)
    {
        // new Conv op
        std::shared_ptr<Conv> conv_ptr = std::make_shared<Conv>();
//        conv_ptr.get()->find_handle((BUFFER_GROUP_S *)conv_cfg_ptr);

        // fill op config
        memcpy(&(conv_ptr->conv_cfg), conv_cfg_ptr, sizeof(CONV_CONFIG_S));

        // // fill op type and op name
        // op_type = conv_cfg_ptr;
        // op_name = conv_cfg_ptr + NAME_LEN;

        op_ptr = conv_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];

        OPERAND_S* out = &operand_stu_map[out_operands[0]];
        out->shape.N = in->shape.N;
        out->shape.C = in->shape.C;
        out->shape.H = (in->shape.H + conv_cfg.pads[0] + conv_cfg.pads[2] - conv_cfg.kernel_shape[0]) / conv_cfg.strides[0];
        out->shape.W = (in->shape.W + conv_cfg.pads[1] + conv_cfg.pads[3] - conv_cfg.kernel_shape[1]) / conv_cfg.strides[1];
        int b = 101;

        BUFFER_INFO_S params;
        params.addr = (int64_t)(&conv_cfg);
        params_vec.push_back(params);

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char *)(&(this->conv_cfg));
        op_name = (char *)((int64_t)&(this->conv_cfg) + NAME_LEN);

        int32_t in_operand_cnt = 3;
        for (size_t i = 0; i < in_operand_cnt; i++)
        {
            std::string in_operand(this->conv_cfg.in_operand_name[i]);
            this->in_operands.push_back(in_operand);
        }

        std::string out_operand(this->conv_cfg.out_operand_name[0]);
        this->out_operands.push_back(out_operand);

        // set the weight and bias
        int32_t *head_ptr = (int32_t *) one_buf_ptr;
        int32_t init_cnt = head_ptr[3];
        char *cur_init_info_ptr = (char *) (one_buf_ptr + head_ptr[4]);

        for (int32_t init_i = 0; init_i < init_cnt; init_i++) {
            OPERAND_S *operand_ptr = (OPERAND_S *) cur_init_info_ptr;
            std::string init_operands = std::string(operand_ptr->operand_name);

            for (std::string this_in_oprand : this->in_operands) {
                if (this_in_oprand == init_operands){
                    int32_t init_operand_elem_size = operand_elem_size(operand_ptr);
                    float *data_ptr = (float *)(cur_init_info_ptr + sizeof(OPERAND_S));
                    if (this_in_oprand.find(weight_char0) != std::string::npos || this_in_oprand.find(weight_char1) != std::string::npos){
                        std::cout << "the init operand is weight of " << this->op_type << "op." << std::endl;
                        memcpy(&weight_operand_desc, operand_ptr, sizeof(OPERAND_S));
                        weight.assign(data_ptr, data_ptr + init_operand_elem_size);

                    } else if (this_in_oprand.find(bias_char0) != std::string::npos || this_in_oprand.find(bias_char1) != std::string::npos){
                        std::cout << "the init operand is bias of " << this->op_type << "op." << std::endl;
                        memcpy(&bias_operand_desc, operand_ptr, sizeof(OPERAND_S));
                        bias.assign(data_ptr, data_ptr + init_operand_elem_size);

                    } else {
                        std::cout << "I can't recognize why it matches " << operand_ptr->operand_name << "in " << this->op_type << std::endl;
                    }
                }
            }

            // update cur_init_info_ptr
            int init_size = operand_buf_size(operand_ptr);
            cur_init_info_ptr += align_buf_size(sizeof(OPERAND_S) + init_size);
        }

        int b = 101;

        return 0;
    }

    int prepare_init_operand_data() override{
        // set desc struct
        // todo What is passed in here is not a real structure, and conv does not need to pass in a structure
        BUFFER_INFO_S weight_desc;
        weight_desc.addr = (int64_t)(&weight_operand_desc);
        this->params_vec.push_back(weight_desc);
        if (conv_cfg.has_bias) {
            BUFFER_INFO_S bias_desc;
            bias_desc.addr = (int64_t)(&bias_operand_desc);
            this->params_vec.push_back(bias_desc);
        }

        // set buf
        BUFFER_INFO_S weight_buf;
        weight_buf.addr = (int64_t)(&(weight[0]));
        this->inputs_vec.push_back(weight_buf);
        if (conv_cfg.has_bias) {
            BUFFER_INFO_S bias_buf;
            bias_buf.addr = (int64_t)(&(bias[0]));
            this->inputs_vec.push_back(bias_buf);
        }


        int c = 101;
        return 0;
    }
};

OP_REGISTER_GLOBAL(Conv, Conv::create_instance);

#endif // OP_CONV_H
