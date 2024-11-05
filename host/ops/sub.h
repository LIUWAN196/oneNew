#ifndef OP_SUB_H
#define OP_SUB_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Sub : public op
{
public:
    SUB_CONFIG_S sub_cfg;

    // 有可能 sub 的第二个输入数据是 init 的
    std::vector<std::vector<float>> initial_datas;  // weight and bias
    std::vector<OPERAND_S> initial_operands;  // weight and bias
    int32_t init_ifmap_idx = -1;
    Sub()
    {
//        printf("new a Sub\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *relu_cfg_ptr)
    {
        // new Sub op
        std::shared_ptr<Sub> sub_ptr = std::make_shared<Sub>();

        // fill op config
        memcpy(&(sub_ptr->sub_cfg), relu_cfg_ptr, sizeof(SUB_CONFIG_S));

        // // fill op type and op name
        // op_type = relu_cfg_ptr;
        // op_name = relu_cfg_ptr + OP_TYPE_LEN;

        op_ptr = sub_ptr;

        return 0;
    }

    virtual int shape_infer(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];
        OPERAND_S* in1 = &operand_stu_map[in_operands[1]];
        OPERAND_S* out = &operand_stu_map[out_operands[0]];

        // the out shape equal in shape
        if (init_ifmap_idx == 0) {  // 第一个输入为 init
            memcpy(&out->shapes[0], &in1->shapes[0], SHAPE_LEN * sizeof(int32_t));
            out->dim_num_of_shapes = in1->dim_num_of_shapes;
        } else {
            memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
            out->dim_num_of_shapes = in->dim_num_of_shapes;
        }

//        std::cout<< "out->dim_num_of_shapes is:" << out->dim_num_of_shapes
//        << ", in->dim_num_of_shapes is: " << in->dim_num_of_shapes << std::endl;


        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&sub_cfg);
        params_vec[0] = params;

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->sub_cfg));
        op_name = (char*)((int64_t)&(this->sub_cfg) + OP_TYPE_LEN);

        if (strcmp(op_name, "/model.10/attn/Sub") == 0) {
            int a = 101;
        }

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->sub_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->sub_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->sub_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        // 下面是看第二个输入操作数是否在 init 中
        // set the weight and bias
        ONE_MODEL_DESC_S *one_model_desc_ptr = (ONE_MODEL_DESC_S *) one_buf_ptr;
        int32_t init_cnt = one_model_desc_ptr->init_cnt;
        char *cur_init_info_ptr = (char *)(one_buf_ptr) + one_model_desc_ptr->init_info_offset;

        // 用于存放 weight bias 的描述和数据
        std::string first_oprand = this->in_operands[0];
        std::string second_oprand = this->in_operands[1];

        for (int32_t init_i = 0; init_i < init_cnt; init_i++) {
            OPERAND_S *operand_ptr = (OPERAND_S *) cur_init_info_ptr;
            std::string init_operands = std::string(operand_ptr->operand_name);

            if (init_operands == first_oprand) {
                init_ifmap_idx = 0;
                ifmap_st_idx = 1;   // 走到这个 if 分支，说明 add 的第一个输入是 init
                initial_operands.resize(1);
                initial_datas.resize(1);
                int32_t init_operand_elem_size = operand_elem_size(operand_ptr);
                float *data_ptr = (float *) (cur_init_info_ptr + sizeof(OPERAND_S));
//                std::cout << "the init operand is weight of " << this->op_type << "op." << std::endl;
                memcpy(&initial_operands[0], operand_ptr, sizeof(OPERAND_S));
                initial_datas[0].assign(data_ptr, data_ptr + init_operand_elem_size);
            }

            if (init_operands == second_oprand) {
                init_ifmap_idx = 1;
                initial_operands.resize(1);
                initial_datas.resize(1);
                int32_t init_operand_elem_size = operand_elem_size(operand_ptr);
                float *data_ptr = (float *) (cur_init_info_ptr + sizeof(OPERAND_S));
//                std::cout << "the init operand is weight of " << this->op_type << "op." << std::endl;
                memcpy(&initial_operands[0], operand_ptr, sizeof(OPERAND_S));
                initial_datas[0].assign(data_ptr, data_ptr + init_operand_elem_size);
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

        if (initial_operands.size() != 0) {
            if (init_ifmap_idx == 0) {
                BUFFER_INFO_S first_operand_desc;
                first_operand_desc.addr = (int64_t) (&initial_operands[0]);
                params_vec[1] = first_operand_desc;    //  [0] is cfg; [1] is init data; [2] is first ifmap

                // set buf
                BUFFER_INFO_S first_operand_buf;
                first_operand_buf.addr = (int64_t) (&(initial_datas[0][0]));
                inputs_vec[0] = first_operand_buf;
            } else {
                BUFFER_INFO_S second_operand_desc;
                second_operand_desc.addr = (int64_t) (&initial_operands[0]);
                params_vec[2] = second_operand_desc;    //  [0] is cfg; [1] is first ifmap; [2] is init data

                // set buf
                BUFFER_INFO_S second_operand_buf;
                second_operand_buf.addr = (int64_t) (&(initial_datas[0][0]));
                inputs_vec[1] = second_operand_buf;
            }
        }

        return 0;
    }

};

OP_REGISTER_GLOBAL(Sub, Sub::create_instance, sizeof(SUB_CONFIG_S));

#endif // OP_SUB_H
