#ifndef OP_GEMM_H
#define OP_GEMM_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Gemm : public op {
public:
    GEMM_CONFIG_S gemm_cfg;
    std::vector<std::vector<float>> initial_datas;  // weight and bias
    std::vector<OPERAND_S> initial_operands;  // weight and bias
//    std::vector<float> weight;
//    std::vector<float> bias;
//    OPERAND_S weight_operand_desc;
//    OPERAND_S bias_operand_desc;

    Gemm() {
//        printf("new a Gemm\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *gemm_cfg_ptr) {
        // new Gemm op
        std::shared_ptr<Gemm> gemm_ptr = std::make_shared<Gemm>();

        // fill op config
        memcpy(&(gemm_ptr->gemm_cfg), gemm_cfg_ptr, sizeof(GEMM_CONFIG_S));


        op_ptr = gemm_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S *in = &operand_stu_map[in_operands[0]];

        OPERAND_S *out = &operand_stu_map[out_operands[0]];
        out->dim_num_of_shapes = in->dim_num_of_shapes;

        for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
            out->shapes[dim_i] = 1;
        }
        memcpy(&out->shapes[0], &in->shapes[0], SHAPE_LEN * sizeof(int32_t));
        out->dim_num_of_shapes = in->dim_num_of_shapes;
        out->shapes[out->dim_num_of_shapes - 1] = initial_datas[1].size();


        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&gemm_cfg);
        params_vec[0] = params;
//        params_vec.push_back(params);

        return 0;
    };

    int fill_operands(char *one_buf_ptr) override {
        // fill op type and op name
        op_type = (char *) (&(this->gemm_cfg));
        op_name = (char *) ((int64_t) &(this->gemm_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S *base_cfg = (BASE_CONFIG_S *) (&(this->gemm_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->gemm_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->gemm_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        // set the weight and bias
        ONE_MODEL_DESC_S *one_model_desc_ptr = (ONE_MODEL_DESC_S *) one_buf_ptr;
        int32_t init_cnt = one_model_desc_ptr->init_cnt;
        char *cur_init_info_ptr = (char *)(one_buf_ptr) + one_model_desc_ptr->init_info_offset;

        // 用于存放 weight bias 的描述和数据
        initial_operands.resize(2);
        initial_datas.resize(2);

        std::string weight_oprand = this->in_operands[1];
        std::string bias_oprand = this->in_operands[2];

        for (int32_t init_i = 0; init_i < init_cnt; init_i++) {
            OPERAND_S *operand_ptr = (OPERAND_S *) cur_init_info_ptr;
            std::string init_operands = std::string(operand_ptr->operand_name);

            if (init_operands == weight_oprand) {
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
        // todo What is passed in here is not a real structure, and conv does not need to pass in a structure

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

//        // set buf
//        BUFFER_INFO_S weight_buf;
//        weight_buf.addr = (int64_t) (&(initial_datas[0][0]));
//        this->inputs_vec.push_back(weight_buf);
//        BUFFER_INFO_S bias_buf;
//        bias_buf.addr = (int64_t) (&(initial_datas[1][0]));
//        this->inputs_vec.push_back(bias_buf);


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

        int32_t in_last_dim_size = 1;
        for (int i = SHAPE_LEN - 1; i >= 0; --i) {
            if (ifmap->shapes[i] != 1) {
                in_last_dim_size = ifmap->shapes[i];
                break;
            }
        }
        int32_t mac = 2; // mul and add, so is 2 computation
        return (double)(out_elem_size * mac * in_last_dim_size * 1e-6);
    };

};

OP_REGISTER_GLOBAL(Gemm, Gemm::create_instance, sizeof(GEMM_CONFIG_S));

#endif // OP_GEMM_H
