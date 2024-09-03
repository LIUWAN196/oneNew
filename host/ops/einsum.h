#ifndef OP_EINSUM_H
#define OP_EINSUM_H

#include "op.h"
#include "math.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Einsum : public op {
public:
    EINSUM_CONFIG_S einsum_cfg;
    std::vector<std::vector<float>> initial_datas;  // weight and bias
    std::vector<OPERAND_S> initial_operands;  // weight and bias

    Einsum() {
//        printf("new a Einsum\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *einsum_cfg_ptr) {
        // new Einsum op
        std::shared_ptr<Einsum> einsum_ptr = std::make_shared<Einsum>();
//        einsum_ptr.get()->find_handle((BUFFER_GROUP_S *)einsum_cfg_ptr);

        // fill op config
        memcpy(&(einsum_ptr->einsum_cfg), einsum_cfg_ptr, sizeof(EINSUM_CONFIG_S));

        op_ptr = einsum_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        // step 1: 从 einsum_cfg.equation 提取字母
        std::string equation_str = std::string(einsum_cfg.equation);
        std::string letters;
        std::remove_copy_if(equation_str.begin(), equation_str.end(), std::back_inserter(letters), [](unsigned char c){ return !std::isalpha(c); }); // 仅保留字母

        OPERAND_S *in0 = &operand_stu_map[in_operands[0]];

        OPERAND_S *in1 = &initial_operands[0];
        OPERAND_S *out = &operand_stu_map[out_operands[0]];

        int ofmap_shapes = letters.length() - (in0->dim_num_of_shapes + in1->dim_num_of_shapes);
        out->dim_num_of_shapes = ofmap_shapes;

        std::unordered_map<char, int32_t> letters_shape_map;
        for (int i = 0; i < in0->dim_num_of_shapes; ++i) {
            char cur_char = letters[i];
            letters_shape_map[cur_char] = in0->shapes[i];
        }

        for (int i = 0; i < in1->dim_num_of_shapes; ++i) {
            char cur_char = letters[in0->dim_num_of_shapes + i];
            letters_shape_map[cur_char] = in1->shapes[i];
        }

        for (int i = 0; i < SHAPE_LEN; ++i) {
            out->shapes[i] = 1;
        }
        for (int i = 0; i < ofmap_shapes; ++i) {
            char cur_char = letters[in0->dim_num_of_shapes + in1->dim_num_of_shapes + i];
            out->shapes[i] = letters_shape_map[cur_char];
        }

//        LOG_MSG("in1 shape is %d", in1->shapes[0]);
//        LOG_ERR("this einsum equation is %s", einsum_cfg.equation);


        params_vec.resize(1 + in_operands.size() + out_operands.size());
        inputs_vec.resize(BUF_MAXNUM);
//        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&einsum_cfg);
        params_vec[0] = params;
//        params_vec.push_back(params);

        return 0;
    };

    int fill_operands(char *one_buf_ptr) override {
        // fill op type and op name
        op_type = (char *) (&(this->einsum_cfg));
        op_name = (char *) ((int64_t) &(this->einsum_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->einsum_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->einsum_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->einsum_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        // set the weight and bias
        int32_t *head_ptr = (int32_t *) one_buf_ptr;
        int32_t init_cnt = head_ptr[3];
        char *cur_init_info_ptr = (char *) (one_buf_ptr + head_ptr[4]);


        initial_operands.resize(1);
        initial_datas.resize(1);

        std::string weigth_oprand = this->in_operands[1];

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

            // update cur_init_info_ptr
            int init_size = operand_buf_size(operand_ptr);
            cur_init_info_ptr += align_buf_size(sizeof(OPERAND_S) + init_size);
        }

        int b = 101;

        return 0;
    }

    int prepare_init_operand_data() override {
        // set desc struct
        // todo What is passed in here is not a real structure, and einsum does not need to pass in a structure

//        params_vec: cfg / ifmap desc / weight desc / bias desc / ofmap desc
        BUFFER_INFO_S weight_desc;
        weight_desc.addr = (int64_t) (&initial_operands[0]);
        params_vec[2] = weight_desc;
//        this->params_vec.push_back(weight_desc);

        // set buf
        BUFFER_INFO_S weight_buf;
        weight_buf.addr = (int64_t) (&(initial_datas[0][0]));
        inputs_vec[1] = weight_buf;

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
        each_ofmap_elem_computation = mac;  // todo: einsum 的计算量比较难统计，看看如何来统计

        return (double)(out_elem_size * each_ofmap_elem_computation * 1e-6);
    };

};

OP_REGISTER_GLOBAL(Einsum, Einsum::create_instance, sizeof(EINSUM_CONFIG_S));

#endif // OP_EINSUM_H