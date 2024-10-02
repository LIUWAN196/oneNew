#ifndef OP_GATHER_H
#define OP_GATHER_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Gather : public op {
public:
    GATHER_CONFIG_S gather_cfg;
    std::vector<std::vector<float>> initial_datas;  // maybe the first ifmap is init
    std::vector<OPERAND_S> initial_operands;  // maybe the first ifmap is init
    int32_t init_ifmap_idx = -1;

    Gather() {
//        printf("new a Gather\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *cfg_ptr) {
        // new Gather op
        std::shared_ptr<Gather> gather_ptr = std::make_shared<Gather>();

        // fill op config
        memcpy(&(gather_ptr->gather_cfg), cfg_ptr, sizeof(GATHER_CONFIG_S));

        op_ptr = gather_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S *data, *idx;
        OPERAND_S* out = &operand_stu_map[out_operands[0]];
        for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
            out->shapes[dim_i] = 1;
        }

        if (gather_cfg.indices_from_ifmap == TRUE) {
            if (initial_operands.size() != 0) {
                data = &initial_operands[0];
            } else {
                data = &operand_stu_map[in_operands[0]];
            }
            idx = &operand_stu_map[in_operands[1]];
            if (gather_cfg.axis == 0) {
                out->dim_num_of_shapes = idx->dim_num_of_shapes + 1;
                for (int i = 0; i < idx->dim_num_of_shapes; ++i) {
                    out->shapes[i] = idx->shapes[i];
                }
                out->shapes[out->dim_num_of_shapes - 1] = data->shapes[data->dim_num_of_shapes - 1];

//                out->dim_num_of_shapes = data->dim_num_of_shapes;
//                for (int i = 0; i < data->dim_num_of_shapes; ++i) {
//                    out->shapes[i] = data->shapes[i];
//                }
//                out->shapes[0] = idx->shapes[0];
            } else if (gather_cfg.axis == 1) {
                out->dim_num_of_shapes = data->dim_num_of_shapes;
                for (int i = 0; i < data->dim_num_of_shapes; ++i) {
                    out->shapes[i] = data->shapes[i];
                }
                out->shapes[1] = idx->shapes[0];
            } else {
                out->dim_num_of_shapes = idx->dim_num_of_shapes + 1;
                int i;
                for (i = 0; i < idx->dim_num_of_shapes; ++i) {
                    out->shapes[i] = idx->shapes[i];
                }
                out->shapes[i] = data->shapes[data->dim_num_of_shapes - 1];
            }

        } else {
            data = &operand_stu_map[in_operands[0]];

            int32_t gather_axis = gather_cfg.axis;

            out->dim_num_of_shapes = data->dim_num_of_shapes - 1;
            for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
                out->shapes[dim_i] = 1;
            }
            for (int dim_i = 0; dim_i < gather_axis; ++dim_i) {
                out->shapes[dim_i] = data->shapes[dim_i];
            }
            for (int dim_i = gather_axis; dim_i < data->dim_num_of_shapes; ++dim_i) {
                out->shapes[dim_i] = data->shapes[dim_i + 1];
            }
        }


        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&gather_cfg);
        params_vec[0] = params;

//        if (strcmp(gather_cfg.op_base_cfg.op_name, "/model.28/Gather_9") == 0) {
//            LOG_DBG("%s", gather_cfg.op_base_cfg.op_name);
//            LOG_DBG("out shape is %d %d %d %d", data->shapes[0], data->shapes[1], data->shapes[2], data->shapes[3]);
//            LOG_DBG("out shape is %d %d %d %d", idx->shapes[0], idx->shapes[1], idx->shapes[2], idx->shapes[3]);
//            LOG_DBG("out shape is %d %d %d %d", out->shapes[0], out->shapes[1], out->shapes[2], out->shapes[3]);
//        }

//
//        if (strcmp(gather_cfg.op_base_cfg.op_name, "/model.28/Gather_10") == 0) {
//            LOG_DBG("%s", gather_cfg.op_base_cfg.op_name);
//        }
//
//        if (gather_cfg.indices_from_ifmap == TRUE) {
//            if (initial_operands.size() != 0) {
//                int32_t gather_axis = gather_cfg.axis;
//
//                OPERAND_S* out = &operand_stu_map[out_operands[0]];
//                for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
//                    out->shapes[dim_i] = 1;
//                }
//                // todo: create shape infer from input
//                out->shapes[1] = 77;
//                out->shapes[2] = 512;
//                out->dim_num_of_shapes = 3;
//
//
//                inputs_vec.resize(in_operands.size());
//                BUFFER_INFO_S params;
//                params.addr = (int64_t) (&gather_cfg);
//                params_vec[0] = params;
//            } else {
//                int32_t gather_axis = gather_cfg.axis;
//
//                OPERAND_S* out = &operand_stu_map[out_operands[0]];
//                for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
//                    out->shapes[dim_i] = 1;
//                }
//                // todo: create shape infer from input
//                out->shapes[1] = 512;
//                out->dim_num_of_shapes = 2;
//
//
//                inputs_vec.resize(in_operands.size());
//                BUFFER_INFO_S params;
//                params.addr = (int64_t) (&gather_cfg);
//                params_vec[0] = params;
//            }
//        } else {
//            int32_t gather_axis = gather_cfg.axis;
//
//            OPERAND_S* out = &operand_stu_map[out_operands[0]];
//
//            for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
//                out->shapes[dim_i] = 1;
//            }
//
//            out->dim_num_of_shapes = in->dim_num_of_shapes - gather_axis;
//            out->shapes[0] = 1;
//            for (int dim_i = 1; dim_i < out->dim_num_of_shapes; ++dim_i) {
//                out->shapes[dim_i] = in->shapes[dim_i + gather_axis];
//            }
//
//
//            inputs_vec.resize(in_operands.size());
//            BUFFER_INFO_S params;
//            params.addr = (int64_t) (&gather_cfg);
//            params_vec[0] = params;
//        }

        return 0;
    };

    int fill_operands(char *one_buf_ptr) override {
        // fill op type and op name
        op_type = (char *) (&(this->gather_cfg));
        op_name = (char *) ((int64_t) &(this->gather_cfg) + OP_TYPE_LEN);

        BASE_CONFIG_S *base_cfg = (BASE_CONFIG_S *) (&(this->gather_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->gather_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->gather_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        // 下面是看第一个输入操作数是否在 init 中
        // set the weight and bias
        ONE_MODEL_DESC_S *one_model_desc_ptr = (ONE_MODEL_DESC_S *) one_buf_ptr;
        int32_t init_cnt = one_model_desc_ptr->init_cnt;
        char *cur_init_info_ptr = (char *)(one_buf_ptr) + one_model_desc_ptr->init_info_offset;
        std::string first_oprand = this->in_operands[0];
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

            // update cur_init_info_ptr
            int init_size = operand_buf_size(operand_ptr);
            cur_init_info_ptr += align_buf_size(sizeof(OPERAND_S) + init_size);
        }

        return 0;
    }

    int prepare_init_operand_data() override {
        // set desc struct
        // todo What is passed in here is not a real structure, and conv does not need to pass in a structure

        if (initial_operands.size() != 0) {
            BUFFER_INFO_S first_operand_desc;
            first_operand_desc.addr = (int64_t) (&initial_operands[0]);
            params_vec[1] = first_operand_desc;    //  [0] is cfg; [1] is init data; [2] is first ifmap

            // set buf
            BUFFER_INFO_S first_operand_buf;
            first_operand_buf.addr = (int64_t) (&(initial_datas[0][0]));
            inputs_vec[0] = first_operand_buf;
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(Gather, Gather::create_instance, sizeof(GATHER_CONFIG_S));

#endif // OP_GATHER_H
