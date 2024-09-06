#ifndef OP_GRID_SAMPLE_H
#define OP_GRID_SAMPLE_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class GridSample : public op
{
public:
    GRID_SAMPLE_CONFIG_S grid_sample_cfg;

    // 有可能 grid_sample 的第二个输入数据是 init 的
    std::vector<std::vector<float>> initial_datas;  // weight and bias
    std::vector<OPERAND_S> initial_operands;  // weight and bias
    GridSample()
    {
//        printf("new a GridSample\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *relu_cfg_ptr)
    {
        // new GridSample op
        std::shared_ptr<GridSample> grid_sample_ptr = std::make_shared<GridSample>();

        // fill op config
        memcpy(&(grid_sample_ptr->grid_sample_cfg), relu_cfg_ptr, sizeof(GRID_SAMPLE_CONFIG_S));

        // // fill op type and op name
        // op_type = relu_cfg_ptr;
        // op_name = relu_cfg_ptr + OP_TYPE_LEN;

        op_ptr = grid_sample_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];
        OPERAND_S* in1 = &operand_stu_map[in_operands[1]];

        OPERAND_S* out = &operand_stu_map[out_operands[0]];

        if (in->dim_num_of_shapes != in1->dim_num_of_shapes) {
            LOG_ERR("the in->dim_num_of_shapes is %d, in1->dim_num_of_shapes is %d, they must be equal",
                    in->dim_num_of_shapes, in1->dim_num_of_shapes);
        }
        out->dim_num_of_shapes = in->dim_num_of_shapes;
        for (int i = 0; i < SHAPE_LEN; ++i) {
            out->shapes[i] = 1;
        }
        // for example: in 8x32x40x40; in1 8x300x4x2;  ------>  out 8x32x300x4
        out->shapes[0] = in->shapes[0];
        out->shapes[1] = in->shapes[1];
        out->shapes[2] = in1->shapes[1];
        out->shapes[3] = in1->shapes[2];

        params_vec.resize(1 + in_operands.size() + out_operands.size());
        inputs_vec.resize(in_operands.size());
        BUFFER_INFO_S params;
        params.addr = (int64_t) (&grid_sample_cfg);
        params_vec[0] = params;

        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->grid_sample_cfg));
        op_name = (char*)((int64_t)&(this->grid_sample_cfg) + OP_TYPE_LEN);

        if (strcmp(op_name, "/model.10/attn/GridSample") == 0) {
            int a = 101;
        }

        BASE_CONFIG_S* base_cfg = (BASE_CONFIG_S*)(&(this->grid_sample_cfg));
        int32_t in_operand_num = base_cfg->in_operand_num;
        int32_t out_operand_num = base_cfg->out_operand_num;

        for (int in_i = 0; in_i < in_operand_num; ++in_i) {
            std::string in_operand(this->grid_sample_cfg.op_base_cfg.in_operand_name[in_i]);
            this->in_operands.push_back(in_operand);
        }

        for (int out_i = 0; out_i < out_operand_num; ++out_i) {
            std::string out_operand(this->grid_sample_cfg.op_base_cfg.out_operand_name[out_i]);
            this->out_operands.push_back(out_operand);
        }

        return 0;
    }


};

OP_REGISTER_GLOBAL(GridSample, GridSample::create_instance, sizeof(GRID_SAMPLE_CONFIG_S));

#endif // OP_GRID_SAMPLE_H
