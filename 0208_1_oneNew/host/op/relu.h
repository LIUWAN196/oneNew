#ifndef OP_RELU_H
#define OP_RELU_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Relu : public op
{
public:
    RELU_CONFIG_S relu_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    Relu()
    {
        printf("new a Relu\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *relu_cfg_ptr)
    {
        // new Relu op
        std::shared_ptr<Relu> relu_ptr = std::make_shared<Relu>();
//        relu_ptr.get()->find_handle((BUFFER_GROUP_S *)relu_cfg_ptr);

        // fill op config
        memcpy(&(relu_ptr->relu_cfg), relu_cfg_ptr, sizeof(RELU_CONFIG_S));

        // // fill op type and op name
        // op_type = relu_cfg_ptr;
        // op_name = relu_cfg_ptr + NAME_LEN;

        op_ptr = relu_ptr;

        return 0;
    }


    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S* in = &operand_stu_map[in_operands[0]];

        OPERAND_S* out = &operand_stu_map[out_operands[0]];
        out->shape = in->shape;

        BUFFER_INFO_S params;
        params.addr = (int64_t)(&relu_cfg);
        params_vec.push_back(params);

        return  0;
    };



    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char*)(&(this->relu_cfg));
        op_name = (char*)((int64_t)&(this->relu_cfg) + NAME_LEN);

        std::string in_operand(this->relu_cfg.in_operand_name[0]);
        this->in_operands.push_back(in_operand);

        std::string out_operand(this->relu_cfg.out_operand_name[0]);
        this->out_operands.push_back(out_operand);




//        memcpy(&in_operand_stu.operand_name, &this->relu_cfg.in_operand_name[0], NAME_LEN);
//        memcpy(&out_operand_stu.operand_name, &this->relu_cfg.out_operand_name[0], NAME_LEN);
//        this->in_operands_s.push_back(&in_operand_stu);
//        this->out_operands_s.push_back(&out_operand_stu);

        return 0;
    }
};

OP_REGISTER_GLOBAL(Relu, Relu::create_instance);

#endif // OP_RELU_H
