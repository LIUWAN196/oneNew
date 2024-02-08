#ifndef OP_ARGMAX_H
#define OP_ARGMAX_H

#include "op.h"
#include "../manager/manager.h"
// namespace one_new {

class Argmax : public op
{
public:
    ARGMAX_CONFIG_S argmax_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    Argmax()
    {
//        printf("new a Argmax\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *argmax_cfg_ptr)
    {
        // new Argmax op
        std::shared_ptr<Argmax> argmax_ptr = std::make_shared<Argmax>();
//        argmax_ptr.get()->find_handle((BUFFER_GROUP_S *)argmax_cfg_ptr);

        // fill op config
        memcpy(&(argmax_ptr->argmax_cfg), argmax_cfg_ptr, sizeof(ARGMAX_CONFIG_S));

        // // fill op type and op name
        // op_type = argmax_cfg_ptr;
        // op_name = argmax_cfg_ptr + OP_TYPE_LEN;

        op_ptr = argmax_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        return  0;

    };

    int fill_operands(char *one_buf_ptr) override
    {

        return 0;
    }


};

OP_REGISTER_GLOBAL(Argmax, Argmax::create_instance, sizeof(ARGMAX_CONFIG_S));

#endif // OP_ARGMAX_H
