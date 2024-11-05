#ifndef OP_IO_H
#define OP_IO_H

#include "op.h"
#include "../manager/manager.h"
// namespace one_new {

class io : public op
{
public:
    IO_CONFIG_S io_cfg;
    OPERAND_S operand;

    io()
    {
//        printf("new a io\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *io_cfg_ptr)
    {
        // new io op
        std::shared_ptr<io> io_ptr = std::make_shared<io>();
//        io_ptr.get()->find_handle((BUFFER_GROUP_S *)io_cfg_ptr);

        // fill op config
        memcpy(&(io_ptr->io_cfg), io_cfg_ptr, sizeof(IO_CONFIG_S));

        // // fill op type and op name
        // op_type = io_cfg_ptr;
        // op_name = io_cfg_ptr + OP_TYPE_LEN;

        op_ptr = io_ptr;

        return 0;
    }

    virtual int rt_prepare(std::unordered_map<std::string, BUF_INFO_S> &operand_buf_map, std::unordered_map<std::string, OPERAND_S> &operand_stu_map, std::set<std::string> &init_operands_list) override
    {
//        std::cout << "this op type is io, dont need rt_prepare. " << std::endl;
        return 0;
    };

    virtual int forward(std::unordered_map<std::string, BUF_INFO_S> &operand_buf_map, std::unordered_map<std::string, OPERAND_S> &operand_stu_map, std::set<std::string> &init_operands_list) override
    {
//        std::cout << "this op type is io, dont need forward. " << std::endl;
        return 0;
    };

    virtual int shape_infer(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S in = operand_stu_map[in_operands[0]];

        OPERAND_S out = operand_stu_map[out_operands[0]];


        return  0;
    };

    int fill_operands(char *one_buf_ptr) override
    {
        // fill op type and op name
        op_type = (char *)(&(this->io_cfg));
        op_name = (char *)((int64_t) & (this->io_cfg) + OP_TYPE_LEN);
        std::string op_name_str(op_name);

        if (op_name_str == "input") //  is input op
        {
            std::string out_operand(this->io_cfg.operand.operand_name);
            this->out_operands.push_back(out_operand);
        }
        else if (op_name_str == "output") //  is output op
        {
            std::string in_operand(this->io_cfg.operand.operand_name);
            this->in_operands.push_back(in_operand);
        }
        else
        {
            std::cout << "the op is not in or output op" << std::endl;
        }

        return 0;
    }
};

OP_REGISTER_GLOBAL(io, io::create_instance, sizeof(IO_CONFIG_S));

#endif // OP_IO_H
