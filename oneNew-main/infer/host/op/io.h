#ifndef OP_IO_H
#define OP_IO_H

#include "op.h"
#include "../manager/manager.h"
// namespace one_new {

class io : public op
{
public:
    IO_CONFIG_S io_cfg;

    io()
    {
        printf("new a io\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *io_cfg_ptr)
    {
        // new io op
        std::shared_ptr<io> io_ptr = std::make_shared<io>();

        // fill op config
        memcpy(&(io_ptr->io_cfg), io_cfg_ptr, sizeof(IO_CONFIG_S));

        // // fill op type and op name
        // op_type = io_cfg_ptr;
        // op_name = io_cfg_ptr + NAME_LEN;

        op_ptr = io_ptr;

        return 0;
    }

    int fill_operands() override
    {
        // fill op type and op name
        op_type = (char *)(&(this->io_cfg));
        op_name = (char *)((int64_t) & (this->io_cfg) + NAME_LEN);
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

OP_REGISTER_GLOBAL(io, io::create_instance);

#endif // OP_IO_H
