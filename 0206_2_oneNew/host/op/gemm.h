#ifndef OP_GEMM_H
#define OP_GEMM_H

#include "op.h"
// #include "../../device/x86/relu6/relu6.h"
#include "../manager/manager.h"
// namespace one_new {

class Gemm : public op
{
public:
    GEMM_CONFIG_S gemm_cfg;

    Gemm()
    {
        printf("new a Relu\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *gemm_cfg_ptr)
    {
        // new Gemm op
        std::shared_ptr<Gemm> gemm_ptr = std::make_shared<Gemm>();

        // fill op config
        memcpy(&(gemm_ptr->gemm_cfg), gemm_cfg_ptr, sizeof(GEMM_CONFIG_S));


        op_ptr = gemm_ptr;

        return 0;
    }

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        OPERAND_S in = operand_stu_map[in_operands[0]];

        OPERAND_S out = operand_stu_map[out_operands[0]];


        return  0;
    };

    int fill_operands() override
    {
        // fill op type and op name
        op_type = (char*)(&(this->gemm_cfg));
        op_name = (char*)((int64_t)&(this->gemm_cfg) + NAME_LEN);

        int32_t in_operand_cnt = 3;
        for (size_t i = 0; i < in_operand_cnt; i++)
        {
        std::string in_operand(this->gemm_cfg.in_operand_name[i]);
        this->in_operands.push_back(in_operand);
        }

        std::string out_operand(this->gemm_cfg.out_operand_name[0]);
        this->out_operands.push_back(out_operand);

        return 0;
    }
};

OP_REGISTER_GLOBAL(Gemm, Gemm::create_instance);

#endif // OP_GEMM_H
