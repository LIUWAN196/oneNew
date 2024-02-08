#ifndef OP_SEGEMET_H
#define OP_SEGEMET_H

#include "op.h"
#include "../manager/manager.h"

class Segment : public op
{
public:
    SEGMENT_CONFIG_S segment_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    Segment()
    {
//        printf("new a Segment\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *segment_cfg_ptr)
    {
        // new Segment op
        std::shared_ptr<Segment> segment_ptr = std::make_shared<Segment>();
//        segment_ptr.get()->find_handle((BUFFER_GROUP_S *)segment_cfg_ptr);

        // fill op config
        memcpy(&(segment_ptr->segment_cfg), segment_cfg_ptr, sizeof(SEGMENT_CONFIG_S));

        // // fill op type and op name
        // op_type = segment_cfg_ptr;
        // op_name = segment_cfg_ptr + OP_TYPE_LEN;

        op_ptr = segment_ptr;

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

OP_REGISTER_GLOBAL(Segment, Segment::create_instance, sizeof(SEGMENT_CONFIG_S));

#endif // OP_SEGEMET_H