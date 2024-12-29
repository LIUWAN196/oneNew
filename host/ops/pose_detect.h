#ifndef OP_POSE_DETECT_H
#define OP_POSE_DETECT_H

#include "op.h"
#include "../manager/manager.h"

class PoseDetect : public op
{
public:
    POSE_DETECT_CONFIG_S pose_detect_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    PoseDetect()
    {
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *pose_detect_cfg_ptr)
    {
        // new Pose_detection op
        std::shared_ptr<PoseDetect> pose_detection_ptr = std::make_shared<PoseDetect>();

        // fill op config
        memcpy(&(pose_detection_ptr->pose_detect_cfg), pose_detect_cfg_ptr, sizeof(POSE_DETECT_CONFIG_S));

        op_ptr = pose_detection_ptr;

        return 0;
    }

    virtual int shape_infer(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        return  0;

    };

    int fill_operands(char *one_buf_ptr) override
    {
        return 0;
    }

};

OP_REGISTER_GLOBAL(PoseDetect, PoseDetect::create_instance, sizeof(POSE_DETECT_CONFIG_S));

#endif // OP_POSE_DETECT_H
