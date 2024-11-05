#ifndef OP_POSE_DETECTION_H
#define OP_POSE_DETECTION_H

#include "op.h"
#include "../manager/manager.h"

class PoseDetection : public op
{
public:
    POSE_DETECTION_CONFIG_S pose_detection_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    PoseDetection()
    {
//        printf("new a PoseDetection\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *Pose_detection_cfg_ptr)
    {
        // new Pose_detection op
        std::shared_ptr<PoseDetection> pose_detection_ptr = std::make_shared<PoseDetection>();
//        Pose_detection_ptr.get()->find_handle((BUFFER_GROUP_S *)Pose_detection_cfg_ptr);

        // fill op config
        memcpy(&(pose_detection_ptr->pose_detection_cfg), Pose_detection_cfg_ptr, sizeof(POSE_DETECTION_CONFIG_S));

        // // fill op type and op name
        // op_type = Pose_detection_cfg_ptr;
        // op_name = Pose_detection_cfg_ptr + OP_TYPE_LEN;

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

OP_REGISTER_GLOBAL(PoseDetection, PoseDetection::create_instance, sizeof(POSE_DETECTION_CONFIG_S));

#endif // OP_POSE_DETECTION_H
