#ifndef OP_OBJECT_DETECT_H
#define OP_OBJECT_DETECT_H

#include "op.h"
#include "../manager/manager.h"

class ObjectDetect : public op
{
public:
    OBJECT_DETECT_CONFIG_S object_detect_cfg;
    OPERAND_S in_operand_stu;
    OPERAND_S out_operand_stu;

    ObjectDetect()
    {
//        printf("new a ObjectDetect\n");
    };

    static int create_instance(std::shared_ptr<op> &op_ptr, char *Pose_detection_cfg_ptr)
    {
        // new Pose_detection op
        std::shared_ptr<ObjectDetect> object_detect_ptr = std::make_shared<ObjectDetect>();
//        Pose_detection_ptr.get()->find_handle((BUFFER_GROUP_S *)Pose_detection_cfg_ptr);

        // fill op config
        memcpy(&(object_detect_ptr->object_detect_cfg), Pose_detection_cfg_ptr, sizeof(OBJECT_DETECT_CONFIG_S));

        // // fill op type and op name
        // op_type = Pose_detection_cfg_ptr;
        // op_name = Pose_detection_cfg_ptr + OP_TYPE_LEN;

        op_ptr = object_detect_ptr;

        return 0;
    }

    virtual int shape_infer(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) override {
        return  0;

    };

    int fill_operands(char *one_buf_ptr) override
    {

        ONE_MODEL_DESC_S *one_model_desc_ptr = (ONE_MODEL_DESC_S *) one_buf_ptr;
        int32_t init_cnt = one_model_desc_ptr->init_cnt;
        char *cur_init_info_ptr = (char *)(one_buf_ptr) + one_model_desc_ptr->init_info_offset;

        USEFUL_INFO_S* useful_ptr =  &one_model_desc_ptr->useful_info;
        BUFFER_INFO_S useful_info;
        useful_info.addr = (int64_t) useful_ptr;
        params_vec[BUF_MAXNUM - 1] = useful_info;

        return 0;
    }


};

OP_REGISTER_GLOBAL(ObjectDetect, ObjectDetect::create_instance, sizeof(OBJECT_DETECT_CONFIG_S));

#endif // OP_OBJECT_DETECT_H
