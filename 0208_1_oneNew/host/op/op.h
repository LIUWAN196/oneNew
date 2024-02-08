#ifndef OP_H
#define OP_H

#include "../../common/nn_common.h"
#include <dlfcn.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>

// namespace one_new
// {
typedef int (*evalaa)(BUFFER_INFO_S *, BUFFER_INFO_S *, BUFFER_INFO_S *);

class op {
public:
    std::string weight_char0 = "weight";
    std::string weight_char1 = "Weight";
    std::string bias_char0 = "bias";
    std::string bias_char1 = "Bias";

    evalaa evla_impl;

    BOOL support_cuda;
    char *op_type;
    char *op_name;
    std::vector<std::string> in_operands;
    std::vector<std::string> out_operands;

//    std::vector<OPERAND_S*> in_operands_s;
//    std::vector<OPERAND_S*> out_operands_s;


    std::vector<BUFFER_INFO_S> params_vec;
    std::vector<BUFFER_INFO_S> inputs_vec;
    std::vector<BUFFER_INFO_S> outputs_vec;

    BUFFER_GROUP_S params = {0};
    BUFFER_GROUP_S inputs = {0};
    BUFFER_GROUP_S outputs = {0};

    op() {
        return;
    };

    int find_handle(char *cfg_ptr) {
        char *op_type = (char *) (cfg_ptr);
        std::string op_lib_name(op_type);

        char op_lib_path[256] = {0};
//        snprintf(op_lib_path, sizeof(op_lib_path), "%s%s%s%s", OP_LIB_DIR, "/lib", op_lib_name.c_str(), ".so");
        snprintf(op_lib_path, sizeof(op_lib_path), "%s%s%s%s%s", OP_LIB_DIR, op_lib_name.c_str(), "/lib",
                 op_lib_name.c_str(), ".so");

        std::string aaa(op_lib_path);
        void *handle = NULL;
        handle = dlopen(op_lib_path, RTLD_LAZY);
        if (handle == NULL) {
            std::cout << "the handle of " << op_type << " op is not find. " << std::endl;
        }

        std::cout << "the handle of " << op_type << " op is find. " << std::endl;

        evla_impl = (int (*)(BUFFER_INFO_S *, BUFFER_INFO_S *, BUFFER_INFO_S *)) dlsym(handle, "eval");

        return 0;
    };

    virtual int calc_out_operand_shape(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) = 0;

    virtual int other_prepare() { return 0; };

    // gen lut, find handle, and so on
    virtual int prepare(char *cfg_ptr) {

        // 1  find handle
        find_handle(cfg_ptr);

        // 2
        other_prepare();
        return 0;
    }

    virtual int prepare_init_operand_data() { return 0; };

    virtual int forward() {
        std::cout << "start forward. " << std::endl;

        int ret = evla_impl(&(this->params_vec[0]), &(this->inputs_vec[0]), &(this->outputs_vec[0]));
        return 0;
    };

    virtual int forward(std::unordered_map<std::string, std::vector<int8_t>> &operand_buf_map,
                        std::unordered_map<std::string, OPERAND_S> &operand_stu_map) {
//        std::cout << "start forward. " << std::endl;
        // step 1： set the operand desc
        for (int i = 0; i < this->in_operands.size(); ++i) {
//        for (int i = 0; i < 1; ++i) {
            if (this->in_operands[i].empty() || this->in_operands[i].find(weight_char0) != std::string::npos ||
                this->in_operands[i].find(weight_char1) != std::string::npos ||
                this->in_operands[i].find(bias_char0) != std::string::npos ||
                this->in_operands[i].find(bias_char1) != std::string::npos) {
                continue;
            }
            BUFFER_INFO_S in_desc;
            in_desc.addr = (int64_t) (&(operand_stu_map[this->in_operands[i]]));
            params_vec.push_back(in_desc);
        }

        for (int i = 0; i < this->out_operands.size(); ++i) {
            BUFFER_INFO_S out_desc;
            out_desc.addr = (int64_t) (&(operand_stu_map[this->out_operands[i]]));
            params_vec.push_back(out_desc);
        }

        // step 2： set the operand buf
        for (int i = 0; i < this->in_operands.size(); ++i) {
//        for (int i = 0; i < 1; ++i) {
            if (this->in_operands[i].empty() || this->in_operands[i].find(weight_char0) != std::string::npos ||
                this->in_operands[i].find(weight_char1) != std::string::npos ||
                this->in_operands[i].find(bias_char0) != std::string::npos ||
                this->in_operands[i].find(bias_char1) != std::string::npos) {
                continue;
            }
            BUFFER_INFO_S in_buf;
            in_buf.addr = (int64_t) (&(operand_buf_map[this->in_operands[i]][0]));
            inputs_vec.push_back(in_buf);
        }

        for (int i = 0; i < this->out_operands.size(); ++i) {
            BUFFER_INFO_S out_buf;
            out_buf.addr = (int64_t) (&(operand_buf_map[this->out_operands[i]][0]));
            outputs_vec.push_back(out_buf);
        }

        // prapaere
        prepare_init_operand_data();


//        std::vector<float> vvv(100,0);
//        std::vector<float> aaa(100,0);

//        int ret = evla_impl((BUFFER_INFO_S*)&(this->params_vec[0]), (BUFFER_INFO_S*)&(vvv[0]), (BUFFER_INFO_S*)&(aaa[0]));
        int ret = evla_impl(&(this->params_vec[0]), &(this->inputs_vec[0]), &(this->outputs_vec[0]));
        return 0;
    };



//    virtual int forward(BUFFER_GROUP_S *params, BUFFER_GROUP_S *inputs, BUFFER_GROUP_S *outputs) const
//    {
//        std::cout << "start forward. " << std::endl;
//
//        int ret = evla_impl(params, inputs, outputs);
//        return 0;
//    };

    virtual int fill_operands(char *one_buf_ptr) = 0;
};

// } // namespace one_new

#endif // OP_H
