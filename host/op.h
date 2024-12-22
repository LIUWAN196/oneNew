#ifndef OP_H
#define OP_H

#include "../../common/nn_common.h"
#include <dlfcn.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include "set"
#include <unordered_map>

// namespace one_new
// {
typedef int (*evalaa)(BUFFER_INFO_S *, BUFFER_INFO_S *, BUFFER_INFO_S *);

class op {
public:

    evalaa evla_impl;

    BOOL support_cuda;
    char *op_type;
    char *op_name;
    std::vector<std::string> in_operands;
    std::vector<std::string> out_operands;

    int32_t ifmap_st_idx = 0;
    int32_t init_st_idx = 0;

    std::vector<BUFFER_INFO_S> params_vec;
    std::vector<BUFFER_INFO_S> inputs_vec;
    std::vector<BUFFER_INFO_S> outputs_vec;

    BUFFER_GROUP_S params = {0};
    BUFFER_GROUP_S inputs = {0};
    BUFFER_GROUP_S outputs = {0};

    op() {
        params_vec.resize(BUF_MAXNUM);
        inputs_vec.resize(BUF_MAXNUM);
        return;
    };

    int find_handle(char *cfg_ptr) {
        char *op_type = (char *) (cfg_ptr);
        std::string op_lib_name(op_type);

//        if (op_lib_name == "Gemm") {
//            std::cout << "the op lib name is " << op_lib_name << std::endl;
//        }

        char op_cuda_lib_path[256] = {0};
        char op_x86_lib_path[256] = {0};

#ifdef USING_GPU
        snprintf(op_cuda_lib_path, sizeof(op_cuda_lib_path), "%s%s%s%s%s%s", OP_CU_LIB_DIR, "/", op_lib_name.c_str(), "/lib",
                 op_lib_name.c_str(), "_cu.so");
#endif
        snprintf(op_x86_lib_path, sizeof(op_x86_lib_path), "%s%s%s%s%s%s", OP_X86_LIB_DIR, "/", op_lib_name.c_str(), "/lib",
                 op_lib_name.c_str(), "_x86.so");

        std::string op_cuda_lib_path_str(op_cuda_lib_path);
        std::string op_x86_lib_path_str(op_x86_lib_path);

#ifdef USING_GPU
        void *cuda_handle = NULL;
        cuda_handle = dlopen(op_cuda_lib_path, RTLD_LAZY);
#endif

        void *x86_handle = NULL;
        x86_handle = dlopen(op_x86_lib_path, RTLD_LAZY);

        if (x86_handle != NULL) {
            evla_impl = (int (*)(BUFFER_INFO_S *, BUFFER_INFO_S *, BUFFER_INFO_S *)) dlsym(x86_handle, "eval");
            return 0;
        }

#ifdef USING_GPU
        if (cuda_handle != NULL) {
            std::cout << "the handle of " << op_type << " op (cuda platform)  is find. " << std::endl;
            evla_impl = (int (*)(BUFFER_INFO_S *, BUFFER_INFO_S *, BUFFER_INFO_S *)) dlsym(cuda_handle, "eval");
            return 0;
        }
#endif

        LOG_ERR("the handle of %s op is not find.", op_type);

//        std::cout << "the handle of " << op_type << " op is not find. " << std::endl;
        return 0;
    };

    virtual int shape_infer(std::unordered_map<std::string, OPERAND_S> &operand_stu_map) = 0;

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

    virtual double get_computation() {

        return 0;
    }

    virtual int forward() {
//        std::cout << "start forward. " << std::endl;

        int ret = evla_impl(&(this->params_vec[0]), &(this->inputs_vec[0]), &(this->outputs_vec[0]));
        return 0;
    };

    virtual int rt_prepare(std::unordered_map<std::string, BUFFER_INFO_S> &operand_buf_map,
                        std::unordered_map<std::string, OPERAND_S> &operand_stu_map, std::set<std::string> &init_operands_list) {
        BASE_CONFIG_S* cfg = (BASE_CONFIG_S*)(this->params_vec[0].addr);

        // step 1： set the operand desc
        int32_t params_ifmap_idx = ifmap_st_idx + 1;      // because [0] is cfg
        for (int i = 0; i < this->in_operands.size(); ++i) {
            auto it = init_operands_list.find(this->in_operands[i]);
            if (this->in_operands[i].empty() || it != init_operands_list.end()) {
                continue;
            }
            BUFFER_INFO_S in_desc;
            in_desc.addr = (int64_t) (&(operand_stu_map[this->in_operands[i]]));
            params_vec[params_ifmap_idx] = in_desc;
            params_ifmap_idx++;

        }

        int32_t ofmap_idx = 1 + this->in_operands.size();
        for (int i = 0; i < this->out_operands.size(); ++i) {
            BUFFER_INFO_S out_desc;
            out_desc.addr = (int64_t) (&(operand_stu_map[this->out_operands[i]]));
            params_vec[ofmap_idx] = out_desc;
            ofmap_idx++;
        }

        // step 2： set the operand buf
        int32_t ifmap_buf_idx = this->ifmap_st_idx;
        for (int i = 0; i < this->in_operands.size(); ++i) {
            auto it = init_operands_list.find(this->in_operands[i]);
            if (this->in_operands[i].empty() || it != init_operands_list.end()) {
                continue;
            }
            BUFFER_INFO_S in_buf;
            in_buf.addr = operand_buf_map[this->in_operands[i]].addr;

            inputs_vec[ifmap_buf_idx] = in_buf;
            ifmap_buf_idx++;
        }

        // 判断是否需要进行计算，例如 reshape / flatten 根本不需要计算
        OPERAND_S* first_ofmap = (OPERAND_S*)&(operand_stu_map[this->out_operands[0]]);
        if (first_ofmap->not_need_buf == TRUE) {
            // 只需要将输入的 buffer 描述拷贝到输出的 buffer 描述即可。后面都不需要计算
            operand_buf_map[this->out_operands[0]] = operand_buf_map[this->in_operands[0]];
            return 0;
        }

        for (int i = 0; i < this->out_operands.size(); ++i) {
            BUFFER_INFO_S out_buf;
            out_buf.addr = operand_buf_map[this->out_operands[i]].addr;
            outputs_vec.push_back(out_buf);
        }

        // prepare
        init_st_idx = ifmap_buf_idx;
        prepare_init_operand_data();

        return 0;
    };


    virtual int forward(std::unordered_map<std::string, BUFFER_INFO_S> &operand_buf_map,
                        std::unordered_map<std::string, OPERAND_S> &operand_stu_map, std::set<std::string> &init_operands_list) {
        BASE_CONFIG_S* cfg = (BASE_CONFIG_S*)(this->params_vec[0].addr);

        // 判断是否需要进行计算，例如 reshape / flatten 根本不需要计算
        OPERAND_S* first_ofmap = (OPERAND_S*)&(operand_stu_map[this->out_operands[0]]);
        if (first_ofmap->not_need_buf == TRUE) {
//            // 只需要将输入的 buffer 描述拷贝到输出的 buffer 描述即可。后面都不需要计算
//            operand_buf_map[this->out_operands[0]] = operand_buf_map[this->in_operands[0]];
            return 0;
        }

        show_dev_input(&(this->params_vec[0]));

        int ret = evla_impl(&(this->params_vec[0]), &(this->inputs_vec[0]), &(this->outputs_vec[0]));
        return 0;
    };

    virtual int forward(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) const
    {
        int ret = evla_impl(params, inputs, outputs);
        return 0;
    };

    virtual int fill_operands(char *one_buf_ptr) = 0;
};

// } // namespace one_new

#endif // OP_H
