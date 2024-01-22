#ifndef OP_H
#define OP_H

#include "../../common/nn_common.h"
#include <dlfcn.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

// namespace one_new
// {
typedef int (*evalaa)(BUFFER_GROUP_S *, BUFFER_GROUP_S *, BUFFER_GROUP_S *);

class op
{
public:
    evalaa evla_impl;

    BOOL support_cuda;
    char* op_type;
    char* op_name;
    std::vector<std::string> in_operands;
    std::vector<std::string> out_operands;
    op()
    {
        return;
    };

    int find_handle(BUFFER_GROUP_S *params)
    {
        char *op_type = (char *)(params->buf_info[0].addr);
        std::string op_lib_name(op_type);

        char op_lib_path[256] = {0};
        snprintf(op_lib_path, sizeof(op_lib_path), "%s%s%s%s", OP_LIB_DIR, "/lib", op_lib_name.c_str(), ".so");

        void *handle = dlopen(op_lib_path, RTLD_LAZY);

        std::cout << "the handle of " << op_type << " op is find. " << std::endl;

        evla_impl = (int (*)(BUFFER_GROUP_S *, BUFFER_GROUP_S *, BUFFER_GROUP_S *))dlsym(handle, "eval");

        return 0;
    };

    virtual int other_prepare() { return 0; };

    // gen lut, find handle, and so on
    virtual int prepare(BUFFER_GROUP_S *params)
    {

        // 1  find handle
        find_handle(params);

        // 2
        other_prepare();
        return 0;
    }

    virtual int forward(BUFFER_GROUP_S *params, BUFFER_GROUP_S *inputs, BUFFER_GROUP_S *outputs) const
    {
        std::cout << "start forward. " << std::endl;

        int ret = evla_impl(params, inputs, outputs);
        return 0;
    };

    virtual int fill_operands() = 0;
};

// } // namespace one_new

#endif // OP_H
