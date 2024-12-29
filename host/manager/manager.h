#ifndef __MANAGER_H__
#define __MANAGER_H__

#include "../../common/nn_common.h"
#include "unordered_map"
#include "../op.h"

typedef int (*creator_)(std::shared_ptr<op>& op_ptr, char *op_cfg_ptr);

class Manager
{
public:
    std::unordered_map<std::string, creator_> Opmap;
    std::unordered_map<std::string, int32_t> op_cfg_size_map;

    std::unordered_map<std::string, creator_> &registeredOp() { return this->Opmap; }
    std::unordered_map<std::string, int32_t> &registeredOpCfgSize() { return this->op_cfg_size_map; }

    static Manager &getInstance();
};

Manager &Manager::getInstance()
{
    static Manager instance;
    return instance;
}

class OpRegistry
{
public:
    explicit OpRegistry(std::string name, creator_ creator_method, int32_t cfg_size)
    {
        Manager &m = Manager::getInstance();
        m.registeredOp()[name] = creator_method;
        m.registeredOpCfgSize()[name] = cfg_size;
    };
};

#define OP_STR_CONCAT_(__x, __y) __x##__y
#define OP_STR_CONCAT(__x, __y) OP_STR_CONCAT_(__x, __y)

#define OP_REGISTER_GLOBAL(name, creator_method, cfg_size) \
    auto OP_STR_CONCAT(__##name__, __COUNTER__) = ::OpRegistry(#name, creator_method, cfg_size)

#endif // __MANAGER_H__