#ifndef __MANAGER_H__
#define __MANAGER_H__

#include "../../common/nn_common.h"
#include "unordered_map"
#include "../op/op.h"

// namespace one_new
//{

typedef int (*creator_)(std::shared_ptr<op>& op_ptr, char *op_cfg_ptr);

class Manager
{
private:
    /* data */
public:
    std::unordered_map<std::string, creator_> Opmap;

    std::unordered_map<std::string, creator_> &registeredOp() { return this->Opmap; }

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
    explicit OpRegistry(std::string name, creator_ creator_method)
    {
        Manager &m = Manager::getInstance();
        m.registeredOp()[name] = creator_method;

        std::cout << "op: " << name << " is registered!" << std::endl;
    };
};

#define OP_STR_CONCAT_(__x, __y) __x##__y
#define OP_STR_CONCAT(__x, __y) OP_STR_CONCAT_(__x, __y)

#define OP_REGISTER_GLOBAL(name, creator_method) \
    auto OP_STR_CONCAT(__##name__, __COUNTER__) = ::OpRegistry(#name, creator_method)

#endif // __MANAGER_H__