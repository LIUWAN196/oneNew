#ifndef NET_H
#define NET_H

#include "../../common/nn_common_cpp.h"
#include <dlfcn.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include "op.h"
#include "../manager/manager.h"
#include <unordered_map>
#include "algorithm"

// namespace one_new
// {
typedef int (*evalaa)(BUFFER_GROUP_S *, BUFFER_GROUP_S *, BUFFER_GROUP_S *);

class net
{
public:
    char *one_buf_ptr; // the one model start addr
    std::unordered_map<std::shared_ptr<op>, std::vector<std::string>> op_in_operands_map;
    std::unordered_map<std::shared_ptr<op>, std::vector<std::string>> op_out_operands_map;
    std::unordered_map<std::shared_ptr<op>, std::vector<std::shared_ptr<op>>> op_pre_node;
    std::vector<std::shared_ptr<op>> op_exec_order;

    int load_one_model(char *one_path)
    {
        // step 1: get one file size
        std::ifstream one_file(one_path, std::ios::ate | std::ios::binary);
        int32_t one_file_size = static_cast<int32_t>(one_file.tellg());
        one_file.close();

        // step 2: load one file
        one_buf_ptr = (char *)malloc(one_file_size);
        FILE *file_p = NULL;

        file_p = fopen(one_path, "r");
        if (file_p == NULL)
        {
            std::cout << "failed: can't open the one file" << std::endl;
            return -1;
        }
        fread(one_buf_ptr, sizeof(char), one_file_size, file_p);
        fclose(file_p);

        return 0;
    }

    int instantiate_op()
    {
        Manager &m = Manager::getInstance();

        int32_t *head_ptr = (int32_t *)one_buf_ptr;

        int32_t io_cnt = head_ptr[5];
        char *cur_io_cfg_ptr = (char *)(one_buf_ptr + head_ptr[6]);

        for (int32_t io_i = 0; io_i < io_cnt; io_i++)
        {
            // get op type
            std::string op_type_str(cur_io_cfg_ptr);
            // get the instance method
            creator_ creator_method = m.Opmap[op_type_str];
            // build this op
            std::shared_ptr<op> op_ptr;
            creator_method(op_ptr, cur_io_cfg_ptr);
            op_ptr->fill_operands();

            op_in_operands_map[op_ptr] = op_ptr.get()->in_operands;
            auto a = op_out_operands_map[op_ptr];
            op_out_operands_map[op_ptr] = op_ptr.get()->out_operands;

            // update cur_io_cfg_ptr
            cur_io_cfg_ptr += align_buf_size(cfg_size_map[op_type_str]);
        }

        int32_t node_cnt = head_ptr[1];
        char *cur_node_cfg_ptr = (char *)(one_buf_ptr + head_ptr[2]);

        for (int32_t node_i = 0; node_i < node_cnt; node_i++)
        {
            // get op type
            std::string op_type_str(cur_node_cfg_ptr);
            // get the instance method
            creator_ creator_method = m.Opmap[op_type_str];
            // build this op
            std::shared_ptr<op> op_ptr;
            creator_method(op_ptr, cur_node_cfg_ptr);
            op_ptr->fill_operands();

            op_in_operands_map[op_ptr] = op_ptr.get()->in_operands;
            auto a = op_out_operands_map[op_ptr];
            op_out_operands_map[op_ptr] = op_ptr.get()->out_operands;

            // update cur_node_cfg_ptr
            cur_node_cfg_ptr += align_buf_size(cfg_size_map[op_type_str]);
        }

        return 0;
    }

    int mv_init_operands()
    {
        int32_t *head_ptr = (int32_t *)one_buf_ptr;
        int32_t init_cnt = head_ptr[3];
        char *cur_init_info_ptr = (char *)(one_buf_ptr + head_ptr[4]);

        for (int32_t init_i = 0; init_i < init_cnt; init_i++)
        {
            OPERAND_S *operand_ptr = (OPERAND_S *)cur_init_info_ptr;
            std::string init_operands = std::string(operand_ptr->operand_name);

            for (auto op_in_operands : op_in_operands_map)
            {
                auto operand_names = op_in_operands.second;
                for (auto operand_name : operand_names)
                {
                    if (operand_name == init_operands)
                    {
                        // operand_names.erase(std::remove(operand_names.begin(), operand_names.end(), operand_name), operand_names.end());
                        std::cout << "erase " << operand_name << " operands" << std::endl;
                    }
                }
            }
        }

        return 0;
    }

    net(char *one_path)
    {
        std::cout << "start load_one_model" << std::endl;
        if (load_one_model(one_path) != 0)
        {
            std::cout << "failed: load_one_model failed! " << std::endl;
        }

        std::cout << "start instantiate net op" << std::endl;
        if (instantiate_op() != 0)
        {
            std::cout << "failed: instantiate object of op class failed! " << std::endl;
        }

        std::cout << "start remove the init operands" << std::endl;
        if (mv_init_operands() != 0)
        {
            std::cout << "failed: remove the init operands failed! " << std::endl;
        }
    };

    int show_operands()
    {
        std::cout << "show op_in_operands_map" << std::endl;
        for (auto op_in_operands : op_in_operands_map)
        {
            std::cout << "op_type: " << op_in_operands.first.get()->op_type << ", in_operands:"
                      << op_in_operands.first.get()->op_name;
            for (auto operand_name : op_in_operands.second)
            {
                std::cout << operand_name << ", ";
            }
            std::cout << std::endl;
        }

        std::cout << "show op_out_operands_map" << std::endl;
        for (auto op_out_operands : op_out_operands_map)
        {
            std::cout << "op_type: " << op_out_operands.first.get()->op_type << ", out_operands:"
                      << op_out_operands.first.get()->op_name;
            for (auto operand_name : op_out_operands.second)
            {
                std::cout << operand_name << ", ";
            }
            std::cout << std::endl;
        }

        return 0;
    }

    int build_op_pre_node()
    {
        for (auto op_in_operands : op_in_operands_map)
        {
            if (op_in_operands.second.empty()){
                op_pre_node.insert(std::make_pair(op_in_operands.first, NULL));
            }
        }

        for (auto op_out_operands : op_out_operands_map)
        {
            // traverse each output operands of this op
            for (auto operand_name : op_out_operands.second)
            {
                for (auto op_in_operands : op_in_operands_map)
                {
                    for (auto in_operand_name : op_in_operands.second)
                    {
                        if (in_operand_name == operand_name)
                        {
                            op_pre_node[op_in_operands.first].push_back(op_out_operands.first);
                            int bbb = 101;
                        }
                    }
                }
            }
        }

//        // tmp add input
//        std::shared_ptr<op> in_op_ptr;
//        for (auto op_out_operands : op_out_operands_map)
//        {
//            in_op_ptr = op_out_operands.first;
//        }
//
//        op_pre_node[in_op_ptr];

        return 0;
    }

    int build_graph_seq()
    {
        while (!op_pre_node.empty())
        {
            auto a = op_pre_node;
            for (auto opa : a)
            {
                if (opa.second.empty())
                {
                    op_exec_order.push_back(opa.first);
                    op_pre_node.erase(opa.first);

                    // erase the previous nodes of op_pre_node
                    for (auto &op_other : op_pre_node)
                    {
                        auto bbbb = op_other.second;
                        op_other.second.erase(std::remove(op_other.second.begin(), op_other.second.end(), opa.first),
                                              op_other.second.end());

                        int c = 101;
                    }

                    break;
                }
                std::cout << "warning: the no op is no input" << std::endl;
            }
        }
        return 0;
    }

    int show_op_exec_order()
    {
        std::cout << "\n=========== start show_op_exec_order ===========" << std::endl;
        for (auto op : op_exec_order)
        {
            std::cout << "op_type: " << op->op_type << ", op_name:" << op->op_name << std::endl;
        }
        std::cout << "=========== end show_op_exec_order ===========" << std::endl;
        return 0;
    }

    int build_graph()
    {
        show_operands();
        build_op_pre_node();
        build_graph_seq();

        show_op_exec_order();

        return 0;
    }
};

// } // namespace one_new

#endif // NET_H
