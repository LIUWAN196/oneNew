#ifndef NET_H
#define NET_H

#include "../../common/nn_common.h"
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
#include <list>
#include <set>
#include "algorithm"
#include <omp.h>

// namespace one_new
// {
typedef int (*evalaa)(BUFFER_INFO_S *, BUFFER_INFO_S *, BUFFER_INFO_S *);

class net;


class extractor {

public:

    std::unordered_map<std::string, std::vector<float>> operand_buf_map;

    net *net_ptr;

    extractor(net *net_ptr_) : net_ptr(net_ptr_) {
//        printf("new a extractor\n");
        new_output_buf();
    };

    int new_output_buf();

    int prof_impl(std::unordered_map<std::string, std::vector<float>> &io_buf_map, std::unordered_map<std::string, std::string> cfg_info_map);

    int impl(std::unordered_map<std::string, std::vector<float>> &io_buf_map, std::unordered_map<std::string, std::string> cfg_info_map);

};


class net {
public:
    char *one_buf_ptr; // the one model start addr
    std::unordered_map<std::shared_ptr<op>, std::vector<std::string>> op_in_operands_map;
    std::unordered_map<std::shared_ptr<op>, std::vector<std::string>> op_out_operands_map;
    std::unordered_map<std::shared_ptr<op>, std::vector<std::shared_ptr<op>>> op_pre_node;
    std::vector<std::shared_ptr<op>> op_exec_order;
    std::set<std::string> operands_list;
    std::set<std::string> init_operands_list;
//    std::vector<OPERAND_S> operand_stu_vec;
    std::unordered_map<std::string, OPERAND_S> operand_stu_map;
//    friend class extractor;

    int load_one_model(const char *one_path) {
        // step 1: get one file size
        std::ifstream one_file(one_path, std::ios::ate | std::ios::binary);
        int32_t one_file_size = static_cast<int32_t>(one_file.tellg());
        one_file.close();

        // step 2: load one file
        one_buf_ptr = (char *) malloc(one_file_size);
        FILE *file_p = NULL;

        file_p = fopen(one_path, "r");
        if (file_p == NULL) {
            std::cout << "failed: can't open the one file" << std::endl;
            return -1;
        }
        fread(one_buf_ptr, sizeof(char), one_file_size, file_p);
        fclose(file_p);

        return 0;
    }

    int instantiate_op() {
        Manager &m = Manager::getInstance();

        ONE_MODEL_DESC_S *one_model_info_ptr = (ONE_MODEL_DESC_S *)one_buf_ptr;
//        int32_t *head_ptr = (int32_t *) one_buf_ptr;

        int32_t io_cnt = one_model_info_ptr->io_cfg_cnt;
        char *cur_io_cfg_ptr = (char *) (one_buf_ptr + one_model_info_ptr->io_cfg_offset);

        for (int32_t io_i = 0; io_i < io_cnt; io_i++) {
            // get op type
            std::string op_type_str(cur_io_cfg_ptr);
            // get the instance method
            creator_ creator_method = m.Opmap[op_type_str];
            // build this op
            std::shared_ptr<op> op_ptr;
            creator_method(op_ptr, cur_io_cfg_ptr);

            // the starting address of the one file is passed in because operators such as conv need to obtain weights and biases through this address
            op_ptr->fill_operands(one_buf_ptr);

            op_in_operands_map[op_ptr] = op_ptr.get()->in_operands;
            auto a = op_out_operands_map[op_ptr];
            op_out_operands_map[op_ptr] = op_ptr.get()->out_operands;

            // update cur_io_cfg_ptr
            cur_io_cfg_ptr += align_buf_size(m.op_cfg_size_map[op_type_str]);
        }

        int32_t node_cnt = one_model_info_ptr->node_cnt;
        char *cur_node_cfg_ptr = (char *) (one_buf_ptr + one_model_info_ptr->node_cfg_offset);

        for (int32_t node_i = 0; node_i < node_cnt; node_i++) {
            // get op type
            std::string op_type_str(cur_node_cfg_ptr);
//            std::cout << "this op type is:" << op_type_str << std::endl;
            // get the instance method
            creator_ creator_method = m.Opmap[op_type_str];
            // build this op
            std::shared_ptr<op> op_ptr;
            creator_method(op_ptr, cur_node_cfg_ptr);

            // the starting address of the one file is passed in because operators such as conv need to obtain weights and biases through this address
            op_ptr->fill_operands(one_buf_ptr);
            op_ptr->prepare(cur_node_cfg_ptr);

            op_in_operands_map[op_ptr] = op_ptr.get()->in_operands;
            auto a = op_out_operands_map[op_ptr];
            op_out_operands_map[op_ptr] = op_ptr.get()->out_operands;

            // update cur_node_cfg_ptr
            cur_node_cfg_ptr += align_buf_size(m.op_cfg_size_map[op_type_str]);
        }

        return 0;
    }

    int mv_init_operands() {
        ONE_MODEL_DESC_S *one_model_info_ptr = (ONE_MODEL_DESC_S *)one_buf_ptr;
//        int32_t *head_ptr = (int32_t *) one_buf_ptr;
//        int32_t *head_ptr = (int32_t *) one_buf_ptr;
        int32_t init_cnt = one_model_info_ptr->init_cnt;
        char *cur_init_info_ptr = (char *) (one_buf_ptr + one_model_info_ptr->init_info_offset);

        for (int32_t init_i = 0; init_i < init_cnt; init_i++) {
            OPERAND_S *operand_ptr = (OPERAND_S *) cur_init_info_ptr;
            std::string init_operands = std::string(operand_ptr->operand_name);

            for (auto &op_in_operands : op_in_operands_map) {
                auto operand_names = op_in_operands.second;
                for (auto operand_name : operand_names) {
                    if (operand_name == init_operands) {
                        init_operands_list.insert(operand_name);
//                        op_in_operands.second.erase(std::remove(op_in_operands.second.begin(), op_in_operands.second.end(), operand_name),
//                                                    op_in_operands.second.end());
//                        operand_names.erase(std::remove(operand_names.begin(), operand_names.end(), operand_name),
//                                            operand_names.end());
//                        std::cout << "erase " << operand_name << " operands" << std::endl;
                    }
                }
            }

            // update cur_init_info_ptr
            int init_size = operand_buf_size(operand_ptr);
            cur_init_info_ptr += align_buf_size(sizeof(OPERAND_S) + init_size);
        }

        int c = 101;

        return 0;
    }

    net(const char *one_path) {
//        std::cout << "start load_one_model" << std::endl;
        if (load_one_model(one_path) != 0) {
            std::cout << "failed: load_one_model failed! " << std::endl;
        }

//        std::cout << "start instantiate net op" << std::endl;
        if (instantiate_op() != 0) {
            std::cout << "failed: instantiate object of op class failed! " << std::endl;
        }

//        std::cout << "start remove the init operands" << std::endl;
        if (mv_init_operands() != 0) {
            std::cout << "failed: remove the init operands failed! " << std::endl;
        }
    };

    int show_operands() {
        std::cout << "========= show op_in_operands_map" << std::endl;
        for (auto op_in_operands : op_in_operands_map) {
            std::cout << "op_type: " << op_in_operands.first.get()->op_type << ", in_operands:"
                      << op_in_operands.first.get()->op_name;
            for (auto operand_name : op_in_operands.second) {
                std::cout << operand_name << ", ";
            }
            std::cout << std::endl;
        }

        std::cout << "=========== show op_out_operands_map" << std::endl;
        for (auto op_out_operands : op_out_operands_map) {
            std::cout << "op_type: " << op_out_operands.first.get()->op_type << ", out_operands:"
                      << op_out_operands.first.get()->op_name;
            for (auto operand_name : op_out_operands.second) {
                std::cout << operand_name << ", ";
            }
            std::cout << std::endl;
        }

        return 0;
    }

    int build_op_pre_node() {
        for (auto op_in_operands : op_in_operands_map) {
            if (op_in_operands.second.empty()) {
                op_pre_node.insert(std::make_pair(op_in_operands.first, NULL));
            }
        }

        for (auto op_out_operands : op_out_operands_map) {
            // traverse each output operands of this op
            for (auto operand_name : op_out_operands.second) {
                for (auto op_in_operands : op_in_operands_map) {
                    for (auto in_operand_name : op_in_operands.second) {
                        if (in_operand_name == operand_name) {
                            op_pre_node[op_in_operands.first].push_back(op_out_operands.first);
                            int bbb = 101;
                        }
                    }
                }
            }
        }

        return 0;
    }

    int build_graph_seq() {
        while (!op_pre_node.empty()) {
            auto a = op_pre_node;
//            printf("op_pre_node.size is %d\n", op_pre_node.size());
            for (auto opa : a) {
                if (opa.second.empty()) {
                    op_exec_order.push_back(opa.first);
                    op_pre_node.erase(opa.first);

                    // erase the previous nodes of op_pre_node
                    for (auto &op_other : op_pre_node) {
                        auto bbbb = op_other.second;
                        op_other.second.erase(std::remove(op_other.second.begin(), op_other.second.end(), opa.first),
                                              op_other.second.end());

                        int c = 101;
                    }

                    break;
                }
//                std::cout << "warning: the no op is no input" << std::endl;
            }
        }
        return 0;
    }

    int show_op_exec_order() {
//        std::cout << "\n=========== start show_op_exec_order ===========" << std::endl;
//        for (auto op : op_exec_order) {
//            std::cout << "op_type: " << op->op_type << ", op_name:" << op->op_name << std::endl;
//        }
//        std::cout << "=========== end show_op_exec_order ===========" << std::endl;
        return 0;
    }

    int gen_operands_list() {
        for (auto operands_vec : op_in_operands_map) {
            for (auto operand : operands_vec.second) {
                if (operand.empty()) {
                    continue;
                }
                operands_list.insert(operand);
            }
        }

        for (auto operands_vec : op_out_operands_map) {
            for (auto operand : operands_vec.second) {
                if (operand.empty()) {
                    continue;
                }
                operands_list.insert(operand);
            }
        }

        for (auto operands_name : operands_list) {
            OPERAND_S operands = {0};
            strcpy(operands.operand_name, operands_name.c_str());
            operand_stu_map[operands_name] = operands;
//            operand_stu_vec.push_back(operands);
        }
        return 0;
    }

    int fill_operand_shape() {

        for (auto op : op_exec_order) {
            std::string op_type_str(op.get()->op_type);
            // if is io op
            if (op_type_str == "io") {
                std::shared_ptr<io> io_ptr = std::dynamic_pointer_cast<io>(op);
                std::string out_operand_name = std::string(io_ptr.get()->io_cfg.operand.operand_name);

                operand_stu_map[out_operand_name] = io_ptr.get()->io_cfg.operand;
            } else {
                op.get()->calc_out_operand_shape(operand_stu_map);
                int a = 101;
            }
        }

        return 0;
    }

    int build_graph() {
//        show_operands();
        build_op_pre_node();
        build_graph_seq();

        show_op_exec_order();

//        std::cout << "===========================================" << std::endl;

        gen_operands_list();

        fill_operand_shape();

        int c = 101;


        return 0;
    }


    extractor *create_exe() {

        extractor *a = new extractor(this);


        return a;

    }

};

// } // namespace one_new


int extractor::new_output_buf() {
//    printf("start new_output_buf\n");

    for (auto operand : net_ptr->operand_stu_map) {
//        std::cout << "new_output_buf for:" << operand.first << std::endl;
        int32_t elem_size = operand_elem_size(&operand.second);
        int32_t buf_size = elem_size * sizeof(float);
//        if (buf_size == 0) {
//            std::cout << "warning: the operand size is 0, check the data type!" << std::endl;
////            buf_size = 1024;  // todo : tmp
//        }
        std::vector<float> operand_vec(elem_size, 0);
        operand_buf_map[operand.first] = operand_vec;
//        std::cout << "end new_output_buf for:" << operand.first << std::endl;

    }

    int a = 101;

    return 0;
}

#include <sys/time.h>

int extractor::prof_impl(std::unordered_map<std::string, std::vector<float>> &io_buf_map, std::unordered_map<std::string, std::string> cfg_info_map) {

    std::unordered_map<std::string, double> op_type_time;
    std::unordered_map<std::string, double> op_name_time;

    struct timeval begin, end;
    double elapsed;

    gettimeofday(&begin, 0);
    for (auto input:io_buf_map) {
        operand_buf_map[input.first] = input.second;
    }
    gettimeofday(&end, 0);

    elapsed = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
    op_type_time["cp_in_data"] = elapsed;
    op_name_time["cp_in_data"] = elapsed;

    std::string ofmap_folder = cfg_info_map["ofmap_folder"];
    // 依次执行 net 中排好序的 op
    int op_idx = 0;
    for (auto op : net_ptr->op_exec_order) {
        gettimeofday(&begin, 0);

        op_idx++;

        double op_st = omp_get_wtime();
//        std::cout << "this op type is: " << op->op_type << ", op->op_name is: " << op->op_name << ", op_idx is: " << op_idx << std::endl;

        op.get()->forward(operand_buf_map, net_ptr->operand_stu_map, net_ptr->init_operands_list);
//        std::cout << "=== end this op type is:" << op->op_type << ", op_idx is: " << op_idx << std::endl;
//
        // dump output, if necessary
        if (cfg_info_map["dump_output4each_node"] == "yes") {
            if (!op.get()->out_operands.empty()) {
                std::string omap_name = op.get()->out_operands[0];
                char* omap_name_c = (char*)omap_name.c_str();
                std::vector<float>* omap_vec = &operand_buf_map[omap_name];
                char* ofmap_ptr = (char *)(&operand_buf_map[omap_name][0]);
                std::string ofmap_name(replace_char(omap_name_c));
                std::string ofmap_path = ofmap_folder + ofmap_name;
                write_bin(ofmap_path.c_str(), omap_vec->size() * sizeof(float), ofmap_ptr);
            }
        }

        gettimeofday(&end, 0);

        double op_ed = omp_get_wtime();
        elapsed = op_ed - op_st;

        op_type_time[std::string(op.get()->op_type)] += elapsed;
        op_name_time[std::string(op.get()->op_name)] += elapsed;
    }

    // 将 out 数据放到 io_buf_map 中
    gettimeofday(&begin, 0);
    for (auto op : net_ptr->op_exec_order) {
        if (std::string(op.get()->op_type) == "io" && op.get()->out_operands.empty()){
            io_buf_map[op.get()->in_operands[0]] = operand_buf_map[op.get()->in_operands[0]];
        }
    }
    gettimeofday(&end, 0);

    elapsed = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
    op_type_time["cp_out_data"] = elapsed;
    op_name_time["cp_out_data"] = elapsed;

    double total_time = 0;
    for (auto op_time : op_name_time) {
        total_time += op_time.second;
    }

    std::cout << "==============  the total_time is: " << total_time << " ==============" << std::endl;
    std::cout << "==============  start show op_type_time ==============" << std::endl;

    for (auto op_time : op_type_time) {
        std::cout << "op name: " << op_time.first << ", time is: " << op_time.second
        << ", ratio is: " << op_time.second * 100 / total_time << " %" << std::endl;
    }

    std::cout << "\n==============  end show the time ==============" << std::endl;

    return 0;
}



int extractor::impl(std::unordered_map<std::string, std::vector<float>> &io_buf_map, std::unordered_map<std::string, std::string> cfg_info_map) {
    for (auto input:io_buf_map) {
        operand_buf_map[input.first] = input.second;
    }

    // 依次执行 net 中排好序的 op
    for (auto op : net_ptr->op_exec_order) {
//        if (std::strcmp(op.get()->op_type, "Flatten") == 0) {
//            std::cout << "this op_type is: " << op.get()->op_type << std::endl;
//            op.get()->forward(operand_buf_map, net_ptr->operand_stu_map, net_ptr->init_operands_list);
//            continue;
//        }
//        std::cout << "this op_type is: " << op.get()->op_type << std::endl;
        op.get()->forward(operand_buf_map, net_ptr->operand_stu_map, net_ptr->init_operands_list);
    }

    // 将 out 数据放到 io_buf_map 中
    for (auto op : net_ptr->op_exec_order) {
        if (std::string(op.get()->op_type) == "io" && op.get()->out_operands.empty()){
            io_buf_map[op.get()->in_operands[0]] = operand_buf_map[op.get()->in_operands[0]];
        }
    }

    return 0;
}




#endif // NET_H