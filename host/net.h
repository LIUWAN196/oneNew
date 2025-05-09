#ifndef NET_H
#define NET_H

#include "../../common/nn_common.h"
#include "../../common/utils_cpp.hpp"
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

#ifdef USING_GPU
#include <cuda_runtime.h>
#endif

#ifdef USING_GPU
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#endif

typedef int (*evalaa)(BUFFER_INFO_S *, BUFFER_INFO_S *, BUFFER_INFO_S *);

class net;

class extractor {
public:
    int* public_buf_ptr;
    std::vector<char*> operand_ptr;
    std::unordered_map<std::string, BUFFER_INFO_S> operand_buf_map;

    net *net_ptr;

    extractor(net *net_ptr_) : net_ptr(net_ptr_) {
        new_output_buf();
    };

    int new_output_buf();

    int impl_dump_ofmap(std::unordered_map<std::string, BUFFER_INFO_S> &io_buf_map,
                        std::unordered_map<std::string, std::string> cfg_info_map);

    int impl_tracing(std::unordered_map<std::string, BUFFER_INFO_S> &io_buf_map,
                     std::unordered_map<std::string, std::string> cfg_info_map);

    int impl(std::unordered_map<std::string, BUFFER_INFO_S> &io_buf_map,
             std::unordered_map<std::string, std::string> cfg_info_map);

    int prepare_for_op(std::unordered_map<std::string, BUFFER_INFO_S> &io_buf_map);

    ~extractor() {
        // release public_buf_ptr
        if (public_buf_ptr != nullptr) {
#ifdef USING_GPU
            cudaFree(public_buf_ptr);
#else
            free(public_buf_ptr);
#endif
            public_buf_ptr = nullptr;
        }

        // release operand_ptr
        for (int i = 0; i < operand_ptr.size(); ++i) {
#ifdef USING_GPU
            cudaFree(operand_ptr[i]);
#else
            free(operand_ptr[i]);
#endif
            operand_ptr[i] = nullptr;
        }
    }
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
    std::unordered_map<std::string, OPERAND_S> operand_stu_map;
    extractor *exc_cls;

    int load_one_model(const char *one_path) {
        // step 1: get one file size
        std::ifstream one_file(one_path, std::ios::ate | std::ios::binary);
        int32_t one_file_size = static_cast<int32_t>(one_file.tellg());
        one_file.close();

        // step 2: load one file
#ifdef USING_GPU
        cudaMallocManaged(&one_buf_ptr, one_file_size, cudaMemAttachGlobal);
#else
        one_buf_ptr = (char *) aligned_alloc(32, one_file_size);
#endif

        FILE *file_p = NULL;
        file_p = fopen(one_path, "r");
        if (file_p == NULL) {
            LOG_ERR("can't open the one file");
            return -1;
        }
        size_t ret = fread(one_buf_ptr, sizeof(char), one_file_size, file_p);
        fclose(file_p);

        // 当模型读取进内存后，先通过 one_model_magic_num 魔法数判断是否为 one 模型
        ONE_MODEL_DESC_S *one_model_info_ptr = (ONE_MODEL_DESC_S *) one_buf_ptr;
        if (one_model_info_ptr->one_model_magic_num != ONE_MAGIC_NUM) {
            LOG_ERR("one_model_magic_num error, should be %d, currently is %d, please check .one model.",
                    ONE_MAGIC_NUM, one_model_info_ptr->one_model_magic_num);
            return -1;
        }
        return 0;
    }

    int load_one_buf(char *one_ptr) {
        one_buf_ptr = one_ptr;
        return 0;
    }

    int instantiate_op() {
        Manager &m = Manager::getInstance();

        ONE_MODEL_DESC_S *one_model_info_ptr = (ONE_MODEL_DESC_S *) one_buf_ptr;

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
        ONE_MODEL_DESC_S *one_model_info_ptr = (ONE_MODEL_DESC_S *) one_buf_ptr;

        int32_t init_cnt = one_model_info_ptr->init_cnt;
        char *cur_init_info_ptr = (char *) (one_buf_ptr + one_model_info_ptr->init_info_offset);

        for (int32_t init_i = 0; init_i < init_cnt; init_i++) {
            OPERAND_S *operand_ptr = (OPERAND_S *) cur_init_info_ptr;
            std::string init_operands = std::string(operand_ptr->operand_name);

            for (auto &op_in_operands: op_in_operands_map) {
                auto operand_names = op_in_operands.second;
                for (auto operand_name: operand_names) {
                    if (operand_name == init_operands) {
                        init_operands_list.insert(operand_name);
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
        if (load_one_model(one_path) != 0) {
            LOG_ERR("load_one_model failed!");
        }

        if (instantiate_op() != 0) {
            std::cout << "failed: instantiate object of op class failed! " << std::endl;
        }

        if (mv_init_operands() != 0) {
            std::cout << "failed: remove the init operands failed! " << std::endl;
        }
    };

    net(void *one_buf_ptr) {
        if (load_one_buf((char *) one_buf_ptr) != 0) {
            std::cout << "failed: load_one_buf failed! " << std::endl;
        }

        if (instantiate_op() != 0) {
            std::cout << "failed: instantiate object of op class failed! " << std::endl;
        }

        if (mv_init_operands() != 0) {
            std::cout << "failed: remove the init operands failed! " << std::endl;
        }
    };

    int show_operands() {
        std::cout << "========= show op_in_operands_map" << std::endl;
        for (auto op_in_operands: op_in_operands_map) {
            std::cout << "op_type: " << op_in_operands.first.get()->op_type << ", in_operands:"
                      << op_in_operands.first.get()->op_name;
            for (auto operand_name: op_in_operands.second) {
                std::cout << operand_name << ", ";
            }
            std::cout << std::endl;
        }

        std::cout << "=========== show op_out_operands_map" << std::endl;
        for (auto op_out_operands: op_out_operands_map) {
            std::cout << "op_type: " << op_out_operands.first.get()->op_type << ", out_operands:"
                      << op_out_operands.first.get()->op_name;
            for (auto operand_name: op_out_operands.second) {
                std::cout << operand_name << ", ";
            }
            std::cout << std::endl;
        }

        return 0;
    }

    int build_op_pre_node() {
        for (auto op_in_operands: op_in_operands_map) {
            if (op_in_operands.second.empty()) {
                op_pre_node.insert(std::make_pair(op_in_operands.first, NULL));
            }
        }

        for (auto op_out_operands: op_out_operands_map) {
            // traverse each output operands of this op
            for (auto operand_name: op_out_operands.second) {
                for (auto op_in_operands: op_in_operands_map) {
                    for (auto in_operand_name: op_in_operands.second) {
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
            for (auto opa: a) {
                if (opa.second.empty()) {
                    op_exec_order.push_back(opa.first);
                    op_pre_node.erase(opa.first);

                    // erase the previous nodes of op_pre_node
                    for (auto &op_other: op_pre_node) {
                        auto bbbb = op_other.second;
                        op_other.second.erase(std::remove(op_other.second.begin(), op_other.second.end(), opa.first),
                                              op_other.second.end());
                    }

                    break;
                }
            }
        }
        return 0;
    }

    int show_op_exec_order() {
        return 0;
    }

    int gen_operands_list() {
        for (auto operands_vec: op_in_operands_map) {
            for (auto operand: operands_vec.second) {
                if (operand.empty()) {
                    continue;
                }
                operands_list.insert(operand);
            }
        }

        for (auto operands_vec: op_out_operands_map) {
            for (auto operand: operands_vec.second) {
                if (operand.empty()) {
                    continue;
                }
                operands_list.insert(operand);
            }
        }

        for (auto operands_name: operands_list) {
            OPERAND_S operands = {0};
            strcpy(operands.operand_name, operands_name.c_str());
            operand_stu_map[operands_name] = operands;
        }
        return 0;
    }

    int fill_operand_shape() {

        for (auto op: op_exec_order) {
            std::string op_type_str(op.get()->op_type);
            // if is io op, 先把输出的 ofmap 的 tensor 放到 operand_stu_map 中
            if (op_type_str == "io") {
                std::shared_ptr<io> io_ptr = std::dynamic_pointer_cast<io>(op);
                std::string out_operand_name = std::string(io_ptr.get()->io_cfg.operand.operand_name);

                operand_stu_map[out_operand_name] = io_ptr.get()->io_cfg.operand;
            }
        }

        // 做 shape infer，推理获得 ofmap 的 shape
        for (auto op: op_exec_order) {
            std::string op_type_str(op.get()->op_type);
            // if is io op
            if (op_type_str != "io") {
                op.get()->shape_infer(operand_stu_map);
                int a = 101;
            }
        }

        return 0;
    }

    int build_graph() {
        build_op_pre_node();
        build_graph_seq();

        show_op_exec_order();

        gen_operands_list();

        fill_operand_shape();

        return 0;
    }

    extractor *create_exe() {

        extractor *_exc_cls = new extractor(this);
        exc_cls = _exc_cls;

        return _exc_cls;
    }

        ~net(){
        // release one_buf_ptr
        if (one_buf_ptr != nullptr) {
#ifdef USING_GPU
            cudaFree(one_buf_ptr);
#else
            free(one_buf_ptr);
#endif
            one_buf_ptr = nullptr;
        }

        // release extractor
        if (exc_cls != nullptr) {
            free(exc_cls);
            exc_cls = nullptr;
        }
    }

};

int extractor::new_output_buf() {
    // 开辟一块公共空间，不允许在设备侧 malloc
    ONE_MODEL_DESC_S *one_model = (ONE_MODEL_DESC_S *) net_ptr->one_buf_ptr;
    PUBLIC_BUF_INFO_S *publice_buf = &one_model->useful_info.public_buf_info;

#ifdef USING_GPU
    int* public_buf_ptr = nullptr;
    CUDA_CHECK(cudaMallocManaged(&public_buf_ptr, publice_buf->public_buf_size, cudaMemAttachGlobal));
    publice_buf->public_buf_ptr = (int64_t)public_buf_ptr;
#else
    publice_buf->public_buf_ptr = (int64_t) aligned_alloc(32, publice_buf->public_buf_size);
#endif

    int64_t total_buf_size = 0;
    int32_t op_idx = 0;
    for (auto operand: net_ptr->operand_stu_map) {
        if (operand.second.not_need_buf == TRUE) {
            continue;
        }
        int32_t elem_size = operand_elem_size(&operand.second);
        int32_t buf_size = elem_size * sizeof(float);

#ifdef USING_GPU
        int* cur_ptr = nullptr;
        CUDA_CHECK(cudaMallocManaged(&cur_ptr, buf_size, cudaMemAttachGlobal));
        int64_t cur_operand_ptr = (int64_t)cur_ptr;
#else
        int64_t cur_operand_ptr = (int64_t) aligned_alloc(32, buf_size);
#endif
        operand_buf_map[operand.first] = {cur_operand_ptr, elem_size, buf_size};
    }

    return 0;
}

#include <sys/time.h>

int extractor::impl_dump_ofmap(std::unordered_map<std::string, BUFFER_INFO_S> &io_buf_map,
                               std::unordered_map<std::string, std::string> cfg_info_map) {

    std::unordered_map<std::string, double> op_type_time;
    std::unordered_map<std::string, double> op_name_time;

    struct timeval begin, end;
    double elapsed;

    gettimeofday(&begin, 0);

    gettimeofday(&end, 0);

    elapsed = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
    op_type_time["cp_in_data"] = elapsed;
    op_name_time["cp_in_data"] = elapsed;

    std::string ofmap_folder = cfg_info_map["ofmap_folder"];
    // 依次执行 net 中排好序的 op
    int op_idx = 0;
    for (auto op: net_ptr->op_exec_order) {
        op_idx++;

        double op_st = omp_get_wtime();

        op.get()->forward(operand_buf_map, net_ptr->operand_stu_map, net_ptr->init_operands_list);

        double op_ed = omp_get_wtime();
        elapsed = op_ed - op_st;

        op_type_time[std::string(op.get()->op_type)] += elapsed;
        op_name_time[std::string(op.get()->op_name)] += elapsed;

        if (!op.get()->out_operands.empty()) {
            std::string omap_name = op.get()->out_operands[0];
            char *omap_name_c = (char *) omap_name.c_str();
            char *ofmap_ptr = (char *) operand_buf_map[omap_name].addr;
            int64_t buf_size = operand_buf_map[omap_name].buf_size;
            std::string ofmap_name(replace_char(omap_name_c));
            std::string ofmap_path = ofmap_folder + ofmap_name;
            write_bin(ofmap_path.c_str(), buf_size, ofmap_ptr);
        }

    }

    // 将 out 数据放到 io_buf_map 中
    gettimeofday(&begin, 0);
    for (auto op: net_ptr->op_exec_order) {
        if (std::string(op.get()->op_type) == "io" && op.get()->out_operands.empty()) {
            io_buf_map[op.get()->in_operands[0]] = operand_buf_map[op.get()->in_operands[0]];
        }
    }
    gettimeofday(&end, 0);

    elapsed = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
    op_type_time["cp_out_data"] = elapsed;
    op_name_time["cp_out_data"] = elapsed;

    double total_time = 0;
    for (auto op_time: op_name_time) {
        total_time += op_time.second;
    }

    std::vector<std::pair<std::string, double>> vec(op_type_time.begin(), op_type_time.end());
    std::sort(vec.begin(), vec.end(),
              [](const std::pair<std::string, double> &a, const std::pair<std::string, double> &b) {
                  return a.second > b.second; // 降序排序
              });

    std::cout << "======================  the total_time is: " << total_time << " ======================" << std::endl;
    const int align_pixel = 24;
    std::cout << std::setw(align_pixel) << std::left << "op_type" << std::setw(align_pixel) << "op_time (ms)"
              << std::setw(align_pixel) << "op_time_ratio (%)" << std::setw(align_pixel) << std::endl;
    for (auto op_time: vec) {
        std::cout << std::setw(align_pixel) << std::left << op_time.first << std::setw(align_pixel)
                  << op_time.second * 1000
                  << std::setw(align_pixel) << op_time.second * 100 / total_time << std::setw(align_pixel) << std::endl;
    }
    std::cout << "======================  end show op_type time ======================" << std::endl;

    return 0;
}

int extractor::impl_tracing(std::unordered_map<std::string, BUFFER_INFO_S> &io_buf_map,
                            std::unordered_map<std::string, std::string> cfg_info_map) {

    std::vector<std::vector<std::string>> op_with_tracing;
    op_with_tracing.resize(net_ptr->op_exec_order.size() + 1);
    const int inner_size = 5;   // op name / op type / op start / op end / computation
    for (auto &row: op_with_tracing) {
        row.resize(inner_size);
    }
    op_with_tracing[0].resize(12);
    op_with_tracing[1].resize(12);

    op_with_tracing[0][0] = "op_name";
    op_with_tracing[0][1] = "op_type";
    op_with_tracing[0][2] = "op_st";
    op_with_tracing[0][3] = "op_ed";
    op_with_tracing[0][4] = "op_computation";

    op_with_tracing[0][6] = "cpu";
    op_with_tracing[0][7] = "cpu_hw_info";
    op_with_tracing[0][8] = "gpu";
    op_with_tracing[0][9] = "gpu_hw_info";
    op_with_tracing[0][10] = "hw_computing_power (GOPS)";
    op_with_tracing[0][11] = "model name";

    op_with_tracing[1][6] = cfg_info_map["cpu"];
    op_with_tracing[1][7] = cfg_info_map["cpu_hw_info"];
    op_with_tracing[1][8] = cfg_info_map["gpu"];
    op_with_tracing[1][9] = cfg_info_map["gpu_hw_info"];
    op_with_tracing[1][10] = cfg_info_map["hw_computing_power (GOPS)"];
    op_with_tracing[1][11] = cfg_info_map["model name"];

    std::string ofmap_folder = cfg_info_map["ofmap_folder"];
    // 依次执行 net 中排好序的 op
    int op_idx = 0;
    double model_st_tamp = omp_get_wtime();
    for (auto op: net_ptr->op_exec_order) {

        double op_st = omp_get_wtime();

        op.get()->forward(operand_buf_map, net_ptr->operand_stu_map, net_ptr->init_operands_list);

        double op_ed = omp_get_wtime();
        op_with_tracing[1 + op_idx][0] = op.get()->op_name;
        op_with_tracing[1 + op_idx][1] = op.get()->op_type;

        double st_stamp = (op_st - model_st_tamp) * 1e6;
        double ed_stamp = (op_ed - model_st_tamp) * 1e6;
        op_with_tracing[1 + op_idx][2] = std::to_string(st_stamp);
        op_with_tracing[1 + op_idx][3] = std::to_string(ed_stamp);
        op_with_tracing[1 + op_idx][4] = std::to_string(op->get_computation());
        op_idx++;
    }

    // 将 out 数据放到 io_buf_map 中
    for (auto op: net_ptr->op_exec_order) {
        if (std::string(op.get()->op_type) == "io" && op.get()->out_operands.empty()) {
            io_buf_map[op.get()->in_operands[0]] = operand_buf_map[op.get()->in_operands[0]];
        }
    }

    std::string csv_file = cfg_info_map["tracing_csv_path"];
    std::ofstream file(csv_file.c_str());
    for (const auto &row: op_with_tracing) {
        for (size_t i = 0; i < row.size(); i++) {
            file << row[i];
            if (i < row.size() - 1) file << ",";
        }
        file << "\n";
    }

    return 0;
}

int extractor::impl(std::unordered_map<std::string, BUFFER_INFO_S> &io_buf_map,
                    std::unordered_map<std::string, std::string> cfg_info_map) {
    // 依次执行 net 中排好序的 op
    for (auto op: net_ptr->op_exec_order) {
        op.get()->forward(operand_buf_map, net_ptr->operand_stu_map, net_ptr->init_operands_list);
    }

    // 将 out 数据放到 io_buf_map 中
    for (auto op: net_ptr->op_exec_order) {
        if (std::string(op.get()->op_type) == "io" && op.get()->out_operands.empty()) {
            io_buf_map[op.get()->in_operands[0]] = operand_buf_map[op.get()->in_operands[0]];
        }
    }

    return 0;
}

int extractor::prepare_for_op(std::unordered_map<std::string, BUFFER_INFO_S> &io_buf_map) {
    for (auto input: io_buf_map) {
        operand_buf_map[input.first] = input.second;
    }

    // 每个 layer 依次准备 op 需要的数据，包括 params / ifmap / ofmap
    for (auto op: net_ptr->op_exec_order) {
        op.get()->rt_prepare(operand_buf_map, net_ptr->operand_stu_map, net_ptr->init_operands_list);
    }

    return 0;
}

#endif // NET_H
