//
// Created by wanzai on 24-8-19.
//

#ifndef ONENEW_UTILS_C_H
#define ONENEW_UTILS_C_H

inline int32_t align_buf_size(int32_t ori_size)
{
    return (ori_size + 63) & (~63);
}

inline int32_t operand_elem_size(OPERAND_S *cur_operand)
{
    int32_t elem_size = 1;
    for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
        elem_size *= cur_operand->shapes[dim_i];
    }
    return elem_size;
}

inline int32_t operand_buf_size(OPERAND_S *cur_operand)
{
    int32_t buf_size = operand_elem_size(cur_operand) * elem_info_map[cur_operand->data_type].size;
    return buf_size;
}

inline int32_t align_operand_buf_size(OPERAND_S *cur_operand)
{
    return (operand_buf_size(cur_operand) + 63) & (~63);
}


int32_t write_bin(const char *filename, const int64_t size, char *buf) {
    FILE *file_p = NULL;

    file_p = fopen(filename, "w");

    size_t bytes_written = fwrite((void *) buf, 1, size, file_p);

    fclose(file_p);

    return 0;
}

char* replace_char(char* str) {
    int i = 0;
    while (str[i]) {
        if (str[i] == '/') {
            str[i] = '_';
        }
        i++;
    }
    return str;
}

void show_dev_input(BUFFER_INFO_S *params) {
    BASE_CONFIG_S *base_op = (BASE_CONFIG_S *) (params[0].addr);
    LOG_MSG("====================================== start print %s op info ===============================", base_op->op_name);
    LOG_MSG("cur op_type is %s, op_name is %s ", base_op->op_type, base_op->op_name);
    for (int ifmap_i = 0; ifmap_i < base_op->in_operand_num; ++ifmap_i) {
        OPERAND_S *ifmap = (OPERAND_S *) (params[ifmap_i + 1].addr);
        LOG_MSG("the %dth ifmap name is %s, dim is %d, shapes is: [%d, %d, %d, %d, %d, %d, %d, %d]",
                ifmap_i, base_op->in_operand_name[ifmap_i], ifmap->dim_num_of_shapes,
                ifmap->shapes[0], ifmap->shapes[1], ifmap->shapes[2], ifmap->shapes[3],
                ifmap->shapes[4], ifmap->shapes[5], ifmap->shapes[6], ifmap->shapes[7]);
    }
    for (int ofmap_i = 0; ofmap_i < base_op->out_operand_num; ++ofmap_i) {
        OPERAND_S *ofmap = (OPERAND_S *) (params[ofmap_i + 1 + base_op->in_operand_num].addr);
        LOG_MSG("the %dth ofmap name is %s, dim is %d, shapes is: [%d, %d, %d, %d, %d, %d, %d, %d]",
                ofmap_i, base_op->out_operand_name[ofmap_i], ofmap->dim_num_of_shapes,
                ofmap->shapes[0], ofmap->shapes[1], ofmap->shapes[2], ofmap->shapes[3],
                ofmap->shapes[4], ofmap->shapes[5], ofmap->shapes[6], ofmap->shapes[7]);
    }
    LOG_MSG("====================================== end   print %s op info ===============================", base_op->op_name);
    LOG_MSG("");
}

#endif //ONENEW_UTILS_C_H
