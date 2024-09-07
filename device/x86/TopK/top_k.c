#include "top_k.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "string.h"
#include "stdint.h"
#include "float.h"


int eval(BUFFER_INFO_S *params, BUFFER_INFO_S *inputs, BUFFER_INFO_S *outputs) {
//    show_dev_input(params);
//    printf("this is x86 mul start\n");
    TOP_K_CONFIG_S *cfg = (TOP_K_CONFIG_S *) (params[0].addr);

    float *input_ptr = (float *) (inputs[0].addr);
    int32_t *out_indices_ptr = (int32_t *) (outputs[1].addr);

    OPERAND_S *in_tensor = (OPERAND_S *) (params[1].addr);
    OPERAND_S *out_indices_tensor = (OPERAND_S *) (params[3].addr);

    int32_t in_elem_size = operand_elem_size(in_tensor);
    int32_t out_elem_size = operand_elem_size(out_indices_tensor);

    float *ifmap_ptr = (float *) malloc(in_elem_size * sizeof(float ));
    memcpy(ifmap_ptr, input_ptr, in_elem_size * sizeof(float ));

    for (int i = 0; i < out_elem_size; ++i) {
        float max_val = -FLT_MAX;
        int32_t indices = 0;
        for (int j = 0; j < in_elem_size; ++j) {
            indices = ifmap_ptr[j] > max_val ? j : indices;
            max_val = ifmap_ptr[j] > max_val ? ifmap_ptr[j] : max_val;
        }
        // 将本次找到的最大值所在位置的数据置为 float 最小值，避免影响下一找最大值
        ifmap_ptr[indices] = -FLT_MAX;

        // 将本次找到的最大值的索引保存下来
        out_indices_ptr[i] = indices;
//        LOG_DBG("top %d  idx is %d", i, indices);
    }

    free(ifmap_ptr);
//    LOG_DBG("这是 topk 算子的结束");
//    LOG_DBG("这是 topk 算子的结束");

    return 0;
}


