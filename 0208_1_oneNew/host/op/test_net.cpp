
#include <dlfcn.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include <sys/time.h>
#include <thread>
#include <pthread.h>
#include <functional>
#include <vector>
#include <algorithm>

#include "../../host/op/relu.h"
#include "../../host/op/maxpool.h"
#include "../../host/op/conv.h"
#include "../../host/op/io.h"
#include "../../host/op/add.h"
#include "../../host/op/global_avgpool.h"
#include "../../host/op/flatten.h"
#include "../../host/op/gemm.h"
#include "net.h"


int main()
{

    net *net_1 = new net("/Project/e0006809/0206_2_oneNew/host/op/relu_conv.one");
    net_1->build_graph();
    extractor* b1 = net_1->create_exe();

    std::unordered_map<std::string, std::vector<int8_t>> io_buf_map;

    int in_elem_size = 3 * 8 * 8;
    float input_ptr[in_elem_size];
    for (int i = 0; i < in_elem_size; ++i) {
        input_ptr[i] = i / 45 - 56 + i % 32 + i;
    }

    std::vector<int8_t> in_buf;
    in_buf.assign(input_ptr, input_ptr + in_elem_size);
    int c = in_buf.size();

    float *aa = (float *)(in_buf[0]);

    for (int i = 0; i < 192; ++i) {
        printf("%f  ", aa[i]);
//        aa[i]

    }


    b1->impl(io_buf_map);


    int a = 101;



    return 0;

}

