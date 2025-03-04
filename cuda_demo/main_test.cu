#include "main_test.h"
#include <iostream>
#include "vector"

void printHelp(const std::string &programName) {
    std::cout << "Usage: " << programName << " [reduce] [transpose] [gemm] or [all]\n";
}

int main(int argc, char **argv) {
    std::vector<std::string> args(argv, argv + argc);
    if (argc != 2 || args[1] == "-h" || args[1] == "--h") {
        printHelp(args[0]);
        return 0;
    }

    if (args[1] == "reduce") {
        reduce_test();
    } else if (args[1] == "transpose") {
        transpose_test();
    } else if (args[1] == "gemm") {
        gemm_test();
    }  else if (args[1] == "all") {
        reduce_test();
        transpose_test();
        gemm_test();
    } else {
        printHelp(args[0]);
    }

    return 0;
}

/* 在 960 显卡上，性能如下所示：
 * reduce_sum: using time is 62.880 ms, band width is 68.305 GB/s
 * reduce_sum op test correct, CUDA output is 1048576.000000, ref is 1048576.000000, and the error is 0.000000
 *
 * transpose: using time is 112.042 ms, band width is 38.333 GB/s
 * transpose op test correct, the max error is 0.000000
 *
 * gemm: using time is 691.177 ms, computing power is  1.591 TFLOPS,
 * gemm op test correct, the max error is 0.093750
 */
