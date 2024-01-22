#include "../../host/op/relu.h"
#include "net.h"
#include "../../host/op/maxpool.h"
#include "../../host/op/conv.h"
#include "../../host/op/io.h"

int main()
{
    net *net_1 = new net("/Project/e0006809/00_one_new/oneNew-main/infer/tools/show_one/44.one");

    net_1->build_graph();
    return 0;
}