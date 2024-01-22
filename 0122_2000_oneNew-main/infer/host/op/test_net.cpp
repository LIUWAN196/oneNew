#include "../../host/op/relu.h"
#include "net.h"
#include "../../host/op/maxpool.h"
#include "../../host/op/conv.h"
#include "../../host/op/io.h"
#include "../../host/op/add.h"

int main()
{
    net *net_1 = new net("/home/e0006809/Desktop/cc/oneNew-main/infer/tools/abuild/res18.one");

    net_1->build_graph();
    return 0;
}