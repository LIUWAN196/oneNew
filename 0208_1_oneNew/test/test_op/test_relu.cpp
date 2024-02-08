#include "../../host/op/relu.h"
// #include "../../host/op/MaxPool.h"
// #include "../../host/op/Conv.h"

int main()
{
    Manager &m = Manager::getInstance();

    creator_ creator_relu = m.Opmap["Relu"];

    std::shared_ptr<op> op_ptr;

    RELU_CONFIG_S relu_cfg;
    // memset(&relu_cfg, 0, sizeof(RELU_CONFIG_S));
    std::string op_type = "Relu";
    strcpy(relu_cfg.op_type, op_type.c_str());

    std::string op_name = "fourth_name";
    strcpy(relu_cfg.op_name, op_name.c_str());

    creator_relu(op_ptr, (char *)(&relu_cfg));

    // C++智能指针父类和子类之间的转换   https://blog.csdn.net/weixin_46222091/article/details/104832221
    std::shared_ptr<Relu> relu_ptr = std::dynamic_pointer_cast<Relu>(op_ptr);
    std::cout << "the Relu op name is: " << relu_ptr->relu_cfg.op_name << std::endl;

    relu_ptr->fill_operands(char *one_buf_ptr);
    
    printf("start of show the all op\n");
    for (auto i : m.Opmap)
    {
        std::cout << i.first << ", and " << i.second << std::endl;
    }
    printf("end of show the all op\n");

    OPERAND_S in;
    in.shape.N = 1;
    in.shape.C = 2;
    in.shape.H = 3;
    in.shape.W = 4;

    OPERAND_S out;
    out.shape.N = 1;
    out.shape.C = 2;
    out.shape.H = 3;
    out.shape.W = 4;

    BUFFER_GROUP_S params;
    BUFFER_GROUP_S input;
    BUFFER_GROUP_S output;

    params.buf_info[0].addr = (int64_t)(&relu_cfg);
    params.buf_info[0].size = sizeof(relu_cfg);

    params.buf_info[1].addr = (int64_t)(&in);
    params.buf_info[1].size = sizeof(OPERAND_S);

    params.buf_info[2].addr = (int64_t)(&out);
    params.buf_info[2].size = sizeof(OPERAND_S);

    int8_t indata[100];
    for (int i = 0; i < 100; ++i)
    {
        indata[i] = i - 23;
    }
    input.buf_info[0].addr = (int64_t)(&indata);

    int8_t outdata[100];
    output.buf_info[0].addr = (int64_t)(&outdata);

    op_ptr->prepare(&params);
    op_ptr->forward(&params, &input, &output);

    printf("shoe the out put data\n");
    for (int i = 0; i < 100; ++i)
    {
        printf("%d  ", outdata[i]);
    }

    return 0;
}