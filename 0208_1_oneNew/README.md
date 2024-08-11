# oneNew

## 仓库目录结构

```shell {.line-numbers}
oneNew
├── infer                           # 推理框架目录
│   ├── common                      # 公共头文件以及 utils 文件暂时存放目录
│   │   ├── log.h
│   │   └── nn_common.h
│   ├── device                      # 算子真正执行的目录
│   │   ├── cuda                    # cuda 算子
│   │   └── x86                     # x86 算子
│   ├── host                        # host 侧目录
│   │   ├── manager                 # 全局管理单例以及 Net 类
│   │   └── op                      # 算子类
│   ├── test                        # 测试目录
│   │   ├── gen_onnx  
│   │   └── py_test.py
│   └── tools                       # 工具目录
│       ├── onnx2one                # 将 onnx 模型转为 one 模型
│       ├── quantize                # 读取 one 模型并对其进行量化
│       └── show_one                # 读取并打印 one 模型内容
└── README.md
```

## tools 工具说明
### onnx2one  
使用方式：  
```shell {.line-numbers}
oneNew-main/infer/tools/onnx2one/build$ ./onnx2one ../example0.onnx ../example0.one
```
目的：将 example0.onnx 模型转为 example0.one 模型。  
为了后续适配 onnx、pnnx、pytorch、tf 等模型。在本项目中定义了一个新的模型存储结构 .one，这个存储结构如下所示：  

|                      |                                             |  |  |  |  |
| :------------------------: | :---------------------------------------------: | :------: | :------: | :------: | :------: |
|      生成该 .one 文件的时间      |      模型的 op 个数       |  第一个 op 的 cfg 起始位置  |    模型的 init 参数个数     |    第一个 init 参数的起始位置     |    ......     |  
|      第一个 op 的 cfg 结构体      |      第二个 op 的 cfg 结构体       |  第三个 op 的 cfg 结构体  |    |    ......     |  
|      第一个 init 参数信息      |      第二个 init 参数信息       |  第三个 init 参数信息  |    |    ......     |   
|      其余必需的参数 a      |      其余必需的参数 b       |  其余必需的参数 c  |    |    ......     |  

目前这里有 3 类信息，依次为第一行的头部信息，标识时间戳、模型的 op 个数、init 参数个数等信息。数据类型均为 int32_t。并且每行都是 64 字节对齐的，所以如果有需要，可以在第一行后面加其他需要的信息。  
第二行是 cfg 的信息，这里将 onnx 模型中的 attributes 参数，例如 kernel_shape、strides 等信息填充到对应算子的 config 中。后续直接将这些参数用于算子的真正执行，不需要再进行转换。这里每个算子的 cfg 也是 64 字节对齐的。  
第三行是 init 参数信息，例如 conv 的 weight 和 bias 参数就存放在这里。对于每个 init 参数，先会填充 OPERAND_S 结构体，用于保存本个 init 参数在 onnx 中的名称、数据的 shape、数据类型为 float 还是 int8_t 等信息，在填充完 OPERAND_S 结构体后，紧接着就将真正的权重和偏置等信息依次往后面填充即可。同理这里每个 init 参数也是 64 字节对齐的。  
第四、五、六、... 行：备用，可以用来存放 input、output 以及其余信息。如果需要存放此类信息，只需要在第一行填写 input/output 个数，在 one 这个文件中存放的起始偏移位置，即可在第四行添加对应信息即可。

### show_one  
使用方式：  
```shell {.line-numbers}
oneNew-main/infer/tools/show_one/build$ ./show_one ../example0.one yes
```
目的：一个朴实无华的功能，输入 example0.one 文件，打印这个 one 文件对应的模型参数信息，例如 op 的个数、每个 op 的参数信息等。如果像上面这样在 example0.one 后还输入了 yes，则会将 init 参数的每个权重都打印出来，用于检查 init 参数的权重是否正确，如果不需要检查的话，则只输入 example0.one 路径即可。

