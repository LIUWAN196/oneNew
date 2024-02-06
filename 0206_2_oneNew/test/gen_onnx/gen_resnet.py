# 这里存放了原始的 res18 和我自己魔改的 res18 模型结构

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import onnx
from onnx import shape_inference

device = torch.device("cuda")



# 定义的残差模块
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1) -> None:
        super(ResBlock, self).__init__()

        # 这里定义了残差块内连续的2个卷积层
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)

        self.downsample = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.downsample(x)
        out = self.relu(out)
        return out



# 原始的 res18 结构
class liu_res18(nn.Module):
    def __init__(self, ResBlock, num_classes=1000) -> None:
        super(liu_res18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:

            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)


if __name__ == '__main__':
    # # 初步导出 onnx 看看，模型是否能正确执行
    # tmp_model = liu_res12(ResBlockLiu)
    #
    # tmp_model18 = liu_res18(ResBlock)

    input = torch.randn(1, 3, 224, 224)


    tmp_model_pt = './tmp_model.pt'
    tmp_2onnx_model = liu_res18(ResBlock)
    tmp_2onnx_model.eval()
    torch.save(tmp_2onnx_model, tmp_model_pt)

    # 通过 .pt 将网络保存为 onnx 形式
    model_path = torch.load('./tmp_model.pt')
    tmp_onnx_path = "tmp_model.onnx"
    torch.onnx.export(model_path,   # pytorch网络模型
                      input,        # 随机的模拟输入
                      tmp_onnx_path,    # 导出的onnx文件位置
                      export_params=True,   # 是否导出网络训练好的参数，一般都为 true，保持训练好的参数
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input_0'],  # 为静态网络图中的输入节点设置别名，在进行onnx推理时，将input_names字段与输入数据绑定
                      output_names=['output_0']
                      )

    # 保存模型的中间节点的 shape 信息
    tmp_onnx = "resnet18.onnx"
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(tmp_onnx_path)), tmp_onnx)



