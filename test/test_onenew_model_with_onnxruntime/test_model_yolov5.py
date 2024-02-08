
import collections
import onnxruntime
import onnxruntime as ort
import onnx
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
# from onnxruntime_extensions import PyOp, onnx_op
import numpy as np
import os

def cosine_similarity(vec1, vec2):
    # 确保两个向量具有相同的维度
    assert len(vec1) == len(vec2), "两个向量的维度必须相同"

    # 计算两个向量的点积
    dot_product = np.dot(vec1, vec2)

    # 计算两个向量的模（长度）
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)

    # 计算余弦相似度
    # 如果norm_a或norm_b为0，则直接返回0（因为无法除以0）
    if norm_a == 0 or norm_b == 0:
        return 0
    else:
        return dot_product / (norm_a * norm_b)


if __name__ == '__main__':
    # step 1: load onnx model
    model_path = '/home/wanzai/桌面/oneNew/model_and_cfg_zoo/model/onnx_model/yolov5s.onnx'
    model = onnx.load(model_path)
    providers = ['CPUExecutionProvider']

    # step 2: load input data
    img = np.fromfile("/home/wanzai/桌面/oneNew/ofmap4debug/yolov5s/model_ifmap.bin", dtype=np.float32)
    img = img.reshape(-1, 3, 640, 640)

    # step 3: model infer to obtain the output of each layer
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())

    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    outputs = [x.name for x in ort_session.get_outputs()]

    ort_out_desc = collections.OrderedDict(zip(outputs, ort_outs))

    # 先不检验最终的输出结果
    out_tensor_list = []  # yolo v5s
    # out_tensor_list = ['/model.24/m.0/Conv_output_0', '/model.24/m.1/Conv_output_0', '/model.24/m.2/Conv_output_0']  # yolo v5s
    # out_tensor_list = ['output', '456', '458']  # yolo v5m
    onenew_bash_folder = '/home/wanzai/桌面/oneNew/ofmap4debug/yolov5s/'

    # step 4: 依次检验每层的 output 的误差是否符合预期
    correct_layer_num = 0
    skip_num = 0
    for (key, value) in ort_out_desc.items():
        if key in out_tensor_list:
            continue
        onnx_ref = value
        replaced_key = key.replace('/', '_')
        onenew_data_path = onenew_bash_folder + replaced_key
        if os.path.exists(onenew_data_path):
            onenew_data = (np.fromfile(onenew_data_path, dtype=np.float32)).reshape(onnx_ref.shape)
        else:
            skip_num = skip_num + 1
            continue

        # compute similarity
        err = onnx_ref - onenew_data
        err_max = abs(err.reshape(-1)).max()
        err_max_idx = abs(err.reshape(-1)).argmax()
        cos_sim = cosine_similarity(onnx_ref.reshape(-1), onenew_data.reshape(-1))
        if cos_sim < 0.98:
            print("=========== the output of {} have big error, it is the {}th tensor, please check, cos_sim is {} ".format(key, correct_layer_num, cos_sim))
            # break
        else:
            print("the output of {} have small error, it is the {}th tensor, cos_sim is {} ".format(key, correct_layer_num, cos_sim))
        correct_layer_num = correct_layer_num + 1

    print("======================  check the final output  ===========================")
    # step 5: 检验最终的输出层的 output 的误差是否符合预期
    correct_layer_num = 0
    for (key, value) in ort_out_desc.items():
        if key not in out_tensor_list:
            continue
        if 'Constant' in key:
            continue
        onnx_ref = value
        replaced_key = key.replace('/', '_')
        onenew_data = (np.fromfile(onenew_bash_folder + replaced_key, dtype=np.float32)).reshape(onnx_ref.shape)

        # compute similarity
        err = onnx_ref - onenew_data
        err_max = abs(err.reshape(-1)).max()
        err_max_idx = abs(err.reshape(-1)).argmax()
        cos_sim = cosine_similarity(onnx_ref.reshape(-1), onenew_data.reshape(-1))
        if cos_sim < 0.98:
            print("=========== the output of {} have big error, it is the {}th tensor, please check, cos_sim is {} ".format(key, correct_layer_num, cos_sim))
            # break
        else:
            print("the output of {} have small error, it is the {}th tensor, cos_sim is {} ".format(key, correct_layer_num, cos_sim))
        correct_layer_num = correct_layer_num + 1


    # step 5: 结束
    a = 101
