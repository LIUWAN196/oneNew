
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
import sys
import os
import math

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

    if len(sys.argv) != 3:
        print("Usage: python example.py onnx_path  ofmap_folder...")
        sys.exit(1)

    model_path = sys.argv[1]
    ofmap_folder = sys.argv[2]

    # step 1: load onnx model
    ifmap_name = 'model_ifmap.bin'
    model = onnx.load(model_path)
    providers = ['CPUExecutionProvider']

    # graph = model.graph
    # node = graph.node
    #
    # for i in range(len(node)):
    #     # 修改Transpose_16 维度参数
    #     if node[i].op_type == 'MatMul':
    #         node[i].name = 'MatMul_' + str(i)
    #     elif node[i].op_type == 'Split':
    #         node[i].name = 'Split_' + str(i)
    #
    # onnx.save(model, "/home/wanzai/桌面/model_and_cfg_and_ofmap_folder/model_and_cfg_zoo/model/todo/vit_t.onnx")




    # step 2: load input data
    img = np.fromfile(ofmap_folder + ifmap_name, dtype=np.float32)
    ifmap_elem_size = img.shape[0]
    ifmap_h_w = np.sqrt(ifmap_elem_size // 3)
    img = img.reshape(-1, 3, int(ifmap_h_w), int(ifmap_h_w))

    # step 3: model infer to obtain the output of each layer
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())

    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    outputs = [x.name for x in ort_session.get_outputs()]

    ort_out_desc = collections.OrderedDict(zip(outputs, ort_outs))

    # step 4: 依次检验每层的 output 的误差是否符合预期
    min_cosmin = 10
    correct_layer_num = 0
    total_layer_num = 0
    fail_layer_num = 0
    for (key, value) in ort_out_desc.items():
        if 'Constant' in key:
            continue
        onnx_ref = value
        replaced_key = key.replace('/', '_')
        onenew_data_path = ofmap_folder + replaced_key
        onenew_data = 1
        if os.path.exists(onenew_data_path):
            onenew_data = (np.fromfile(onenew_data_path, dtype=np.float32)).reshape(onnx_ref.shape)
        else:
            continue

        if (key == "/vit/encoder/layer.0/layernorm_before/Div_output_0") :
            a = 101
            b = 102

        # compute similarity
        err = onnx_ref - onenew_data
        err_max = abs(err.reshape(-1)).max()
        err_max_idx = abs(err.reshape(-1)).argmax()
        cos_sim = cosine_similarity(onnx_ref.reshape(-1), onenew_data.reshape(-1))
        if (cos_sim < 0.9998) | (math.isnan(cos_sim)):
            print("=========== the output of {} have big error, it is the {}th tensor, please check, cos_sim is {} ".format(key, correct_layer_num, cos_sim))
            fail_layer_num = fail_layer_num + 1
            # break
        else:
            print("the output of {} have small error, it is the {}th tensor, cos_sim is {} ".format(key, correct_layer_num, cos_sim))
        correct_layer_num = correct_layer_num + 1
        total_layer_num = total_layer_num + 1

print("end check, total_layer_num is {}, fail_layer_num is {} ".format(total_layer_num, fail_layer_num))


