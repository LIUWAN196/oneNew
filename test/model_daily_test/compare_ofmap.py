import sys
import os

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

    if len(sys.argv) < 3:
        print("Usage: python example.py onnx_path ofmap_folder...")
        sys.exit(1)

    model_path = sys.argv[1]
    ofmap_folder = sys.argv[2]
    con_sim_min_threshold = 0.99
    con_sim_max_threshold = 1.01
    if (len(sys.argv) == 4):
        if (sys.argv[3] == 'opt_model'):
            con_sim_min_threshold = 0.98

    # step 1: load onnx model
    ifmap_name = 'model_ifmap.bin'
    model = onnx.load(model_path)
    providers = ['CPUExecutionProvider']

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
    min_cos_sim = 10
    max_cos_sim = -10
    correct_layer_num = 0
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

        # compute similarity
        err = onnx_ref - onenew_data
        err_max = abs(err.reshape(-1)).max()
        err_max_idx = abs(err.reshape(-1)).argmax()
        cos_sim = cosine_similarity(onnx_ref.reshape(-1), onenew_data.reshape(-1))
        if cos_sim < min_cos_sim:
            min_cos_sim = cos_sim
        if cos_sim > max_cos_sim:
            max_cos_sim = cos_sim
        correct_layer_num = correct_layer_num + 1

model_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
if ((min_cos_sim > con_sim_min_threshold) & (max_cos_sim < con_sim_max_threshold) & (not np.isnan(cos_sim))):
    part1 = model_name
    part2 = " daily test SUCCESS"
    part3 = "min cosine similarity of ofmap is: " + str(min_cos_sim)
    print("{:<36} {:<36} {:<56}".format(part1, part2, part3))
else:
    part0 = "==========  "
    part1 = model_name
    part2 = " daily test FAIL"
    part3 = "min cosine similarity of ofmap is: " + str(min_cos_sim)
    print("\033[91m{:<36} {:<30} {:<36} {:<56}\033[0m".format(part0, part1, part2, part3))


