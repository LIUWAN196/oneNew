
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

# 加载 imageNet 的时候需要进行 resize 和 normalize 操作
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 用于在自己测试单张图片时，读取图片并进行 resize 和 normalize 操作
def pre_image(image_file):
    input_image = Image.open(image_file)
    input_tensor = transform_img(input_image)
    inputs = input_tensor.unsqueeze(0)

    return inputs

if __name__ == '__main__':


    clip_img_onnx = "/media/wanzai/File/model_and_cfg_and_ofmap_folder/model_and_cfg_zoo/model/todo/clip_img.onnx"
    clip_txt_onnx = "/media/wanzai/File/model_and_cfg_and_ofmap_folder/model_and_cfg_zoo/model/todo/clip_txt.onnx"

    # img_ifmap = np.array(pre_image("/home/wanzai/桌面/clip_model-img/cat.jpeg"))
    # img_ifmap = np.array(pre_image("/home/wanzai/桌面/clip_model-img/dark_horse.jpeg"))
    # img_ifmap = np.array(pre_image("/home/wanzai/桌面/clip_model-img/dog.png"))
    img_ifmap = np.array(pre_image("/home/wanzai/桌面/clip_model-img/truck.jpeg"))
    # img_ifmap = np.array(pre_image("/home/wanzai/桌面/clip_model-img/white_horse.jpg"))

    txt_seq = np.array([[49406,   589,   533,   320,  1125,   539,   320,  1579,  4558, 49407,
                         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                         0,     0,     0,     0,     0,     0,     0]]).astype(np.int32)

    img_model = onnx.load(clip_img_onnx)
    img_ort_session = onnxruntime.InferenceSession(img_model.SerializeToString())

    img_ort_inputs = {img_ort_session.get_inputs()[0].name: img_ifmap}
    img_ort_outs = img_ort_session.run(None, img_ort_inputs)
    img_outputs = [x.name for x in img_ort_session.get_outputs()]

    img_ort_out_desc = collections.OrderedDict(zip(img_outputs, img_ort_outs))

    img_final_ofmp = img_ort_out_desc[img_outputs[0]]
    # print(img_final_ofmp)
    img_final_ofmp_div = np.sqrt(np.sum(img_final_ofmp * img_final_ofmp))
    img_final_ofmp = img_final_ofmp / img_final_ofmp_div * 100


    txt_model = onnx.load(clip_txt_onnx)
    txt_ort_session = onnxruntime.InferenceSession(txt_model.SerializeToString())

    txt_ort_inputs = {txt_ort_session.get_inputs()[0].name: txt_seq}
    txt_ort_outs = txt_ort_session.run(None, txt_ort_inputs)
    txt_outputs = [x.name for x in txt_ort_session.get_outputs()]

    txt_ort_out_desc = collections.OrderedDict(zip(txt_outputs, txt_ort_outs))

    txt_final_ofmp = txt_ort_out_desc[txt_outputs[0]]

    before_logits = np.sum(img_final_ofmp * txt_final_ofmp)

    # cos_sim = cosine_similarity(img_final_ofmp.reshape(-1), txt_final_ofmp.reshape(-1))
    print("before_logits is: ", before_logits)

    a = 101
