import cv2
import numpy as np
from PIL import Image, ImageDraw


pathImg = "/home/wanzai/桌面/oneNew/imgs/src_img/person&car.png"

a = cv2.imread(pathImg)

width = 640
height = 640
len_result = 7
len_result_bisic = 7
point_colors = ["red","blue"]

img = Image.open(pathImg)
image_width, image_height = img.size
draw = ImageDraw.Draw(img)

box_info = np.fromfile("/home/wanzai/桌面/oneNew/cmake-build-debug/example/detect_ofmap.bin", dtype=np.float32)
box_num = 20

for i in range(0, box_num):
    # 定义方框的四个浮点值
    (x1, y1, x2, y2) = (
        box_info[i * 7 + 3] * image_width,
        box_info[i * 7 + 4] * image_height,
        box_info[i * 7 + 5] * image_width,
        box_info[i * 7 + 6] * image_height
    )
    # 在原始图片上绘制方框
    draw.rectangle([(x1, y1), (x2, y2)], outline="yellow", width=2)

img.save('/home/wanzai/桌面/oneNew/imgs/dst_img/person&car_yolov5m_detect.jpg')