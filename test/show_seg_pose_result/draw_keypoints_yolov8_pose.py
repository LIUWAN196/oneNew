import cv2
import numpy as np
from PIL import Image, ImageDraw

pathImg = "/home/wanzai/桌面/oneNew/imgs/src_img/xinxin2.jpg"
a = cv2.imread(pathImg)

width = 640
height = 640
len_result = 7
len_result_bisic = 7
point_colors = ["red","blue"]

img = Image.open(pathImg)
image_width, image_height = img.size
draw = ImageDraw.Draw(img)


box_info = np.fromfile("/home/wanzai/桌面/oneNew/cmake-build-debug/example/pose_ofmap.bin", dtype=np.float32)
box_num = 1

key_point = box_info[7:7+51]
key_point = key_point.reshape(-1, 3)

for i in range(0, box_num):
    # 定义方框的四个浮点值
    (x1, y1, x2, y2) = (
        box_info[i * 7 + 3] * image_width,
        box_info[i * 7 + 4] * image_height,
        box_info[i * 7 + 5] * image_width,
        box_info[i * 7 + 6] * image_height
    )

    len = 30
    for j in range(0, 17):
        if key_point[j, 2] > 0.5:
            x_point = int(key_point[j, 0] * image_width)
            y_point = int(key_point[j, 1] * image_height)
            draw.ellipse((x_point - len, y_point - len, x_point + len, y_point + len), outline="red", width=12)

    # 在原始图片上绘制方框
    draw.rectangle([(x1, y1), (x2, y2)], outline="yellow", width=12)

img.save('/home/wanzai/桌面/oneNew/imgs/dst_img/xinxin2_yolov8m_pose.jpg')