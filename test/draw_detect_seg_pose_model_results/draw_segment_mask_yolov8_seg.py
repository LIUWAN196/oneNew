import cv2
import numpy as np
from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt

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

box_info = np.fromfile("/home/wanzai/桌面/oneNew/cmake-build-debug/example/segment_ofmap.bin", dtype=np.float32)

mask_info = np.fromfile("/home/wanzai/桌面/oneNew/ofmap4debug/yolov8m_seg/output1", dtype=np.float32)
box_num = 1

per_box_info_size = 7 + 32
box_info_ = box_info.reshape(-1, per_box_info_size)

mask = box_info[7:7+32].reshape(-1, 32)
mask_info_ = mask_info.reshape(32, -1)
a = np.dot(mask, mask_info_)
b = (1 / (1 + np.exp(-a))).reshape(160, 160)
b = (b * 255).astype(np.uint8)
b[b < 127] = 0
b[b > 127] = 255

left = int(box_info[3] * 160)
top = int(box_info[4] * 160)
right = int(box_info[5] * 160)
botton = int(box_info[6] * 160)

new_b = np.ones((160, 160), np.uint8) * 0
new_b[top:botton, left:right] = b[top:botton, left:right]
mask_data = new_b

# draw mask
h = image_height
w = image_width

mask_resized = cv2.resize(mask_data,(w, h), fx=0, fy=0, interpolation=cv2.INTER_AREA)
# mask_resized = cv2.warpAffine(mask, IM, (w, h), flags=cv2.INTER_LINEAR)  # 1080x810

colored_mask = (np.ones((h, w, 3)) * (255, 192, 233))

mask_resized[mask_resized > 127] = 255
mask_resized[mask_resized < 127] = 0

# mask0_img = Image.fromarray(mask_resized, 'L')  # 'L'模式表示灰度图像
# mask0_img.show()

# masked_colored_mask = np.ones((h, w, 3)
masked_colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask= mask_resized)

# masked_colored_mask_ = masked_colored_mask.reshape(500,-1)
img_cv2 = cv2.imread(pathImg)

mask_indices = mask_resized == 255
img_cv2[mask_indices] = (img_cv2[mask_indices] * 0.2 + masked_colored_mask[mask_indices] * 0.8).astype(np.uint8)

# img_cv2 = (img_cv2 * 0.60 + masked_colored_mask * 0.4).astype(np.uint8)
cv2.imwrite("/home/wanzai/桌面/oneNew/imgs/dst_img/xinxin2_yolov8m_seg.jpg", img_cv2)

# # mask_indices = mask_resized == 1
# # img[mask_indices] = (img[mask_indices] * 0.6 + masked_colored_mask[mask_indices] * 0.4).astype(np.uint8)
#
# #
# # img = Image.fromarray(mask_data, 'L')  # 'L'模式表示灰度图像
# # img.show()
#
# for i in range(0, box_num):
#     # 定义方框的四个浮点值
#     (x1, y1, x2, y2) = (
#         box_info[i * per_box_info_size + 3] * image_width,
#         box_info[i * per_box_info_size + 4] * image_height,
#         box_info[i * per_box_info_size + 5] * image_width,
#         box_info[i * per_box_info_size + 6] * image_height
#     )
#     # 在原始图片上绘制方框
#     draw.rectangle([(x1, y1), (x2, y2)], outline="yellow", width=2)
#
# img.save('yolov8_dog_boxes.jpg')